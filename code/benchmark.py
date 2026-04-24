"""benchmark.py — Greedy baseline and Medusa propose/verify decoding loop."""
import argparse
import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import MedusaModel
from utils import (
    build_candidate_tree, build_linear_tree, build_naive_tree,
    generate_tree_mask, generate_position_ids,
    greedy_accept, typical_accept,
)

DEFAULT_PROMPTS = [
    "Write a Python script to sort a list.",
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of eating healthy?",
    "Compose a short poem about the moon.",
    "Give me a step-by-step recipe for chocolate chip cookies.",
]

S_K = [10, 3, 2, 2]  # top-k widths per Medusa head (paper §2.3)


# ----- KV cache utilities -------------------------------------------------

def _to_legacy_kv(past_kv):
    """Convert a Cache (transformers>=4.36) into a legacy tuple of (key, value) pairs.

    Covers three shapes that have shipped over the years:
      - already a tuple of (k, v) pairs (legacy format) — pass through;
      - DynamicCache with `to_legacy_cache()` (official API);
      - DynamicCache with `key_cache`/`value_cache` lists (4.36–4.49);
      - DynamicCache with `layers` of DynamicLayer objects (>=4.50).
    """
    if past_kv is None or isinstance(past_kv, tuple):
        return past_kv
    to_legacy = getattr(past_kv, "to_legacy_cache", None)
    if callable(to_legacy):
        legacy = to_legacy()
        if legacy is not None:
            return legacy
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        return tuple(
            (past_kv.key_cache[i], past_kv.value_cache[i])
            for i in range(len(past_kv.key_cache))
        )
    if hasattr(past_kv, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_kv.layers)
    raise TypeError(f"Unsupported past_key_values type: {type(past_kv).__name__}")


def _from_legacy_kv(legacy):
    """Wrap a legacy ((k, v), ...) tuple back into a DynamicCache.

    Recent transformers no longer auto-convert legacy tuples inside `forward`
    (LlamaModel calls `past_key_values.get_seq_length()` directly), so we
    have to round-trip through DynamicCache ourselves.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return legacy
    from_legacy = getattr(DynamicCache, "from_legacy_cache", None)
    if callable(from_legacy):
        try:
            return from_legacy(legacy)
        except Exception:
            pass
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        cache.update(k, v, layer_idx)
    return cache


def _gather_kv_positions(past_kv, keep_positions):
    """Return new past_key_values keeping only the specified sequence positions."""
    legacy = _to_legacy_kv(past_kv)
    sliced = tuple(
        (k.index_select(2, keep_positions), v.index_select(2, keep_positions))
        for k, v in legacy
    )
    return _from_legacy_kv(sliced)


def _bool_to_additive_4d(mask_bool, dtype=torch.float16):
    """(tree_size, total_len) bool → (1, 1, tree_size, total_len) additive float."""
    float_mask = torch.zeros(mask_bool.shape, dtype=dtype)
    float_mask[~mask_bool] = float("-inf")
    return float_mask.unsqueeze(0).unsqueeze(0)


def _avg(results, key):
    return sum(r[key] for r in results) / len(results) if results else 0.0


# ----- Prompt formatting --------------------------------------------------

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _format_prompt(tokenizer, prompt, model_id):
    """Wrap a raw instruction in the model's expected chat format.

    Vicuna path uses FastChat's `get_conversation_template` — the same call
    the official MEDUSA repo uses (medusa/inference/cli.py, train_legacy.py).
    Falls back to a hand-rolled Vicuna string if fschat isn't importable.
    """
    mid = (model_id or "").lower()
    if "vicuna" in mid:
        try:
            # Lightweight FastChat import: only depends on dataclasses/enum,
            # so `pip install --no-deps fschat` is enough.
            from fastchat.conversation import get_conv_template
            conv = get_conv_template("vicuna_v1.1")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        except ImportError:
            return f"{VICUNA_SYSTEM} USER: {prompt} ASSISTANT:"
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


# ----- Greedy baseline ----------------------------------------------------

def greedy_baseline(model, tokenizer, prompts, max_new_tokens, model_id=""):
    """Baseline greedy decoding via HF generate. Returns per-prompt result list."""
    if prompts:
        warmup_text = _format_prompt(tokenizer, prompts[0], model_id)
        warmup = tokenizer(warmup_text, return_tensors="pt").to(model.device)
        _ = model.generate(**warmup, max_new_tokens=10, do_sample=False)

    results = []
    for prompt in prompts:
        formatted = _format_prompt(tokenizer, prompt, model_id)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        total_time = time.time() - start

        gen_tokens = outputs.shape[1] - input_len
        tps = gen_tokens / total_time if total_time > 0 else 0.0
        gen_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        results.append({
            "prompt": prompt, "generated_text": gen_text,
            "tps": tps, "total_time": total_time, "generated_tokens": gen_tokens,
        })
        print(f"  [{prompt[:30]}...] tps={tps:.2f} time={total_time:.2f}s tokens={gen_tokens}")
    return results


# ----- Medusa decode ------------------------------------------------------

def _build_tree_for_mode(top_per_head, tree_mode, tree_budget):
    """Dispatch to the right tree builder for each Table 3 ablation mode."""
    if tree_mode == "none":
        return build_linear_tree(top_per_head)
    if tree_mode == "naive":
        return build_naive_tree(top_per_head)
    return build_candidate_tree(top_per_head, tree_budget=tree_budget)


def medusa_decode(model, tokenizer, prompt, max_new_tokens,
                  acceptance="greedy", tree_budget=64, tree_mode="optimized",
                  design="paper", model_id=""):
    """MEDUSA propose → tree → verify → accept loop with explicit KV surgery.

    Two propose/verify structures are supported; both use all K trained heads
    at tree depths 1..K:

      design="paper" (default, paper-faithful, §2.3 / Figure 6):
        Head predictions are read from the hidden state of the last committed
        token (prefill's last position, then verify's last-accepted position).
        One forward per step (verify only), which also advances lm_token
        into the KV cache.

      design="extra_forward" (stretch-goal alternative):
        Separate proposal forward on lm_token alone gives head logits used
        to build the tree. The base-LM argmax from that proposal is
        committed as an unconditional bonus token (lm_token_2), and the
        verify pass forwards [lm_token_2, tree_tokens]. Two forwards per
        step; guaranteed ≥2 new tokens per step before tree acceptance.

    Returns (generated_text, avg_extra_accepted_per_step, tps).
    """
    if design == "paper":
        return _medusa_decode_paper(
            model, tokenizer, prompt, max_new_tokens,
            acceptance=acceptance, tree_budget=tree_budget, tree_mode=tree_mode,
            model_id=model_id,
        )
    if design == "extra_forward":
        return _medusa_decode_extra_forward(
            model, tokenizer, prompt, max_new_tokens,
            acceptance=acceptance, tree_budget=tree_budget, tree_mode=tree_mode,
            model_id=model_id,
        )
    raise ValueError(f"Unknown design={design!r}; expected 'paper' or 'extra_forward'")


def _medusa_decode_paper(model, tokenizer, prompt, max_new_tokens,
                         acceptance="greedy", tree_budget=64, tree_mode="optimized",
                         model_id=""):
    """Paper-faithful Medusa decode. One forward per step; all K heads used.

    Invariant at the top of each iteration:
        `past_kv` holds positions [0..prefix_len-1]. `lm_token` is committed
        (appended to `generated`) but not yet in `past_kv`. `head_preds[k]` are
        head k's logits from the hidden state at position prefix_len-1, i.e.
        they predict position prefix_len + k + 1 — which is tree depth k+1.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eos_id = tokenizer.eos_token_id
    accept_fn = typical_accept if acceptance == "typical" else greedy_accept

    formatted = _format_prompt(tokenizer, prompt, model_id)
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated = []
    acceptance_lengths = []

    start = time.time()
    with torch.no_grad():
        # Prefill — KEEP head logits (seeds the first tree's proposals).
        orig_logits, head_logits, past_kv = model(input_ids, use_cache=True)
        lm_token = orig_logits[0, -1, :].argmax().item()
        generated.append(lm_token)
        prefix_len = prompt_len  # past_kv excludes lm_token
        # Head k at hidden position L-1 predicts token at L+k+1 → tree depth k+1.
        head_preds = [head_logits[k][0, -1, :] for k in range(model.num_heads)]

        while len(generated) < max_new_tokens and lm_token != eos_id:
            top_per_head = [
                head_preds[k].topk(S_K[k]).indices.unsqueeze(0)
                for k in range(len(S_K))
            ]
            tree_tokens, tree_parent_indices = _build_tree_for_mode(
                top_per_head, tree_mode, tree_budget,
            )
            tree_size = tree_tokens.shape[0]

            # Verify input = [lm_token, tree_tokens]. lm_token lands at position
            # prefix_len; tree depth-d lands at prefix_len + d.
            verify_input = torch.cat([
                torch.tensor([lm_token], device=device),
                tree_tokens.to(device),
            ]).unsqueeze(0)

            # Attention mask: lm_token row attends to the real prefix + itself.
            # Tree rows attend to (prefix + lm_token) as the "prefix" and ancestors.
            tree_mask = generate_tree_mask(tree_parent_indices, prefix_len + 1)
            lm_row = torch.zeros(prefix_len + 1 + tree_size, dtype=torch.bool)
            lm_row[: prefix_len + 1] = True
            full_mask = torch.cat([lm_row.unsqueeze(0), tree_mask], dim=0)
            full_mask_4d = _bool_to_additive_4d(full_mask, dtype=dtype).to(device)

            tree_positions = generate_position_ids(tree_parent_indices, prefix_len + 1)
            full_positions = torch.cat([
                torch.tensor([prefix_len], dtype=torch.long),
                tree_positions,
            ]).unsqueeze(0).to(device)

            verify_orig_logits, verify_head_logits, past_kv_verify = model(
                verify_input,
                attention_mask=full_mask_4d,
                position_ids=full_positions,
                past_key_values=past_kv,
                use_cache=True,
            )

            # lm_token's LM prediction checks depth-1 roots; tree-node
            # predictions (indices 1..tree_size) check their children.
            prop_like = verify_orig_logits[0, 0, :]
            tree_verify = verify_orig_logits[0, 1:, :]
            accepted_path = accept_fn(
                prop_like, tree_verify, tree_tokens, tree_parent_indices,
            )
            accepted_tokens = [tree_tokens[i].item() for i in accepted_path]

            # KV surgery: keep [prefix + lm_token] + accepted tree positions.
            keep = list(range(prefix_len + 1)) + [
                prefix_len + 1 + i for i in accepted_path
            ]
            past_kv = _gather_kv_positions(
                past_kv_verify,
                torch.tensor(keep, dtype=torch.long, device=device),
            )

            generated.extend(accepted_tokens)
            acceptance_lengths.append(len(accepted_path))
            prefix_len = prefix_len + 1 + len(accepted_path)

            if eos_id is not None and eos_id in accepted_tokens:
                break

            # Re-seed lm_token and head_preds from the hidden state at the last
            # committed position. Verify-output index 0 = lm_token; 1+i = tree node i.
            last_idx = 1 + accepted_path[-1] if accepted_path else 0
            lm_token = verify_orig_logits[0, last_idx, :].argmax().item()
            head_preds = [
                verify_head_logits[k][0, last_idx, :] for k in range(model.num_heads)
            ]
            generated.append(lm_token)

    total_time = time.time() - start
    tps = len(generated) / total_time if total_time > 0 else 0.0
    avg_acceptance = (sum(acceptance_lengths) / len(acceptance_lengths)
                      if acceptance_lengths else 0.0)

    if eos_id is not None and eos_id in generated:
        generated = generated[: generated.index(eos_id) + 1]
    generated = generated[:max_new_tokens]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, avg_acceptance, tps


def _medusa_decode_extra_forward(model, tokenizer, prompt, max_new_tokens,
                                 acceptance="greedy", tree_budget=64,
                                 tree_mode="optimized", model_id=""):
    """Stretch-goal alternative: dedicated proposal forward, guaranteed bonus token.

    Two forwards per step:
      1. Proposal forward (`lm_token` alone) → prop_orig_logits + prop_head_logits
         from the hidden state at `lm_token`'s position.
      2. Verify forward (`[lm_token_2, tree_tokens]`) where
         `lm_token_2 = argmax(prop_orig_logits)` is committed unconditionally.

    The tree roots at position `prefix_len + 1` (one past lm_token_2) and its
    4 depths are sourced from heads 0-3 (all K heads). Each step guarantees
    at least 2 new tokens (lm_token_2 + next lm_token from verify) plus any
    accepted tree tokens.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    eos_id = tokenizer.eos_token_id
    accept_fn = typical_accept if acceptance == "typical" else greedy_accept

    formatted = _format_prompt(tokenizer, prompt, model_id)
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated = []
    acceptance_lengths = []

    start = time.time()
    with torch.no_grad():
        # Prefill
        orig_logits, _, past_kv = model(input_ids, use_cache=True)
        lm_token = orig_logits[0, -1, :].argmax().item()
        generated.append(lm_token)
        prefix_len = prompt_len  # tokens currently in past_kv (excludes lm_token)

        while len(generated) < max_new_tokens and lm_token != eos_id:
            # Proposal pass: forward lm_token alone.
            lm_input = torch.tensor([[lm_token]], device=device)
            prop_orig_logits, prop_head_logits, past_kv = model(
                lm_input, past_key_values=past_kv, use_cache=True,
            )
            prefix_len += 1  # past_kv now includes lm_token

            # Guaranteed bonus: base-LM argmax at lm_token's hidden state.
            # This will be committed unconditionally at position prefix_len.
            lm_token_2 = prop_orig_logits[0, 0, :].argmax().item()

            # Tree uses all K heads. Head k at lm_token's hidden state predicts
            # position prefix_len + k + 1 — so tree depth-1 at prefix_len+1 is
            # head 0, depth-4 at prefix_len+4 is head 3.
            top_per_head = [
                prop_head_logits[k][0, 0, :].topk(S_K[k]).indices.unsqueeze(0)
                for k in range(len(S_K))
            ]
            tree_tokens, tree_parent_indices = _build_tree_for_mode(
                top_per_head, tree_mode, tree_budget,
            )
            tree_size = tree_tokens.shape[0]

            # Verify input = [lm_token_2, tree_tokens]. lm_token_2 at position
            # prefix_len; tree depth-d at prefix_len + d.
            verify_input = torch.cat([
                torch.tensor([lm_token_2], device=device),
                tree_tokens.to(device),
            ]).unsqueeze(0)

            tree_mask = generate_tree_mask(tree_parent_indices, prefix_len + 1)
            lm_row = torch.zeros(prefix_len + 1 + tree_size, dtype=torch.bool)
            lm_row[: prefix_len + 1] = True
            full_mask = torch.cat([lm_row.unsqueeze(0), tree_mask], dim=0)
            full_mask_4d = _bool_to_additive_4d(full_mask, dtype=dtype).to(device)

            tree_positions = generate_position_ids(tree_parent_indices, prefix_len + 1)
            full_positions = torch.cat([
                torch.tensor([prefix_len], dtype=torch.long),
                tree_positions,
            ]).unsqueeze(0).to(device)

            verify_orig_logits, _, past_kv_verify = model(
                verify_input,
                attention_mask=full_mask_4d,
                position_ids=full_positions,
                past_key_values=past_kv,
                use_cache=True,
            )

            # Accept: lm_token_2 always accepted. Depth-1 roots checked against
            # lm_token_2's LM prediction; tree nodes check their children.
            prop_like = verify_orig_logits[0, 0, :]
            tree_verify = verify_orig_logits[0, 1:, :]
            accepted_path = accept_fn(
                prop_like, tree_verify, tree_tokens, tree_parent_indices,
            )
            accepted_tokens = [tree_tokens[i].item() for i in accepted_path]

            # KV surgery: keep prefix + lm_token_2 + accepted tree positions.
            keep = list(range(prefix_len + 1)) + [
                prefix_len + 1 + i for i in accepted_path
            ]
            past_kv = _gather_kv_positions(
                past_kv_verify,
                torch.tensor(keep, dtype=torch.long, device=device),
            )

            generated.append(lm_token_2)
            generated.extend(accepted_tokens)
            acceptance_lengths.append(len(accepted_path))
            prefix_len = prefix_len + 1 + len(accepted_path)

            if eos_id is not None and (lm_token_2 == eos_id or eos_id in accepted_tokens):
                break

            # Next lm_token: base LM at the last committed verify position.
            last_idx = 1 + accepted_path[-1] if accepted_path else 0
            lm_token = verify_orig_logits[0, last_idx, :].argmax().item()
            generated.append(lm_token)

    total_time = time.time() - start
    tps = len(generated) / total_time if total_time > 0 else 0.0
    avg_acceptance = (sum(acceptance_lengths) / len(acceptance_lengths)
                      if acceptance_lengths else 0.0)

    # Trim output: stop at first EOS, then cap at max_new_tokens
    if eos_id is not None and eos_id in generated:
        generated = generated[: generated.index(eos_id) + 1]
    generated = generated[:max_new_tokens]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, avg_acceptance, tps


# ----- Aggregation --------------------------------------------------------

def run_medusa(model, tokenizer, prompts, max_new_tokens, acceptance, tree_budget,
               tree_mode="optimized", design="paper", model_id=""):
    """Run medusa_decode on each prompt and return a per-prompt result list."""
    results = []
    for prompt in prompts:
        text, acc_rate, tps = medusa_decode(
            model, tokenizer, prompt, max_new_tokens,
            acceptance=acceptance, tree_budget=tree_budget, tree_mode=tree_mode,
            design=design, model_id=model_id,
        )
        snippet = text[:80].replace("\n", " ")
        print(f"  [{prompt[:30]}...] acc={acc_rate:.2f} tps={tps:.1f} | {snippet}")
        results.append({
            "prompt": prompt, "generated_text": text,
            "acceptance_rate": acc_rate, "tps": tps,
        })
    return results


def run_comparison(model, tokenizer, prompts, max_new_tokens, tree_budget=64,
                   design="paper", model_id=""):
    """Greedy vs typical acceptance comparison."""
    out = {}
    for mode in ("greedy", "typical"):
        print(f"\n--- Acceptance: {mode} ---")
        results = run_medusa(
            model, tokenizer, prompts, max_new_tokens, mode, tree_budget,
            design=design, model_id=model_id,
        )
        avg_acc = _avg(results, "acceptance_rate")
        avg_tps = _avg(results, "tps")
        print(f"  => avg acceptance_rate={avg_acc:.3f}  avg_tps={avg_tps:.2f}")
        out[f"{mode}_acceptance"] = {
            "avg_acceptance_rate": avg_acc, "avg_tps": avg_tps, "results": results,
        }
    return out


def run_table3_ablation(medusa_model, tokenizer, prompts, max_new_tokens,
                        output_dir, model_id, acceptance="greedy",
                        design="paper"):
    """Table 3 ablation (paper §3.2): measures speedup for three tree modes
    sharing one greedy baseline. Saves `table3_<model_short>.json`.

    Rows (relative to greedy):
      - none:      heads-only, linear chain (paper Row 1, ~1.5x target)
      - naive:     full 220-node Cartesian product (paper Row 2, ~1.9x target)
      - optimized: 64-node pruned tree (paper Row 3, ~2.2x target)
    """
    print("\n--- Greedy baseline (shared across ablation rows) ---")
    greedy_results = greedy_baseline(
        medusa_model.backbone, tokenizer, prompts, max_new_tokens, model_id=model_id,
    )
    greedy_tps = _avg(greedy_results, "tps")
    print(f"Avg greedy TPS: {greedy_tps:.2f}")

    rows = {}
    for mode in ("none", "naive", "optimized"):
        print(f"\n--- Medusa tree={mode} ---")
        results = run_medusa(
            medusa_model, tokenizer, prompts, max_new_tokens,
            acceptance=acceptance, tree_budget=64, tree_mode=mode,
            design=design, model_id=model_id,
        )
        medusa_tps = _avg(results, "tps")
        avg_acc = _avg(results, "acceptance_rate")
        speedup = medusa_tps / greedy_tps if greedy_tps > 0 else 0.0
        print(f"  => tps={medusa_tps:.2f}  speedup={speedup:.2f}x  acc_rate={avg_acc:.3f}")
        rows[mode] = {
            "medusa_tps": medusa_tps,
            "speedup": speedup,
            "avg_acceptance_rate": avg_acc,
            "results": results,
        }

    short = model_id.split("/")[-1]
    out_file = os.path.join(output_dir, f"table3_{short}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": model_id,
            "acceptance": acceptance,
            "design": design,
            "greedy_tps": greedy_tps,
            "greedy_results": greedy_results,
            "rows": rows,
        }, f, indent=4)
    print(f"\nTable 3 ablation saved to {out_file}")
    return {"greedy_tps": greedy_tps, "rows": rows}


def compute_head_accuracy(model, tokenizer, prompts, max_length=512, model_id=""):
    """Per-head top-1 accuracy measured over *generated* assistant-turn tokens.

    Generates 64 tokens per prompt (greedy, on-distribution with training), then
    runs a single forward over prompt+generation and measures each head's
    prediction accuracy starting from the last prompt position.
    Head k at hidden position t predicts token t+k+2.
    """
    device = next(model.parameters()).device
    correct = [0] * model.num_heads
    total = [0] * model.num_heads
    gen_tokens = 64

    with torch.no_grad():
        for prompt in prompts:
            formatted = _format_prompt(tokenizer, prompt, model_id)
            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
            prompt_len = input_ids.shape[1]

            # Generate assistant-turn tokens — same distribution the heads are trained on.
            gen_ids = model.backbone.generate(
                input_ids, max_new_tokens=gen_tokens, do_sample=False, use_cache=True,
            )
            gen_ids = gen_ids[:, :max_length]
            if gen_ids.shape[1] <= prompt_len + 4:
                continue

            # Single forward over prompt + generated sequence.
            _, head_logits = model(gen_ids)

            seq_len = gen_ids.shape[1]
            for k in range(model.num_heads):
                shift = k + 2
                # Measure from the last prompt position onward so targets are generated tokens.
                start_pos = prompt_len - 1
                end_pos = seq_len - shift
                if start_pos >= end_pos:
                    continue
                preds = head_logits[k][0, start_pos:end_pos, :].argmax(dim=-1)
                targets = gen_ids[0, start_pos + shift:end_pos + shift]
                correct[k] += (preds == targets).sum().item()
                total[k] += targets.shape[0]

    return [correct[k] / max(1, total[k]) for k in range(model.num_heads)]


def run_full_benchmark(medusa_model, tokenizer, prompts, max_new_tokens,
                       output_dir, model_id, out_filename=None,
                       design="paper"):
    """Greedy baseline + Medusa + per-head accuracy in one pass; saves JSON."""
    if out_filename is None:
        out_filename = f"{model_id.split('/')[-1]}_benchmark.json"

    print("\n--- Greedy baseline ---")
    greedy_results = greedy_baseline(
        medusa_model.backbone, tokenizer, prompts, max_new_tokens, model_id=model_id,
    )
    greedy_tps = _avg(greedy_results, "tps")
    print(f"Avg greedy TPS: {greedy_tps:.2f}")

    print(f"\n--- Medusa inference (greedy acceptance, 64-node tree, design={design}) ---")
    medusa_results_greedy = run_medusa(
        medusa_model, tokenizer, prompts, max_new_tokens, "greedy", 64,
        design=design, model_id=model_id,
    )
    greedy_medusa_tps = _avg(medusa_results_greedy, "tps")
    greedy_avg_acc = _avg(medusa_results_greedy, "acceptance_rate")
    greedy_speedup = greedy_medusa_tps / greedy_tps if greedy_tps > 0 else 0.0
    print(f"Avg Medusa TPS: {greedy_medusa_tps:.2f} | Speedup: {greedy_speedup:.2f}x | "
          f"Avg acceptance: {greedy_avg_acc:.3f}")

    print(f"\n--- Medusa inference (typical acceptance, 64-node tree, design={design}) ---")
    medusa_results_typical = run_medusa(
        medusa_model, tokenizer, prompts, max_new_tokens, "typical", 64,
        design=design, model_id=model_id,
    )
    typical_medusa_tps = _avg(medusa_results_typical, "tps")
    typical_avg_acc = _avg(medusa_results_typical, "acceptance_rate")
    typical_speedup = typical_medusa_tps / greedy_tps if greedy_tps > 0 else 0.0
    print(f"Avg Medusa TPS: {typical_medusa_tps:.2f} | Speedup: {typical_speedup:.2f}x | "
          f"Avg acceptance: {typical_avg_acc:.3f}")

    print("\n--- Per-head accuracy ---")
    head_accs = compute_head_accuracy(medusa_model, tokenizer, prompts, model_id=model_id)
    for k, a in enumerate(head_accs):
        print(f"  Head {k}: {a:.3f}")

    results = {
        "model": model_id,
        "design": design,
        "greedy_tps": greedy_tps,
        # Greedy acceptance
        "medusa_tps": greedy_medusa_tps,
        "speedup_ratio": greedy_speedup,
        "avg_acceptance_rate": greedy_avg_acc,
        # Typical acceptance (paper-comparable)
        "medusa_typical_tps": typical_medusa_tps,
        "speedup_ratio_typical": typical_speedup,
        "avg_acceptance_rate_typical": typical_avg_acc,
        "head_accuracies": {f"head_{k}": a for k, a in enumerate(head_accs)},
        "greedy_results": medusa_results_greedy,
        "typical_results": medusa_results_typical,
    }
    out_file = os.path.join(output_dir, out_filename)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")
    return results


# ----- Model loading ------------------------------------------------------

def load_backbone(model_id, device, quantize=False):
    if quantize:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            attn_implementation="eager",
            device_map={"": device},
        )
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="eager",
        ).to(device)
    backbone.eval()
    return backbone


def load_medusa(backbone, device, checkpoint_path):
    """Wrap backbone with Medusa heads and load checkpoint if present."""
    medusa_model = MedusaModel(backbone, num_heads=4)
    # If the backbone was loaded in 4-bit, bitsandbytes forbids .to() on it.
    # Detect via getattr and only move the heads in that case.
    is_quantized = getattr(backbone, "is_quantized", False) or getattr(backbone, "is_loaded_in_4bit", False)
    if is_quantized:
        medusa_model.heads.to(device)
    else:
        medusa_model.to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        medusa_model.heads.load_state_dict(state_dict)
        print(f"Loaded head checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. Using untrained heads.")
    medusa_model.eval()
    return medusa_model


# ----- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MEDUSA Benchmark")
    parser.add_argument("--mode", choices=["greedy", "medusa", "both", "full", "table3"],
                        default="both",
                        help="greedy|medusa|both: individual runs; "
                             "full: greedy+medusa+head-accuracy in one pass; "
                             "table3: Table 3 ablation across tree modes")
    parser.add_argument("--acceptance", choices=["greedy", "typical"], default="greedy",
                        help="Acceptance criterion for Medusa decoding")
    parser.add_argument("--tree", choices=["optimized", "naive", "none"], default="optimized",
                        help="Tree topology: optimized (64-node pruned), naive "
                             "(220-node full Cartesian), none (linear heads-only chain). "
                             "Covers Table 3 Row 3 / Row 2 / Row 1 respectively.")
    parser.add_argument("--tree_budget", type=int, choices=[32, 64], default=64,
                        help="Optimized-tree node budget (32 or 64). Ignored for --tree naive/none.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to medusa_heads.pt; defaults to results/medusa_heads.pt")
    parser.add_argument("--model_id", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--compare", action="store_true",
                        help="With --mode medusa: run greedy vs typical and save comparison")
    parser.add_argument("--quantize", action="store_true",
                        help="Load backbone in 4-bit (bitsandbytes nf4). Required for Vicuna-7B on 24 GB GPUs.")
    parser.add_argument("--design", choices=["paper", "extra_forward"], default="paper",
                        help="Propose/verify structure. 'paper' = single verify forward, "
                             "heads 0-K-1 at tree depths 1-K (§2.3 / Figure 6). "
                             "'extra_forward' = stretch-goal variant with a separate proposal "
                             "forward and a guaranteed bonus token per step (still uses all heads).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Llama-family models: prefer the slow tokenizer for decoding. The fast
    # tokenizer (`LlamaTokenizerFast`) has a known issue where decoded text
    # keeps the SentencePiece word-boundary marker (▁) and inserts spaces
    # between every token ("▁Sure , ▁here ' s ▁a"). The slow tokenizer routes
    # decode through sentencepiece directly and reassembles cleanly.
    is_llama_family = any(
        tag in args.model_id.lower()
        for tag in ("llama", "vicuna", "tinyllama", "alpaca", "wizard")
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, use_fast=not is_llama_family, legacy=False,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = args.checkpoint or os.path.join(output_dir, "medusa_heads.pt")

    prompts = DEFAULT_PROMPTS
    backbone = load_backbone(args.model_id, device, quantize=args.quantize)

    if args.mode == "full":
        print(f"\n=== Full Benchmark — {args.model_id} ===")
        medusa_model = load_medusa(backbone, device, ckpt_path)
        run_full_benchmark(
            medusa_model, tokenizer, prompts,
            args.max_new_tokens, output_dir, model_id=args.model_id,
            design=args.design,
        )
        return

    if args.mode == "table3":
        print(f"\n=== Table 3 Ablation — {args.model_id} ===")
        medusa_model = load_medusa(backbone, device, ckpt_path)
        run_table3_ablation(
            medusa_model, tokenizer, prompts, args.max_new_tokens,
            output_dir, model_id=args.model_id, acceptance=args.acceptance,
            design=args.design,
        )
        return

    if args.mode in ("greedy", "both"):
        print("\n=== Greedy Baseline ===")
        results = greedy_baseline(
            backbone, tokenizer, prompts, args.max_new_tokens, model_id=args.model_id,
        )
        avg_tps = _avg(results, "tps")
        print(f"Average greedy TPS: {avg_tps:.2f}")
        with open(os.path.join(output_dir, "greedy_baseline.json"), "w") as f:
            json.dump({"average_tps": avg_tps, "results": results}, f, indent=4)

    if args.mode in ("medusa", "both"):
        print("\n=== Medusa Inference ===")
        medusa_model = load_medusa(backbone, device, ckpt_path)

        if args.compare:
            comparison = run_comparison(
                medusa_model, tokenizer, prompts, args.max_new_tokens,
                tree_budget=args.tree_budget, design=args.design,
                model_id=args.model_id,
            )
            comparison["selected_tree_budget"] = args.tree_budget
            comparison["tree_mode"] = args.tree
            comparison["model_id"] = args.model_id
            comparison["design"] = args.design
            out_file = os.path.join(output_dir, "comparison.json")
            with open(out_file, "w") as f:
                json.dump(comparison, f, indent=4)
            print(f"\nComparison saved to {out_file}")
        else:
            results = run_medusa(
                medusa_model, tokenizer, prompts, args.max_new_tokens,
                args.acceptance, args.tree_budget, tree_mode=args.tree,
                design=args.design, model_id=args.model_id,
            )
            avg_tps = _avg(results, "tps")
            avg_acc = _avg(results, "acceptance_rate")
            print(f"\nAverage Medusa TPS: {avg_tps:.2f}")
            print(f"Average extra tokens accepted per step: {avg_acc:.3f}")
            out_file = os.path.join(output_dir, "medusa_inference.json")
            with open(out_file, "w") as f:
                json.dump({
                    "average_tps": avg_tps,
                    "average_acceptance_rate": avg_acc,
                    "acceptance_mode": args.acceptance,
                    "tree_mode": args.tree,
                    "tree_budget": args.tree_budget,
                    "design": args.design,
                    "results": results,
                }, f, indent=4)
            print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
