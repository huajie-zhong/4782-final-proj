"""
benchmark.py

Greedy baseline generation and Medusa inference loop with KV cache management.
Supports greedy and typical acceptance criteria (Sprint 7 & 8).
"""
import time
import json
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import MedusaModel
from utils import (
    build_candidate_tree, generate_tree_mask, generate_position_ids,
    greedy_accept, typical_accept,
)


def greedy_baseline(model, tokenizer, prompts, max_new_tokens):
    """
    Measures baseline greedy decoding speed on a set of prompts.
    Returns TPS (Tokens Per Second) and logs the output.
    """
    results = []

    if len(prompts) > 0:
        inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()

        generated_tokens = outputs.shape[1] - input_len
        total_time = end_time - start_time
        tps = generated_tokens / total_time

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "tps": tps,
            "total_time": total_time,
            "generated_tokens": generated_tokens
        })
        print(f"Prompt: {prompt[:30]}... | TPS: {tps:.2f} | Time: {total_time:.2f}s")

    return results


# ---------------------------------------------------------------------------
# KV cache utilities
# ---------------------------------------------------------------------------

def _to_legacy_kv(past_kv):
    """Convert DynamicCache (transformers>=4.36) to a tuple of (key, value) pairs."""
    if past_kv is None:
        return None
    if hasattr(past_kv, "key_cache"):
        return tuple(
            (past_kv.key_cache[i], past_kv.value_cache[i])
            for i in range(len(past_kv.key_cache))
        )
    return past_kv


def _gather_kv_positions(past_kv, keep_positions_tensor):
    """Return a new past_key_values keeping only the specified sequence positions."""
    legacy = _to_legacy_kv(past_kv)
    return tuple(
        (k.index_select(2, keep_positions_tensor), v.index_select(2, keep_positions_tensor))
        for k, v in legacy
    )


def _bool_to_additive_4d(mask_bool, dtype=torch.float16):
    """
    Convert a bool mask (tree_size, total_seq_len) to a 4D additive float mask
    (1, 1, tree_size, total_seq_len) where True→0 and False→-inf.
    """
    float_mask = torch.zeros(mask_bool.shape, dtype=dtype)
    float_mask[~mask_bool] = float("-inf")
    return float_mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Medusa decode
# ---------------------------------------------------------------------------

def medusa_decode(model, tokenizer, prompt, max_new_tokens,
                  acceptance="greedy", tree_budget=64):
    """
    Full MEDUSA propose → build tree → verify → accept decoding loop.

    Args:
        model: MedusaModel instance (eval mode, on device).
        tokenizer: corresponding tokenizer.
        prompt: input string.
        max_new_tokens: maximum number of tokens to generate.
        acceptance: "greedy" or "typical".
        tree_budget: 32 or 64 tree candidates.

    Returns:
        generated_text (str): full decoded output including prompt.
        avg_acceptance_rate (float): avg extra tree tokens accepted per step.
        tps (float): total generated tokens / wall-clock seconds.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    S_K = [10, 3, 2, 2]

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated = []          # token IDs generated beyond the prompt
    acceptance_lengths = [] # extra tokens accepted from tree each step

    start_time = time.time()

    with torch.no_grad():
        # ------------------------------------------------------------------
        # Prefill: process entire prompt once
        # ------------------------------------------------------------------
        orig_logits, _, past_kv = model(input_ids, use_cache=True)
        # orig_logits: (1, prompt_len, vocab_size)

        lm_token = orig_logits[0, -1, :].argmax().item()
        generated.append(lm_token)
        # prefix_len = number of tokens in KV cache (does NOT yet include lm_token)
        prefix_len = prompt_len

        # ------------------------------------------------------------------
        # Decoding loop
        # ------------------------------------------------------------------
        while len(generated) < max_new_tokens:
            if lm_token == tokenizer.eos_token_id:
                break

            # === PROPOSAL PASS (forward lm_token, 1 token) ================
            lm_input = torch.tensor([[lm_token]], device=device)
            prop_orig_logits, prop_head_logits, past_kv = model(
                lm_input, past_key_values=past_kv, use_cache=True
            )
            # past_kv now covers prefix_len + 1 positions (lm_token appended)
            prefix_len += 1
            # prop_orig_logits: (1, 1, vocab_size) — predicts token after lm_token
            # prop_head_logits[k]: (1, 1, vocab_size) — head k's speculative prediction

            # === BUILD TREE ===============================================
            top_tokens_per_head = [
                prop_head_logits[k][0, 0, :].topk(S_K[k]).indices.unsqueeze(0)
                for k in range(len(S_K))
            ]
            tree_tokens, tree_parent_indices = build_candidate_tree(
                top_tokens_per_head, tree_budget=tree_budget
            )
            tree_size = len(tree_tokens)

            # === VERIFY PASS (all tree_size candidates in one forward) ====
            # Mask: each tree node may attend only to prefix + own ancestors
            tree_mask_bool = generate_tree_mask(tree_parent_indices, prefix_len)
            # Convert to (1, 1, tree_size, prefix_len+tree_size) additive float mask
            tree_mask_4d = _bool_to_additive_4d(tree_mask_bool, dtype=dtype).to(device)
            position_ids = generate_position_ids(tree_parent_indices, prefix_len).unsqueeze(0).to(device)

            verify_orig_logits, _, past_kv_verify = model(
                tree_tokens.unsqueeze(0).to(device),
                attention_mask=tree_mask_4d,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            # verify_orig_logits: (1, tree_size, vocab_size)

            # === ACCEPT ===================================================
            if acceptance == "greedy":
                accepted_path = greedy_accept(
                    prop_orig_logits[0, 0, :],
                    verify_orig_logits[0],
                    tree_tokens,
                    tree_parent_indices,
                )
            else:
                accepted_path = typical_accept(
                    prop_orig_logits[0, 0, :],
                    verify_orig_logits[0],
                    tree_tokens,
                    tree_parent_indices,
                )

            # === KV CACHE UPDATE ==========================================
            # past_kv_verify has prefix_len + tree_size positions.
            # Keep: prefix (0..prefix_len-1) + accepted tree nodes.
            accepted_kv_positions = [prefix_len + idx for idx in accepted_path]
            keep_positions = list(range(prefix_len)) + accepted_kv_positions
            keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=device)
            past_kv = _gather_kv_positions(past_kv_verify, keep_tensor)

            # === ADVANCE ==================================================
            accepted_tokens = [tree_tokens[idx].item() for idx in accepted_path]
            generated.extend(accepted_tokens)
            acceptance_lengths.append(len(accepted_path))
            prefix_len += len(accepted_path)

            # Next lm_token: from verify logits at last accepted node if path non-empty
            if accepted_path:
                lm_token = verify_orig_logits[0, accepted_path[-1], :].argmax().item()
            else:
                lm_token = prop_orig_logits[0, 0, :].argmax().item()
            generated.append(lm_token)

            if lm_token == tokenizer.eos_token_id:
                break

    total_time = time.time() - start_time
    tps = len(generated) / total_time if total_time > 0 else 0.0
    avg_acceptance = (sum(acceptance_lengths) / len(acceptance_lengths)
                      if acceptance_lengths else 0.0)

    # Decode prompt + generated tokens together
    all_ids = (tokenizer(prompt, return_tensors="pt").input_ids[0].tolist() + generated)
    generated_text = tokenizer.decode(all_ids, skip_special_tokens=True)

    return generated_text, avg_acceptance, tps


# ---------------------------------------------------------------------------
# Sprint 8: comparison benchmark
# ---------------------------------------------------------------------------

def run_comparison(model, tokenizer, prompts, max_new_tokens, tree_budget=64):
    """
    Run greedy vs typical acceptance on all prompts.
    Returns a dict suitable for saving as sprint8_comparison.json.
    """
    comparison = {}
    for mode in ("greedy", "typical"):
        print(f"\n--- Acceptance: {mode} ---")
        mode_results = []
        for prompt in prompts:
            text, acc_rate, tps = medusa_decode(
                model, tokenizer, prompt, max_new_tokens,
                acceptance=mode, tree_budget=tree_budget,
            )
            snippet = text[len(prompt):len(prompt) + 80].replace("\n", " ")
            print(f"  [{prompt[:30]}...] acc={acc_rate:.2f} tps={tps:.1f} | {snippet}...")
            mode_results.append({
                "prompt": prompt,
                "acceptance_rate": acc_rate,
                "tps": tps,
            })
        avg_acc = sum(r["acceptance_rate"] for r in mode_results) / len(mode_results)
        avg_tps = sum(r["tps"] for r in mode_results) / len(mode_results)
        print(f"  => avg acceptance_rate={avg_acc:.3f}  avg_tps={avg_tps:.2f}")
        comparison[f"{mode}_acceptance"] = {
            "avg_acceptance_rate": avg_acc,
            "avg_tps": avg_tps,
            "results": mode_results,
        }
    return comparison


# ---------------------------------------------------------------------------
# Full benchmark (model-agnostic)
# ---------------------------------------------------------------------------

def compute_head_accuracy(model, tokenizer, prompts, max_length=512):
    """
    Compute per-head top-1 accuracy on the evaluation prompts.
    Each head k predicts token at position t+k+2; we compare against ground-truth.
    Returns a list of floats, one per head.
    """
    device = next(model.parameters()).device
    head_correct = [0] * model.num_heads
    head_total = [0] * model.num_heads

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]
            if input_ids.shape[1] < 4:
                continue

            _, head_logits = model(input_ids)

            for k in range(model.num_heads):
                shift = k + 2
                if input_ids.shape[1] <= shift:
                    continue
                logits_k = head_logits[k][0, :-shift, :]   # (seq-shift, vocab)
                targets_k = input_ids[0, shift:]            # (seq-shift,)
                preds = logits_k.argmax(dim=-1)
                head_correct[k] += (preds == targets_k).sum().item()
                head_total[k] += targets_k.shape[0]

    return [head_correct[k] / max(1, head_total[k]) for k in range(model.num_heads)]


def run_full_benchmark(medusa_model, backbone, tokenizer, prompts, max_new_tokens,
                       output_dir, model_id="unknown", out_filename=None):
    """
    Model-agnostic full benchmark: greedy baseline vs Medusa inference.
    Collects greedy TPS, Medusa TPS, speedup ratio, acceptance rate,
    and per-head top-1 accuracy. Saves a JSON to output_dir/out_filename.

    Args:
        model_id: HuggingFace model identifier string (used for labelling output).
        out_filename: output JSON filename; defaults to <model_short_name>_benchmark.json.
    """
    if out_filename is None:
        short_name = model_id.split("/")[-1]
        out_filename = f"{short_name}_benchmark.json"
    # Greedy baseline
    print("\n--- Greedy baseline ---")
    greedy_results = greedy_baseline(backbone, tokenizer, prompts, max_new_tokens)
    greedy_tps = sum(r["tps"] for r in greedy_results) / len(greedy_results)
    print(f"Avg greedy TPS: {greedy_tps:.2f}")

    # Medusa inference (greedy acceptance, 64-node tree)
    print("\n--- Medusa inference (greedy acceptance, 64-node tree) ---")
    medusa_results = []
    for prompt in prompts:
        text, acc_rate, tps = medusa_decode(
            medusa_model, tokenizer, prompt, max_new_tokens,
            acceptance="greedy", tree_budget=64,
        )
        snippet = text[len(prompt):len(prompt) + 80].replace("\n", " ")
        print(f"  [{prompt[:30]}...] acc={acc_rate:.2f} tps={tps:.1f} | {snippet}")
        medusa_results.append({"prompt": prompt, "acceptance_rate": acc_rate, "tps": tps})

    medusa_tps = sum(r["tps"] for r in medusa_results) / len(medusa_results)
    avg_acc_rate = sum(r["acceptance_rate"] for r in medusa_results) / len(medusa_results)
    speedup = medusa_tps / greedy_tps if greedy_tps > 0 else 0.0
    print(f"Avg Medusa TPS: {medusa_tps:.2f} | Speedup: {speedup:.2f}x | Avg acceptance: {avg_acc_rate:.3f}")

    # Per-head accuracy on eval prompts
    print("\n--- Per-head accuracy (on eval prompts) ---")
    head_accs = compute_head_accuracy(medusa_model, tokenizer, prompts)
    for k, acc in enumerate(head_accs):
        print(f"  Head {k}: {acc:.3f}")

    results = {
        "model": model_id,
        "greedy_tps": greedy_tps,
        "medusa_tps": medusa_tps,
        "speedup_ratio": speedup,
        "avg_acceptance_rate": avg_acc_rate,
        "head_accuracies": {f"head_{k}": head_accs[k] for k in range(len(head_accs))},
        "greedy_results": greedy_results,
        "medusa_results": medusa_results,
    }

    out_file = os.path.join(output_dir, out_filename)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_file}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MEDUSA Benchmark")
    parser.add_argument("--mode", choices=["greedy", "medusa", "both", "full"], default="both",
                        help="greedy|medusa|both: individual runs; full: greedy+medusa+head-accuracy in one pass")
    parser.add_argument("--acceptance", choices=["greedy", "typical"], default="greedy",
                        help="Acceptance criterion for Medusa decoding")
    parser.add_argument("--tree_budget", type=int, choices=[32, 64], default=64,
                        help="Number of candidate tree nodes (32 or 64)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to medusa_heads.pt; defaults to results/medusa_heads.pt")
    parser.add_argument("--model_id", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--compare", action="store_true",
                        help="Sprint 8: compare greedy vs typical acceptance and save results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        "Write a Python script to sort a list.",
        "Explain the theory of relativity in simple terms.",
        "What are the benefits of eating healthy?",
        "Compose a short poem about the moon.",
        "Give me a step-by-step recipe for chocolate chip cookies.",
    ]

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Greedy baseline
    # ------------------------------------------------------------------
    if args.mode in ("greedy", "both"):
        print("\n=== Greedy Baseline ===")
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        results = greedy_baseline(backbone, tokenizer, prompts, args.max_new_tokens)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        print(f"Average greedy TPS: {avg_tps:.2f}")
        with open(os.path.join(output_dir, "greedy_baseline.json"), "w") as f:
            json.dump({"average_tps": avg_tps, "results": results}, f, indent=4)
        del backbone

    # ------------------------------------------------------------------
    # Medusa inference
    # ------------------------------------------------------------------
    if args.mode in ("medusa", "both"):
        print("\n=== Medusa Inference ===")
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        medusa_model = MedusaModel(backbone, num_heads=4).to(device)

        ckpt_path = args.checkpoint or os.path.join(output_dir, "medusa_heads.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            medusa_model.heads.load_state_dict(state_dict)
            print(f"Loaded head checkpoint from {ckpt_path}")
        else:
            print(f"WARNING: No checkpoint at {ckpt_path}. Using untrained (init) heads.")

        medusa_model.eval()

        if args.compare:
            # Sprint 8: greedy vs typical comparison
            comparison = run_comparison(
                medusa_model, tokenizer, prompts, args.max_new_tokens,
                tree_budget=args.tree_budget,
            )
            comparison["selected_tree_budget"] = args.tree_budget
            comparison["model_id"] = args.model_id
            out_file = os.path.join(output_dir, "sprint8_comparison.json")
            with open(out_file, "w") as f:
                json.dump(comparison, f, indent=4)
            print(f"\nComparison saved to {out_file}")
        else:
            # Single-mode run
            medusa_results = []
            for prompt in prompts:
                text, acc_rate, tps = medusa_decode(
                    medusa_model, tokenizer, prompt, args.max_new_tokens,
                    acceptance=args.acceptance, tree_budget=args.tree_budget,
                )
                snippet = text[len(prompt):len(prompt) + 100].replace("\n", " ")
                print(f"[{prompt[:30]}...] acc={acc_rate:.2f} tps={tps:.1f}")
                print(f"  >> {snippet}")
                medusa_results.append({
                    "prompt": prompt,
                    "generated_text": text,
                    "acceptance_rate": acc_rate,
                    "tps": tps,
                })

            avg_tps = sum(r["tps"] for r in medusa_results) / len(medusa_results)
            avg_acc = sum(r["acceptance_rate"] for r in medusa_results) / len(medusa_results)
            print(f"\nAverage Medusa TPS: {avg_tps:.2f}")
            print(f"Average extra tokens accepted per step: {avg_acc:.3f}")

            out_file = os.path.join(output_dir, "medusa_inference.json")
            with open(out_file, "w") as f:
                json.dump({
                    "average_tps": avg_tps,
                    "average_acceptance_rate": avg_acc,
                    "acceptance_mode": args.acceptance,
                    "tree_budget": args.tree_budget,
                    "results": medusa_results,
                }, f, indent=4)
            print(f"Results saved to {out_file}")


    # ------------------------------------------------------------------
    # Full benchmark (greedy + Medusa + head accuracy in one pass)
    # ------------------------------------------------------------------
    if args.mode == "full":
        print(f"\n=== Full Benchmark — {args.model_id} ===")
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        medusa_model = MedusaModel(backbone, num_heads=4).to(device)

        ckpt_path = args.checkpoint or os.path.join(output_dir, "medusa_heads.pt")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            medusa_model.heads.load_state_dict(state_dict)
            print(f"Loaded head checkpoint from {ckpt_path}")
        else:
            print(f"WARNING: No checkpoint at {ckpt_path}. Using untrained heads.")

        medusa_model.eval()
        run_full_benchmark(
            medusa_model, backbone, tokenizer, prompts,
            args.max_new_tokens, output_dir, model_id=args.model_id,
        )


if __name__ == "__main__":
    main()
