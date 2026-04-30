"""MEDUSA live demo — Modal.com backend with T4 GPU (primary fast endpoint).

Deploy from the final/code/ directory:
    pip install modal
    modal token new          # one-time auth
    modal deploy modal_app.py

After deploy, copy the printed URL into final/code/demo/.env.production as VITE_API_URL,
then rebuild the frontend: cd code/demo && npm run build.

The Modal Volume caches TinyLlama weights so cold starts after the first request
take ~5 s instead of ~2 min.
"""
import json
import os
import sys
import time

import modal

# ── Local path setup (works locally; in container PYTHONPATH=/app handles it) ──

_HERE = os.path.dirname(os.path.abspath(__file__))  # final/code/
_RESULTS = os.path.normpath(os.path.join(_HERE, "..", "results"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from model import MedusaModel  # noqa: E402  (final/code/model.py locally, /app/model.py in container)
from utils import (  # noqa: E402
    build_candidate_tree,
    generate_tree_mask,
    generate_position_ids,
    greedy_accept,
    typical_accept,
)

# ── Modal app ──────────────────────────────────────────────────────────────────

app = modal.App("medusa-inference")

# Volume caches HF downloads so cold starts after the first are fast
_volume = modal.Volume.from_name("medusa-tinyllama", create_if_missing=True)

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HOME": "/hf-cache", "PYTHONPATH": "/app"})
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "torch",
        "transformers>=4.40",
        "accelerate",
        "sentencepiece",
    )
    .add_local_file(os.path.join(_HERE, "model.py"), "/app/model.py")
    .add_local_file(os.path.join(_HERE, "utils.py"), "/app/utils.py")
    .add_local_file(
        os.path.join(_RESULTS, "medusa_heads_tinyllama.pt"),
        "/app/medusa_heads_tinyllama.pt",
    )
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHECKPOINT = "/app/medusa_heads_tinyllama.pt"
S_K = [10, 3, 2, 2]
MAX_TOKENS_CAP = 512

# ── Model globals — None until serve() populates them on container start ───────

_tokenizer: AutoTokenizer | None = None
_model: MedusaModel | None = None

# ── KV cache helpers (identical to space/app.py) ───────────────────────────────

def _to_legacy_kv(past_kv):
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
    legacy = _to_legacy_kv(past_kv)
    sliced = tuple(
        (k.index_select(2, keep_positions), v.index_select(2, keep_positions))
        for k, v in legacy
    )
    return _from_legacy_kv(sliced)


def _bool_to_additive_4d(mask_bool, dtype=torch.float16):
    float_mask = torch.zeros(mask_bool.shape, dtype=dtype)
    float_mask[~mask_bool] = float("-inf")
    return float_mask.unsqueeze(0).unsqueeze(0)


# ── Inference generator ────────────────────────────────────────────────────────

def _format_prompt(prompt: str) -> str:
    try:
        return _tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _run_inference(prompt: str, max_new_tokens: int, mode: str, acceptance: str, device: str):
    """Generator yielding one event dict per decoding step.
    mode: "medusa" or "base"
    """
    dtype = torch.float16
    eos_id = _tokenizer.eos_token_id
    accept_fn = typical_accept if acceptance == "typical" else greedy_accept

    # Initial encoding
    full_prompt = _format_prompt(prompt)
    input_ids = _tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    
    # We keep track of all generated token IDs to decode with context (preserves spaces)
    all_token_ids = input_ids[0].tolist()
    generated_count = 0
    acceptance_lengths: list[int] = []
    start = time.time()

    def decode_new_tokens(new_ids):
        nonlocal all_token_ids
        results = []
        for tid in new_ids:
            # Decode with full prefix to ensure correct spacing/subword merging
            prev_text = _tokenizer.decode(all_token_ids, skip_special_tokens=True)
            all_token_ids.append(tid)
            curr_text = _tokenizer.decode(all_token_ids, skip_special_tokens=True)
            results.append(curr_text[len(prev_text):])
        return results

    with torch.no_grad():
        if mode == "base":
            # Standard Autoregressive Decoding
            past_kv = None
            curr_input = input_ids
            
            step = 0
            while generated_count < max_new_tokens:
                outputs = _model.backbone(curr_input, past_key_values=past_kv, use_cache=True)
                logits = outputs.logits
                past_kv = outputs.past_key_values
                
                next_token = logits[0, -1, :].argmax().item()
                if next_token == eos_id:
                    break
                    
                step_tokens = decode_new_tokens([next_token])
                generated_count += 1
                step += 1
                curr_input = torch.tensor([[next_token]], device=device)
                
                elapsed = time.time() - start
                tps = generated_count / elapsed if elapsed > 0 else 0.0
                
                yield {
                    "type": "tokens", "step": step, 
                    "tokens": [{"text": t, "spec": False} for t in step_tokens],
                    "accepted": 0, "avg_rate": 1.0, "tps": round(tps, 1),
                    "total": generated_count,
                }
        else:
            # Medusa Speculative Decoding
            orig_logits, head_logits, past_kv = _model(input_ids, use_cache=True)
            lm_token: int = orig_logits[0, -1, :].argmax().item()
            
            # Initial LM token
            if lm_token != eos_id:
                step_tokens = decode_new_tokens([lm_token])
                generated_count += 1
                yield {
                    "type": "tokens", "step": 0, 
                    "tokens": [{"text": t, "spec": False} for t in step_tokens],
                    "accepted": 0, "avg_rate": 0.0, "tps": 0.0, "total": 1,
                }
            
            prefix_len = prompt_len
            head_preds = [head_logits[k][0, -1, :] for k in range(_model.num_heads)]

            step = 1
            while generated_count < max_new_tokens and lm_token != eos_id:
                top_per_head = [
                    head_preds[k].topk(S_K[k]).indices.unsqueeze(0)
                    for k in range(len(S_K))
                ]
                tree_tokens, tree_parent_indices = build_candidate_tree(top_per_head, tree_budget=64)
                tree_size = tree_tokens.shape[0]

                verify_input = torch.cat([
                    torch.tensor([lm_token], device=device),
                    tree_tokens.to(device),
                ]).unsqueeze(0)

                tree_mask = generate_tree_mask(tree_parent_indices, prefix_len + 1)
                lm_row = torch.zeros(prefix_len + 1 + tree_size, dtype=torch.bool)
                lm_row[: prefix_len + 1] = True
                full_mask = torch.cat([lm_row.unsqueeze(0), tree_mask], dim=0)
                full_mask_4d = _bool_to_additive_4d(full_mask, dtype=dtype).to(device)

                tree_positions = generate_position_ids(tree_parent_indices, prefix_len + 1)
                full_positions = torch.cat([
                    torch.tensor([prefix_len], dtype=torch.long), tree_positions,
                ]).unsqueeze(0).to(device)

                verify_orig, verify_heads, past_kv_verify = _model(
                    verify_input,
                    attention_mask=full_mask_4d,
                    position_ids=full_positions,
                    past_key_values=past_kv,
                    use_cache=True,
                )

                accepted_path = accept_fn(
                    verify_orig[0, 0, :], verify_orig[0, 1:, :],
                    tree_tokens, tree_parent_indices,
                )
                accepted_ids = [tree_tokens[i].item() for i in accepted_path]

                keep = list(range(prefix_len + 1)) + [prefix_len + 1 + i for i in accepted_path]
                past_kv = _gather_kv_positions(
                    past_kv_verify,
                    torch.tensor(keep, dtype=torch.long, device=device),
                )

                acceptance_lengths.append(len(accepted_ids))
                prefix_len = prefix_len + 1 + len(accepted_ids)

                has_eos = eos_id is not None and eos_id in accepted_ids
                if not has_eos:
                    last_idx = 1 + accepted_path[-1] if accepted_path else 0
                    lm_token = verify_orig[0, last_idx, :].argmax().item()
                    head_preds = [verify_heads[k][0, last_idx, :] for k in range(_model.num_heads)]
                    
                    # We yield speculative tokens then the base token
                    spec_tokens = decode_new_tokens(accepted_ids)
                    base_tokens = decode_new_tokens([lm_token])
                    
                    step_data = []
                    for t in spec_tokens: step_data.append({"text": t, "spec": True})
                    for t in base_tokens: step_data.append({"text": t, "spec": False})
                else:
                    # EOS hit in speculative tokens
                    eos_pos = accepted_ids.index(eos_id)
                    accepted_ids = accepted_ids[:eos_pos]
                    spec_tokens = decode_new_tokens(accepted_ids)
                    step_data = [{"text": t, "spec": True} for t in spec_tokens]
                    lm_token = eos_id

                generated_count += len(step_data)
                elapsed = time.time() - start
                avg_rate = sum(acceptance_lengths) / len(acceptance_lengths)
                tps = generated_count / elapsed if elapsed > 0 else 0.0

                yield {
                    "type": "tokens", "step": step, "tokens": step_data,
                    "accepted": len(accepted_ids),
                    "avg_rate": round(avg_rate, 3),
                    "tps": round(tps, 1),
                    "total": generated_count,
                }
                step += 1
                if has_eos or lm_token == eos_id:
                    break

    elapsed = time.time() - start
    tps = generated_count / elapsed if elapsed > 0 else 0.0
    avg = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
    yield {
        "type": "done",
        "total_tokens": generated_count,
        "avg_acceptance": round(avg, 3),
        "tps": round(tps, 1),
        "elapsed": round(elapsed, 2),
    }


# ── FastAPI web app ────────────────────────────────────────────────────────────

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@web_app.get("/")
async def health():
    return {"status": "ok", "backend": "modal-t4", "model": MODEL_ID}


@web_app.get("/generate")
async def generate(prompt: str, max_new_tokens: int = 64, mode: str = "medusa", acceptance: str = "typical"):
    max_new_tokens = max(4, min(int(max_new_tokens), MAX_TOKENS_CAP))

    def stream():
        for event in _run_inference(prompt, max_new_tokens, mode, acceptance, device="cuda"):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Modal endpoint ─────────────────────────────────────────────────────────────

@app.function(
    image=_image,
    gpu="T4",
    volumes={"/hf-cache": _volume},
    scaledown_window=300,
    timeout=300,
    allow_concurrent_inputs=4,
)
@modal.asgi_app()
def serve():
    """Called once per container start; loads model then returns the FastAPI app."""
    global _tokenizer, _model

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, legacy=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    _tokenizer = tok

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()
    backbone.eval()

    mdl = MedusaModel(backbone, num_heads=4)
    state = torch.load(CHECKPOINT, map_location="cuda")
    mdl.heads.load_state_dict(state)
    mdl.eval()
    _model = mdl

    print("Model ready on GPU.")
    return web_app
