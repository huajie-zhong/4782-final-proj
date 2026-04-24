"""
train.py

Handles the parameter-efficient fine-tuning (PEFT) of the Medusa heads while
keeping the backbone model frozen. Implements label shifting and weighted loss
for the different heads.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from model import MedusaModel
from data_utils import download_and_preprocess, MedusaDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer and model
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Mixed precision: bf16 backbone (matches the reference Medusa-1 train_legacy.py
    # which loads with torch_dtype=bfloat16) + fp32 heads so AdamW state and loss
    # accumulation stay in fp32. Compute runs under autocast(bf16).
    compute_dtype = torch.bfloat16
    head_dtype = torch.float32

    # SDPA at training time: fused scaled-dot-product attention. Mathematically
    # equivalent to eager (same softmax, same math), ~1.3-1.8× faster on long seq.
    # CLAUDE.md note: eager is only required at *inference* for the tree-attention
    # 4D mask (benchmark.py), so training can safely use SDPA.
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        model = MedusaModel(backbone, num_heads=4, head_dtype=head_dtype)
        # Backbone is already placed by device_map; only move heads (bnb forbids .to on the whole model).
        model.heads.to(device)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=compute_dtype, attn_implementation="sdpa"
        )
        model = MedusaModel(backbone, num_heads=4, head_dtype=head_dtype)
        model.to(device)

    # Optional torch.compile. Falls through cleanly on older torch / graph-break
    # failures so training still runs if compile misbehaves.
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile failed ({e}); running eager.")

    # Verify what is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # 2. Data loading
    print("Preparing data...")
    tokenized_data = download_and_preprocess(
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        model_name=args.model_name,
    )
    dataset = MedusaDataset(tokenized_data, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)

    # Split train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Optimizer and Scheduler
    optimizer = AdamW(model.heads.parameters(), lr=args.lr)
    
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )

    # Lambda weights: λ_k = 0.8^k with k starting at 1 (paper §2.2.1, 1-indexed sum).
    # K=4 heads → [0.8, 0.64, 0.512, 0.4096].
    lambda_weights = [0.8**(k + 1) for k in range(model.num_heads)]
    print(f"Loss weights per head: {lambda_weights}")

    # 4. Training Loop
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Accumulators stay on GPU until log time to avoid per-step .item() syncs.
    use_autocast = (device.type == "cuda")
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=compute_dtype)) \
                   if use_autocast else (lambda: torch.cuda.amp.autocast(enabled=False))

    for epoch in range(args.epochs):
        model.train()
        head_acc_sums = torch.zeros(model.num_heads, device=device)
        total_tokens = torch.zeros((), device=device, dtype=torch.long)
        last_loss_scalar = torch.zeros((), device=device)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            # AND of attention (drops padding) and loss_mask (assistant-only).
            loss_mask = (batch["loss_mask"].to(device, non_blocking=True) & attention_mask)

            # Skip original_logits — it's unused during training and costs a full
            # (B, T, vocab) matmul through the backbone's LM head per step.
            with autocast_ctx():
                _, head_logits = model(
                    input_ids, attention_mask=attention_mask, compute_original_logits=False
                )

                loss = input_ids.new_zeros((), dtype=torch.float32)

                for k in range(model.num_heads):
                    # Label shifting: Head k predicts t + k + 2
                    shift = k + 2

                    logits_k = head_logits[k][:, :-shift, :].contiguous().view(-1, head_logits[k].size(-1))
                    targets_k = input_ids[:, shift:].contiguous().view(-1)
                    mask_k = loss_mask[:, shift:].contiguous().view(-1)

                    valid_indices = mask_k == 1
                    logits_k = logits_k[valid_indices]
                    targets_k = targets_k[valid_indices]

                    if targets_k.numel() == 0:
                        continue

                    # cross_entropy is autocast-promoted to fp32 for stability.
                    ce_loss = F.cross_entropy(logits_k, targets_k)
                    loss = loss + lambda_weights[k] * ce_loss

                    # Accumulate accuracy on-device; no .item() per step.
                    with torch.no_grad():
                        preds = torch.argmax(logits_k, dim=-1)
                        head_acc_sums[k] += (preds == targets_k).sum()
                        if k == 0:
                            total_tokens += targets_k.numel()

            loss = loss / args.grad_accum_steps
            loss.backward()
            last_loss_scalar = loss.detach() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % args.log_interval == 0:
                # Single sync point per log interval.
                denom = total_tokens.clamp(min=1).to(torch.float32)
                avg_accs = (head_acc_sums / denom).tolist()
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {last_loss_scalar.item():.4f} | "
                      f"Head Acc: {[f'{a:.3f}' for a in avg_accs]}")

        # Validation
        model.eval()
        val_head_accs = torch.zeros(model.num_heads, device=device)
        val_tokens = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad(), autocast_ctx():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                loss_mask = (batch["loss_mask"].to(device, non_blocking=True) & attention_mask)

                _, head_logits = model(
                    input_ids, attention_mask=attention_mask, compute_original_logits=False
                )

                for k in range(model.num_heads):
                    shift = k + 2
                    logits_k = head_logits[k][:, :-shift, :].contiguous().view(-1, head_logits[k].size(-1))
                    targets_k = input_ids[:, shift:].contiguous().view(-1)
                    mask_k = loss_mask[:, shift:].contiguous().view(-1)

                    valid_indices = mask_k == 1
                    logits_k = logits_k[valid_indices]
                    targets_k = targets_k[valid_indices]

                    if targets_k.numel() > 0:
                        preds = torch.argmax(logits_k, dim=-1)
                        val_head_accs[k] += (preds == targets_k).sum()
                        if k == 0:
                            val_tokens += targets_k.numel()

        denom = val_tokens.clamp(min=1).to(torch.float32)
        val_accs = (val_head_accs / denom).tolist()
        print(f"--- Epoch {epoch+1} Validation ---")
        print(f"Head Accuracies: {[f'{a:.3f}' for a in val_accs]}")

    # Save Checkpoint
    print(f"Saving Medusa heads to {args.save_path}")
    torch.save(model.heads.state_dict(), args.save_path)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medusa Heads")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # 1000 = smoke-test default; paper uses ~60k. Pass --max_samples 60000 for the full run.
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of ShareGPT samples to use (1000=smoke test, 60000=paper-scale)")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    # NOTE: effective batch size (batch_size * grad_accum_steps) is the paper-relevant
    # knob for optimizer dynamics — the paper uses 32. Keep the product at 32 across any
    # override (e.g. 8×4, 16×2, 32×1) so reproducibility is not affected.
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between logging")
    parser.add_argument("--save_path", type=str, default="../results/medusa_heads.pt", help="Path to save weights")
    parser.add_argument("--quantize", action="store_true", help="Load backbone in 4-bit (bitsandbytes nf4). Required for Vicuna-7B on 24 GB GPUs; on H100 80GB prefer bf16 (faster).")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (0 disables multiprocessing).")
    parser.add_argument("--compile", action="store_true", help="torch.compile the MedusaModel. Falls back to eager on failure.")

    args = parser.parse_args()
    
    # Adjust save_path to be relative to the script location or working dir
    # Usually running from final/code, so ../results/ is final/results/
    if not os.path.isabs(args.save_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.save_path = os.path.join(base_dir, "results", os.path.basename(args.save_path))
        
    train(args)
