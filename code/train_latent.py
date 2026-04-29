"""
train_latent.py

Unified training script that supports both original MedusaModel and the
MLA-inspired LatentMedusaModel.  All original training behaviour is preserved
when --head_type=original (the default).  The --head_type=latent flag switches
to the latent bottleneck heads with configurable d_latent and per-head dims.

The --flop_equiv flag adjusts the number of epochs so that total training FLOPs
approximately match between original and latent heads (since the bottleneck
reduces per-step compute, more steps are "free" within the same FLOP budget).
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from model import MedusaModel
from latent_model import LatentMedusaModel, compute_flop_ratio
from data_utils import download_and_preprocess, MedusaDataset
import time


def profile_baseline(args, backbone, device, head_dtype, loader):
    """Briefly profile the original MedusaModel to get time-per-step baseline."""
    print("\n--- Profiling Original Medusa baseline for time-equivalence ---")
    from model import MedusaHead

    # Extract the original LM head weights to initialize the dummy head
    if hasattr(backbone, "lm_head"):
        lm_head = backbone.lm_head
    elif hasattr(backbone, "embed_out"):
        lm_head = backbone.embed_out
    else:
        raise ValueError("Could not find the language modeling head in the backbone.")

    # Build just ONE standard head temporarily to simulate the computation 4 times
    # This saves massive VRAM compared to instantiating 4 heads with vocab_size weights
    orig_head = MedusaHead(backbone.config.hidden_size, backbone.config.vocab_size, lm_head.weight.data, head_dtype=head_dtype).to(device)

    # Use SGD for profiling to avoid Adam overhead (moments), we only need time-per-step
    optimizer = torch.optim.SGD(orig_head.parameters(), lr=args.lr)

    # Identify the base model for hidden state extraction
    base_model = getattr(backbone, "model",
                         getattr(backbone, "transformer", backbone.base_model))

    # Warmup step
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    orig_head.train()
    with torch.no_grad():
        hidden_states = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    # Warmup forward/backward
    shift = 2
    t = torch.full_like(input_ids, -100)
    t[:, :-shift] = input_ids[:, shift:]
    t = t.masked_fill(loss_mask == 0, -100)
    logits_k = orig_head(hidden_states.to(orig_head.linear1.weight.dtype))
    loss_k = F.cross_entropy(logits_k.view(-1, logits_k.size(-1)), t.view(-1))
    loss_k.backward()

    # Cleanup warmup
    del hidden_states, input_ids, attention_mask, loss_mask, batch, logits_k, loss_k, t
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5: break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        optimizer.zero_grad()

        # 1. Get hidden states once
        with torch.no_grad():
            hidden_states = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2. Process head sequentially to simulate 4 heads
        for k in range(4): # Standard MedusaModel uses 4 heads
            shift = k + 2
            # Target calculation
            t = torch.full_like(input_ids, -100)
            t[:, :-shift] = input_ids[:, shift:]
            t = t.masked_fill(loss_mask == 0, -100)

            # Forward + Backward for one head
            logits_k = orig_head(hidden_states.to(orig_head.linear1.weight.dtype))
            loss_k = F.cross_entropy(logits_k.view(-1, logits_k.size(-1)), t.view(-1))
            loss_k.backward()

            # logits_k and loss_k are freed here
            del logits_k, loss_k, t

        optimizer.step()

        # Cleanup loop step
        del hidden_states, input_ids, attention_mask, loss_mask, batch

    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / 5
    print(f"Original baseline time-per-step: {avg_time:.4f}s")

    # Cleanup
    del orig_head
    del optimizer
    torch.cuda.empty_cache()
    return avg_timedef build_model(args, backbone, device, head_dtype):
    """Instantiate either MedusaModel or LatentMedusaModel based on args."""
    if args.head_type == "latent":
        d_latent = args.d_latent
        if d_latent is None:
            d_latent = backbone.config.hidden_size // 4
        model = LatentMedusaModel(
            backbone,
            num_heads=4,
            d_latent=d_latent,
            per_head_latent=args.per_head_latent,
            head_dtype=head_dtype,
        )
        print(f"Using LatentMedusaModel: d_latent={d_latent}, "
              f"per_head={args.per_head_latent}, "
              f"latent_dims={model.latent_dims}")
    else:
        model = MedusaModel(backbone, num_heads=4, head_dtype=head_dtype)
        print("Using original MedusaModel")
    return model
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 1. Load tokenizer and model
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16
    head_dtype = torch.float32

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
        model = build_model(args, backbone, device, head_dtype)
        model.heads.to(device)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=compute_dtype, attn_implementation="sdpa"
        )
        model = build_model(args, backbone, device, head_dtype)
        model.to(device)

    # Optional torch.compile
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile failed ({e}); running eager.")

    # Param counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    if args.head_type == "latent":
        total, per_head = model.count_head_params()
        print(f"  Per-head params: {[f'{n:,}' for n in per_head]}")

    # 2. Data loading
    print("Preparing data...")
    tokenized_data = download_and_preprocess(
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        model_name=args.model_name,
    )
    dataset = MedusaDataset(tokenized_data, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)

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

    # FLOP/Time-equivalent epoch scaling
    epochs = args.epochs
    if args.head_type == "latent":
        if args.time_equiv:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            t_orig = profile_baseline(args, backbone, device, head_dtype, train_loader)
            
            # Profile latent for 5 steps
            print("--- Profiling Latent Medusa for calibration ---")
            optimizer_temp = torch.optim.SGD(model.heads.parameters(), lr=args.lr)
            base_model = getattr(model.backbone, "model", 
                                 getattr(model.backbone, "transformer", model.backbone.base_model))
            model.train()
            torch.cuda.synchronize()
            start = time.time()
            for i, batch in enumerate(train_loader):
                if i >= 5: break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                
                optimizer_temp.zero_grad()
                with torch.no_grad():
                    hidden_states = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                
                for k in range(model.num_heads):
                    shift = k + 2
                    t = torch.full_like(input_ids, -100)
                    t[:, :-shift] = input_ids[:, shift:]
                    t = t.masked_fill(loss_mask == 0, -100)

                    logits_k = model.heads[k](hidden_states.to(model.heads[k].W_down.weight.dtype))
                    loss_k = F.cross_entropy(logits_k.view(-1, logits_k.size(-1)), t.view(-1))
                    loss_k.backward()
                    del logits_k, loss_k, t

                optimizer_temp.step()
                # Explicitly clean up to save memory
                del hidden_states, input_ids, attention_mask, loss_mask, batch

            torch.cuda.synchronize()
            t_lat = (time.time() - start) / 5

            # Reset model grad status after profiling
            model.zero_grad()
            optimizer_temp.zero_grad()
            del optimizer_temp
            gc.collect()
            torch.cuda.empty_cache()

            ratio = t_orig / t_lat
            print(f"Time ratio (original/latent): {ratio:.2f}x")
            if ratio > 1.0:
                epochs = max(1, round(args.epochs * ratio))
                print(f"Time-equivalent training: {args.epochs} → {epochs} epochs (latent faster)")
        elif args.flop_equiv:
            d_latent = args.d_latent or (backbone.config.hidden_size // 4)
            ratio = compute_flop_ratio(
                backbone.config.hidden_size, backbone.config.vocab_size,
                d_latent, per_head_latent=args.per_head_latent,
            )
            if ratio > 1.0:
                epochs = max(1, round(args.epochs * ratio))
                print(f"FLOP-equivalent training: {args.epochs} → {epochs} epochs (ratio={ratio:.2f}x, latent cheaper)")
            else:
                print(f"Latent heads are {1/ratio:.1f}x more expensive per step — keeping {epochs} epoch(s)")

    # 3. Optimizer and Scheduler
    optimizer = AdamW(model.heads.parameters(), lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Lambda weights: λ_k = 0.8^k with k starting at 1 (paper §2.2.1, 1-indexed sum).
    lambda_weights = [0.8 ** (k + 1) for k in range(model.num_heads)]
    print(f"Loss weights per head: {lambda_weights}")

    # 4. Training Loop
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    use_autocast = (device.type == "cuda")
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=compute_dtype)) \
                   if use_autocast else (lambda: torch.cuda.amp.autocast(enabled=False))

    # Identify the base model for hidden state extraction
    base_model = getattr(model.backbone, "model", 
                         getattr(model.backbone, "transformer", model.backbone.base_model))

    for epoch in range(epochs):
        model.train()
        head_acc_sums = torch.zeros(model.num_heads, device=device)
        total_tokens = torch.zeros((), device=device, dtype=torch.long)
        last_loss_scalar = torch.zeros((), device=device)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            loss_mask = (batch["loss_mask"].to(device, non_blocking=True) & attention_mask)

            with autocast_ctx():
                # 1. Get hidden states once
                with torch.no_grad():
                    hidden_states = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                
                total_loss = input_ids.new_zeros((), dtype=torch.float32)

                # 2. Process heads sequentially
                for k in range(model.num_heads):
                    shift = k + 2
                    
                    # Compute one head's logits
                    # Use the correct head parameter to determine target dtype
                    h_dtype = model.heads[k].W_down.weight.dtype if hasattr(model.heads[k], "W_down") else model.heads[k].linear1.weight.dtype
                    logits_k = model.heads[k](hidden_states.to(h_dtype))
                    
                    # Shift and mask
                    l_k = logits_k[:, :-shift, :].contiguous().view(-1, logits_k.size(-1))
                    t_k = input_ids[:, shift:].contiguous().view(-1)
                    m_k = loss_mask[:, shift:].contiguous().view(-1)

                    valid_indices = m_k == 1
                    l_k = l_k[valid_indices]
                    t_k = t_k[valid_indices]

                    if t_k.numel() > 0:
                        ce_loss = F.cross_entropy(l_k, t_k)
                        weighted_loss = (lambda_weights[k] * ce_loss) / args.grad_accum_steps
                        weighted_loss.backward()
                        
                        total_loss = total_loss + ce_loss.detach()

                        with torch.no_grad():
                            preds = torch.argmax(l_k, dim=-1)
                            head_acc_sums[k] += (preds == t_k).sum()
                            if k == 0:
                                total_tokens += t_k.numel()
                    
                    del logits_k, l_k, t_k, m_k

            last_loss_scalar = total_loss.detach()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % args.log_interval == 0:
                denom = total_tokens.clamp(min=1).to(torch.float32)
                avg_accs = (head_acc_sums / denom).tolist()
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {last_loss_scalar.item():.4f} | "
                      f"Head Acc: {[f'{a:.3f}' for a in avg_accs]}")
            
            # Step cleanup
            del hidden_states, total_loss, batch, input_ids, attention_mask, loss_mask

        # Validation
        model.eval()
        val_head_accs = torch.zeros(model.num_heads, device=device)
        val_tokens = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad(), autocast_ctx():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                loss_mask = (batch["loss_mask"].to(device, non_blocking=True) & attention_mask)

                # 1. Get hidden states once
                hidden_states = base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

                # 2. Process heads sequentially
                for k in range(model.num_heads):
                    shift = k + 2
                    
                    # Compute one head's logits
                    h_dtype = model.heads[k].W_down.weight.dtype if hasattr(model.heads[k], "W_down") else model.heads[k].linear1.weight.dtype
                    logits_k = model.heads[k](hidden_states.to(h_dtype))
                    
                    l_k = logits_k[:, :-shift, :].contiguous().view(-1, logits_k.size(-1))
                    t_k = input_ids[:, shift:].contiguous().view(-1)
                    m_k = loss_mask[:, shift:].contiguous().view(-1)

                    valid_indices = m_k == 1
                    l_k = l_k[valid_indices]
                    t_k = t_k[valid_indices]

                    if t_k.numel() > 0:
                        preds = torch.argmax(l_k, dim=-1)
                        val_head_accs[k] += (preds == t_k).sum()
                        if k == 0:
                            val_tokens += t_k.numel()
                    
                    del logits_k, l_k, t_k, m_k
                
                del hidden_states, batch, input_ids, attention_mask, loss_mask

        denom = val_tokens.clamp(min=1).to(torch.float32)
        val_accs = (val_head_accs / denom).tolist()
        print(f"--- Epoch {epoch+1} Validation ---")
        print(f"Head Accuracies: {[f'{a:.3f}' for a in val_accs]}")
        torch.cuda.empty_cache()

    # Save Checkpoint
    print(f"Saving heads to {args.save_path}")
    # Save extra metadata for latent heads so we can reconstruct the model
    save_dict = model.heads.state_dict()
    if args.head_type == "latent":
        metadata = {
            "head_type": "latent",
            "d_latent": args.d_latent or (backbone.config.hidden_size // 4),
            "per_head_latent": args.per_head_latent,
            "latent_dims": model.latent_dims,
        }
        torch.save({"heads_state_dict": save_dict, "metadata": metadata}, args.save_path)
    else:
        torch.save(save_dict, args.save_path)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medusa Heads (original or latent)")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Number of ShareGPT samples (1000=smoke, 60000=paper)")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save weights (defaults to results/medusa_heads.pt or latent_medusa_heads.pt)")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile", action="store_true")

    # --- Latent head options ---
    parser.add_argument("--head_type", choices=["original", "latent"], default="original",
                        help="Head architecture: 'original' = standard Medusa, 'latent' = MLA-style bottleneck")
    parser.add_argument("--d_latent", type=int, default=None,
                        help="Latent bottleneck dimension (default: hidden_size // 4). Only used with --head_type latent.")
    parser.add_argument("--per_head_latent", action="store_true",
                        help="Use decreasing latent dims per head: d_latent, d_latent//2, d_latent//4, d_latent//8")
    parser.add_argument("--flop_equiv", action="store_true",
                        help="Scale epochs so total training FLOPs match original heads")
    parser.add_argument("--time_equiv", action="store_true",
                        help="Profile and scale epochs so total training wall-clock time matches original heads")

    args = parser.parse_args()

    # Default save paths
    if args.save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.head_type == "latent":
            fname = "latent_medusa_heads.pt"
        else:
            fname = "medusa_heads.pt"
        args.save_path = os.path.join(base_dir, "results", fname)
    elif not os.path.isabs(args.save_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.save_path = os.path.join(base_dir, "results", os.path.basename(args.save_path))

    train(args)
