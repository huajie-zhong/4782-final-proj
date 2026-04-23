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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
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
        
    backbone = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = MedusaModel(backbone, num_heads=4)
    model.to(device)

    # Verify what is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # 2. Data loading
    print("Preparing data...")
    tokenized_data = download_and_preprocess(tokenizer, max_samples=args.max_samples)
    dataset = MedusaDataset(tokenized_data, max_length=args.max_length, pad_token_id=tokenizer.pad_token_id)

    # Split train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        head_accuracies = [0.0] * model.num_heads
        total_tokens = 0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            # We don't need the original logits for training the Medusa heads
            _, head_logits = model(input_ids, attention_mask=attention_mask)
            
            loss = 0
            batch_head_accs = [0.0] * model.num_heads
            batch_valid_tokens = 0

            for k in range(model.num_heads):
                # Label shifting: Head k predicts t + k + 2
                shift = k + 2
                
                # Truncate logits and targets
                # shape: (batch_size, seq_len, vocab_size)
                logits_k = head_logits[k][:, :-shift, :].contiguous().view(-1, head_logits[k].size(-1))
                targets_k = input_ids[:, shift:].contiguous().view(-1)
                mask_k = attention_mask[:, shift:].contiguous().view(-1)

                # Filter by mask
                valid_indices = mask_k == 1
                logits_k = logits_k[valid_indices]
                targets_k = targets_k[valid_indices]

                if targets_k.numel() == 0:
                    continue

                # Compute Cross-Entropy Loss
                ce_loss = F.cross_entropy(logits_k, targets_k)
                loss += lambda_weights[k] * ce_loss

                # Compute Top-1 Accuracy for logging
                preds = torch.argmax(logits_k, dim=-1)
                acc = (preds == targets_k).float().sum().item()
                batch_head_accs[k] = acc
                
                if k == 0:
                    batch_valid_tokens = targets_k.numel()

            # Normalize loss by grad_accum_steps
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            total_loss += loss.item() * args.grad_accum_steps
            for k in range(model.num_heads):
                head_accuracies[k] += batch_head_accs[k]
            total_tokens += batch_valid_tokens

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % args.log_interval == 0:
                avg_accs = [acc / max(1, total_tokens) for acc in head_accuracies]
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item() * args.grad_accum_steps:.4f} | Head Acc: {[f'{a:.3f}' for a in avg_accs]}")

        # Validation
        model.eval()
        val_head_accs = [0.0] * model.num_heads
        val_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                _, head_logits = model(input_ids, attention_mask=attention_mask)
                
                for k in range(model.num_heads):
                    shift = k + 2
                    logits_k = head_logits[k][:, :-shift, :].contiguous().view(-1, head_logits[k].size(-1))
                    targets_k = input_ids[:, shift:].contiguous().view(-1)
                    mask_k = attention_mask[:, shift:].contiguous().view(-1)
                    
                    valid_indices = mask_k == 1
                    logits_k = logits_k[valid_indices]
                    targets_k = targets_k[valid_indices]
                    
                    if targets_k.numel() > 0:
                        preds = torch.argmax(logits_k, dim=-1)
                        val_head_accs[k] += (preds == targets_k).float().sum().item()
                        if k == 0:
                            val_tokens += targets_k.numel()
                            
        val_accs = [acc / max(1, val_tokens) for acc in val_head_accs]
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between logging")
    parser.add_argument("--save_path", type=str, default="../results/medusa_heads.pt", help="Path to save weights")
    
    args = parser.parse_args()
    
    # Adjust save_path to be relative to the script location or working dir
    # Usually running from final/code, so ../results/ is final/results/
    if not os.path.isabs(args.save_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.save_path = os.path.join(base_dir, "results", os.path.basename(args.save_path))
        
    train(args)
