"""
compare_heads.py

A/B comparison between original MedusaModel and LatentMedusaModel.

Usage:
    # Quick smoke test (200 samples each)
    python compare_heads.py --max_samples 200

    # Paper-scale comparison
    python compare_heads.py --max_samples 60000 --model_name lmsys/vicuna-7b-v1.5 --quantize

Runs:
    1. Train original Medusa heads
    2. Train latent Medusa heads (FLOP-equivalent epochs)
    3. Benchmark both on the same prompts
    4. Print comparison table and save to results/head_comparison.json
"""

import argparse
import json
import os
import subprocess
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")


def run_cmd(cmd, desc):
    """Run a subprocess and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)


def load_benchmark_json(path):
    """Load a benchmark JSON, returning None if it doesn't exist."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="A/B compare original vs latent Medusa heads")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--d_latent", type=int, default=None, help="Latent dim (default: hidden//4)")
    parser.add_argument("--per_head_latent", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, use existing checkpoints")
    parser.add_argument("--skip_original", action="store_true",
                        help="Skip original head training/benchmark (assume already done)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    quant_flag = "--quantize" if args.quantize else ""
    ph_flag = "--per_head_latent" if args.per_head_latent else ""
    d_latent_flag = f"--d_latent {args.d_latent}" if args.d_latent else ""

    # ---- Train original ----
    if not args.skip_train and not args.skip_original:
        run_cmd(
            f"python train_latent.py "
            f"--head_type original "
            f"--model_name {args.model_name} "
            f"--max_samples {args.max_samples} "
            f"--max_length {args.max_length} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"{quant_flag}",
            "Training ORIGINAL Medusa heads",
        )

    # ---- Train latent ----
    if not args.skip_train:
        run_cmd(
            f"python train_latent.py "
            f"--head_type latent "
            f"--model_name {args.model_name} "
            f"--max_samples {args.max_samples} "
            f"--max_length {args.max_length} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--flop_equiv "
            f"{d_latent_flag} {ph_flag} {quant_flag}",
            "Training LATENT Medusa heads (FLOP-equivalent)",
        )

    # ---- Benchmark original ----
    orig_out = os.path.join(RESULTS_DIR, "original_benchmark.json")
    if not args.skip_original:
        run_cmd(
            f"python benchmark.py "
            f"--head_type original "
            f"--mode full "
            f"--model_id {args.model_name} "
            f"{quant_flag}",
            "Benchmarking ORIGINAL Medusa heads",
        )
        # Rename output
        default_out = os.path.join(RESULTS_DIR, f"{args.model_name.split('/')[-1]}_benchmark.json")
        if os.path.exists(default_out):
            os.rename(default_out, orig_out)

    # ---- Benchmark latent ----
    latent_out = os.path.join(RESULTS_DIR, "latent_benchmark.json")
    run_cmd(
        f"python benchmark.py "
        f"--head_type latent "
        f"--mode full "
        f"--model_id {args.model_name} "
        f"{d_latent_flag} {ph_flag} {quant_flag}",
        "Benchmarking LATENT Medusa heads",
    )
    default_out = os.path.join(RESULTS_DIR, f"{args.model_name.split('/')[-1]}_benchmark.json")
    if os.path.exists(default_out):
        os.rename(default_out, latent_out)

    # ---- Compare ----
    orig = load_benchmark_json(orig_out)
    latent = load_benchmark_json(latent_out)

    if orig and latent:
        print("\n" + "=" * 70)
        print("  COMPARISON: Original vs Latent Medusa Heads")
        print("=" * 70)

        headers = ["Metric", "Original", "Latent", "Δ"]
        rows = []

        def add_row(name, orig_val, latent_val, fmt=".2f", higher_better=True):
            delta = latent_val - orig_val
            sign = "+" if delta >= 0 else ""
            better = "✓" if (delta > 0 and higher_better) or (delta < 0 and not higher_better) else ""
            rows.append([
                name,
                f"{orig_val:{fmt}}",
                f"{latent_val:{fmt}}",
                f"{sign}{delta:{fmt}} {better}",
            ])

        add_row("Greedy TPS", orig["greedy_tps"], latent["greedy_tps"])
        add_row("Medusa TPS (greedy acc)", orig["medusa_tps"], latent["medusa_tps"])
        add_row("Speedup (greedy acc)", orig["speedup_ratio"], latent["speedup_ratio"])
        add_row("Medusa TPS (typical acc)", orig["medusa_typical_tps"], latent["medusa_typical_tps"])
        add_row("Speedup (typical acc)", orig["speedup_ratio_typical"], latent["speedup_ratio_typical"])
        add_row("Avg acceptance (greedy)", orig["avg_acceptance_rate"], latent["avg_acceptance_rate"], ".3f")
        add_row("Avg acceptance (typical)", orig["avg_acceptance_rate_typical"], latent["avg_acceptance_rate_typical"], ".3f")

        for k in range(4):
            key = f"head_{k}"
            add_row(f"Head {k} accuracy", orig["head_accuracies"][key], latent["head_accuracies"][key], ".3f")

        # Print table
        col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(4)]
        fmt_row = lambda r: " | ".join(r[i].ljust(col_widths[i]) for i in range(4))
        print(fmt_row(headers))
        print("-" * (sum(col_widths) + 9))
        for row in rows:
            print(fmt_row(row))

        # Save comparison
        comparison = {
            "model": args.model_name,
            "max_samples": args.max_samples,
            "d_latent": args.d_latent,
            "per_head_latent": args.per_head_latent,
            "original": orig,
            "latent": latent,
        }
        comp_path = os.path.join(RESULTS_DIR, "head_comparison.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=4)
        print(f"\nFull comparison saved to {comp_path}")
    else:
        print("\nCould not load both benchmark results for comparison.")


if __name__ == "__main__":
    main()
