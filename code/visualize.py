"""
visualize.py

Loads metrics from results/ and generates publication-quality PNGs:
  - head_accuracies.png      (per-head top-1 accuracy per model)
  - speedup_comparison.png   (greedy vs Medusa TPS + speedup ratio)
  - acceptance_histogram.png (per-prompt acceptance rate distribution)
  - table3_comparison.png    (ablation speedups vs paper targets)
  - acceptance_comparison.png (greedy vs typical acceptance, per tree budget)

All plots are optional: each skips gracefully if its source JSON is missing.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

# Paper Table 3 targets (MEDUSA-1 on Vicuna-7B).
PAPER_TABLE3_SPEEDUP = {"none": 1.54, "naive": 1.92, "optimized": 2.18}


def _short(model_id):
    return model_id.split("/")[-1] if isinstance(model_id, str) and "/" in model_id else str(model_id)


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_all(pattern):
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        data = _load_json(path)
        if data is not None:
            out[path] = data
    return out


def plot_head_accuracies(benchmarks):
    if not benchmarks:
        print("[skip] head_accuracies — no *_benchmark.json")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, d in benchmarks.items():
        accs = d.get("head_accuracies", {})
        if not accs:
            continue
        keys = sorted(accs.keys(), key=lambda s: int(s.split("_")[-1]))
        vals = [accs[k] for k in keys]
        ax.plot(range(1, len(vals) + 1), vals, marker="o",
                label=_short(d.get("model", "?")))
    ax.set_xlabel("Medusa head index (k)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Per-head top-1 accuracy")
    ax.set_xticks(range(1, 5))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(RESULTS_DIR, "head_accuracies.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_speedup_comparison(benchmarks):
    if not benchmarks:
        print("[skip] speedup_comparison — no *_benchmark.json")
        return
    labels, greedy, medusa, speedups = [], [], [], []
    for _, d in benchmarks.items():
        labels.append(_short(d.get("model", "?")))
        greedy.append(d.get("greedy_tps", 0.0))
        # Prefer typical acceptance (paper-comparable); fall back to greedy acceptance.
        medusa.append(d.get("medusa_typical_tps", d.get("medusa_tps", 0.0)))
        speedups.append(d.get("speedup_ratio_typical", d.get("speedup_ratio", 0.0)))
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, greedy, w, label="Greedy baseline", color="#888")
    ax.bar(x + w / 2, medusa, w, label="Medusa (typical accept)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Greedy vs Medusa throughput")
    top = max(max(greedy + medusa) * 1.12, 1.0)
    ax.set_ylim(0, top)
    for i, s in enumerate(speedups):
        ax.text(i, max(greedy[i], medusa[i]) * 1.02,
                f"{s:.2f}x", ha="center", fontweight="bold")
    ax.legend()
    out = os.path.join(RESULTS_DIR, "speedup_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_acceptance_histogram(benchmarks):
    if not benchmarks:
        print("[skip] acceptance_histogram — no *_benchmark.json")
        return
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (_, d) in zip(axes[0], benchmarks.items()):
        # Prefer typical_results (paper-comparable); fall back to greedy_results / legacy key.
        result_list = d.get("typical_results", d.get("greedy_results", d.get("medusa_results", [])))
        rates = [r.get("acceptance_rate", 0.0) for r in result_list]
        model = _short(d.get("model", "?"))
        if not rates:
            ax.set_title(f"{model} (no data)")
            continue
        ax.hist(rates, bins=min(10, max(3, len(rates))), edgecolor="black", color="#4c72b0")
        ax.axvline(float(np.mean(rates)), color="red", ls="--", label=f"mean={np.mean(rates):.2f}")
        ax.set_xlabel("Extra tokens accepted per step")
        ax.set_ylabel("Prompt count")
        ax.set_title(model)
        ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "acceptance_histogram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_table3_comparison(tables):
    if not tables:
        print("[skip] table3_comparison — no table3_*.json")
        return
    modes = ["none", "naive", "optimized"]
    labels = ["Heads only", "+ Tree attn (naive)", "+ Optimized tree"]
    runs = [(_short(d.get("model", path)), d) for path, d in tables.items()]

    n_groups = len(runs) + 1  # +1 for paper column
    x = np.arange(len(modes))
    w = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, d) in enumerate(runs):
        vals = [d.get("rows", {}).get(m, {}).get("speedup", 0.0) for m in modes]
        ax.bar(x + (i - (n_groups - 1) / 2) * w, vals, w, label=name)
    paper_vals = [PAPER_TABLE3_SPEEDUP[m] for m in modes]
    ax.bar(x + ((n_groups - 1) - (n_groups - 1) / 2) * w, paper_vals, w,
           label="Paper (Vicuna-7B)", color="#dd8452", hatch="//", alpha=0.8)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup vs greedy")
    ax.set_title("Table 3 ablation — tree mode vs speedup")
    ax.legend()
    out = os.path.join(RESULTS_DIR, "table3_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_acceptance_comparison(comparisons):
    """Greedy vs typical acceptance — grouped by tree budget / model."""
    if not comparisons:
        print("[skip] acceptance_comparison — no comparison_*.json")
        return
    labels, greedy_tps, typical_tps, greedy_acc, typical_acc = [], [], [], [], []
    for path, d in comparisons.items():
        tag = os.path.basename(path).replace("comparison_", "").replace(".json", "")
        labels.append(tag)
        greedy_tps.append(d.get("greedy_acceptance", {}).get("avg_tps", 0.0))
        typical_tps.append(d.get("typical_acceptance", {}).get("avg_tps", 0.0))
        greedy_acc.append(d.get("greedy_acceptance", {}).get("avg_acceptance_rate", 0.0))
        typical_acc.append(d.get("typical_acceptance", {}).get("avg_acceptance_rate", 0.0))

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x - w / 2, greedy_tps, w, label="Greedy accept", color="#4c72b0")
    ax1.bar(x + w / 2, typical_tps, w, label="Typical accept", color="#dd8452")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Tokens / second"); ax1.set_title("Throughput by acceptance criterion")
    ax1.legend()

    ax2.bar(x - w / 2, greedy_acc, w, label="Greedy accept", color="#4c72b0")
    ax2.bar(x + w / 2, typical_acc, w, label="Typical accept", color="#dd8452")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Acceptance rate (extra tokens / step)")
    ax2.set_title("Acceptance rate by criterion")
    ax2.legend()

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "acceptance_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmarks = _load_all("*_benchmark.json")
    tables = _load_all("table3_*.json")
    comparisons = _load_all("comparison_*.json")

    print(f"Loaded {len(benchmarks)} benchmark(s), "
          f"{len(tables)} Table 3 run(s), {len(comparisons)} comparison(s).")

    plot_head_accuracies(benchmarks)
    plot_speedup_comparison(benchmarks)
    plot_acceptance_histogram(benchmarks)
    plot_table3_comparison(tables)
    plot_acceptance_comparison(comparisons)


if __name__ == "__main__":
    main()
