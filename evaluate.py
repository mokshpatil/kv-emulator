"""Evaluation script: runs all workloads across all modes and generates plots."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import InliningConfig, CMTConfig, small_config
from src.kvssd import KVSSD
from src.workload import synthetic_workload, WORKLOAD_PROFILES

MODES = ["baseline", "kvpack_s", "kvpack_d", "ml_linear", "ml_bandit"]
MODE_LABELS = {
    "baseline": "Baseline",
    "kvpack_s": "KVPack-S",
    "kvpack_d": "KVPack-D",
    "ml_linear": "ML-Linear",
    "ml_bandit": "ML-Bandit",
}
MODE_COLORS = {
    "baseline": "#444444",
    "kvpack_s": "#2196F3",
    "kvpack_d": "#1565C0",
    "ml_linear": "#FF9800",
    "ml_bandit": "#F44336",
}


def run_workload(workload_name, mode, num_keys=10000, num_ops=50000,
                 read_ratio=0.5):
    config = small_config()
    desired_read = num_keys // 10
    desired_total = desired_read * 2
    desired_budget = desired_total * config.mapping.entry_size
    cmt_ratio = desired_budget / config.capacity_bytes

    config.cmt = CMTConfig(budget_ratio=cmt_ratio)
    config.inlining = InliningConfig(
        mode=mode,
        profiler_warmup=num_keys // 10,
        profiler_interval=num_keys // 10,
    )
    ssd = KVSSD(config)

    for op in synthetic_workload(workload_name, num_keys, num_ops, read_ratio):
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)

    return ssd


def collect_results(workloads, modes, num_keys, num_ops, read_ratio):
    results = {}
    total = len(workloads) * len(modes)
    done = 0
    for wl in workloads:
        results[wl] = {}
        for mode in modes:
            done += 1
            print(f"  [{done}/{total}] {wl} / {mode}...")
            ssd = run_workload(wl, mode, num_keys, num_ops, read_ratio)
            m = ssd.metrics
            results[wl][mode] = {
                "flash_reads": m.total_flash_reads,
                "tp_reads": m.tp_reads,
                "data_reads": m.data_reads,
                "flash_writes": m.flash_writes,
                "cmt_hit_rate": m.cmt_hit_rate,
                "inline_ratio": m.inline_ratio,
                "reads_leq_1": m.reads_with_one_or_fewer,
                "waf": m.waf,
                "avg_latency": m.avg_read_latency,
                "p50_latency": m.p50_read_latency,
                "p99_latency": m.p99_read_latency,
                "p999_latency": m.p999_read_latency,
                "latency_cdf": m.latency_cdf(),
            }
    return results


def plot_flash_reads(results, workloads, modes, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(workloads))
    width = 0.15
    for i, mode in enumerate(modes):
        vals = [results[wl][mode]["flash_reads"] for wl in workloads]
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.set_ylabel("Total Flash Reads")
    ax.set_title("Flash Reads by Workload")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "flash_reads.png"), dpi=150)
    plt.close()


def plot_flash_reduction(results, workloads, modes, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    non_base = [m for m in modes if m != "baseline"]
    x = range(len(workloads))
    width = 0.18
    for i, mode in enumerate(non_base):
        vals = []
        for wl in workloads:
            base = results[wl]["baseline"]["flash_reads"]
            if base > 0:
                vals.append((1.0 - results[wl][mode]["flash_reads"] / base) * 100)
            else:
                vals.append(0.0)
        offset = (i - len(non_base) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.set_ylabel("Flash Read Reduction (%)")
    ax.set_title("Flash Read Reduction vs Baseline")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "flash_reduction.png"), dpi=150)
    plt.close()


def plot_inline_ratio(results, workloads, modes, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(workloads))
    width = 0.15
    for i, mode in enumerate(modes):
        vals = [results[wl][mode]["inline_ratio"] * 100 for wl in workloads]
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.set_ylabel("Inline Ratio (%)")
    ax.set_title("Inline Entry Ratio by Workload")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "inline_ratio.png"), dpi=150)
    plt.close()


def plot_latency(results, workloads, modes, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("avg_latency", "Average"), ("p50_latency", "P50"), ("p99_latency", "P99")]

    for ax, (metric, title) in zip(axes, metrics):
        x = range(len(workloads))
        width = 0.15
        for i, mode in enumerate(modes):
            vals = [results[wl][mode][metric] for wl in workloads]
            offset = (i - len(modes) / 2 + 0.5) * width
            ax.bar([xi + offset for xi in x], vals, width,
                   label=MODE_LABELS[mode], color=MODE_COLORS[mode])
        ax.set_xticks(x)
        ax.set_xticklabels(workloads, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Latency (us)")
        ax.set_title(f"{title} Read Latency")

    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency.png"), dpi=150)
    plt.close()


def plot_latency_cdf(results, workload, modes, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        cdf = results[workload][mode]["latency_cdf"]
        if cdf:
            xs, ys = zip(*cdf)
            ax.plot(xs, ys, label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xlabel("Latency (us)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Read Latency CDF - {workload}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"latency_cdf_{workload}.png"), dpi=150)
    plt.close()


def plot_waf(results, workloads, modes, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(workloads))
    width = 0.15
    for i, mode in enumerate(modes):
        vals = [results[wl][mode]["waf"] for wl in workloads]
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=MODE_LABELS[mode], color=MODE_COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right")
    ax.set_ylabel("Write Amplification Factor")
    ax.set_title("WAF by Workload")
    ax.legend()
    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "waf.png"), dpi=150)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation and generate plots")
    parser.add_argument("--num-keys", type=int, default=10000)
    parser.add_argument("--num-ops", type=int, default=50000)
    parser.add_argument("--read-ratio", type=float, default=0.5)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--workloads", default=None,
                        help="Comma-separated workload names (default: all)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    workloads = args.workloads.split(",") if args.workloads else list(WORKLOAD_PROFILES.keys())

    print("Running evaluation...")
    results = collect_results(workloads, MODES, args.num_keys, args.num_ops,
                              args.read_ratio)

    # save raw results
    serializable = {}
    for wl in workloads:
        serializable[wl] = {}
        for mode in MODES:
            r = dict(results[wl][mode])
            r.pop("latency_cdf", None)
            serializable[wl][mode] = r
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {args.outdir}/results.json")

    # generate plots
    print("Generating plots...")
    plot_flash_reads(results, workloads, MODES, args.outdir)
    plot_flash_reduction(results, workloads, MODES, args.outdir)
    plot_inline_ratio(results, workloads, MODES, args.outdir)
    plot_latency(results, workloads, MODES, args.outdir)
    plot_waf(results, workloads, MODES, args.outdir)

    # CDF plots for a few representative workloads
    for wl in ["ZippyDB", "Cache", "RTDATA"]:
        if wl in workloads:
            plot_latency_cdf(results, wl, MODES, args.outdir)

    print(f"Plots saved to {args.outdir}/")

    # print summary table
    print(f"\n{'='*70}")
    print("Flash read reduction vs baseline")
    print(f"{'='*70}")
    non_base = [m for m in MODES if m != "baseline"]
    header = f"  {'workload':<12}"
    for m in non_base:
        header += f" {MODE_LABELS[m]:>12}"
    print(header)
    print(f"  {'-'*12}" + f" {'-'*12}" * len(non_base))
    for wl in workloads:
        base = results[wl]["baseline"]["flash_reads"]
        if base > 0:
            line = f"  {wl:<12}"
            for m in non_base:
                red = (1.0 - results[wl][m]["flash_reads"] / base) * 100
                line += f" {red:>11.1f}%"
            print(line)


if __name__ == "__main__":
    main()
