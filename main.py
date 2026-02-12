import sys
import os
import argparse

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import InliningConfig, CMTConfig, small_config
from src.kvssd import KVSSD
from src.workload import synthetic_workload, trace_workload, WORKLOAD_PROFILES

ALL_MODES = ["baseline", "kvpack_s", "kvpack_d", "ml_linear", "ml_bandit"]


def run_synthetic(workload_name, mode, num_keys=10000, num_ops=50000,
                   read_ratio=0.5):
    config = small_config()

    # size CMT so working set is 10x CMT capacity
    desired_read_entries = num_keys // 10
    desired_total_entries = desired_read_entries * 2
    desired_budget = desired_total_entries * config.mapping.entry_size
    cmt_ratio = desired_budget / config.capacity_bytes

    config.cmt = CMTConfig(budget_ratio=cmt_ratio)
    config.inlining = InliningConfig(
        mode=mode,
        profiler_warmup=num_keys // 10,
        profiler_interval=num_keys // 10,
    )
    ssd = KVSSD(config)

    ops = synthetic_workload(
        workload_name=workload_name,
        num_keys=num_keys,
        num_ops=num_ops,
        read_ratio=read_ratio,
    )

    for op in ops:
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)

    return ssd


def run_trace(trace_path, mode, max_ops=0):
    config = small_config()
    config.inlining = InliningConfig(
        mode=mode,
        profiler_warmup=10000,
        profiler_interval=10000,
    )
    ssd = KVSSD(config)

    for op in trace_workload(trace_path, max_ops=max_ops):
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)

    return ssd


def print_comparison(label, results, modes):
    print(f"\n{'='*70}")
    print(label)
    print(f"{'='*70}")

    header = f"  {'metric':<22}"
    for mode in modes:
        header += f" {mode:>12}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*12}" * len(modes))

    ml = [results[m].metrics for m in modes]

    rows = [
        ("flash reads", [str(m.total_flash_reads) for m in ml]),
        ("  tp reads", [str(m.tp_reads) for m in ml]),
        ("  data reads", [str(m.data_reads) for m in ml]),
        ("flash writes", [str(m.flash_writes) for m in ml]),
        ("CMT hit rate", [f"{m.cmt_hit_rate:.4f}" for m in ml]),
        ("inline ratio", [f"{m.inline_ratio:.4f}" for m in ml]),
        ("reads<=1 flash", [f"{m.reads_with_one_or_fewer:.4f}" for m in ml]),
        ("conversions", [str(m.inline_to_regular) for m in ml]),
        ("WAF", [f"{m.waf:.2f}" for m in ml]),
        ("avg latency (us)", [f"{m.avg_read_latency:.1f}" for m in ml]),
        ("p50 latency (us)", [f"{m.p50_read_latency:.1f}" for m in ml]),
        ("p99 latency (us)", [f"{m.p99_read_latency:.1f}" for m in ml]),
        ("p99.9 latency (us)", [f"{m.p999_read_latency:.1f}" for m in ml]),
    ]

    for label_row, vals in rows:
        line = f"  {label_row:<22}"
        for v in vals:
            line += f" {v:>12}"
        print(line)


def cmd_synthetic(args):
    modes = args.modes.split(",") if args.modes else ALL_MODES

    if args.workload == "all":
        workloads = list(WORKLOAD_PROFILES.keys())
    else:
        workloads = [args.workload]

    all_results = {}
    for wl in workloads:
        profile = WORKLOAD_PROFILES[wl]
        label = (f"Workload: {wl} (key={profile['key_size']}B, "
                 f"value={profile['value_size']}B, source={profile['source']})")
        all_results[wl] = {}
        for mode in modes:
            all_results[wl][mode] = run_synthetic(
                wl, mode, num_keys=args.num_keys, num_ops=args.num_ops,
                read_ratio=args.read_ratio,
            )
        print_comparison(label, all_results[wl], modes)

    # flash read reduction summary
    if len(workloads) > 1 and "baseline" in modes:
        non_base = [m for m in modes if m != "baseline"]
        print(f"\n{'='*70}")
        print("Flash read reduction vs baseline")
        print(f"{'='*70}")
        header = f"  {'workload':<12}"
        for m in non_base:
            header += f" {m:>12}"
        print(header)
        print(f"  {'-'*12}" + f" {'-'*12}" * len(non_base))
        for wl in workloads:
            base = all_results[wl]["baseline"].metrics.total_flash_reads
            if base > 0:
                line = f"  {wl:<12}"
                for m in non_base:
                    red = 1.0 - all_results[wl][m].metrics.total_flash_reads / base
                    line += f" {red:>11.1%}"
                print(line)


def cmd_trace(args):
    modes = args.modes.split(",") if args.modes else ALL_MODES
    results = {}
    for mode in modes:
        print(f"  running {mode}...")
        results[mode] = run_trace(args.trace_path, mode, max_ops=args.max_ops)
    label = f"Trace: {os.path.basename(args.trace_path)}"
    print_comparison(label, results, modes)


def main():
    parser = argparse.ArgumentParser(description="KV-SSD Emulator")
    subparsers = parser.add_subparsers(dest="command")

    # synthetic workload command
    syn = subparsers.add_parser("synthetic", help="Run synthetic workloads")
    syn.add_argument("--workload", default="all",
                     choices=list(WORKLOAD_PROFILES.keys()) + ["all"])
    syn.add_argument("--num-keys", type=int, default=10000)
    syn.add_argument("--num-ops", type=int, default=50000)
    syn.add_argument("--read-ratio", type=float, default=0.5,
                     help="Fraction of ops that are reads (0.0-1.0)")
    syn.add_argument("--modes", default=None,
                     help=f"Comma-separated modes (default: all). Options: {','.join(ALL_MODES)}")

    # trace replay command
    tr = subparsers.add_parser("trace", help="Replay a trace file")
    tr.add_argument("trace_path", help="Path to trace CSV (or .zst)")
    tr.add_argument("--max-ops", type=int, default=0, help="Limit ops (0=all)")
    tr.add_argument("--modes", default=None,
                     help=f"Comma-separated modes (default: all). Options: {','.join(ALL_MODES)}")

    args = parser.parse_args()

    if args.command == "synthetic":
        cmd_synthetic(args)
    elif args.command == "trace":
        cmd_trace(args)
    else:
        # default: run all synthetic workloads
        args.workload = "all"
        args.num_keys = 10000
        args.num_ops = 50000
        args.read_ratio = 0.5
        args.modes = None
        cmd_synthetic(args)


if __name__ == "__main__":
    main()
