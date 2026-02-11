import sys
import os

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SSDConfig, InliningConfig, CMTConfig, small_config
from src.kvssd import KVSSD
from src.workload import synthetic_workload, WORKLOAD_PROFILES


def run_workload(workload_name, mode, num_keys=10000, num_ops=50000):
    config = small_config()

    # size CMT so working set is 10x CMT capacity (matching KVPack eval)
    desired_read_entries = num_keys // 10
    desired_total_entries = desired_read_entries * 2
    desired_budget = desired_total_entries * config.mapping.entry_size
    cmt_ratio = desired_budget / config.capacity_bytes

    config.cmt = CMTConfig(budget_ratio=cmt_ratio)
    config.inlining = InliningConfig(
        mode=mode,
        # profile over first 10% of keys
        profiler_warmup=num_keys // 10,
        profiler_interval=num_keys // 10,
    )
    ssd = KVSSD(config)

    ops = synthetic_workload(
        workload_name=workload_name,
        num_keys=num_keys,
        num_ops=num_ops,
        read_ratio=1.0,  # read-only after populate, matching KVPack eval
    )

    for op in ops:
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)

    return ssd


def main():
    workloads = ["ZippyDB", "Cache15", "RTDATA", "Dedup"]
    modes = ["baseline", "kvpack_s", "kvpack_d"]

    for wl in workloads:
        profile = WORKLOAD_PROFILES[wl]
        print(f"\n{'='*60}")
        print(f"Workload: {wl} (key={profile['key_size']}B, "
              f"value={profile['value_size']}B)")
        print(f"{'='*60}")

        for mode in modes:
            ssd = run_workload(wl, mode)
            m = ssd.metrics
            print(f"\n  [{mode}]")
            print(f"    flash reads  : {m.total_flash_reads} "
                  f"(tp={m.tp_reads}, data={m.data_reads})")
            print(f"    CMT hit rate : {m.cmt_hit_rate:.4f}")
            print(f"    inline ratio : {m.inline_ratio:.4f}")
            print(f"    reads<=1     : {m.reads_with_one_or_fewer:.4f}")
            print(f"    conversions  : {m.inline_to_regular}")


if __name__ == "__main__":
    main()
