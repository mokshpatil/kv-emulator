import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config, InliningConfig, CMTConfig
from src.kvssd import KVSSD
from src.workload import synthetic_workload


def make_ssd(mode, num_keys=1000):
    config = small_config()
    desired_read = num_keys // 10
    desired_total = desired_read * 2
    budget = desired_total * config.mapping.entry_size
    config.cmt = CMTConfig(budget_ratio=budget / config.capacity_bytes)
    config.inlining = InliningConfig(
        mode=mode, profiler_warmup=100, profiler_interval=100,
    )
    return KVSSD(config)


def run_workload(ssd, workload_name, num_keys, num_ops, read_ratio=0.5):
    for op in synthetic_workload(workload_name, num_keys, num_ops, read_ratio):
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)


def test_ml_linear_learns_to_inline():
    ssd = make_ssd("ml_linear", num_keys=2000)
    run_workload(ssd, "ZippyDB", 2000, 10000, read_ratio=0.5)
    # after learning, should have some inline entries
    assert ssd.metrics.inline_entries > 0
    assert ssd.metrics.inline_ratio > 0.3


def test_ml_bandit_learns_to_inline():
    ssd = make_ssd("ml_bandit", num_keys=2000)
    run_workload(ssd, "ZippyDB", 2000, 10000, read_ratio=0.5)
    assert ssd.metrics.inline_entries > 0
    assert ssd.metrics.inline_ratio > 0.3


def test_ml_linear_reduces_flash_reads():
    # compare ml_linear against baseline
    ssd_base = make_ssd("baseline", num_keys=2000)
    run_workload(ssd_base, "ZippyDB", 2000, 10000, read_ratio=0.5)

    ssd_ml = make_ssd("ml_linear", num_keys=2000)
    run_workload(ssd_ml, "ZippyDB", 2000, 10000, read_ratio=0.5)

    assert ssd_ml.metrics.total_flash_reads < ssd_base.metrics.total_flash_reads


def test_ml_bandit_reduces_flash_reads():
    ssd_base = make_ssd("baseline", num_keys=2000)
    run_workload(ssd_base, "ZippyDB", 2000, 10000, read_ratio=0.5)

    ssd_ml = make_ssd("ml_bandit", num_keys=2000)
    run_workload(ssd_ml, "ZippyDB", 2000, 10000, read_ratio=0.5)

    assert ssd_ml.metrics.total_flash_reads < ssd_base.metrics.total_flash_reads


def test_ml_feedback_called():
    ssd = make_ssd("ml_linear", num_keys=500)
    run_workload(ssd, "ZippyDB", 500, 2000, read_ratio=0.5)
    # feedback buffer should have entries from GET operations
    assert len(ssd.policy._buffer) > 0


if __name__ == "__main__":
    test_ml_linear_learns_to_inline()
    test_ml_bandit_learns_to_inline()
    test_ml_linear_reduces_flash_reads()
    test_ml_bandit_reduces_flash_reads()
    test_ml_feedback_called()
    print("all ml policy tests passed")
