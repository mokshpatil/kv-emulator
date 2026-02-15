import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config, gc_config, InliningConfig, CMTConfig
from src.kvssd import KVSSD
from src.workload import synthetic_workload


def make_ssd(mode="baseline", num_keys=1000):
    config = small_config()
    desired_read = num_keys // 10
    desired_total = desired_read * 2
    budget = desired_total * config.mapping.entry_size
    config.cmt = CMTConfig(budget_ratio=budget / config.capacity_bytes)
    config.inlining = InliningConfig(
        mode=mode, profiler_warmup=100, profiler_interval=100,
    )
    return KVSSD(config)


def make_gc_ssd(mode="baseline"):
    config = gc_config()
    config.cmt = CMTConfig(budget_ratio=0.01)
    config.inlining = InliningConfig(
        mode=mode, profiler_warmup=20, profiler_interval=20,
    )
    return KVSSD(config)


def test_gc_victim_selection():
    ssd = make_ssd()
    for i in range(100):
        ssd.put(f"key{i:04d}".encode(), 64)
    for i in range(50):
        ssd.delete(f"key{i:04d}".encode())

    victim = ssd.gc._select_victim()
    assert victim is not None
    assert ssd.flash.invalid_count_in_block(victim) > 0


def test_gc_collects_and_erases():
    ssd = make_ssd()
    for i in range(200):
        ssd.put(f"key{i:04d}".encode(), 64)
    for i in range(100):
        ssd.delete(f"key{i:04d}".encode())

    erases_before = ssd.metrics.flash_erases
    ssd.gc.threshold = 0.0001
    rounds = ssd.gc.run(max_rounds=5)
    assert rounds > 0
    assert ssd.metrics.flash_erases > erases_before
    assert ssd.metrics.gc_invocations > 0


def test_gc_preserves_data():
    ssd = make_ssd()
    keys = [f"key{i:04d}".encode() for i in range(200)]
    for k in keys:
        ssd.put(k, 64)
    for i in range(1, 200, 2):
        ssd.delete(keys[i])

    victim = ssd.gc._select_victim()
    assert victim is not None
    ssd.gc._collect_block(victim)

    for i in range(0, 200, 2):
        assert ssd.get(keys[i]) is True, f"key {i} lost after GC"
    for i in range(1, 200, 2):
        assert ssd.get(keys[i]) is False, f"deleted key {i} reappeared after GC"


def test_gc_waf_tracking():
    ssd = make_ssd()
    for i in range(200):
        ssd.put(f"key{i:04d}".encode(), 64)
    assert ssd.metrics.host_writes == 200
    assert ssd.metrics.waf >= 1.0


def test_gc_metrics_pages_copied():
    ssd = make_ssd()
    for i in range(200):
        ssd.put(f"key{i:04d}".encode(), 64)
    for i in range(0, 200, 2):
        ssd.delete(f"key{i:04d}".encode())

    copies_before = ssd.metrics.gc_pages_copied
    victim = ssd.gc._select_victim()
    assert victim is not None
    valid_in_victim = ssd.flash.valid_pages_in_block(victim)
    ssd.gc._collect_block(victim)
    if len(valid_in_victim) > 0:
        assert ssd.metrics.gc_pages_copied > copies_before


def test_gc_triggers_naturally():
    # use gc_config (2MB SSD) with enough overwrites to fill the flash
    ssd = make_gc_ssd()
    # populate 60 keys (each gets 1 data page + shared TPs)
    for i in range(60):
        ssd.put(f"key{i:04d}".encode(), 64)
    # overwrite keys repeatedly to create invalidated pages and fill the SSD
    for _ in range(5):
        for i in range(60):
            ssd.put(f"key{i:04d}".encode(), 64)
    # GC should have triggered at least once
    assert ssd.metrics.gc_invocations > 0, (
        f"GC never triggered (util={ssd.flash.utilization:.2f}, "
        f"erases={ssd.metrics.flash_erases})"
    )
    assert ssd.metrics.flash_erases > 0


def test_gc_with_deletes_in_workload():
    # use synthetic workload with delete_ratio
    ssd = make_gc_ssd()
    ops = list(synthetic_workload("RTDATA", num_keys=50, num_ops=500,
                                  read_ratio=0.3, delete_ratio=0.2, seed=1))
    for op in ops:
        if op.op_type == "put":
            ssd.put(op.key, op.value_size)
        elif op.op_type == "get":
            ssd.get(op.key)
        elif op.op_type == "delete":
            ssd.delete(op.key)
    # with deletes creating fragmentation, GC may trigger
    # at minimum, verify the workload ran without crashing
    assert ssd.metrics.total_ops > 0


def test_gc_retry_on_flash_full():
    # disable normal GC threshold so only the flash-full retry triggers GC
    ssd = make_gc_ssd()
    ssd.gc.threshold = 2.0  # never trigger proactively — only flash-full retry

    # populate 60 keys then overwrite to fill all physical pages
    num_keys = 60
    for i in range(num_keys):
        ssd.put(f"k{i:05d}".encode(), 64)

    # overwrite keys — old pages stay physically occupied, new pages allocated
    # eventually allocator wraps around and hits flash-full → GC retry
    for _ in range(10):
        for i in range(num_keys):
            ssd.put(f"k{i:05d}".encode(), 64)

    # GC should have been triggered by flash-full retry
    assert ssd.metrics.gc_invocations > 0
    assert ssd._gc_retries > 0

    # all keys should still be readable
    for i in range(num_keys):
        assert ssd.get(f"k{i:05d}".encode()) is True


if __name__ == "__main__":
    test_gc_victim_selection()
    test_gc_collects_and_erases()
    test_gc_preserves_data()
    test_gc_waf_tracking()
    test_gc_metrics_pages_copied()
    test_gc_triggers_naturally()
    test_gc_with_deletes_in_workload()
    test_gc_retry_on_flash_full()
    print("all gc tests passed")
