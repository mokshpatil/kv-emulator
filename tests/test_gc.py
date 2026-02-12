import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config, InliningConfig, CMTConfig
from src.kvssd import KVSSD


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
    # lower threshold so GC actually triggers
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
    # delete odd-indexed to create fragmentation
    for i in range(1, 200, 2):
        ssd.delete(keys[i])

    # force a single GC round by calling _collect_block directly
    victim = ssd.gc._select_victim()
    assert victim is not None
    ssd.gc._collect_block(victim)

    # even-indexed keys should still be accessible
    for i in range(0, 200, 2):
        assert ssd.get(keys[i]) is True, f"key {i} lost after GC"
    # odd-indexed should still be gone
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
    # delete alternating keys to create fragmentation within blocks
    for i in range(0, 200, 2):
        ssd.delete(f"key{i:04d}".encode())

    copies_before = ssd.metrics.gc_pages_copied
    victim = ssd.gc._select_victim()
    assert victim is not None
    valid_in_victim = ssd.flash.valid_pages_in_block(victim)
    ssd.gc._collect_block(victim)
    # if the victim had valid pages, they should have been copied
    if len(valid_in_victim) > 0:
        assert ssd.metrics.gc_pages_copied > copies_before


if __name__ == "__main__":
    test_gc_victim_selection()
    test_gc_collects_and_erases()
    test_gc_preserves_data()
    test_gc_waf_tracking()
    test_gc_metrics_pages_copied()
    print("all gc tests passed")
