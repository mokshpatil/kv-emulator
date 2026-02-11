import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config, InliningConfig, CMTConfig
from src.kvssd import KVSSD


def make_ssd(mode="baseline", num_keys=1000, cmt_ratio_denom=10):
    config = small_config()
    desired_read = num_keys // cmt_ratio_denom
    desired_total = desired_read * 2
    budget = desired_total * config.mapping.entry_size
    config.cmt = CMTConfig(budget_ratio=budget / config.capacity_bytes)
    config.inlining = InliningConfig(
        mode=mode, profiler_warmup=100, profiler_interval=100,
    )
    return KVSSD(config)


def test_put_get_baseline():
    ssd = make_ssd("baseline")
    key = b"testkey123"
    ssd.put(key, 64)
    assert ssd.get(key) is True
    assert ssd.get(b"nonexistent") is False


def test_delete():
    ssd = make_ssd("baseline")
    key = b"deletetest"
    ssd.put(key, 64)
    assert ssd.delete(key) is True
    assert ssd.get(key) is False
    assert ssd.delete(key) is False


def test_baseline_no_inlining():
    # baseline only inlines values <= 8B, so 64B values should be regular
    ssd = make_ssd("baseline")
    for i in range(500):
        ssd.put(f"key{i:04d}".encode(), 64)
    assert ssd.metrics.inline_entries == 0
    assert ssd.metrics.regular_entries == 500


def test_baseline_tiny_values_inlined():
    # values <= 8B should be inlined even in baseline
    ssd = make_ssd("baseline")
    for i in range(100):
        ssd.put(f"key{i:04d}".encode(), 4)
    assert ssd.metrics.inline_entries == 100
    assert ssd.metrics.regular_entries == 0


def test_kvpack_s_inlining():
    ssd = make_ssd("kvpack_s", num_keys=2000)
    # insert 200 entries during profiler warmup (regular)
    for i in range(200):
        ssd.put(f"key{i:04d}".encode(), 40)
    # profiler should now be done, remaining entries should be inline
    for i in range(200, 500):
        ssd.put(f"key{i:04d}".encode(), 40)
    assert ssd.metrics.inline_entries > 0
    assert ssd.metrics.inline_ratio > 0.5


def test_flash_reads_reduced_with_inlining():
    # baseline: CMT miss -> 2 flash reads (TP + data)
    # kvpack: CMT miss on inline -> 1 flash read (TP only)
    num_keys = 1000
    num_reads = 5000

    # run baseline
    ssd_base = make_ssd("baseline", num_keys=num_keys)
    for i in range(num_keys):
        ssd_base.put(f"key{i:04d}".encode(), 50)
    for i in range(num_reads):
        ssd_base.get(f"key{i % num_keys:04d}".encode())
    baseline_reads = ssd_base.metrics.total_flash_reads

    # run kvpack_s
    ssd_kv = make_ssd("kvpack_s", num_keys=num_keys)
    for i in range(num_keys):
        ssd_kv.put(f"key{i:04d}".encode(), 50)
    for i in range(num_reads):
        ssd_kv.get(f"key{i % num_keys:04d}".encode())
    kvpack_reads = ssd_kv.metrics.total_flash_reads

    # kvpack should have fewer total flash reads
    assert kvpack_reads < baseline_reads, (
        f"kvpack ({kvpack_reads}) should have fewer reads than "
        f"baseline ({baseline_reads})"
    )


def test_cmt_hit_rate_with_small_working_set():
    # if CMT can hold the entire working set, hit rate should be high
    config = small_config()
    config.cmt = CMTConfig(budget_ratio=0.01)  # large CMT
    config.inlining = InliningConfig(mode="baseline")
    ssd = KVSSD(config)

    for i in range(100):
        ssd.put(f"key{i:03d}".encode(), 64)
    for i in range(1000):
        ssd.get(f"key{i % 100:03d}".encode())

    assert ssd.metrics.cmt_hit_rate > 0.9


def test_metrics_reads_by_flash_count():
    ssd = make_ssd("baseline", num_keys=500)
    for i in range(500):
        ssd.put(f"key{i:04d}".encode(), 64)
    for i in range(100):
        ssd.get(f"key{i:04d}".encode())

    total = sum(ssd.metrics.reads_by_flash_count.values())
    assert total == 100


if __name__ == "__main__":
    test_put_get_baseline()
    test_delete()
    test_baseline_no_inlining()
    test_baseline_tiny_values_inlined()
    test_kvpack_s_inlining()
    test_flash_reads_reduced_with_inlining()
    test_cmt_hit_rate_with_small_working_set()
    test_metrics_reads_by_flash_count()
    print("all kvssd tests passed")
