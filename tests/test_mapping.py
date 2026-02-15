import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config
from src.metrics import Metrics
from src.flash import Flash
from src.mapping import TranslationPage, GMD, MappingEntry, compute_key_hash


def make_entry(key_hash=1, key_size=32, value_size=64, inline=False, data_page=0):
    frames = 1 if not inline else max(1, (12 + key_size + value_size + 31) // 32)
    return MappingEntry(
        key_hash=key_hash,
        key_size=key_size,
        value_size=value_size,
        is_inline=inline,
        data_page_id=-1 if inline else data_page,
        frames_used=frames,
    )


def test_tp_insert_and_find():
    tp = TranslationPage(0, total_frames=512)
    entry = make_entry(key_hash=100)
    tp.insert(entry)
    assert tp.find(100) is entry
    assert tp.find(999) is None
    assert tp.num_entries == 1
    assert tp.used_frames == 1


def test_tp_remove():
    tp = TranslationPage(0, total_frames=512)
    tp.insert(make_entry(key_hash=1))
    tp.insert(make_entry(key_hash=2))
    assert tp.num_entries == 2

    removed = tp.remove(1)
    assert removed is not None
    assert removed.key_hash == 1
    assert tp.num_entries == 1
    assert tp.find(1) is None

    # removing non-existent key returns None
    assert tp.remove(999) is None


def test_tp_overwrite_replaces_entry():
    tp = TranslationPage(0, total_frames=512)
    old = make_entry(key_hash=1, data_page=10)
    tp.insert(old)
    assert tp.used_frames == 1

    new = make_entry(key_hash=1, data_page=20)
    tp.insert(new)
    assert tp.num_entries == 1
    assert tp.find(1).data_page_id == 20
    assert tp.used_frames == 1


def test_tp_has_space():
    tp = TranslationPage(0, total_frames=4)
    assert tp.has_space(1)
    assert tp.has_space(4)
    assert not tp.has_space(5)

    for i in range(4):
        tp.insert(make_entry(key_hash=i))
    assert not tp.has_space(1)
    assert tp.free_frames == 0


def test_tp_utilization():
    tp = TranslationPage(0, total_frames=10)
    assert tp.utilization == 0.0
    tp.insert(make_entry(key_hash=1))
    assert tp.utilization == 0.1
    tp.insert(make_entry(key_hash=2))
    assert tp.utilization == 0.2


def test_tp_inline_ratio():
    tp = TranslationPage(0, total_frames=512)
    tp.insert(make_entry(key_hash=1, inline=False))
    assert tp.inline_ratio == 0.0

    tp.insert(make_entry(key_hash=2, inline=True, key_size=8, value_size=8))
    assert tp.inline_ratio == 0.5


def test_tp_evict_one_inline():
    tp = TranslationPage(0, total_frames=512)
    tp.insert(make_entry(key_hash=1, inline=False))
    tp.insert(make_entry(key_hash=2, inline=True, key_size=8, value_size=8))
    tp.insert(make_entry(key_hash=3, inline=True, key_size=8, value_size=8))

    evicted = tp.evict_one_inline()
    assert evicted is not None
    assert evicted.is_inline
    assert tp.num_inline == 1

    # no inline entries to evict after removing both
    tp.evict_one_inline()
    assert tp.evict_one_inline() is None


def test_gmd_quadratic_probing():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)
    gmd = GMD(config, flash, metrics)

    # two keys that map to the same TP should use quadratic probing
    key_hash_a = 0
    key_hash_b = config.num_translation_pages  # same tp_id mod num_tps
    assert gmd._get_tp_id(key_hash_a, 0) == gmd._get_tp_id(key_hash_b, 0)

    # fill tp for key_hash_a
    tp_a = gmd.get_or_create_tp(gmd._get_tp_id(key_hash_a, 0))
    entry_a = make_entry(key_hash=key_hash_a, data_page=10)
    tp_a.insert(entry_a)

    # find_entry should locate entry_a
    found_tp, found_entry = gmd.find_entry(key_hash_a)
    assert found_entry is not None
    assert found_entry.data_page_id == 10


def test_gmd_find_tp_for_insert():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)
    gmd = GMD(config, flash, metrics)

    tp = gmd.find_tp_for_insert(42, frames_needed=1)
    assert tp is not None
    assert tp.has_space(1)


def test_gmd_compute_frames():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)
    gmd = GMD(config, flash, metrics)

    # 32 bytes → 1 frame (entry_size=32)
    assert gmd.compute_frames(32) == 1
    # 33 bytes → 2 frames
    assert gmd.compute_frames(33) == 2
    # 64 bytes → 2 frames
    assert gmd.compute_frames(64) == 2


def test_key_hash_deterministic():
    mask = (1 << 27) - 1
    h1 = compute_key_hash(b"test_key", mask)
    h2 = compute_key_hash(b"test_key", mask)
    assert h1 == h2
    # different keys should (almost certainly) produce different hashes
    h3 = compute_key_hash(b"other_key", mask)
    assert h1 != h3


if __name__ == "__main__":
    test_tp_insert_and_find()
    test_tp_remove()
    test_tp_overwrite_replaces_entry()
    test_tp_has_space()
    test_tp_utilization()
    test_tp_inline_ratio()
    test_tp_evict_one_inline()
    test_gmd_quadratic_probing()
    test_gmd_find_tp_for_insert()
    test_gmd_compute_frames()
    test_key_hash_deterministic()
    print("all mapping tests passed")
