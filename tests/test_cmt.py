import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cmt import CMT
from src.mapping import MappingEntry


def make_entry(key_hash=1, data_page=0, inline=False):
    return MappingEntry(
        key_hash=key_hash,
        key_size=32,
        value_size=64,
        is_inline=inline,
        data_page_id=-1 if inline else data_page,
        frames_used=1,
    )


def test_lookup_hit_and_miss():
    cmt = CMT(read_capacity=10, write_capacity=10)
    entry = make_entry(key_hash=1, data_page=100)
    cmt.insert(1, entry)

    assert cmt.lookup(1) is entry
    assert cmt.lookup(999) is None


def test_lru_eviction():
    cmt = CMT(read_capacity=3, write_capacity=3)
    for i in range(3):
        cmt.insert(i, make_entry(key_hash=i, data_page=i))
    assert cmt.size == 3

    # inserting a 4th should evict the LRU (key_hash=0)
    cmt.insert(3, make_entry(key_hash=3, data_page=3))
    assert cmt.size == 3
    assert cmt.lookup(0) is None
    assert cmt.lookup(3) is not None


def test_lru_access_refreshes():
    cmt = CMT(read_capacity=3, write_capacity=3)
    for i in range(3):
        cmt.insert(i, make_entry(key_hash=i, data_page=i))

    # access key 0 to refresh it
    cmt.lookup(0)

    # insert key 3 — should evict key 1 (now the LRU), not key 0
    cmt.insert(3, make_entry(key_hash=3, data_page=3))
    assert cmt.lookup(0) is not None
    assert cmt.lookup(1) is None


def test_inline_entries_not_cached():
    cmt = CMT(read_capacity=10, write_capacity=10)
    inline_entry = make_entry(key_hash=1, inline=True)
    cmt.insert(1, inline_entry)
    assert cmt.lookup(1) is None
    assert cmt.size == 0


def test_invalidate():
    cmt = CMT(read_capacity=10, write_capacity=10)
    cmt.insert(1, make_entry(key_hash=1, data_page=100))
    assert cmt.lookup(1) is not None

    cmt.invalidate(1)
    assert cmt.lookup(1) is None
    assert cmt.size == 0

    # invalidating non-existent key is a no-op
    cmt.invalidate(999)


def test_update_data_page():
    cmt = CMT(read_capacity=10, write_capacity=10)
    cmt.insert(1, make_entry(key_hash=1, data_page=100))
    cmt.insert(2, make_entry(key_hash=2, data_page=200))

    # GC relocates page 100 → 500
    cmt.update_data_page(100, 500)

    assert cmt.lookup(1).data_page_id == 500
    assert cmt.lookup(2).data_page_id == 200  # unchanged


def test_update_data_page_skips_inline():
    cmt = CMT(read_capacity=10, write_capacity=10)
    # force-insert an inline entry for testing
    inline = make_entry(key_hash=1, inline=True)
    cmt._read_cache[1] = inline

    cmt.update_data_page(-1, 500)
    assert cmt.lookup(1).data_page_id == -1  # unchanged


if __name__ == "__main__":
    test_lookup_hit_and_miss()
    test_lru_eviction()
    test_lru_access_refreshes()
    test_inline_entries_not_cached()
    test_invalidate()
    test_update_data_page()
    test_update_data_page_skips_inline()
    print("all cmt tests passed")
