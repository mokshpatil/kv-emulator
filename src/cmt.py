from collections import OrderedDict

from src.mapping import MappingEntry


class CMT:
    def __init__(self, read_capacity: int, write_capacity: int):
        self.read_capacity = read_capacity
        self.write_capacity = write_capacity

        # LRU caches: key_hash -> MappingEntry
        self._read_cache = OrderedDict()
        # write cache tracks dirty translation page IDs
        self._write_cache = OrderedDict()

    def lookup(self, key_hash: int):
        if key_hash in self._read_cache:
            self._read_cache.move_to_end(key_hash)
            return self._read_cache[key_hash]
        return None

    def insert(self, key_hash: int, entry: MappingEntry, cache_inline: bool = False):
        # KVPack policy: inline entries are NOT cached in CMT
        if entry.is_inline and not cache_inline:
            return
        # evict LRU if at capacity
        while len(self._read_cache) >= self.read_capacity:
            self._read_cache.popitem(last=False)
        self._read_cache[key_hash] = entry

    def invalidate(self, key_hash: int):
        self._read_cache.pop(key_hash, None)

    def update_data_page(self, old_page_id: int, new_page_id: int):
        # update cached entries when GC relocates a data page
        for entry in self._read_cache.values():
            if not entry.is_inline and entry.data_page_id == old_page_id:
                entry.data_page_id = new_page_id

    def mark_dirty(self, tp_id: int):
        # track translation pages that have pending writes
        while len(self._write_cache) >= self.write_capacity:
            self._write_cache.popitem(last=False)
        self._write_cache[tp_id] = True

    def flush_dirty(self):
        # return list of dirty tp_ids and clear
        dirty = list(self._write_cache.keys())
        self._write_cache.clear()
        return dirty

    @property
    def read_size(self):
        return len(self._read_cache)

    @property
    def write_size(self):
        return len(self._write_cache)
