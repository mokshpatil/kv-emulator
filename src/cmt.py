from collections import OrderedDict

from src.mapping import MappingEntry


class CMT:
    def __init__(self, read_capacity: int, write_capacity: int):
        self.read_capacity = read_capacity
        # LRU cache: key_hash -> MappingEntry
        self._read_cache = OrderedDict()

    def lookup(self, key_hash: int):
        if key_hash in self._read_cache:
            self._read_cache.move_to_end(key_hash)
            return self._read_cache[key_hash]
        return None

    def insert(self, key_hash: int, entry: MappingEntry):
        # KVPack policy: inline entries are NOT cached in CMT
        if entry.is_inline:
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

    @property
    def size(self):
        return len(self._read_cache)
