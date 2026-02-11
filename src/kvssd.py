from src.config import SSDConfig
from src.metrics import Metrics
from src.flash import Flash
from src.mapping import GMD, MappingEntry, compute_key_hash
from src.cmt import CMT
from src.inlining import InlineContext, create_policy


class KVSSD:
    def __init__(self, config: SSDConfig):
        self.config = config
        self.metrics = Metrics()
        self.flash = Flash(config, self.metrics)
        self.gmd = GMD(config, self.flash, self.metrics)

        read_cap, write_cap = config.cmt_entry_capacity
        self.cmt = CMT(read_cap, write_cap)
        self.policy = create_policy(config)

    def get(self, key: bytes) -> bool:
        self.metrics.total_gets += 1
        self.metrics.begin_request()

        key_hash = compute_key_hash(key, self.config.hash_mask)

        # check CMT first
        entry = self.cmt.lookup(key_hash)
        if entry is not None:
            self.metrics.cmt_hits += 1
            if not entry.is_inline:
                # need to read the data page
                self.flash.read_page(entry.data_page_id, "data")
            # inline entries in CMT: value was already available (but KVPack
            # policy doesn't cache inline entries, so this path is rare)
            self.metrics.end_get_request()
            return True

        # CMT miss: find translation page via GMD
        self.metrics.cmt_misses += 1
        tp, entry = self.gmd.find_entry(key_hash)

        if tp is None or entry is None:
            self.metrics.end_get_request()
            return False

        # reading the translation page from flash
        self.flash.read_page(tp.flash_page_id, "translation")

        if entry.is_inline:
            # value is in the translation page, no extra read needed
            self.metrics.end_get_request()
            return True

        # regular entry: cache it in CMT, then read data page
        self.cmt.insert(key_hash, entry)
        self.flash.read_page(entry.data_page_id, "data")
        self.metrics.end_get_request()
        return True

    def put(self, key: bytes, value_size: int):
        self.metrics.total_puts += 1
        key_hash = compute_key_hash(key, self.config.hash_mask)
        key_size = len(key) if isinstance(key, bytes) else len(key.encode())

        # feed the profiler
        ctx = InlineContext(
            key_size=key_size,
            value_size=value_size,
            cmt_hit_rate=self.metrics.cmt_hit_rate,
        )
        self.policy.update(ctx)

        # decide inline or regular
        should_inline = self.policy.should_inline(ctx)

        if should_inline:
            self._put_inline(key_hash, key_size, value_size)
        else:
            self._put_regular(key_hash, key_size, value_size)

    def _put_inline(self, key_hash, key_size, value_size):
        # compute frames needed: 8B hash + 2B key_len + 2B val_len + value
        total_size = 12 + key_size + value_size
        frames = self.gmd.compute_frames(total_size)

        tp = self.gmd.find_tp_for_insert(key_hash, frames)
        if tp is None:
            # no space with probing, fall back to regular
            self._put_regular(key_hash, key_size, value_size)
            return

        # check if we need to evict an inline entry to make room
        if not tp.has_space(frames):
            evicted = tp.evict_one_inline()
            if evicted is not None:
                self._convert_to_regular(tp, evicted)
            if not tp.has_space(frames):
                self._put_regular(key_hash, key_size, value_size)
                return

        entry = MappingEntry(
            key_hash=key_hash,
            key_size=key_size,
            value_size=value_size,
            is_inline=True,
            data_page_id=-1,
            frames_used=frames,
        )
        tp.insert(entry)
        self.flash.write_page(tp.flash_page_id, "translation")
        self.cmt.invalidate(key_hash)
        self.metrics.inline_entries += 1

    def _put_regular(self, key_hash, key_size, value_size):
        tp = self.gmd.find_tp_for_insert(key_hash, 1)
        if tp is None:
            return

        # allocate a data page and write the value
        data_page = self.flash.allocate_page()
        self.flash.write_page(data_page, "data")

        entry = MappingEntry(
            key_hash=key_hash,
            key_size=key_size,
            value_size=value_size,
            is_inline=False,
            data_page_id=data_page,
            frames_used=1,
        )
        tp.insert(entry)
        self.flash.write_page(tp.flash_page_id, "translation")
        self.cmt.insert(key_hash, entry)
        self.metrics.regular_entries += 1

    def _convert_to_regular(self, tp, old_entry):
        # convert an evicted inline entry to a regular entry
        data_page = self.flash.allocate_page()
        self.flash.write_page(data_page, "data")

        new_entry = MappingEntry(
            key_hash=old_entry.key_hash,
            key_size=old_entry.key_size,
            value_size=old_entry.value_size,
            is_inline=False,
            data_page_id=data_page,
            frames_used=1,
        )
        tp.insert(new_entry)
        self.cmt.insert(old_entry.key_hash, new_entry)
        self.metrics.inline_entries -= 1
        self.metrics.regular_entries += 1
        self.metrics.inline_to_regular += 1

    def delete(self, key: bytes) -> bool:
        self.metrics.total_deletes += 1
        key_hash = compute_key_hash(key, self.config.hash_mask)

        tp, entry = self.gmd.find_entry(key_hash)
        if tp is None or entry is None:
            return False

        was_inline = entry.is_inline
        tp.remove(key_hash)
        self.cmt.invalidate(key_hash)

        if was_inline:
            self.metrics.inline_entries -= 1
        else:
            self.metrics.regular_entries -= 1
            if entry.data_page_id >= 0:
                self.flash.free_page(entry.data_page_id)
        return True
