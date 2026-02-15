from src.config import SSDConfig
from src.metrics import Metrics
from src.flash import Flash
from src.mapping import GMD, MappingEntry, compute_key_hash
from src.cmt import CMT
from src.gc import GarbageCollector
from src.inlining import InlineContext, create_policy


class KVSSD:
    def __init__(self, config: SSDConfig):
        self.config = config
        self.metrics = Metrics(read_latency_us=config.flash.read_latency_us)
        self.flash = Flash(config, self.metrics)
        self.gmd = GMD(config, self.flash, self.metrics)

        read_cap, write_cap = config.cmt_entry_capacity
        self.cmt = CMT(read_cap, write_cap)
        self.policy = create_policy(config)

        self.gc = GarbageCollector(self.flash, self.metrics)
        self.gc.set_relocators(self._relocate_data_page, self._relocate_tp_page)
        self._epoch = 0
        self._gc_retries = 0

    def _allocate_with_gc(self):
        # try to allocate; on flash-full, force GC and retry
        try:
            return self.flash.allocate_page()
        except RuntimeError:
            rounds = self.gc.run(max_rounds=self.flash.total_blocks, force=True)
            self._gc_retries += 1
            if rounds == 0:
                raise RuntimeError("flash full: GC could not reclaim any space")
            return self.flash.allocate_page()

    def get(self, key: bytes) -> bool:
        self.metrics.total_gets += 1
        self.metrics.begin_request()

        key_hash = compute_key_hash(key, self.config.hash_mask)

        # check CMT first
        entry = self.cmt.lookup(key_hash)
        if entry is not None:
            self.metrics.cmt_hits += 1
            if not entry.is_inline:
                self.flash.read_page(entry.data_page_id, "data")
            self.metrics.end_get_request()
            return True

        # CMT miss: find translation page via GMD
        self.metrics.cmt_misses += 1
        tp, entry = self.gmd.find_entry(key_hash)

        if tp is None or entry is None:
            self.metrics.end_get_request()
            return False

        # read the translation page from flash
        self.flash.read_page(tp.flash_page_id, "translation")

        if entry.is_inline:
            self._send_feedback(entry, flash_reads=1)
            self.metrics.end_get_request()
            return True

        # regular entry: cache it in CMT, then read data page
        self.cmt.insert(key_hash, entry)
        self.flash.read_page(entry.data_page_id, "data")
        self._send_feedback(entry, flash_reads=2)
        self.metrics.end_get_request()
        return True

    def _send_feedback(self, entry, flash_reads):
        # provide reward signal to ML policies
        if not hasattr(self.policy, "feedback"):
            return
        ctx = self._build_context(entry.key_hash, entry.key_size, entry.value_size)
        self.policy.feedback(ctx, entry.is_inline, flash_reads)

    def _build_context(self, key_hash, key_size, value_size):
        # look up TP for utilization/inline_ratio context
        tp_util, tp_inl = 0.0, 0.0
        tp = self.gmd.get_tp(self.gmd._get_tp_id(key_hash, 0))
        if tp is not None:
            tp_util = tp.utilization
            tp_inl = tp.inline_ratio
        return InlineContext(
            key_size=key_size,
            value_size=value_size,
            tp_utilization=tp_util,
            tp_inline_ratio=tp_inl,
            cmt_hit_rate=self.metrics.cmt_hit_rate,
            epoch=self._epoch,
        )

    def put(self, key: bytes, value_size: int):
        self.metrics.total_puts += 1
        self.metrics.host_writes += 1
        self._epoch += 1
        key_hash = compute_key_hash(key, self.config.hash_mask)
        key_size = len(key) if isinstance(key, bytes) else len(key.encode())

        # free old data page if overwriting an existing regular entry
        self._free_old_entry(key_hash)

        ctx = self._build_context(key_hash, key_size, value_size)
        self.policy.update(ctx)
        should_inline = self.policy.should_inline(ctx)

        if should_inline:
            self._put_inline(key_hash, key_size, value_size)
        else:
            self._put_regular(key_hash, key_size, value_size)

        # trigger GC if flash utilization is high
        if self.gc.should_run():
            self.gc.run()

    def _free_old_entry(self, key_hash):
        # find and free old data page when overwriting a key
        _tp, old_entry = self.gmd.find_entry(key_hash)
        if old_entry is not None and not old_entry.is_inline:
            if old_entry.data_page_id >= 0:
                self.flash.free_page(old_entry.data_page_id)

    def _put_inline(self, key_hash, key_size, value_size):
        # compute frames needed: 8B hash + 2B key_len + 2B val_len + value
        total_size = 12 + key_size + value_size
        frames = self.gmd.compute_frames(total_size)

        tp = self.gmd.find_tp_for_insert(key_hash, frames)
        if tp is None:
            self._put_regular(key_hash, key_size, value_size)
            return

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

        data_page = self._allocate_with_gc()
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
        data_page = self._allocate_with_gc()
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

    def _relocate_data_page(self, old_page_id, new_page_id):
        # update all mapping entries and CMT entries pointing to old_page_id
        for tp in self.gmd._pages.values():
            for entry in tp.entries.values():
                if not entry.is_inline and entry.data_page_id == old_page_id:
                    entry.data_page_id = new_page_id
        # update CMT cached entries
        self.cmt.update_data_page(old_page_id, new_page_id)

    def _relocate_tp_page(self, old_page_id, new_page_id):
        # update the translation page's flash_page_id
        for tp in self.gmd._pages.values():
            if tp.flash_page_id == old_page_id:
                tp.flash_page_id = new_page_id
                return
