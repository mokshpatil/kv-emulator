import hashlib
import math
import struct
from dataclasses import dataclass

from src.config import SSDConfig
from src.flash import Flash
from src.metrics import Metrics


def compute_key_hash(key: bytes, mask: int) -> int:
    # deterministic hash using md5 truncated to required bits
    digest = hashlib.md5(key if isinstance(key, bytes) else key.encode()).digest()
    value = struct.unpack("<I", digest[:4])[0]
    return value & mask


@dataclass
class MappingEntry:
    key_hash: int
    key_size: int
    value_size: int
    is_inline: bool
    data_page_id: int       # -1 if inline
    frames_used: int        # 1 for regular, >1 for inline


class TranslationPage:
    def __init__(self, tp_id: int, total_frames: int):
        self.tp_id = tp_id
        self.total_frames = total_frames
        self.used_frames = 0
        self.entries = {}       # key_hash -> MappingEntry
        self.num_inline = 0
        self.flash_page_id = -1 # assigned when persisted to flash

    @property
    def free_frames(self):
        return self.total_frames - self.used_frames

    @property
    def num_entries(self):
        return len(self.entries)

    @property
    def utilization(self):
        return self.used_frames / self.total_frames if self.total_frames > 0 else 0.0

    @property
    def inline_ratio(self):
        return self.num_inline / self.num_entries if self.num_entries > 0 else 0.0

    def has_space(self, frames_needed):
        return self.free_frames >= frames_needed

    def find(self, key_hash):
        return self.entries.get(key_hash)

    def insert(self, entry: MappingEntry):
        # if key already exists, remove old entry first
        if entry.key_hash in self.entries:
            self.remove(entry.key_hash)
        self.entries[entry.key_hash] = entry
        self.used_frames += entry.frames_used
        if entry.is_inline:
            self.num_inline += 1

    def remove(self, key_hash):
        entry = self.entries.pop(key_hash, None)
        if entry is None:
            return None
        self.used_frames -= entry.frames_used
        if entry.is_inline:
            self.num_inline -= 1
        return entry

    def evict_one_inline(self):
        # pick an arbitrary inline entry to evict (convert to regular)
        for kh, entry in self.entries.items():
            if entry.is_inline:
                return self.remove(kh)
        return None


class GMD:
    def __init__(self, config: SSDConfig, flash: Flash, metrics: Metrics):
        self.config = config
        self.flash = flash
        self.metrics = metrics
        self.num_tps = config.num_translation_pages
        self.frames_per_tp = config.frames_per_tp
        self.hash_mask = config.hash_mask
        self.max_retry = config.mapping.max_retry

        # lazily allocated translation pages
        self._pages = {}

    def _get_tp_id(self, key_hash, retry=0):
        # quadratic probing
        return (key_hash + retry * retry) % self.num_tps

    def get_or_create_tp(self, tp_id):
        if tp_id not in self._pages:
            tp = TranslationPage(tp_id, self.frames_per_tp)
            tp.flash_page_id = self.flash.allocate_page()
            self._pages[tp_id] = tp
        return self._pages[tp_id]

    def get_tp(self, tp_id):
        return self._pages.get(tp_id)

    def find_entry(self, key_hash):
        # search for a key across translation pages using quadratic probing
        for retry in range(self.max_retry):
            tp_id = self._get_tp_id(key_hash, retry)
            tp = self.get_tp(tp_id)
            if tp is None:
                return None, None
            entry = tp.find(key_hash)
            if entry is not None:
                return tp, entry
        return None, None

    def find_tp_for_insert(self, key_hash, frames_needed):
        # find a translation page with space for the entry
        for retry in range(self.max_retry):
            tp_id = self._get_tp_id(key_hash, retry)
            tp = self.get_or_create_tp(tp_id)

            # check if key already exists here
            if tp.find(key_hash) is not None:
                return tp

            # check if there is space
            if tp.has_space(frames_needed):
                return tp
        return None

    def compute_frames(self, total_size):
        # how many frames needed for a given total entry size
        entry_size = self.config.mapping.entry_size
        return max(1, math.ceil(total_size / entry_size))

    @property
    def total_entries(self):
        return sum(tp.num_entries for tp in self._pages.values())

    @property
    def total_inline(self):
        return sum(tp.num_inline for tp in self._pages.values())
