from dataclasses import dataclass, field
import math


@dataclass
class FlashConfig:
    page_size: int = 16384          # 16KB
    pages_per_block: int = 256      # block = 4MB
    read_latency_us: float = 45.0
    write_latency_us: float = 200.0
    erase_latency_us: float = 2000.0


@dataclass
class MappingConfig:
    entry_size: int = 32            # regular mapping entry size in bytes
    hash_bits: int = 27
    max_retry: int = 8              # quadratic probing bound
    ppa_size: int = 8               # physical page address size in bytes
    data_alignment: int = 512       # sector-aligned KV pair storage


@dataclass
class CMTConfig:
    budget_ratio: float = 0.005     # 0.5% of SSD capacity
    read_write_ratio: float = 1.0   # 1:1 read:write cache split


@dataclass
class InliningConfig:
    mode: str = "kvpack_s"          # baseline, kvpack_s, kvpack_d
    profiler_warmup: int = 1000000  # I/Os before setting threshold
    profiler_interval: int = 100000 # KVPack-D re-query interval


@dataclass
class SSDConfig:
    capacity_bytes: int = 64 * (1024 ** 3)  # 64GB
    flash: FlashConfig = field(default_factory=FlashConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    cmt: CMTConfig = field(default_factory=CMTConfig)
    inlining: InliningConfig = field(default_factory=InliningConfig)

    @property
    def frames_per_tp(self):
        return self.flash.page_size // self.mapping.entry_size

    @property
    def max_kv_pairs(self):
        return self.capacity_bytes // self.mapping.data_alignment

    @property
    def num_translation_pages(self):
        return self.max_kv_pairs // self.frames_per_tp

    @property
    def mapping_table_size(self):
        return self.num_translation_pages * self.flash.page_size

    @property
    def cmt_entry_capacity(self):
        # total CMT budget in entries, split between read and write
        budget = int(self.capacity_bytes * self.cmt.budget_ratio)
        total_entries = budget // self.mapping.entry_size
        ratio = self.cmt.read_write_ratio
        read_entries = int(total_entries * ratio / (1 + ratio))
        write_entries = total_entries - read_entries
        return read_entries, write_entries

    @property
    def hash_mask(self):
        return (1 << self.mapping.hash_bits) - 1


def small_config():
    # smaller config for development and testing
    return SSDConfig(
        capacity_bytes=256 * (1024 ** 2),  # 256MB
        inlining=InliningConfig(profiler_warmup=1000, profiler_interval=500),
    )


def gc_config():
    # tight config that forces GC to trigger with modest workloads
    return SSDConfig(
        capacity_bytes=2 * (1024 ** 2),  # 2MB = 128 pages
        flash=FlashConfig(page_size=16384, pages_per_block=16),
        inlining=InliningConfig(profiler_warmup=50, profiler_interval=50),
    )
