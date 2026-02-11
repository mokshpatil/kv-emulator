from dataclasses import dataclass, field


@dataclass
class Metrics:
    # flash read counters
    tp_reads: int = 0               # translation page reads
    data_reads: int = 0             # data page reads
    flash_writes: int = 0           # total flash page writes
    flash_erases: int = 0           # block erases

    # CMT counters
    cmt_hits: int = 0
    cmt_misses: int = 0

    # mapping counters
    total_puts: int = 0
    total_gets: int = 0
    total_deletes: int = 0
    inline_entries: int = 0         # current count of inline entries
    regular_entries: int = 0        # current count of regular entries
    inline_to_regular: int = 0      # eviction conversions

    # per-request tracking for current request
    _request_flash_reads: int = field(default=0, repr=False)

    # histogram: requests by flash read count
    reads_by_flash_count: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})

    @property
    def total_flash_reads(self):
        return self.tp_reads + self.data_reads

    @property
    def total_ops(self):
        return self.total_puts + self.total_gets + self.total_deletes

    @property
    def cmt_hit_rate(self):
        total = self.cmt_hits + self.cmt_misses
        return self.cmt_hits / total if total > 0 else 0.0

    @property
    def inline_ratio(self):
        total = self.inline_entries + self.regular_entries
        return self.inline_entries / total if total > 0 else 0.0

    @property
    def reads_with_one_or_fewer(self):
        # percentage of GET requests completing in <= 1 flash read
        total = sum(self.reads_by_flash_count.values())
        if total == 0:
            return 0.0
        good = self.reads_by_flash_count.get(0, 0) + self.reads_by_flash_count.get(1, 0)
        return good / total

    def begin_request(self):
        self._request_flash_reads = 0

    def record_flash_read(self, page_type):
        self._request_flash_reads += 1
        if page_type == "translation":
            self.tp_reads += 1
        else:
            self.data_reads += 1

    def end_get_request(self):
        count = self._request_flash_reads
        if count not in self.reads_by_flash_count:
            self.reads_by_flash_count[count] = 0
        self.reads_by_flash_count[count] += 1

    def summary(self):
        return {
            "total_ops": self.total_ops,
            "total_gets": self.total_gets,
            "total_puts": self.total_puts,
            "flash_reads": self.total_flash_reads,
            "tp_reads": self.tp_reads,
            "data_reads": self.data_reads,
            "flash_writes": self.flash_writes,
            "cmt_hit_rate": f"{self.cmt_hit_rate:.4f}",
            "inline_ratio": f"{self.inline_ratio:.4f}",
            "inline_entries": self.inline_entries,
            "regular_entries": self.regular_entries,
            "inline_to_regular": self.inline_to_regular,
            "reads_leq_1_flash": f"{self.reads_with_one_or_fewer:.4f}",
            "reads_by_flash_count": dict(sorted(self.reads_by_flash_count.items())),
        }

    def print_summary(self):
        print("\n--- Metrics Summary ---")
        for k, v in self.summary().items():
            print(f"  {k}: {v}")
        print()
