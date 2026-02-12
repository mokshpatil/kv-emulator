from dataclasses import dataclass, field


@dataclass
class Metrics:
    # flash read counters
    tp_reads: int = 0
    data_reads: int = 0
    flash_writes: int = 0
    flash_erases: int = 0

    # CMT counters
    cmt_hits: int = 0
    cmt_misses: int = 0

    # mapping counters
    total_puts: int = 0
    total_gets: int = 0
    total_deletes: int = 0
    inline_entries: int = 0
    regular_entries: int = 0
    inline_to_regular: int = 0

    # GC counters
    gc_invocations: int = 0
    gc_pages_copied: int = 0
    host_writes: int = 0

    # per-request state
    _request_flash_reads: int = field(default=0, repr=False)

    # histogram: GET requests by flash read count
    reads_by_flash_count: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})

    # latency tracking: list of per-GET latencies in microseconds
    _get_latencies: list = field(default_factory=list, repr=False)
    read_latency_us: float = field(default=45.0, repr=False)

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
    def waf(self):
        if self.host_writes == 0:
            return 0.0
        return self.flash_writes / self.host_writes

    @property
    def reads_with_one_or_fewer(self):
        total = sum(self.reads_by_flash_count.values())
        if total == 0:
            return 0.0
        good = self.reads_by_flash_count.get(0, 0) + self.reads_by_flash_count.get(1, 0)
        return good / total

    @property
    def avg_read_latency(self):
        if not self._get_latencies:
            return 0.0
        return sum(self._get_latencies) / len(self._get_latencies)

    @property
    def p50_read_latency(self):
        return self._percentile(50)

    @property
    def p99_read_latency(self):
        return self._percentile(99)

    @property
    def p999_read_latency(self):
        return self._percentile(99.9)

    def _percentile(self, pct):
        if not self._get_latencies:
            return 0.0
        s = sorted(self._get_latencies)
        idx = int(len(s) * pct / 100)
        idx = min(idx, len(s) - 1)
        return s[idx]

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
        latency = count * self.read_latency_us
        self._get_latencies.append(latency)

    def latency_cdf(self, buckets=50):
        # return (latency, fraction) pairs for plotting
        if not self._get_latencies:
            return []
        s = sorted(self._get_latencies)
        n = len(s)
        step = max(1, n // buckets)
        points = []
        for i in range(0, n, step):
            points.append((s[i], (i + 1) / n))
        if points[-1][1] < 1.0:
            points.append((s[-1], 1.0))
        return points

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
            "avg_latency_us": f"{self.avg_read_latency:.1f}",
            "p50_latency_us": f"{self.p50_read_latency:.1f}",
            "p99_latency_us": f"{self.p99_read_latency:.1f}",
            "waf": f"{self.waf:.2f}",
        }

    def print_summary(self):
        print("\n--- Metrics Summary ---")
        for k, v in self.summary().items():
            print(f"  {k}: {v}")
        print()
