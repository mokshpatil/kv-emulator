from dataclasses import dataclass
import math


@dataclass
class InlineContext:
    key_size: int
    value_size: int
    tp_utilization: float = 0.0
    tp_inline_ratio: float = 0.0
    cmt_hit_rate: float = 0.0
    epoch: int = 0


class BaselinePolicy:
    """TurboHash-style: inline only if value fits in PPA field (<= 8B)."""

    def __init__(self, ppa_size: int = 8):
        self.ppa_size = ppa_size

    def should_inline(self, ctx: InlineContext) -> bool:
        return ctx.value_size <= self.ppa_size

    def update(self, ctx: InlineContext):
        pass


class KVPackSPolicy:
    """Static threshold set once after profiling warmup period."""

    def __init__(self, entry_size: int = 32, warmup: int = 1000000):
        self.entry_size = entry_size
        self.warmup = warmup
        self.threshold = 0           # set after profiling
        self._frame_counts = {}      # frame_count -> access_count
        self._ios_seen = 0
        self._profiling_done = False

    def should_inline(self, ctx: InlineContext) -> bool:
        if not self._profiling_done:
            return False
        total = ctx.key_size + ctx.value_size + 12  # 8B hash + 2B key_len + 2B val_len
        return total <= self.threshold

    def update(self, ctx: InlineContext):
        if self._profiling_done:
            return
        self._ios_seen += 1
        # track how many frames this KV pair would need
        total = ctx.key_size + ctx.value_size + 12
        frames = max(1, math.ceil(total / self.entry_size))
        self._frame_counts[frames] = self._frame_counts.get(frames, 0) + 1

        if self._ios_seen >= self.warmup:
            self._set_threshold()

    def _set_threshold(self):
        if not self._frame_counts:
            self._profiling_done = True
            return
        # find most popular frame count
        popular_frames = max(self._frame_counts, key=self._frame_counts.get)
        self.threshold = popular_frames * self.entry_size
        self._profiling_done = True


class KVPackDPolicy:
    """Dynamic threshold adjusted periodically via continuous profiling."""

    def __init__(self, entry_size: int = 32, warmup: int = 1000000,
                 interval: int = 100000):
        self.entry_size = entry_size
        self.warmup = warmup
        self.interval = interval
        self.threshold = 0
        self._frame_counts = {}
        self._ios_seen = 0
        self._total_ios = 0
        self._initialized = False

    def should_inline(self, ctx: InlineContext) -> bool:
        if not self._initialized:
            return False
        total = ctx.key_size + ctx.value_size + 12
        return total <= self.threshold

    def update(self, ctx: InlineContext):
        self._total_ios += 1
        self._ios_seen += 1

        total = ctx.key_size + ctx.value_size + 12
        frames = max(1, math.ceil(total / self.entry_size))
        self._frame_counts[frames] = self._frame_counts.get(frames, 0) + 1

        if not self._initialized and self._total_ios >= self.warmup:
            self._recompute_threshold()
            self._initialized = True
        elif self._initialized and self._ios_seen >= self.interval:
            self._recompute_threshold()

    def _recompute_threshold(self):
        if not self._frame_counts:
            return
        popular_frames = max(self._frame_counts, key=self._frame_counts.get)
        new_threshold = popular_frames * self.entry_size
        # only increase threshold, never decrease (per KVPack-D design)
        if new_threshold > self.threshold:
            self.threshold = new_threshold
        # reset counters for next interval
        self._frame_counts.clear()
        self._ios_seen = 0


def create_policy(config):
    mode = config.inlining.mode
    entry_size = config.mapping.entry_size
    if mode == "baseline":
        return BaselinePolicy(ppa_size=config.mapping.ppa_size)
    elif mode == "kvpack_s":
        return KVPackSPolicy(
            entry_size=entry_size,
            warmup=config.inlining.profiler_warmup,
        )
    elif mode == "kvpack_d":
        return KVPackDPolicy(
            entry_size=entry_size,
            warmup=config.inlining.profiler_warmup,
            interval=config.inlining.profiler_interval,
        )
    elif mode == "ml_linear":
        from src.ml_policies import LinearRegressionPolicy
        return LinearRegressionPolicy(
            entry_size=entry_size,
            warmup=config.inlining.profiler_warmup,
            retrain_interval=config.inlining.profiler_interval,
        )
    elif mode == "ml_bandit":
        from src.ml_policies import EpsilonGreedyPolicy
        return EpsilonGreedyPolicy(
            entry_size=entry_size,
            warmup=config.inlining.profiler_warmup,
        )
    else:
        raise ValueError(f"unknown inlining mode: {mode}")
