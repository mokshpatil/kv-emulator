import math
import random
from src.inlining import InlineContext


class LinearRegressionPolicy:
    """Online linear regression to decide inlining based on KV features."""

    def __init__(self, entry_size=32, warmup=1000, lr=0.01, retrain_interval=500):
        self.entry_size = entry_size
        self.warmup = warmup
        self.lr = lr
        self.retrain_interval = retrain_interval

        # weight vector: [value_size_norm, total_size_norm, cmt_miss_rate]
        self.weights = [0.0, 0.0, 0.0]
        self.bias = 0.0

        self._rng = random.Random(42)
        self._feedback_count = 0
        self._trained = False
        self._buffer = []
        self._steps_since_train = 0

    def _extract_features(self, ctx):
        total = ctx.key_size + ctx.value_size + 12
        return [
            ctx.value_size / self.entry_size,
            total / (self.entry_size * 16),
            1.0 - ctx.cmt_hit_rate,
        ]

    def should_inline(self, ctx: InlineContext) -> bool:
        if not self._trained:
            # explore: randomly inline 50% during warmup
            return self._rng.random() < 0.5
        features = self._extract_features(ctx)
        score = self.bias
        for w, f in zip(self.weights, features):
            score += w * f
        return score > 0.0

    def update(self, _ctx: InlineContext):
        pass

    def feedback(self, ctx: InlineContext, was_inline: bool, flash_reads: int):
        # reward: positive means inlining was beneficial
        # on CMT miss: inline=1 read, regular=2 reads
        if was_inline:
            reward = 1.0 if flash_reads <= 1 else -0.5
        else:
            reward = -1.0 if flash_reads >= 2 else 0.5

        features = self._extract_features(ctx)
        self._buffer.append((features, reward))
        self._feedback_count += 1
        self._steps_since_train += 1

        if not self._trained and self._feedback_count >= self.warmup:
            self._train()
            self._trained = True
        elif self._trained and self._steps_since_train >= self.retrain_interval:
            self._train()

    def _train(self):
        # SGD over recent buffer
        batch = self._buffer[-self.retrain_interval:] if self._trained else self._buffer
        for features, reward in batch:
            pred = self.bias
            for w, f in zip(self.weights, features):
                pred += w * f
            error = reward - pred
            self.bias += self.lr * error
            for i in range(len(self.weights)):
                self.weights[i] += self.lr * error * features[i]
        self._steps_since_train = 0


class EpsilonGreedyPolicy:
    """Contextual bandit with epsilon-greedy exploration."""

    def __init__(self, entry_size=32, warmup=500, epsilon=0.2, decay=0.999,
                 num_bins=8):
        self.entry_size = entry_size
        self.warmup = warmup
        self.epsilon = epsilon
        self.decay = decay
        self.num_bins = num_bins

        self._ios_seen = 0
        self._rng = random.Random(42)

        # per-bin reward tracking: bin -> {inline: [sum, count], regular: [sum, count]}
        self._rewards = {}

    def _get_bin(self, ctx):
        total = ctx.key_size + ctx.value_size + 12
        frames = max(1, math.ceil(total / self.entry_size))
        return min(frames, self.num_bins) - 1

    def should_inline(self, ctx: InlineContext) -> bool:
        self._ios_seen += 1
        if self._ios_seen < self.warmup:
            # explore during warmup
            return self._rng.random() < 0.5

        bin_id = self._get_bin(ctx)
        stats = self._rewards.get(bin_id)
        if stats is None:
            return self._rng.random() < 0.5

        # epsilon-greedy
        if self._rng.random() < self.epsilon:
            return self._rng.random() < 0.5

        inline_avg = stats["inline"][0] / max(1, stats["inline"][1])
        regular_avg = stats["regular"][0] / max(1, stats["regular"][1])
        return inline_avg > regular_avg

    def update(self, _ctx: InlineContext):
        self.epsilon = max(0.01, self.epsilon * self.decay)

    def feedback(self, ctx: InlineContext, was_inline: bool, flash_reads: int):
        bin_id = self._get_bin(ctx)
        if bin_id not in self._rewards:
            self._rewards[bin_id] = {
                "inline": [0.0, 0],
                "regular": [0.0, 0],
            }

        # reward = negative flash reads (fewer reads = higher reward)
        reward = -flash_reads
        arm = "inline" if was_inline else "regular"
        self._rewards[bin_id][arm][0] += reward
        self._rewards[bin_id][arm][1] += 1
