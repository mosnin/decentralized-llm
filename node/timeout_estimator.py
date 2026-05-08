import time
from collections import deque
from dataclasses import dataclass


@dataclass
class LatencySample:
    latency_ms: float
    model_name: str
    token_count: int
    timestamp: float


class TimeoutEstimator:
    """
    Estimates appropriate timeout values using exponential moving averages
    and percentile-based safety margins.

    Timeout = p95_latency * safety_multiplier, clamped to [min_timeout_ms, max_timeout_ms].
    Falls back to default_timeout_ms when insufficient data.
    """

    def __init__(
        self,
        min_timeout_ms: float = 1_000.0,
        max_timeout_ms: float = 300_000.0,
        default_timeout_ms: float = 30_000.0,
        safety_multiplier: float = 1.5,
        window_size: int = 100,
    ):
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        self.default_timeout_ms = default_timeout_ms
        self.safety_multiplier = safety_multiplier
        self.window_size = window_size
        self._samples: dict[str, deque[LatencySample]] = {}

    def record(self, model_name: str, latency_ms: float, token_count: int = 0) -> None:
        """Record a latency observation for a model."""
        if model_name not in self._samples:
            self._samples[model_name] = deque(maxlen=self.window_size)
        self._samples[model_name].append(
            LatencySample(
                latency_ms=latency_ms,
                model_name=model_name,
                token_count=token_count,
                timestamp=time.time(),
            )
        )

    def _percentile(self, values: list[float], p: float) -> float:
        """Compute p-th percentile (0-100) of a sorted or unsorted list."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, int(len(sorted_vals) * p / 100) - 1)
        return sorted_vals[idx]

    def estimate(self, model_name: str, token_count: int = 0) -> float:
        """
        Return estimated timeout in milliseconds for a job.

        If token_count > 0 and we have enough samples, scale by tokens-per-ms rate.
        Falls back to default if fewer than 5 samples available.
        """
        samples = self._samples.get(model_name)
        if not samples or len(samples) < 5:
            return self.default_timeout_ms

        latencies = [s.latency_ms for s in samples]
        p95 = self._percentile(latencies, 95)
        raw = p95 * self.safety_multiplier

        # Scale by token count if we have token data
        if token_count > 0:
            token_samples = [s for s in samples if s.token_count > 0]
            if len(token_samples) >= 5:
                avg_ms_per_token = sum(s.latency_ms / s.token_count for s in token_samples) / len(
                    token_samples
                )
                raw = max(raw, avg_ms_per_token * token_count * self.safety_multiplier)

        return max(self.min_timeout_ms, min(self.max_timeout_ms, raw))

    def reset(self, model_name: str | None = None) -> None:
        """Clear samples for a model, or all models if None."""
        if model_name is None:
            self._samples.clear()
        else:
            self._samples.pop(model_name, None)

    def sample_count(self, model_name: str) -> int:
        samples = self._samples.get(model_name)
        return len(samples) if samples else 0
