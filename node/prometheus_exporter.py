"""
Pure-Python Prometheus text-format metrics exporter for the decentralized-LLM network.

No external ``prometheus_client`` dependency required — works standalone.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class MetricType(StrEnum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ---------------------------------------------------------------------------
# Histogram helper
# ---------------------------------------------------------------------------


class HistogramMetric:
    """Tracks observations across configurable buckets (thread-safe)."""

    def __init__(self, buckets: list[float]) -> None:
        # Buckets must be sorted; append +Inf sentinel
        self._bounds: list[float] = sorted(buckets)
        if self._bounds[-1] != float("inf"):
            self._bounds.append(float("inf"))
        self._lock = threading.Lock()
        self._counts: list[int] = [0] * len(self._bounds)
        self._sum: float = 0.0
        self._total: int = 0

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._total += 1
            for i, bound in enumerate(self._bounds):
                if value <= bound:
                    self._counts[i] += 1

    def render(self, name: str, labels_str: str, help_text: str) -> str:
        """Return Prometheus text lines for this histogram."""
        lines: list[str] = []
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")

        with self._lock:
            counts = list(self._counts)
            total_sum = self._sum
            total_count = self._total

        # _counts is already cumulative (observe() increments every bucket >= value)
        for bound, count in zip(self._bounds, counts):
            le_val = "+Inf" if bound == float("inf") else _fmt_float(bound)
            if labels_str:
                bucket_labels = f'{labels_str},le="{le_val}"'
            else:
                bucket_labels = f'le="{le_val}"'
            lines.append(f"{name}_bucket{{{bucket_labels}}} {count}")

        if labels_str:
            lines.append(f"{name}_sum{{{labels_str}}} {_fmt_float(total_sum)}")
            lines.append(f"{name}_count{{{labels_str}}} {total_count}")
        else:
            lines.append(f"{name}_sum {_fmt_float(total_sum)}")
            lines.append(f"{name}_count {total_count}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PrometheusMetric dataclass
# ---------------------------------------------------------------------------


@dataclass
class PrometheusMetric:
    name: str
    help_text: str
    metric_type: MetricType
    labels: list[str] = field(default_factory=list)
    # Callable returns either a scalar or a dict mapping label-value tuples -> scalar
    value_fn: Callable[[], object] = field(default_factory=lambda: lambda: 0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PrometheusRegistry:
    """Thread-safe registry that renders metrics in Prometheus text exposition format."""

    def __init__(self) -> None:
        self._metrics: list[PrometheusMetric] = []
        self._lock = threading.Lock()

    def register(self, metric: PrometheusMetric) -> None:
        with self._lock:
            self._metrics.append(metric)

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics)

        if not metrics:
            return ""

        parts: list[str] = []
        for metric in metrics:
            value = metric.value_fn()

            if isinstance(value, str) and value.startswith("# HELP"):
                # Pre-rendered block (e.g. from HistogramMetric.render)
                parts.append(value)
                continue

            parts.append(f"# HELP {metric.name} {metric.help_text}")
            parts.append(f"# TYPE {metric.name} {metric.metric_type.value}")

            if isinstance(value, dict):
                for label_values, scalar in value.items():
                    if metric.labels and label_values:
                        kv = ",".join(f'{k}="{v}"' for k, v in zip(metric.labels, label_values))
                        parts.append(f"{metric.name}{{{kv}}} {_fmt_scalar(scalar)}")
                    else:
                        parts.append(f"{metric.name} {_fmt_scalar(scalar)}")
            else:
                parts.append(f"{metric.name} {_fmt_scalar(value)}")

        return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_float(v: float) -> str:
    """Format a float without unnecessary trailing zeros."""
    if v == float("inf"):
        return "+Inf"
    # Use repr-style but strip trailing zeros after decimal
    s = f"{v:.10g}"
    return s


def _fmt_scalar(v: object) -> str:
    if isinstance(v, float):
        return _fmt_float(v)
    return str(v)


# ---------------------------------------------------------------------------
# NetworkMetrics
# ---------------------------------------------------------------------------


class NetworkMetrics:
    """
    Holds the actual in-memory counters/gauges/histograms for the dLLM network.

    All operations are thread-safe.
    """

    INFERENCE_LATENCY_BUCKETS: list[float] = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters stored as defaultdict so label combinations are auto-created
        # Key: tuple of label values
        self._inference_requests_total: dict[tuple, int] = defaultdict(int)
        self._tokens_generated_total: dict[tuple, int] = defaultdict(int)
        self._p2p_messages_total: dict[tuple, int] = defaultdict(int)
        self._circuit_breaker_trips_total: dict[tuple, int] = defaultdict(int)
        self._settlement_amount_lamports_total: int = 0

        # Gauges
        self._active_nodes_total: int = 0
        self._active_jobs_total: dict[tuple, int] = defaultdict(int)
        self._kv_cache_hit_ratio: float = 0.0

        # Histograms
        self.inference_latency_seconds = HistogramMetric(self.INFERENCE_LATENCY_BUCKETS)
        self.shard_forward_duration_seconds = HistogramMetric(self.INFERENCE_LATENCY_BUCKETS)

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_inference(
        self,
        model: str,
        latency_s: float,
        success: bool,
        tokens: int,
    ) -> None:
        status = "success" if success else "failure"
        with self._lock:
            self._inference_requests_total[(model, status)] += 1
            self._tokens_generated_total[(model,)] += tokens
        self.inference_latency_seconds.observe(latency_s)

    def record_settlement(self, lamports: int) -> None:
        with self._lock:
            self._settlement_amount_lamports_total += lamports

    def record_p2p_message(self, msg_type: str) -> None:
        with self._lock:
            self._p2p_messages_total[(msg_type,)] += 1

    def record_shard_forward(self, duration_s: float) -> None:
        self.shard_forward_duration_seconds.observe(duration_s)

    def record_circuit_breaker_trip(self, node_id: str) -> None:
        with self._lock:
            self._circuit_breaker_trips_total[(node_id,)] += 1

    def set_active_nodes(self, n: int) -> None:
        with self._lock:
            self._active_nodes_total = n

    def set_active_jobs(self, status: str, n: int) -> None:
        with self._lock:
            self._active_jobs_total[(status,)] = n

    def set_kv_cache_hit_ratio(self, ratio: float) -> None:
        with self._lock:
            self._kv_cache_hit_ratio = ratio

    # ------------------------------------------------------------------
    # Snapshot accessors (return copies for thread safety)
    # ------------------------------------------------------------------

    def snapshot_inference_requests(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._inference_requests_total)

    def snapshot_tokens_generated(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._tokens_generated_total)

    def snapshot_p2p_messages(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._p2p_messages_total)

    def snapshot_circuit_breaker_trips(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._circuit_breaker_trips_total)

    def snapshot_settlement(self) -> int:
        with self._lock:
            return self._settlement_amount_lamports_total

    def snapshot_active_nodes(self) -> int:
        with self._lock:
            return self._active_nodes_total

    def snapshot_active_jobs(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._active_jobs_total)

    def snapshot_kv_cache_hit_ratio(self) -> float:
        with self._lock:
            return self._kv_cache_hit_ratio

    # ------------------------------------------------------------------
    # Registry builder
    # ------------------------------------------------------------------

    def build_registry(self) -> PrometheusRegistry:
        """Build and return a PrometheusRegistry wired to this NetworkMetrics instance."""
        registry = PrometheusRegistry()

        registry.register(
            PrometheusMetric(
                name="dllm_inference_requests_total",
                help_text="Total inference requests",
                metric_type=MetricType.COUNTER,
                labels=["model", "status"],
                value_fn=self.snapshot_inference_requests,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_tokens_generated_total",
                help_text="Total tokens generated",
                metric_type=MetricType.COUNTER,
                labels=["model"],
                value_fn=self.snapshot_tokens_generated,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_active_nodes_total",
                help_text="Number of active nodes in the network",
                metric_type=MetricType.GAUGE,
                labels=[],
                value_fn=self.snapshot_active_nodes,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_active_jobs_total",
                help_text="Number of active jobs by status",
                metric_type=MetricType.GAUGE,
                labels=["status"],
                value_fn=self.snapshot_active_jobs,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_settlement_amount_lamports_total",
                help_text="Total settlement amount in lamports",
                metric_type=MetricType.COUNTER,
                labels=[],
                value_fn=self.snapshot_settlement,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_p2p_messages_total",
                help_text="Total P2P messages by type",
                metric_type=MetricType.COUNTER,
                labels=["type"],
                value_fn=self.snapshot_p2p_messages,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_circuit_breaker_trips_total",
                help_text="Total circuit breaker trips by node",
                metric_type=MetricType.COUNTER,
                labels=["node_id"],
                value_fn=self.snapshot_circuit_breaker_trips,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_kv_cache_hit_ratio",
                help_text="KV cache hit ratio",
                metric_type=MetricType.GAUGE,
                labels=[],
                value_fn=self.snapshot_kv_cache_hit_ratio,
            )
        )

        # Histograms are rendered via pre-rendered blocks
        def _render_inference_latency() -> str:
            return self.inference_latency_seconds.render(
                "dllm_inference_latency_seconds",
                "",
                "Inference latency in seconds",
            )

        def _render_shard_forward() -> str:
            return self.shard_forward_duration_seconds.render(
                "dllm_shard_forward_duration_seconds",
                "",
                "Shard forward duration in seconds",
            )

        registry.register(
            PrometheusMetric(
                name="dllm_inference_latency_seconds",
                help_text="Inference latency in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=[],
                value_fn=_render_inference_latency,
            )
        )
        registry.register(
            PrometheusMetric(
                name="dllm_shard_forward_duration_seconds",
                help_text="Shard forward duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=[],
                value_fn=_render_shard_forward,
            )
        )

        return registry


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_metrics = NetworkMetrics()


def get_metrics() -> NetworkMetrics:
    """Return the module-level NetworkMetrics singleton."""
    return _global_metrics
