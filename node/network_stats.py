"""Aggregate network statistics for the decentralized LLM network."""

import time
from dataclasses import dataclass


@dataclass
class NetworkSnapshot:
    """Point-in-time snapshot of network state."""

    timestamp: float
    total_nodes: int
    active_nodes: int  # nodes with heartbeat in last 5 min
    total_jobs_24h: int
    completed_jobs_24h: int
    failed_jobs_24h: int
    avg_latency_ms: float
    total_staked_tokens: int
    models_available: list[str]


class NetworkStatsCollector:
    """Collects and aggregates network statistics."""

    _WINDOW_24H: float = 86400.0

    def __init__(self) -> None:
        # Each entry: (timestamp, success: bool, latency_ms: float, model: str)
        self._jobs: list[tuple[float, bool, float, str]] = []
        # node_pubkey -> last heartbeat timestamp
        self._heartbeats: dict[str, float] = {}
        # node_pubkey -> staked tokens
        self._staked: dict[str, int] = {}
        # node_pubkey -> set of models
        self._node_models: dict[str, set[str]] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def record_job(self, success: bool, latency_ms: float, model: str) -> None:
        """Record the outcome of a completed (or failed) job."""
        self._jobs.append((time.time(), success, latency_ms, model))

    def record_node_heartbeat(self, node_pubkey: str) -> None:
        """Update the last-seen timestamp for a node."""
        self._heartbeats[node_pubkey] = time.time()

    def record_node_registered(self, node_pubkey: str, staked: int, models: list[str]) -> None:
        """Register (or update) a node's stake and supported models."""
        self._staked[node_pubkey] = staked
        self._node_models[node_pubkey] = set(models)

    def snapshot(self) -> NetworkSnapshot:
        """Return current aggregate stats. Jobs outside the 24h window are pruned."""
        self._prune_old_jobs()
        now = time.time()

        jobs = self._jobs
        total = len(jobs)
        completed = sum(1 for _, success, _, _ in jobs if success)
        failed = total - completed

        latencies = [lat for _, success, lat, _ in jobs if success]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        all_models: set[str] = set()
        for model_set in self._node_models.values():
            all_models |= model_set

        return NetworkSnapshot(
            timestamp=now,
            total_nodes=len(self._staked),
            active_nodes=self.active_node_count(),
            total_jobs_24h=total,
            completed_jobs_24h=completed,
            failed_jobs_24h=failed,
            avg_latency_ms=avg_latency,
            total_staked_tokens=sum(self._staked.values()),
            models_available=sorted(all_models),
        )

    def active_node_count(self, window_seconds: float = 300) -> int:
        """Count nodes whose last heartbeat is within *window_seconds*."""
        cutoff = time.time() - window_seconds
        return sum(1 for ts in self._heartbeats.values() if ts >= cutoff)

    # ── internals ─────────────────────────────────────────────────────────────

    def _prune_old_jobs(self) -> None:
        cutoff = time.time() - self._WINDOW_24H
        self._jobs = [entry for entry in self._jobs if entry[0] >= cutoff]
