"""Live reputation engine with feedback from completed/failed jobs."""

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node.settlement import SettlementBatch

# Error severity penalties
_ERROR_PENALTIES: dict[str, float] = {
    "timeout": -0.05,
    "wrong_result": -0.20,
    "crash": -0.10,
}

_DISPUTE_PENALTY: float = -0.30
_SUCCESS_BOOST: float = 0.05
_STALE_THRESHOLD_S: float = 3600.0  # 1 hour


@dataclass
class NodeRecord:
    node_id: str
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_disputed: int = 0
    total_latency_ms: float = 0.0
    last_seen: float = field(default_factory=time.time)
    stake_lamports: int = 0
    score: float = 0.5  # starts neutral


class ReputationEngine:
    """
    Thread-safe reputation engine that updates node scores based on job outcomes.

    Scores are clamped to [min_score, max_score] (default [0.0, 1.0]).
    decay_factor controls how quickly idle nodes drift back toward neutral (0.5).
    """

    def __init__(
        self,
        decay_factor: float = 0.95,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> None:
        self.decay_factor = decay_factor
        self.min_score = min_score
        self.max_score = max_score
        self._records: dict[str, NodeRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, node_id: str) -> NodeRecord:
        """Return existing record or create a new neutral one (lock must be held)."""
        if node_id not in self._records:
            self._records[node_id] = NodeRecord(node_id=node_id)
        return self._records[node_id]

    def _clamp(self, value: float) -> float:
        return max(self.min_score, min(self.max_score, value))

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def record_success(self, node_id: str, latency_ms: float, tokens_generated: int) -> None:
        """Increase score on successful job completion."""
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.jobs_completed += 1
            rec.total_latency_ms += latency_ms
            rec.last_seen = time.time()
            rec.score = self._clamp(rec.score + _SUCCESS_BOOST)

    def record_failure(self, node_id: str, error_type: str) -> None:
        """
        Decrease score based on error severity.

        Known error_type values and their penalties:
          "timeout"      → -0.05
          "wrong_result" → -0.20
          "crash"        → -0.10
        Unknown types default to the "crash" penalty.
        """
        penalty = _ERROR_PENALTIES.get(error_type, _ERROR_PENALTIES["crash"])
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.jobs_failed += 1
            rec.last_seen = time.time()
            rec.score = self._clamp(rec.score + penalty)

    def record_dispute(self, node_id: str) -> None:
        """Apply a large penalty for a disputed/fraudulent job."""
        with self._lock:
            rec = self._get_or_create(node_id)
            rec.jobs_disputed += 1
            rec.last_seen = time.time()
            rec.score = self._clamp(rec.score + _DISPUTE_PENALTY)

    # ------------------------------------------------------------------
    # Score queries
    # ------------------------------------------------------------------

    def get_score(self, node_id: str) -> float:
        """Return score in [min_score, max_score]. Unknown nodes default to 0.5."""
        with self._lock:
            rec = self._records.get(node_id)
            return rec.score if rec is not None else 0.5

    def get_all_scores(self) -> dict[str, float]:
        """Return a snapshot of all node scores."""
        with self._lock:
            return {node_id: rec.score for node_id, rec in self._records.items()}

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def apply_decay(self, node_id: str) -> None:
        """
        Multiply the deviation from neutral (0.5) by decay_factor,
        so scores drift back toward 0.5 over time.
        """
        with self._lock:
            rec = self._records.get(node_id)
            if rec is None:
                return
            deviation = rec.score - 0.5
            rec.score = self._clamp(0.5 + deviation * self.decay_factor)

    def bulk_decay(self) -> None:
        """Apply decay to all nodes not seen in the last hour."""
        cutoff = time.time() - _STALE_THRESHOLD_S
        with self._lock:
            for rec in self._records.values():
                if rec.last_seen < cutoff:
                    deviation = rec.score - 0.5
                    rec.score = self._clamp(0.5 + deviation * self.decay_factor)

    # ------------------------------------------------------------------
    # Settlement integration
    # ------------------------------------------------------------------

    def update_from_settlement(self, settlement_batch: "SettlementBatch") -> None:
        """
        Process a SettlementBatch from settlement.py.

        Each PaymentRecord in the batch represents a completed job; the
        client_pubkey field identifies the paying node/client. We treat
        every record in a CONFIRMED batch as a success, and every record
        in a FAILED batch as a failure (crash-level).
        """
        from node.settlement import SettlementStatus

        for record in settlement_batch.records:
            node_id = record.client_pubkey
            if settlement_batch.status == SettlementStatus.CONFIRMED:
                self.record_success(
                    node_id=node_id,
                    latency_ms=0.0,
                    tokens_generated=0,
                )
            elif settlement_batch.status == SettlementStatus.FAILED:
                self.record_failure(node_id=node_id, error_type="crash")


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_global_engine: ReputationEngine = ReputationEngine()


def get_engine() -> ReputationEngine:
    """Return the module-level global ReputationEngine instance."""
    return _global_engine


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------


class ReputationMiddleware:
    """
    Wraps a Router or Scheduler instance, injecting live reputation scores
    before each routing decision and recording outcomes afterwards.

    Usage:
        middleware = ReputationMiddleware(router, engine=get_engine())
        middleware.refresh_scores(nodes)         # push scores into node objects
        decision = middleware.wrapped.route(nodes, model)
        middleware.on_success(node_id, latency_ms, tokens)
        middleware.on_failure(node_id, error_type)
    """

    def __init__(self, wrapped, engine: ReputationEngine | None = None) -> None:
        self.wrapped = wrapped
        self.engine = engine or get_engine()

    def refresh_scores(self, nodes: list) -> None:
        """
        Update each node's ``reputation`` attribute with the live engine score.

        Works for any object that exposes a ``node_id`` attribute and allows
        attribute assignment (e.g. dataclass instances, SimpleNamespace).
        Also calls ``set_reputation`` on JobScheduler instances to sync its
        internal reputation dict.
        """
        for node in nodes:
            node_id = getattr(node, "node_id", None)
            if node_id is not None:
                score = self.engine.get_score(node_id)
                try:
                    node.reputation = score
                except AttributeError:
                    pass
                # Sync JobScheduler internal dict if applicable
                if hasattr(self.wrapped, "set_reputation"):
                    self.wrapped.set_reputation(node_id, score)

    def on_success(self, node_id: str, latency_ms: float = 0.0, tokens_generated: int = 0) -> None:
        """Record a successful job outcome."""
        self.engine.record_success(node_id, latency_ms, tokens_generated)

    def on_failure(self, node_id: str, error_type: str = "crash") -> None:
        """Record a failed job outcome."""
        self.engine.record_failure(node_id, error_type)

    def on_dispute(self, node_id: str) -> None:
        """Record a disputed job outcome."""
        self.engine.record_dispute(node_id)
