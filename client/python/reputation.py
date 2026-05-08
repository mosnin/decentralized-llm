"""Local reputation cache for observed node performance."""

import statistics
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class NodeObservation:
    node_pubkey: str
    timestamp: float
    success: bool
    latency_ms: float
    error_type: str = ""  # e.g. "TimeoutError", "IntegrityError", ""


class ReputationCache:
    """
    Maintains a sliding window of observations per node.

    window_seconds: float = 300.0  # 5-minute sliding window
    max_observations_per_node: int = 50
    """

    window_seconds: float = 300.0
    max_observations_per_node: int = 50

    def __init__(
        self,
        window_seconds: float = 300.0,
        max_observations_per_node: int = 50,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_observations_per_node = max_observations_per_node
        self._observations: dict[str, deque[NodeObservation]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        node_pubkey: str,
        success: bool,
        latency_ms: float,
        error_type: str = "",
    ) -> None:
        """Add an observation. Prune observations older than window_seconds."""
        if node_pubkey not in self._observations:
            self._observations[node_pubkey] = deque()

        obs = NodeObservation(
            node_pubkey=node_pubkey,
            timestamp=time.time(),
            success=success,
            latency_ms=latency_ms,
            error_type=error_type,
        )
        dq = self._observations[node_pubkey]
        dq.append(obs)

        # Cap at max_observations_per_node (drop oldest first)
        while len(dq) > self.max_observations_per_node:
            dq.popleft()

        self._prune(node_pubkey)

    def score(self, node_pubkey: str) -> float:
        """
        Returns a score in [0.0, 1.0].

        Formula: (success_rate * 0.7) + (latency_score * 0.3)
        where success_rate = successes / total observations
        and latency_score = max(0, 1 - median_latency_ms / 10_000)
        Returns 0.5 (neutral) if no observations.
        """
        self._prune(node_pubkey)
        dq = self._observations.get(node_pubkey)
        if not dq:
            return 0.5

        total = len(dq)
        successes = sum(1 for o in dq if o.success)
        success_rate = successes / total

        latencies = [o.latency_ms for o in dq]
        median_latency = statistics.median(latencies)
        latency_score = max(0.0, 1.0 - median_latency / 10_000.0)

        return (success_rate * 0.7) + (latency_score * 0.3)

    def rank_nodes(self, pubkeys: list[str]) -> list[str]:
        """Return pubkeys sorted by score() descending."""
        return sorted(pubkeys, key=self.score, reverse=True)

    def is_penalized(self, node_pubkey: str) -> bool:
        """Returns True if the last 3 observations are all failures."""
        self._prune(node_pubkey)
        dq = self._observations.get(node_pubkey)
        if not dq or len(dq) < 3:
            return False
        last_three = list(dq)[-3:]
        return all(not o.success for o in last_three)

    def clear(self, node_pubkey: str | None = None) -> None:
        """Clear observations for one node, or all if node_pubkey is None."""
        if node_pubkey is None:
            self._observations.clear()
        else:
            self._observations.pop(node_pubkey, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune(self, node_pubkey: str) -> None:
        """Remove observations older than window_seconds for a single node."""
        dq = self._observations.get(node_pubkey)
        if not dq:
            return
        cutoff = time.time() - self.window_seconds
        while dq and dq[0].timestamp < cutoff:
            dq.popleft()
