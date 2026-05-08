import time
from dataclasses import dataclass
from typing import Protocol


class JobLike(Protocol):
    """Minimal interface a job must satisfy."""

    job_id: int
    payment_amount: int
    deadline: float
    model_id: bytes


@dataclass
class SchedulerConfig:
    payment_weight: float = 0.4  # weight for payment score
    urgency_weight: float = 0.35  # weight for deadline urgency
    reputation_weight: float = 0.25  # weight for node reputation
    urgency_horizon_s: float = 300.0  # deadline within this → max urgency score


class JobScheduler:
    """
    Scores and sorts jobs by a weighted combination of payment, urgency, and reputation.

    score(job) = payment_weight * payment_score
               + urgency_weight * urgency_score
               + reputation_weight * reputation_score

    All component scores are normalized to [0, 1].
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self._reputation: dict[str, float] = {}  # node_id → score in [0,1]

    def set_reputation(self, node_id: str, score: float) -> None:
        """Register reputation score for a node (0.0–1.0)."""
        self._reputation[node_id] = max(0.0, min(1.0, score))

    def payment_score(self, job, max_payment: int) -> float:
        """Normalize payment against the highest-paying job in the batch."""
        if max_payment <= 0:
            return 0.0
        return min(1.0, job.payment_amount / max_payment)

    def urgency_score(self, job) -> float:
        """1.0 if past deadline or within urgency_horizon_s; 0.0 if far future."""
        remaining = job.deadline - time.time()
        if remaining <= 0:
            return 1.0
        if remaining >= self.config.urgency_horizon_s:
            return 0.0
        return 1.0 - (remaining / self.config.urgency_horizon_s)

    def reputation_score(self, node_id: str) -> float:
        """Return registered reputation or 0.5 (neutral) for unknown nodes."""
        return self._reputation.get(node_id, 0.5)

    def score(self, job, node_id: str, max_payment: int) -> float:
        """Compute the composite scheduling score."""
        ps = self.payment_score(job, max_payment)
        us = self.urgency_score(job)
        rs = self.reputation_score(node_id)
        c = self.config
        return c.payment_weight * ps + c.urgency_weight * us + c.reputation_weight * rs

    def rank(self, jobs: list, node_id: str) -> list:
        """
        Sort jobs highest-score-first.
        max_payment is computed from the batch for normalization.
        """
        if not jobs:
            return []
        max_payment = max(j.payment_amount for j in jobs)
        return sorted(
            jobs,
            key=lambda j: self.score(j, node_id, max_payment),
            reverse=True,
        )
