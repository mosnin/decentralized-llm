"""Tests for node/reputation_engine.py."""

import threading
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from node.reputation_engine import (
    NodeRecord,
    ReputationEngine,
    ReputationMiddleware,
    get_engine,
)
from node.router import InferenceRouter, RoutingConfig
from node.settlement import PaymentRecord, SettlementBatch, SettlementStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_engine(**kwargs) -> ReputationEngine:
    """Return a new isolated engine (not the global singleton)."""
    return ReputationEngine(**kwargs)


# ---------------------------------------------------------------------------
# NodeRecord dataclass
# ---------------------------------------------------------------------------


def test_node_record_defaults():
    rec = NodeRecord(node_id="n1")
    assert rec.jobs_completed == 0
    assert rec.jobs_failed == 0
    assert rec.jobs_disputed == 0
    assert rec.total_latency_ms == 0.0
    assert rec.stake_lamports == 0
    assert rec.score == 0.5


# ---------------------------------------------------------------------------
# 1. New node has neutral score
# ---------------------------------------------------------------------------


def test_new_node_has_neutral_score():
    engine = fresh_engine()
    assert engine.get_score("brand-new-node") == 0.5


# ---------------------------------------------------------------------------
# 2. Success increases score
# ---------------------------------------------------------------------------


def test_success_increases_score():
    engine = fresh_engine()
    initial = engine.get_score("n1")
    engine.record_success("n1", latency_ms=100.0, tokens_generated=50)
    assert engine.get_score("n1") > initial


# ---------------------------------------------------------------------------
# 3. Failure (timeout) decreases score
# ---------------------------------------------------------------------------


def test_failure_timeout_decreases_score():
    engine = fresh_engine()
    # Give node a known starting point
    engine.record_success("n1", latency_ms=0, tokens_generated=0)  # ~0.55
    before = engine.get_score("n1")
    engine.record_failure("n1", error_type="timeout")
    assert engine.get_score("n1") < before


# ---------------------------------------------------------------------------
# 4. Failure (wrong_result) causes large decrease
# ---------------------------------------------------------------------------


def test_failure_wrong_result_large_decrease():
    engine = fresh_engine()
    engine.record_failure("n1", error_type="timeout")  # -0.05
    score_after_timeout = engine.get_score("n1")

    engine2 = fresh_engine()
    engine2.record_failure("n2", error_type="wrong_result")  # -0.20
    score_after_wrong = engine2.get_score("n2")

    assert score_after_wrong < score_after_timeout


# ---------------------------------------------------------------------------
# 5. Dispute causes large decrease
# ---------------------------------------------------------------------------


def test_dispute_large_decrease():
    engine = fresh_engine()
    engine.record_dispute("n1")
    # -0.30 from neutral 0.5 → 0.20
    assert engine.get_score("n1") == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 6. Score clamped to [0, 1]
# ---------------------------------------------------------------------------


def test_score_clamped_to_zero_one():
    engine = fresh_engine()
    # Drive score to max
    for _ in range(30):
        engine.record_success("n1", latency_ms=1.0, tokens_generated=1)
    assert engine.get_score("n1") <= 1.0

    # Drive score to min
    for _ in range(30):
        engine.record_failure("n2", error_type="wrong_result")
    assert engine.get_score("n2") >= 0.0


# ---------------------------------------------------------------------------
# 7. Decay moves score toward neutral (0.5)
# ---------------------------------------------------------------------------


def test_decay_moves_toward_neutral():
    engine = fresh_engine(decay_factor=0.9)

    # Score above 0.5 → decays down toward 0.5
    engine.record_success("high", latency_ms=0, tokens_generated=0)  # 0.55
    score_before = engine.get_score("high")
    engine.apply_decay("high")
    score_after = engine.get_score("high")
    assert score_after < score_before
    assert score_after > 0.5  # hasn't crossed neutral

    # Score below 0.5 → decays up toward 0.5
    engine.record_failure("low", error_type="wrong_result")  # 0.30
    score_before_low = engine.get_score("low")
    engine.apply_decay("low")
    score_after_low = engine.get_score("low")
    assert score_after_low > score_before_low
    assert score_after_low < 0.5  # hasn't crossed neutral


# ---------------------------------------------------------------------------
# 8. bulk_decay affects stale nodes
# ---------------------------------------------------------------------------


def test_bulk_decay_affects_stale_nodes():
    engine = fresh_engine(decay_factor=0.9)
    engine.record_success("stale", latency_ms=0, tokens_generated=0)
    score_before = engine.get_score("stale")

    # Manually backdate last_seen so the node appears stale (> 1 hour ago)
    engine._records["stale"].last_seen = time.time() - 7201

    engine.bulk_decay()
    assert engine.get_score("stale") < score_before


def test_bulk_decay_ignores_recent_nodes():
    engine = fresh_engine(decay_factor=0.9)
    engine.record_success("fresh", latency_ms=0, tokens_generated=0)
    score_before = engine.get_score("fresh")
    # last_seen is recent (default = time.time()), so no decay
    engine.bulk_decay()
    assert engine.get_score("fresh") == score_before


# ---------------------------------------------------------------------------
# 9. get_all_scores returns dict
# ---------------------------------------------------------------------------


def test_get_all_scores_returns_dict():
    engine = fresh_engine()
    engine.record_success("a", latency_ms=10, tokens_generated=5)
    engine.record_failure("b", error_type="timeout")
    scores = engine.get_all_scores()
    assert isinstance(scores, dict)
    assert "a" in scores
    assert "b" in scores
    assert all(isinstance(v, float) for v in scores.values())


# ---------------------------------------------------------------------------
# 10. Thread safety — 50 concurrent updates, no corruption
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_updates():
    engine = fresh_engine()
    errors: list[Exception] = []

    def worker(node_id: str) -> None:
        try:
            for _ in range(20):
                engine.record_success(node_id, latency_ms=5.0, tokens_generated=10)
                engine.record_failure(node_id, error_type="timeout")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"n{i % 5}",)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions in threads: {errors}"
    scores = engine.get_all_scores()
    for node_id, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Score out of range for {node_id}: {score}"


# ---------------------------------------------------------------------------
# 11. Router uses ReputationEngine
# ---------------------------------------------------------------------------


@dataclass
class FakeNode:
    node_id: str
    reputation: float = 0.5
    current_load: int = 0
    max_load: int = 4
    cost_per_token: int = 100
    supported_models: list = field(default_factory=lambda: ["llama"])


def test_router_uses_reputation_engine():
    """Router._score should call get_engine().get_score(node_id)."""
    engine = fresh_engine()
    # Give node a distinctive score
    for _ in range(5):
        engine.record_success("high-rep", latency_ms=1, tokens_generated=1)
    for _ in range(3):
        engine.record_failure("low-rep", error_type="wrong_result")

    with patch("node.router.get_engine", return_value=engine):
        router = InferenceRouter(
            config=RoutingConfig(
                reputation_weight=1.0,
                availability_weight=0.0,
                cost_weight=0.0,
            )
        )
        high_node = FakeNode(node_id="high-rep")
        low_node = FakeNode(node_id="low-rep")
        decision = router.route([high_node, low_node], "llama")

    assert decision is not None
    assert decision.node_id == "high-rep"


# ---------------------------------------------------------------------------
# 12. update_from_settlement_batch
# ---------------------------------------------------------------------------


def _make_batch(status: SettlementStatus, node_ids: list[str]) -> SettlementBatch:
    records = [
        PaymentRecord(
            job_id=i,
            amount_lamports=1000,
            client_pubkey=nid,
            earned_at=time.time(),
        )
        for i, nid in enumerate(node_ids)
    ]
    return SettlementBatch(
        batch_id="batch-000001",
        records=records,
        created_at=time.time(),
        status=status,
    )


def test_update_from_settlement_batch_confirmed():
    engine = fresh_engine()
    batch = _make_batch(SettlementStatus.CONFIRMED, ["node-a", "node-b"])
    engine.update_from_settlement(batch)

    # Confirmed → success → scores should be above neutral
    assert engine.get_score("node-a") > 0.5
    assert engine.get_score("node-b") > 0.5


def test_update_from_settlement_batch_failed():
    engine = fresh_engine()
    batch = _make_batch(SettlementStatus.FAILED, ["node-c"])
    engine.update_from_settlement(batch)

    # Failed → crash → score should drop below neutral
    assert engine.get_score("node-c") < 0.5


def test_update_from_settlement_batch_pending_is_noop():
    engine = fresh_engine()
    batch = _make_batch(SettlementStatus.PENDING, ["node-d"])
    engine.update_from_settlement(batch)

    # Pending batch → no update → neutral
    assert engine.get_score("node-d") == 0.5


# ---------------------------------------------------------------------------
# 13. Module-level singleton
# ---------------------------------------------------------------------------


def test_get_engine_returns_same_instance():
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2
    assert isinstance(e1, ReputationEngine)


# ---------------------------------------------------------------------------
# 14. ReputationMiddleware refreshes scores into node objects
# ---------------------------------------------------------------------------


def test_reputation_middleware_refresh_scores():
    engine = fresh_engine()
    engine.record_success("n1", latency_ms=10, tokens_generated=5)
    expected = engine.get_score("n1")

    node = FakeNode(node_id="n1", reputation=0.0)
    mock_router = MagicMock()
    mw = ReputationMiddleware(mock_router, engine=engine)
    mw.refresh_scores([node])

    assert node.reputation == pytest.approx(expected)


def test_reputation_middleware_on_success_delegates():
    engine = fresh_engine()
    mw = ReputationMiddleware(MagicMock(), engine=engine)
    mw.on_success("n1", latency_ms=50.0, tokens_generated=100)
    assert engine.get_score("n1") > 0.5


def test_reputation_middleware_on_failure_delegates():
    engine = fresh_engine()
    mw = ReputationMiddleware(MagicMock(), engine=engine)
    mw.on_failure("n1", error_type="crash")
    assert engine.get_score("n1") < 0.5
