import time
from dataclasses import dataclass, field

from node.scheduler import JobScheduler, SchedulerConfig


@dataclass
class FakeJob:
    job_id: int
    payment_amount: int
    deadline: float
    model_id: bytes = field(default_factory=lambda: b"\x00" * 32)


# ---------------------------------------------------------------------------
# payment_score
# ---------------------------------------------------------------------------


def test_payment_score_max_payment_is_one():
    scheduler = JobScheduler()
    job = FakeJob(job_id=1, payment_amount=100, deadline=time.time() + 3600)
    assert scheduler.payment_score(job, max_payment=100) == 1.0


def test_payment_score_zero_payment_zero():
    scheduler = JobScheduler()
    job = FakeJob(job_id=1, payment_amount=0, deadline=time.time() + 3600)
    assert scheduler.payment_score(job, max_payment=0) == 0.0


# ---------------------------------------------------------------------------
# urgency_score
# ---------------------------------------------------------------------------


def test_urgency_score_past_deadline():
    scheduler = JobScheduler()
    job = FakeJob(job_id=1, payment_amount=50, deadline=time.time() - 1.0)
    assert scheduler.urgency_score(job) == 1.0


def test_urgency_score_far_future():
    scheduler = JobScheduler()
    # deadline 1 hour out, horizon is 300 s
    job = FakeJob(job_id=1, payment_amount=50, deadline=time.time() + 3600)
    assert scheduler.urgency_score(job) == 0.0


def test_urgency_score_within_horizon():
    scheduler = JobScheduler()
    horizon = scheduler.config.urgency_horizon_s  # 300 s
    # deadline exactly halfway through the horizon
    job = FakeJob(job_id=1, payment_amount=50, deadline=time.time() + horizon / 2)
    score = scheduler.urgency_score(job)
    # Should be approximately 0.5 (within small tolerance for execution time)
    assert abs(score - 0.5) < 0.05


# ---------------------------------------------------------------------------
# reputation_score / set_reputation
# ---------------------------------------------------------------------------


def test_reputation_score_unknown_node():
    scheduler = JobScheduler()
    assert scheduler.reputation_score("unknown-node") == 0.5


def test_reputation_score_known_node():
    scheduler = JobScheduler()
    scheduler.set_reputation("node-abc", 0.85)
    assert scheduler.reputation_score("node-abc") == 0.85


def test_set_reputation_clamped():
    scheduler = JobScheduler()
    scheduler.set_reputation("node-high", 1.5)
    assert scheduler.reputation_score("node-high") == 1.0

    scheduler.set_reputation("node-low", -0.5)
    assert scheduler.reputation_score("node-low") == 0.0


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------


def test_rank_high_payment_first():
    scheduler = JobScheduler()
    far_future = time.time() + 7200  # well beyond urgency horizon
    low_pay = FakeJob(job_id=1, payment_amount=10, deadline=far_future)
    high_pay = FakeJob(job_id=2, payment_amount=100, deadline=far_future)

    ranked = scheduler.rank([low_pay, high_pay], node_id="node-x")
    assert ranked[0].job_id == high_pay.job_id


def test_rank_urgent_job_promoted():
    """A job about to expire should rank above an equal-payment far-future job."""
    scheduler = JobScheduler()
    far_future = FakeJob(job_id=1, payment_amount=50, deadline=time.time() + 7200)
    urgent = FakeJob(job_id=2, payment_amount=50, deadline=time.time() + 10)

    ranked = scheduler.rank([far_future, urgent], node_id="node-x")
    assert ranked[0].job_id == urgent.job_id


def test_rank_empty_returns_empty():
    scheduler = JobScheduler()
    assert scheduler.rank([], node_id="node-x") == []


# ---------------------------------------------------------------------------
# composite score weights
# ---------------------------------------------------------------------------


def test_composite_score_weights_sum_to_one():
    config = SchedulerConfig()
    total = config.payment_weight + config.urgency_weight + config.reputation_weight
    assert abs(total - 1.0) < 1e-9
