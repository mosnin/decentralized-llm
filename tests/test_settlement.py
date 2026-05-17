"""Tests for node.settlement — SettlementTracker and related dataclasses."""

from node.settlement import SettlementBatch, SettlementStatus, SettlementTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracker(**kwargs) -> SettlementTracker:
    return SettlementTracker(**kwargs)


def add_records(tracker: SettlementTracker, n: int, amount: int = 100) -> None:
    for i in range(n):
        tracker.record(job_id=i, amount_lamports=amount, client_pubkey=f"pubkey-{i}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_record_adds_to_pending():
    tracker = make_tracker()
    tracker.record(job_id=1, amount_lamports=500, client_pubkey="pk1")
    assert tracker.pending_count() == 1


def test_should_flush_by_size():
    tracker = make_tracker(min_batch_lamports=10_000, max_batch_size=3)
    add_records(tracker, 3, amount=1)
    assert tracker.should_flush() is True


def test_should_flush_by_lamports():
    tracker = make_tracker(min_batch_lamports=1000, max_batch_size=50)
    tracker.record(job_id=1, amount_lamports=1000, client_pubkey="pk1")
    assert tracker.should_flush() is True


def test_should_not_flush_when_small():
    tracker = make_tracker(min_batch_lamports=1000, max_batch_size=50)
    tracker.record(job_id=1, amount_lamports=500, client_pubkey="pk1")
    assert tracker.should_flush() is False


def test_flush_creates_batch():
    tracker = make_tracker()
    tracker.record(job_id=1, amount_lamports=300, client_pubkey="pk1")
    tracker.record(job_id=2, amount_lamports=700, client_pubkey="pk2")
    batch = tracker.flush()
    assert batch is not None
    assert isinstance(batch, SettlementBatch)
    assert batch.total_lamports == 1000
    assert batch.job_count == 2


def test_flush_clears_pending():
    tracker = make_tracker()
    add_records(tracker, 5)
    tracker.flush()
    assert tracker.pending_count() == 0


def test_flush_empty_returns_none():
    tracker = make_tracker()
    result = tracker.flush()
    assert result is None


def test_confirm_marks_confirmed():
    tracker = make_tracker()
    add_records(tracker, 1)
    batch = tracker.flush()
    assert batch is not None
    tracker.confirm(batch.batch_id, tx_signature="sig123")
    assert batch.status == SettlementStatus.CONFIRMED


def test_confirm_stores_tx_sig():
    tracker = make_tracker()
    add_records(tracker, 1)
    batch = tracker.flush()
    assert batch is not None
    tracker.confirm(batch.batch_id, tx_signature="abc456")
    assert batch.tx_signature == "abc456"


def test_fail_marks_failed():
    tracker = make_tracker()
    add_records(tracker, 1)
    batch = tracker.flush()
    assert batch is not None
    tracker.fail(batch.batch_id)
    assert batch.status == SettlementStatus.FAILED


def test_get_batch_known():
    tracker = make_tracker()
    add_records(tracker, 1)
    batch = tracker.flush()
    assert batch is not None
    retrieved = tracker.get_batch(batch.batch_id)
    assert retrieved is batch


def test_get_batch_unknown():
    tracker = make_tracker()
    assert tracker.get_batch("nonexistent-batch") is None


def test_batches_by_status():
    tracker = make_tracker()

    add_records(tracker, 2)
    b1 = tracker.flush()
    add_records(tracker, 2)
    b2 = tracker.flush()
    add_records(tracker, 2)
    b3 = tracker.flush()

    assert b1 is not None and b2 is not None and b3 is not None

    tracker.confirm(b1.batch_id, tx_signature="sig1")
    tracker.fail(b2.batch_id)
    # b3 stays PENDING

    confirmed = tracker.batches_by_status(SettlementStatus.CONFIRMED)
    failed = tracker.batches_by_status(SettlementStatus.FAILED)
    pending = tracker.batches_by_status(SettlementStatus.PENDING)

    assert len(confirmed) == 1 and confirmed[0].batch_id == b1.batch_id
    assert len(failed) == 1 and failed[0].batch_id == b2.batch_id
    assert len(pending) == 1 and pending[0].batch_id == b3.batch_id


def test_total_confirmed_lamports():
    tracker = make_tracker()

    tracker.record(job_id=1, amount_lamports=400, client_pubkey="pk1")
    b1 = tracker.flush()
    tracker.record(job_id=2, amount_lamports=600, client_pubkey="pk2")
    b2 = tracker.flush()
    tracker.record(job_id=3, amount_lamports=200, client_pubkey="pk3")
    b3 = tracker.flush()

    assert b1 is not None and b2 is not None and b3 is not None

    tracker.confirm(b1.batch_id, tx_signature="s1")
    tracker.confirm(b2.batch_id, tx_signature="s2")
    # b3 stays pending

    assert tracker.total_confirmed_lamports() == 1000


def test_batch_total_lamports():
    tracker = make_tracker()
    tracker.record(job_id=1, amount_lamports=250, client_pubkey="pk1")
    tracker.record(job_id=2, amount_lamports=750, client_pubkey="pk2")
    tracker.record(job_id=3, amount_lamports=500, client_pubkey="pk3")
    batch = tracker.flush()
    assert batch is not None
    assert batch.total_lamports == 1500
