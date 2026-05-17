import time

from node.earnings import EarningsTracker


def test_record_and_lifetime_summary():
    tracker = EarningsTracker()
    tracker.record(1, 100, model_name="gpt2", tokens_generated=50)
    tracker.record(2, 200, model_name="gpt2", tokens_generated=100)
    tracker.record(3, 300, model_name="llama", tokens_generated=150)

    summary = tracker.lifetime_summary()
    assert summary["total_jobs"] == 3
    assert summary["total_lamports"] == 600


def test_window_summary_basic():
    tracker = EarningsTracker(window_seconds=60.0)
    tracker.record(1, 100, model_name="gpt2", tokens_generated=10)
    tracker.record(2, 200, model_name="gpt2", tokens_generated=20)

    summary = tracker.window_summary()
    assert summary["jobs_completed"] == 2
    assert summary["total_lamports"] == 300
    assert summary["total_tokens"] == 30
    assert summary["window_seconds"] == 60.0


def test_window_summary_excludes_old():
    tracker = EarningsTracker(window_seconds=0.1)
    tracker.record(1, 500, model_name="old_model", tokens_generated=100)
    time.sleep(0.15)
    tracker.record(2, 100, model_name="new_model", tokens_generated=10)

    summary = tracker.window_summary()
    assert summary["jobs_completed"] == 1
    assert summary["total_lamports"] == 100
    assert summary["total_tokens"] == 10


def test_window_avg_lamports_per_job():
    tracker = EarningsTracker(window_seconds=60.0)
    tracker.record(1, 100)
    tracker.record(2, 300)

    summary = tracker.window_summary()
    assert summary["avg_lamports_per_job"] == 200.0


def test_top_models():
    tracker = EarningsTracker(window_seconds=60.0)
    tracker.record(1, 100, model_name="modelA")
    tracker.record(2, 200, model_name="modelA")
    tracker.record(3, 150, model_name="modelA")
    tracker.record(4, 100, model_name="modelB")
    tracker.record(5, 200, model_name="modelB")

    summary = tracker.window_summary()
    top = summary["top_models"]
    assert len(top) >= 2
    assert top[0]["model"] == "modelA"
    assert top[0]["jobs"] == 3
    assert top[0]["lamports"] == 450
    assert top[1]["model"] == "modelB"
    assert top[1]["jobs"] == 2
    assert top[1]["lamports"] == 300


def test_recent_jobs_n():
    tracker = EarningsTracker(window_seconds=60.0)
    for i in range(5):
        tracker.record(i, i * 10)

    recent = tracker.recent_jobs(2)
    assert len(recent) == 2
    assert recent[-1].job_id == 4
    assert recent[-2].job_id == 3


def test_recent_jobs_empty():
    tracker = EarningsTracker()
    assert tracker.recent_jobs() == []


def test_tokens_tracked_in_window():
    tracker = EarningsTracker(window_seconds=60.0)
    tracker.record(1, 100, tokens_generated=50)
    tracker.record(2, 200, tokens_generated=75)
    tracker.record(3, 300, tokens_generated=25)

    summary = tracker.window_summary()
    assert summary["total_tokens"] == 150


def test_lifetime_not_affected_by_window():
    tracker = EarningsTracker(window_seconds=0.1)
    tracker.record(1, 500)
    tracker.record(2, 300)
    time.sleep(0.15)

    # window should be empty
    window = tracker.window_summary()
    assert window["jobs_completed"] == 0

    # lifetime should still reflect all records
    lifetime = tracker.lifetime_summary()
    assert lifetime["total_jobs"] == 2
    assert lifetime["total_lamports"] == 800


def test_avg_lamports_zero_jobs():
    tracker = EarningsTracker(window_seconds=0.1)
    tracker.record(1, 100)
    time.sleep(0.15)

    summary = tracker.window_summary()
    assert summary["jobs_completed"] == 0
    assert summary["avg_lamports_per_job"] == 0.0
