"""Tests for client.python.reputation.ReputationCache."""

import time

from client.python.reputation import ReputationCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_n(
    cache: ReputationCache, pubkey: str, n: int, success: bool, latency_ms: float
) -> None:
    for _ in range(n):
        cache.record(pubkey, success=success, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_neutral_no_observations():
    cache = ReputationCache()
    assert cache.score("node_unknown") == 0.5


def test_score_perfect_fast_node():
    cache = ReputationCache()
    # 10 successes, very fast (100 ms)
    _record_n(cache, "node_fast", 10, success=True, latency_ms=100.0)
    s = cache.score("node_fast")
    # success_rate=1.0 → 0.7; latency_score=max(0, 1-100/10000)=0.99 → 0.297
    # Expected ≈ 0.997
    assert s > 0.95


def test_score_slow_node_penalized():
    cache = ReputationCache()
    # All successes but very slow (9000 ms median)
    _record_n(cache, "node_slow", 10, success=True, latency_ms=9000.0)
    s = cache.score("node_slow")
    # latency_score = max(0, 1 - 9000/10000) = 0.1 → contributes 0.03
    # success_rate = 1.0 → 0.7; total ≈ 0.73
    assert s < 0.80


def test_score_failure_rate_lowers_score():
    cache = ReputationCache()
    # 5 successes, 5 failures; fast latency
    for i in range(10):
        cache.record("node_mixed", success=(i % 2 == 0), latency_ms=100.0)
    s = cache.score("node_mixed")
    # success_rate=0.5 → 0.35; latency_score≈0.99 → 0.297; total≈0.647
    assert s < 0.70
    assert s > 0.30


def test_rank_nodes_by_score():
    cache = ReputationCache()
    _record_n(cache, "node_bad", 5, success=False, latency_ms=5000.0)
    _record_n(cache, "node_good", 5, success=True, latency_ms=200.0)
    _record_n(cache, "node_mid", 5, success=True, latency_ms=5000.0)

    ranked = cache.rank_nodes(["node_bad", "node_good", "node_mid"])
    assert ranked[0] == "node_good"
    assert ranked[-1] == "node_bad"


def test_is_penalized_after_three_failures():
    cache = ReputationCache()
    cache.record("node_x", success=True, latency_ms=100.0)
    cache.record("node_x", success=False, latency_ms=100.0)
    cache.record("node_x", success=False, latency_ms=100.0)
    cache.record("node_x", success=False, latency_ms=100.0)
    assert cache.is_penalized("node_x") is True


def test_is_not_penalized_after_success():
    cache = ReputationCache()
    cache.record("node_y", success=False, latency_ms=100.0)
    cache.record("node_y", success=False, latency_ms=100.0)
    cache.record("node_y", success=True, latency_ms=100.0)
    assert cache.is_penalized("node_y") is False


def test_old_observations_pruned():
    cache = ReputationCache(window_seconds=0.1)
    cache.record("node_z", success=True, latency_ms=50.0)
    # Wait for the observation to age out
    time.sleep(0.15)
    # Trigger pruning via score()
    assert cache.score("node_z") == 0.5


def test_clear_specific_node():
    cache = ReputationCache()
    _record_n(cache, "node_a", 5, success=True, latency_ms=100.0)
    _record_n(cache, "node_b", 5, success=True, latency_ms=100.0)
    cache.clear("node_a")
    assert cache.score("node_a") == 0.5
    # node_b should be unaffected
    assert cache.score("node_b") > 0.5


def test_clear_all_nodes():
    cache = ReputationCache()
    _record_n(cache, "node_a", 5, success=True, latency_ms=100.0)
    _record_n(cache, "node_b", 5, success=True, latency_ms=100.0)
    cache.clear()
    assert cache.score("node_a") == 0.5
    assert cache.score("node_b") == 0.5
