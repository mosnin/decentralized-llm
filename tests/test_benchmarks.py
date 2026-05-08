"""
Performance benchmarks for critical-path components.
All tests assert that operations complete within acceptable time bounds.
"""

import asyncio
import random
import time
from dataclasses import dataclass

from client.python.reputation import ReputationCache
from node.audit_log import AuditEventType, AuditLog
from node.earnings import EarningsTracker
from node.gossip import GossipMessageType, GossipNode
from node.merkle import MerkleTree
from node.rate_limiter import RateLimitConfig, RateLimiter
from node.result_cache import ResultCache
from node.scheduler import JobScheduler


@dataclass
class FakeJob:
    job_id: int
    payment_amount: int
    deadline: float
    model_id: bytes = b"\x00" * 32


def test_result_cache_throughput():
    """Put 10,000 items into ResultCache, then get all 10,000. Total time < 1.0s."""
    cache = ResultCache(max_size=10_000)
    n = 10_000

    start = time.perf_counter()

    for i in range(n):
        cache.put("model", f"prompt_{i}", f"result_{i}")

    for i in range(n):
        val = cache.get("model", f"prompt_{i}")
        assert val == f"result_{i}"

    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"result_cache_throughput took {elapsed:.3f}s (limit 1.0s)"


def test_reputation_cache_throughput():
    """Record 1,000 observations for 100 nodes, score all 100. Total time < 0.5s."""
    cache = ReputationCache()
    n_nodes = 100
    n_obs = 1_000

    start = time.perf_counter()

    for i in range(n_obs):
        node_id = f"node_{i % n_nodes}"
        cache.record(node_id, success=(i % 5 != 0), latency_ms=float(100 + i % 500))

    for i in range(n_nodes):
        score = cache.score(f"node_{i}")
        assert 0.0 <= score <= 1.0

    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"reputation_cache_throughput took {elapsed:.3f}s (limit 0.5s)"


def test_audit_log_append_throughput():
    """Append 1,000 entries to AuditLog, verify chain. Total time < 2.0s."""
    log = AuditLog()
    n = 1_000

    start = time.perf_counter()

    for i in range(n):
        log.append(
            AuditEventType.JOB_COMPLETED,
            actor=f"node_{i % 10}",
            payload={"job_id": i, "tokens": 512},
        )

    assert log.verify_chain()

    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"audit_log_append_throughput took {elapsed:.3f}s (limit 2.0s)"


def test_merkle_tree_construction():
    """Build MerkleTree from 1,024 leaves, generate and verify all proofs. Total time < 1.0s."""
    n = 1_024
    leaves = [f"leaf_{i}".encode() for i in range(n)]

    start = time.perf_counter()

    tree = MerkleTree(leaves)
    for i in range(n):
        proof = tree.get_proof(i)
        assert proof.verify()

    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"merkle_tree_construction took {elapsed:.3f}s (limit 1.0s)"


def test_scheduler_rank_throughput():
    """Create 100 FakeJobs, call rank() 100 times. Total time < 0.5s."""
    scheduler = JobScheduler()
    now = time.time()
    jobs = [
        FakeJob(
            job_id=i,
            payment_amount=random.randint(1, 1_000),
            deadline=now + random.uniform(10, 600),
        )
        for i in range(100)
    ]

    start = time.perf_counter()

    for _ in range(100):
        ranked = scheduler.rank(jobs, node_id="node_0")
        assert len(ranked) == 100

    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"scheduler_rank_throughput took {elapsed:.3f}s (limit 0.5s)"


def test_gossip_propagation_speed():
    """20 GossipNodes in a ring; broadcast from node 0 reaches all 20 within 0.1s."""
    n = 20
    nodes = [GossipNode(node_id=f"node_{i}", fanout=3) for i in range(n)]

    # Ring topology: each node connected to its two neighbours
    for i in range(n):
        nodes[i].add_peer(nodes[(i - 1) % n])
        nodes[i].add_peer(nodes[(i + 1) % n])

    start = time.perf_counter()

    nodes[0].broadcast(GossipMessageType.NODE_HEALTH, payload={"status": "ok"}, ttl=20)

    elapsed = time.perf_counter() - start

    for node in nodes:
        assert node.inbox_count() >= 1, f"{node.node_id} did not receive the message"

    assert elapsed < 0.1, f"gossip_propagation_speed took {elapsed:.3f}s (limit 0.1s)"


def test_earnings_tracker_window_summary():
    """Record 500 earnings events, call window_summary() 100 times. Total time < 0.5s."""
    tracker = EarningsTracker()
    n = 500

    start = time.perf_counter()

    for i in range(n):
        tracker.record(
            job_id=i,
            amount_lamports=random.randint(100, 10_000),
            model_name=f"model_{i % 5}",
            tokens_generated=random.randint(50, 2_000),
        )

    for _ in range(100):
        summary = tracker.window_summary()
        assert summary["jobs_completed"] == n

    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"earnings_tracker_window_summary took {elapsed:.3f}s (limit 0.5s)"


def test_rate_limiter_concurrent_checks():
    """500 concurrent is_allowed() checks with burst_size=1000; all allowed. Total time < 1.0s."""
    config = RateLimitConfig(requests_per_minute=60_000, burst_size=1_000)
    limiter = RateLimiter(config)

    async def _run():
        tasks = [limiter.is_allowed(f"key_{i}") for i in range(500)]
        results = await asyncio.gather(*tasks)
        return results

    start = time.perf_counter()

    results = asyncio.run(_run())

    elapsed = time.perf_counter() - start

    assert all(results), "Some requests were unexpectedly rate-limited"
    assert elapsed < 1.0, f"rate_limiter_concurrent_checks took {elapsed:.3f}s (limit 1.0s)"
