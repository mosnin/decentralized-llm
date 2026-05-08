"""End-to-end integration tests wiring multiple components together."""

import asyncio
from dataclasses import dataclass

from client.python.reputation import ReputationCache
from node.audit_log import AuditEventType, AuditLog
from node.capabilities import CapabilityMatcher, JobRequirements, NodeCapabilities, PrecisionType
from node.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError
from node.earnings import EarningsTracker
from node.peer_manager import PeerManager, PeerStatus
from node.rate_limiter import RateLimitConfig, RateLimiter
from node.result_cache import ResultCache
from node.router import InferenceRouter, RoutingConfig
from node.settlement import SettlementTracker

# ---------------------------------------------------------------------------
# Scenario 1: Full job lifecycle simulation
# ---------------------------------------------------------------------------


def test_full_job_lifecycle():
    reputation = ReputationCache()
    result_cache = ResultCache()
    earnings = EarningsTracker()
    rate_limiter = RateLimiter(RateLimitConfig(requests_per_minute=60, burst_size=10))
    audit = AuditLog()

    async def run():
        for i in range(5):
            # Record reputation: 4 successes, 1 failure
            success = i < 4
            reputation.record("node-1", success=success, latency_ms=200.0)

            # Cache result for each job
            model = "llama"
            prompt = f"prompt-{i}"
            result_cache.put(model, prompt, f"response-{i}")

            # Record earnings
            earnings.record(job_id=i, amount_lamports=1000 * (i + 1), model_name=model)

            # Append audit event
            audit.append(
                AuditEventType.JOB_COMPLETED,
                actor="node-1",
                payload={"job_id": i, "tokens": 100},
            )

            # Record rate limiter check for client-1
            await rate_limiter.is_allowed("client-1")

    asyncio.run(run())

    # Verify reputation score > 0.7 (4 out of 5 successes)
    score = reputation.score("node-1")
    assert score > 0.7, f"Expected score > 0.7, got {score}"

    # Verify result cache has all 5 entries
    stats = result_cache.stats()
    assert stats["size"] == 5, f"Expected 5 cache entries, got {stats['size']}"

    # Verify earnings window shows 5 jobs
    summary = earnings.window_summary()
    assert summary["jobs_completed"] == 5, f"Expected 5 jobs, got {summary['jobs_completed']}"

    # Verify audit log has 5 JOB_COMPLETED entries
    completed = audit.entries_by_type(AuditEventType.JOB_COMPLETED)
    assert len(completed) == 5, f"Expected 5 audit entries, got {len(completed)}"

    # Verify audit chain integrity
    assert audit.verify_chain(), "Audit chain verification failed"

    # Verify rate limiter still allows requests (burst_size=10, used 5)
    async def check_allowed():
        return await rate_limiter.is_allowed("client-1")

    allowed = asyncio.run(check_allowed())
    assert allowed, "Rate limiter should still allow requests within burst"


# ---------------------------------------------------------------------------
# Scenario 2: PeerManager + CapabilityMatcher + InferenceRouter pipeline
# ---------------------------------------------------------------------------


@dataclass
class FakeNode:
    node_id: str
    reputation: float
    current_load: int
    max_load: int
    cost_per_token: int
    supported_models: list


def test_peer_routing_pipeline():
    peer_manager = PeerManager()
    capability_matcher = CapabilityMatcher()
    router = InferenceRouter(RoutingConfig(min_reputation=0.0))

    node_ids = ["node-a", "node-b", "node-c"]

    # Register 3 nodes in PeerManager and CapabilityMatcher
    for i, nid in enumerate(node_ids):
        peer_manager.add_peer(
            peer_id=nid,
            host=f"host-{i}",
            port=8000 + i,
            supported_models=["llama"],
        )
        capability_matcher.register(
            NodeCapabilities(
                node_id=nid,
                gpu_memory_gb=8.0,
                cpu_cores=4,
                supported_precisions=[PrecisionType.FP16],
                supported_models=["llama"],
                max_batch_size=4,
                max_sequence_length=2048,
            )
        )

    # Tick twice → nodes become SUSPECTED but not DEAD
    peer_manager.tick()
    peer_manager.tick()

    # Confirm all are SUSPECTED, not DEAD
    for nid in node_ids:
        status = peer_manager.status(nid)
        assert status == PeerStatus.SUSPECTED, f"Expected SUSPECTED for {nid}, got {status}"

    # Send heartbeat for node-a and node-b (they become ACTIVE again)
    peer_manager.heartbeat("node-a")
    peer_manager.heartbeat("node-b")

    # Find eligible nodes via CapabilityMatcher for "llama"
    req = JobRequirements(
        model_name="llama",
        min_gpu_memory_gb=4.0,
        required_precision=PrecisionType.FP16,
    )
    eligible_caps = capability_matcher.find_eligible(req)
    assert len(eligible_caps) == 3, f"Expected 3 eligible nodes, got {len(eligible_caps)}"

    # Build FakeNode list only for ACTIVE peers (those that sent heartbeats)
    active_ids = {p.peer_id for p in peer_manager.active_peers()}
    assert "node-a" in active_ids
    assert "node-b" in active_ids
    assert "node-c" not in active_ids

    fake_nodes = [
        FakeNode(
            node_id=caps.node_id,
            reputation=0.9,
            current_load=0,
            max_load=10,
            cost_per_token=10,
            supported_models=caps.supported_models,
        )
        for caps in eligible_caps
        if caps.node_id in active_ids
    ]

    # Route to best node
    decision = router.route(fake_nodes, "llama")
    assert decision is not None, "Router should return a decision"
    assert decision.node_id in {"node-a", "node-b"}, (
        f"Routed to {decision.node_id}, expected one of the healthy nodes"
    )


# ---------------------------------------------------------------------------
# Scenario 3: CircuitBreaker + with_retry composition
# ---------------------------------------------------------------------------


def test_circuit_breaker_with_retry():
    cb = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=60.0),
        name="test-cb",
    )

    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ValueError("transient failure")
        return "ok"

    # First 2 calls fail, 3rd succeeds — circuit stays CLOSED (< threshold=3)
    async def run_sequence():
        results = []
        for _ in range(3):
            try:
                r = await cb.call(flaky)
                results.append(r)
            except ValueError:
                results.append("error")
        return results

    results = asyncio.run(run_sequence())
    assert results == ["error", "error", "ok"]

    from node.circuit_breaker import CircuitState

    assert cb.state == CircuitState.CLOSED, f"Circuit should be CLOSED, got {cb.state}"

    # Now trigger 3 consecutive failures → circuit opens
    call_count = 0  # reset so flaky always fails for a while

    always_fail_count = 0

    async def always_fail():
        nonlocal always_fail_count
        always_fail_count += 1
        raise RuntimeError("always fails")

    async def trigger_open():
        for _ in range(3):
            try:
                await cb.call(always_fail)
            except (RuntimeError, CircuitOpenError):
                pass

    asyncio.run(trigger_open())

    assert cb.state == CircuitState.OPEN, f"Circuit should be OPEN, got {cb.state}"

    # Next call should raise CircuitOpenError
    async def next_call():
        await cb.call(always_fail)

    try:
        asyncio.run(next_call())
        assert False, "Expected CircuitOpenError"
    except CircuitOpenError:
        pass  # expected


# ---------------------------------------------------------------------------
# Scenario 4: Settlement + EarningsTracker + AuditLog
# ---------------------------------------------------------------------------


def test_settlement_earnings_audit():
    settlement = SettlementTracker(min_batch_lamports=1, max_batch_size=100)
    earnings = EarningsTracker()
    audit = AuditLog()

    # Record 5 payments in both EarningsTracker and SettlementTracker
    for i in range(5):
        amount = 2000 * (i + 1)
        settlement.record(job_id=i, amount_lamports=amount, client_pubkey="client-pub-key")
        earnings.record(job_id=i, amount_lamports=amount, model_name="llama")
        audit.append(
            AuditEventType.PAYMENT_RECEIVED,
            actor="node-1",
            payload={"job_id": i, "amount": amount},
        )

    # Flush the settlement batch
    batch = settlement.flush()
    assert batch is not None, "Flush should return a batch"
    assert batch.job_count == 5, f"Expected 5 records in batch, got {batch.job_count}"

    # Confirm the batch with a fake tx signature
    confirmed = settlement.confirm(batch.batch_id, tx_signature="fake-tx-sig-abc123")
    assert confirmed, "Confirm should return True"

    # Verify total_confirmed_lamports > 0
    total = settlement.total_confirmed_lamports()
    assert total > 0, f"Expected total_confirmed_lamports > 0, got {total}"

    # Verify earnings summary has 5 jobs
    summary = earnings.window_summary()
    assert summary["jobs_completed"] == 5

    # Verify audit chain is valid
    payment_entries = audit.entries_by_type(AuditEventType.PAYMENT_RECEIVED)
    assert len(payment_entries) == 5, (
        f"Expected 5 PAYMENT_RECEIVED events, got {len(payment_entries)}"
    )
    assert audit.verify_chain(), "Audit chain verification failed"
