from dataclasses import dataclass, field

from node.router import InferenceRouter, RouteDecision, RoutingConfig


@dataclass
class FakeNode:
    node_id: str
    reputation: float = 0.5
    current_load: int = 0
    max_load: int = 4
    cost_per_token: int = 100
    supported_models: list = field(default_factory=lambda: ["llama"])


def make_router(**kwargs):
    return InferenceRouter(config=RoutingConfig(**kwargs))


# ---------------------------------------------------------------------------
# Basic routing tests
# ---------------------------------------------------------------------------


def test_route_returns_none_no_nodes():
    router = InferenceRouter()
    assert router.route([], "llama") is None


def test_route_returns_none_no_match():
    router = InferenceRouter()
    node = FakeNode(node_id="n1", supported_models=["gpt2"])
    assert router.route([node], "llama") is None


def test_route_returns_decision():
    router = InferenceRouter()
    node = FakeNode(node_id="n1")
    decision = router.route([node], "llama")
    assert isinstance(decision, RouteDecision)
    assert decision.node_id == "n1"


def test_route_excludes_full_nodes():
    router = InferenceRouter()
    full_node = FakeNode(node_id="full", current_load=4, max_load=4)
    good_node = FakeNode(node_id="good", current_load=0, max_load=4)
    decision = router.route([full_node, good_node], "llama")
    assert decision is not None
    assert decision.node_id == "good"


def test_route_excludes_below_reputation():
    router = make_router(min_reputation=0.7)
    low_rep = FakeNode(node_id="low", reputation=0.5)
    high_rep = FakeNode(node_id="high", reputation=0.9)
    decision = router.route([low_rep, high_rep], "llama")
    assert decision is not None
    assert decision.node_id == "high"


def test_route_excludes_above_cost():
    router = make_router(max_cost_lamports=150)
    expensive = FakeNode(node_id="expensive", cost_per_token=200)
    cheap = FakeNode(node_id="cheap", cost_per_token=100)
    decision = router.route([expensive, cheap], "llama")
    assert decision is not None
    assert decision.node_id == "cheap"


def test_route_prefers_high_reputation():
    # Use weights that make reputation dominate; keep cost & load equal
    router = make_router(reputation_weight=1.0, availability_weight=0.0, cost_weight=0.0)
    low_rep = FakeNode(node_id="low", reputation=0.2)
    high_rep = FakeNode(node_id="high", reputation=0.9)
    decision = router.route([low_rep, high_rep], "llama")
    assert decision is not None
    assert decision.node_id == "high"


# ---------------------------------------------------------------------------
# route_multi tests
# ---------------------------------------------------------------------------


def test_route_multi_returns_top_n():
    router = InferenceRouter()
    nodes = [
        FakeNode(node_id="n1", reputation=0.9),
        FakeNode(node_id="n2", reputation=0.5),
        FakeNode(node_id="n3", reputation=0.1),
    ]
    decisions = router.route_multi(nodes, "llama", count=2)
    assert len(decisions) == 2
    # Top 2 by reputation (with equal other factors) should be n1 and n2
    ids = {d.node_id for d in decisions}
    assert "n1" in ids
    assert "n2" in ids


def test_route_multi_fewer_than_count():
    router = InferenceRouter()
    node = FakeNode(node_id="only")
    decisions = router.route_multi([node], "llama", count=3)
    assert len(decisions) == 1
    assert decisions[0].node_id == "only"


# ---------------------------------------------------------------------------
# Reason field tests
# ---------------------------------------------------------------------------


def test_route_only_option_reason():
    router = InferenceRouter()
    node = FakeNode(node_id="solo")
    decision = router.route([node], "llama")
    assert decision is not None
    assert decision.reason == "only_option"


def test_route_best_score_reason():
    router = InferenceRouter()
    nodes = [FakeNode(node_id="n1"), FakeNode(node_id="n2")]
    decision = router.route(nodes, "llama")
    assert decision is not None
    assert decision.reason == "best_score"
