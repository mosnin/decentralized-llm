"""
Tests for node/topology_optimizer.py

Covers:
  - NodeLatencyMatrix: update, get, default fallback, avg_path_latency
  - ShardAssignment dataclass
  - TopologyOptimizer: score_assignment, optimize, should_rebalance,
    get_pipeline_order, estimate_e2e_latency
  - Simulated annealing behaviour (explores worse states; escapes local minima)
  - Churn penalty (discourages moving shards that are already well-placed)
  - 3-shard, 5-node end-to-end optimisation scenario
"""

import math

import pytest

from node.topology_optimizer import (
    NodeLatencyMatrix,
    ShardAssignment,
    TopologyOptimizer,
)


# ============================================================ NodeLatencyMatrix

class TestNodeLatencyMatrix:
    def test_update_and_get_stored_latency(self):
        m = NodeLatencyMatrix()
        m.update("A", "B", 42.0)
        assert m.get("A", "B") == 42.0

    def test_get_unknown_pair_returns_default(self):
        m = NodeLatencyMatrix()
        assert m.get("X", "Y") == NodeLatencyMatrix.DEFAULT_LATENCY_MS

    def test_get_self_to_self_is_zero(self):
        m = NodeLatencyMatrix()
        assert m.get("A", "A") == 0.0

    def test_update_overwrites_previous_value(self):
        m = NodeLatencyMatrix()
        m.update("A", "B", 50.0)
        m.update("A", "B", 75.0)
        assert m.get("A", "B") == 75.0

    def test_asymmetric_latencies(self):
        """Latency A→B and B→A are stored independently."""
        m = NodeLatencyMatrix()
        m.update("A", "B", 30.0)
        m.update("B", "A", 60.0)
        assert m.get("A", "B") == 30.0
        assert m.get("B", "A") == 60.0

    def test_avg_path_latency_empty_path(self):
        m = NodeLatencyMatrix()
        assert m.get_avg_path_latency([]) == 0.0

    def test_avg_path_latency_single_node(self):
        m = NodeLatencyMatrix()
        assert m.get_avg_path_latency(["A"]) == 0.0

    def test_avg_path_latency_two_nodes(self):
        m = NodeLatencyMatrix()
        m.update("A", "B", 20.0)
        assert m.get_avg_path_latency(["A", "B"]) == 20.0

    def test_avg_path_latency_three_nodes(self):
        """Path A→B→C should sum latencies of both hops."""
        m = NodeLatencyMatrix()
        m.update("A", "B", 10.0)
        m.update("B", "C", 25.0)
        # C→? is not queried; path only has two hops
        assert m.get_avg_path_latency(["A", "B", "C"]) == 35.0

    def test_avg_path_latency_uses_default_for_unknown(self):
        m = NodeLatencyMatrix()
        # Neither hop is stored → both use the 100 ms default
        total = m.get_avg_path_latency(["X", "Y", "Z"])
        assert total == 200.0


# ============================================================ ShardAssignment

class TestShardAssignment:
    def test_shard_assignment_fields(self):
        sa = ShardAssignment(shard_index=2, node_id="node-7", model_id="llama-3")
        assert sa.shard_index == 2
        assert sa.node_id == "node-7"
        assert sa.model_id == "llama-3"
        assert sa.score == 0.0

    def test_shard_assignment_custom_score(self):
        sa = ShardAssignment(shard_index=0, node_id="n0", model_id="m", score=3.14)
        assert sa.score == 3.14


# ============================================================ TopologyOptimizer

def _make_assignments(mapping: dict[int, str], model_id: str = "m") -> list[ShardAssignment]:
    """Build a list of ShardAssignments from {shard_index: node_id}."""
    return [ShardAssignment(shard_index=k, node_id=v, model_id=model_id) for k, v in mapping.items()]


def _latency_matrix_from_dict(pairs: dict[tuple[str, str], float]) -> NodeLatencyMatrix:
    m = NodeLatencyMatrix()
    for (a, b), lat in pairs.items():
        m.update(a, b, lat)
    return m


class TestScoreAssignment:
    def test_score_known_latencies_no_load(self):
        """
        With zero load everywhere and no previous assignment, the score should
        be dominated by alpha * pipeline_latency.
        """
        opt = TopologyOptimizer(num_shards=2, alpha=1.0, beta=0.0, gamma=0.0)
        lm = NodeLatencyMatrix()
        lm.update("A", "B", 30.0)
        assignments = _make_assignments({0: "A", 1: "B"})
        node_loads = {"A": 0.0, "B": 0.0}
        score = opt.score_assignment(assignments, lm, node_loads)
        assert score == pytest.approx(30.0)

    def test_score_wrong_number_of_shards_returns_inf(self):
        opt = TopologyOptimizer(num_shards=3)
        lm = NodeLatencyMatrix()
        assignments = _make_assignments({0: "A", 1: "B"})  # only 2, need 3
        score = opt.score_assignment(assignments, lm, {})
        assert score == float("inf")

    def test_score_load_imbalance_increases_score(self):
        """
        Two identical pipelines, one with balanced load, one unbalanced.
        The unbalanced one should score higher (worse).
        """
        opt = TopologyOptimizer(num_shards=2, alpha=0.0, beta=1.0, gamma=0.0)
        lm = NodeLatencyMatrix()
        assignments = _make_assignments({0: "A", 1: "B"})

        balanced_loads = {"A": 0.5, "B": 0.5}
        unbalanced_loads = {"A": 0.0, "B": 1.0}

        score_balanced = opt.score_assignment(assignments, lm, balanced_loads)
        score_unbalanced = opt.score_assignment(assignments, lm, unbalanced_loads)

        assert score_balanced < score_unbalanced


class TestOptimize:
    def test_optimize_returns_correct_number_of_shards(self):
        opt = TopologyOptimizer(num_shards=3)
        lm = NodeLatencyMatrix()
        nodes = ["n0", "n1", "n2"]
        result = opt.optimize(nodes, lm, {}, current_assignment=None, max_iterations=20)
        assert len(result) == 3

    def test_optimize_shards_are_ordered(self):
        opt = TopologyOptimizer(num_shards=3)
        lm = NodeLatencyMatrix()
        nodes = ["n0", "n1", "n2"]
        result = opt.optimize(nodes, lm, {}, current_assignment=None, max_iterations=20)
        indices = [a.shard_index for a in result]
        assert indices == sorted(indices)

    def test_optimize_nodes_from_candidates(self):
        """Every assigned node must be from the candidate list."""
        opt = TopologyOptimizer(num_shards=3)
        lm = NodeLatencyMatrix()
        nodes = ["n0", "n1", "n2", "n3"]
        result = opt.optimize(nodes, lm, {}, current_assignment=None, max_iterations=50)
        for a in result:
            assert a.node_id in nodes

    def test_optimize_finds_lower_latency_assignment(self):
        """
        Set up a star topology where n0 is a central hub with low latency to
        all other nodes.  The optimizer should prefer routing through n0.
        The best pipeline for 2 shards is n0→n1 (10 ms) vs alternatives (~200 ms).
        """
        opt = TopologyOptimizer(num_shards=2, alpha=1.0, beta=0.0, gamma=0.0)
        lm = NodeLatencyMatrix()
        # n0 ↔ n1 is very fast
        lm.update("n0", "n1", 5.0)
        lm.update("n1", "n0", 5.0)
        # All other pairs are slow (default 100 ms)

        nodes = ["n0", "n1", "n2"]
        node_loads = {n: 0.0 for n in nodes}
        result = opt.optimize(
            nodes, lm, node_loads, current_assignment=None, max_iterations=200
        )
        e2e = opt.estimate_e2e_latency(result, lm)
        # The best possible latency is 5 ms (n0→n1 or n1→n0)
        assert e2e <= 50.0  # optimizer should find something much better than random


class TestShouldRebalance:
    def test_rebalance_if_improvement_above_threshold(self):
        opt = TopologyOptimizer(num_shards=2)
        # 20 % improvement > 15 % threshold
        assert opt.should_rebalance(100.0, 80.0, threshold=0.15) is True

    def test_no_rebalance_if_improvement_below_threshold(self):
        opt = TopologyOptimizer(num_shards=2)
        # 5 % improvement < 15 % threshold
        assert opt.should_rebalance(100.0, 95.0, threshold=0.15) is False

    def test_no_rebalance_if_proposed_is_worse(self):
        opt = TopologyOptimizer(num_shards=2)
        assert opt.should_rebalance(80.0, 100.0) is False

    def test_rebalance_at_exact_threshold_is_false(self):
        """Exactly at the threshold is not strictly greater → no rebalance."""
        opt = TopologyOptimizer(num_shards=2)
        assert opt.should_rebalance(100.0, 85.0, threshold=0.15) is False

    def test_rebalance_zero_current_score(self):
        """When current score is 0, only rebalance if proposed is strictly lower."""
        opt = TopologyOptimizer(num_shards=2)
        assert opt.should_rebalance(0.0, 0.0) is False
        assert opt.should_rebalance(0.0, -1.0) is True


class TestGetPipelineOrder:
    def test_returns_node_ids_in_shard_order(self):
        opt = TopologyOptimizer(num_shards=3)
        assignments = [
            ShardAssignment(shard_index=2, node_id="n2", model_id="m"),
            ShardAssignment(shard_index=0, node_id="n0", model_id="m"),
            ShardAssignment(shard_index=1, node_id="n1", model_id="m"),
        ]
        order = opt.get_pipeline_order(assignments)
        assert order == ["n0", "n1", "n2"]

    def test_single_shard_pipeline_order(self):
        opt = TopologyOptimizer(num_shards=1)
        assignments = [ShardAssignment(shard_index=0, node_id="solo", model_id="m")]
        assert opt.get_pipeline_order(assignments) == ["solo"]


class TestEstimateE2eLatency:
    def test_e2e_latency_two_shards(self):
        opt = TopologyOptimizer(num_shards=2)
        lm = NodeLatencyMatrix()
        lm.update("n0", "n1", 55.0)
        assignments = _make_assignments({0: "n0", 1: "n1"})
        assert opt.estimate_e2e_latency(assignments, lm) == pytest.approx(55.0)

    def test_e2e_latency_single_shard_is_zero(self):
        opt = TopologyOptimizer(num_shards=1)
        lm = NodeLatencyMatrix()
        assignments = [ShardAssignment(shard_index=0, node_id="n0", model_id="m")]
        assert opt.estimate_e2e_latency(assignments, lm) == 0.0

    def test_e2e_latency_three_shards_sums_hops(self):
        opt = TopologyOptimizer(num_shards=3)
        lm = NodeLatencyMatrix()
        lm.update("n0", "n1", 10.0)
        lm.update("n1", "n2", 20.0)
        assignments = _make_assignments({0: "n0", 1: "n1", 2: "n2"})
        assert opt.estimate_e2e_latency(assignments, lm) == pytest.approx(30.0)


# ============================================================ Simulated annealing

class TestSimulatedAnnealing:
    def test_annealing_explores_worse_states_temporarily(self):
        """
        With high temperature the optimizer should occasionally accept worse
        moves.  We check this by running many short trials (1 iteration each)
        from a known good starting point and verifying that at least some
        result in a worse immediate outcome — demonstrating the probabilistic
        acceptance.

        Because the test is probabilistic we use a large number of trials to
        make the failure probability negligible.
        """
        import random as _random

        _random.seed(42)

        opt = TopologyOptimizer(num_shards=2, alpha=1.0, beta=0.0, gamma=0.0)
        lm = NodeLatencyMatrix()
        # n0→n1 is very good; n0→n2 and n1→n2 are bad
        lm.update("n0", "n1", 1.0)
        lm.update("n0", "n2", 2.0)
        lm.update("n1", "n2", 2.0)
        lm.update("n1", "n0", 1.0)
        lm.update("n2", "n0", 2.0)
        lm.update("n2", "n1", 2.0)

        best_assignments = _make_assignments({0: "n0", 1: "n1"})
        nodes = ["n0", "n1", "n2"]

        accepted_worse_at_least_once = False
        for _ in range(200):
            opt.optimize(
                nodes, lm, {}, current_assignment=best_assignments, max_iterations=5
            )
            if opt.last_exploration_stats["accepted_worse"] > 0:
                accepted_worse_at_least_once = True
                break

        assert accepted_worse_at_least_once, (
            "Simulated annealing never accepted a worse state — "
            "probabilistic acceptance may be broken"
        )

    def test_annealing_converges_to_good_solution(self):
        """After enough iterations the optimizer should find a low-latency path."""
        import random as _random

        _random.seed(0)

        opt = TopologyOptimizer(num_shards=3, alpha=1.0, beta=0.0, gamma=0.0)
        lm = NodeLatencyMatrix()
        # Chain n0→n1→n2 is the only fast path
        lm.update("n0", "n1", 2.0)
        lm.update("n1", "n2", 2.0)
        # All other pairs are slow
        for a in ["n0", "n1", "n2", "n3"]:
            for b in ["n0", "n1", "n2", "n3"]:
                if (a, b) not in [("n0", "n1"), ("n1", "n2")]:
                    lm.update(a, b, 300.0)

        nodes = ["n0", "n1", "n2", "n3"]
        result = opt.optimize(nodes, lm, {}, current_assignment=None, max_iterations=500)
        e2e = opt.estimate_e2e_latency(result, lm)
        # Optimal is n0→n1→n2 = 4 ms.  Allow some slack for stochastic search.
        assert e2e < 100.0


# ============================================================ Churn penalty

class TestChurnPenalty:
    def test_churn_penalty_discourages_unnecessary_rebalancing(self):
        """
        When the current assignment is already good, a non-zero gamma (churn
        weight) should make the optimizer reluctant to move shards.

        We compare a high-gamma optimizer (prefers stability) with a zero-gamma
        one (ignores churn) when both start from a good assignment.
        The high-gamma optimizer should keep more shards in place.
        """
        import random as _random

        _random.seed(7)

        # All nodes have similar latency to each other
        lm = NodeLatencyMatrix()
        nodes = ["n0", "n1", "n2"]
        for a in nodes:
            for b in nodes:
                lm.update(a, b, 10.0)

        current = _make_assignments({0: "n0", 1: "n1", 2: "n2"})
        node_loads = {n: 0.3 for n in nodes}

        opt_stable = TopologyOptimizer(num_shards=3, alpha=0.3, beta=0.3, gamma=0.4)
        opt_mobile = TopologyOptimizer(num_shards=3, alpha=0.9, beta=0.1, gamma=0.0)

        runs_stable = 0
        runs_mobile = 0
        trials = 50

        for _ in range(trials):
            result_stable = opt_stable.optimize(
                nodes, lm, node_loads, current_assignment=current, max_iterations=50
            )
            result_mobile = opt_mobile.optimize(
                nodes, lm, node_loads, current_assignment=current, max_iterations=50
            )

            cur_map = {a.shard_index: a.node_id for a in current}
            stable_changes = sum(
                1 for a in result_stable if cur_map.get(a.shard_index) != a.node_id
            )
            mobile_changes = sum(
                1 for a in result_mobile if cur_map.get(a.shard_index) != a.node_id
            )
            runs_stable += stable_changes
            runs_mobile += mobile_changes

        # The stable (high gamma) optimizer should move fewer shards on average
        # across all trials than the mobile (zero gamma) one.
        assert runs_stable <= runs_mobile, (
            f"High-churn-penalty optimizer moved {runs_stable} shards "
            f"vs low-penalty optimizer {runs_mobile} — churn penalty not working"
        )


# ============================================================ 3-shard 5-node scenario

class TestThreeShardFiveNodeScenario:
    """
    End-to-end scenario: 3 shards, 5 candidate nodes.

    Topology:
      - n0 and n1 are colocated with 1 ms between them.
      - n1 and n2 are colocated with 2 ms between them.
      - All other pairs have high latency (200 ms).

    The optimal pipeline is n0 → n1 → n2 with total latency 3 ms.
    Nodes n3 and n4 are decoys with high cross-latency.
    """

    def setup_method(self):
        self.nodes = ["n0", "n1", "n2", "n3", "n4"]
        self.lm = NodeLatencyMatrix()

        # Fast chain
        self.lm.update("n0", "n1", 1.0)
        self.lm.update("n1", "n2", 2.0)
        # Reverse (pipeline goes one direction, but SA might swap)
        self.lm.update("n1", "n0", 1.0)
        self.lm.update("n2", "n1", 2.0)

        # Everything else is slow
        for a in self.nodes:
            for b in self.nodes:
                if (a, b) not in [
                    ("n0", "n1"), ("n1", "n2"), ("n1", "n0"), ("n2", "n1")
                ] and a != b:
                    self.lm.update(a, b, 200.0)

        self.node_loads = {n: 0.2 for n in self.nodes}

    def test_optimizer_finds_low_latency_path(self):
        import random as _random

        _random.seed(99)

        opt = TopologyOptimizer(num_shards=3, alpha=1.0, beta=0.0, gamma=0.0)
        result = opt.optimize(
            self.nodes, self.lm, self.node_loads,
            current_assignment=None,
            max_iterations=500,
        )
        e2e = opt.estimate_e2e_latency(result, self.lm)
        # Optimal is 3 ms; allow generous slack for stochastic search
        assert e2e < 100.0

    def test_pipeline_order_length_equals_num_shards(self):
        opt = TopologyOptimizer(num_shards=3)
        result = opt.optimize(
            self.nodes, self.lm, self.node_loads,
            current_assignment=None,
            max_iterations=20,
        )
        order = opt.get_pipeline_order(result)
        assert len(order) == 3

    def test_should_rebalance_accepts_significantly_better_proposal(self):
        opt = TopologyOptimizer(num_shards=3)
        bad_assignment = _make_assignments({0: "n3", 1: "n4", 2: "n3"})
        good_assignment = _make_assignments({0: "n0", 1: "n1", 2: "n2"})

        score_bad = opt._score_with_churn(bad_assignment, self.lm, self.node_loads, None)
        score_good = opt._score_with_churn(good_assignment, self.lm, self.node_loads, None)

        assert opt.should_rebalance(score_bad, score_good)

    def test_should_rebalance_rejects_marginal_improvement(self):
        opt = TopologyOptimizer(num_shards=3)
        # Two nearly identical assignments — very small score delta
        a1 = _make_assignments({0: "n0", 1: "n1", 2: "n2"})
        # Slightly perturb one node to a node that's only 1 ms worse
        lm2 = NodeLatencyMatrix()
        lm2.update("n0", "n1", 1.0)
        lm2.update("n1", "n2", 2.0)
        lm2.update("n0", "n1b", 1.5)
        lm2.update("n1b", "n2", 2.5)

        score1 = opt._score_with_churn(a1, lm2, self.node_loads, None)
        # score1 is the "current"; propose a trivially better score (0.1% better)
        score_marginal = score1 * 0.999
        assert not opt.should_rebalance(score1, score_marginal)
