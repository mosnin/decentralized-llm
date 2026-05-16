"""
Network topology optimizer for decentralized LLM inference.

Minimizes end-to-end pipeline latency by dynamically reassigning model shards
to nodes using a scoring function that balances latency, load, and churn, then
uses simulated annealing to search for a better shard-to-node mapping.

Pipeline flow:  shard 0 → shard 1 → shard 2 → … → shard N-1
The total transfer cost is the sum of pairwise latencies between consecutive
nodes in the pipeline order.
"""

import math
import random
from dataclasses import dataclass, field


class NodeLatencyMatrix:
    """
    Stores pairwise ping latencies (ms) between nodes.

    Unknown pairs default to 100 ms.
    """

    DEFAULT_LATENCY_MS: float = 100.0

    def __init__(self) -> None:
        # (from_node, to_node) -> latency_ms
        self._data: dict[tuple[str, str], float] = {}

    def update(self, from_node: str, to_node: str, latency_ms: float) -> None:
        """Record or overwrite the latency from *from_node* to *to_node*."""
        self._data[(from_node, to_node)] = latency_ms

    def get(self, from_node: str, to_node: str) -> float:
        """
        Return the stored latency (ms) between the two nodes.

        Falls back to ``DEFAULT_LATENCY_MS`` (100 ms) when the pair has not
        been recorded.  Self-to-self latency is always 0.
        """
        if from_node == to_node:
            return 0.0
        return self._data.get((from_node, to_node), self.DEFAULT_LATENCY_MS)

    def get_avg_path_latency(self, path: list[str]) -> float:
        """
        Return the total latency (ms) for a sequence of nodes by summing the
        latency of each consecutive pair.

        An empty or single-node path has zero latency.
        """
        if len(path) < 2:
            return 0.0
        return sum(self.get(path[i], path[i + 1]) for i in range(len(path) - 1))


@dataclass
class ShardAssignment:
    """Maps a model shard to the node that should serve it."""

    shard_index: int
    node_id: str
    model_id: str
    score: float = field(default=0.0)


class TopologyOptimizer:
    """
    Finds a shard-to-node assignment that minimises end-to-end inference
    latency using a weighted score and simulated annealing.

    Score (lower is better)
    -----------------------
    score = alpha  * pipeline_latency
          + beta   * load_imbalance
          + gamma  * churn_penalty

    Parameters
    ----------
    num_shards:
        Number of pipeline shards.
    alpha:
        Weight for cumulative inter-node transfer latency.
    beta:
        Weight for node-load imbalance (std-dev of load fractions).
    gamma:
        Weight for churn penalty (number of shards moved from current
        assignment, normalised to [0, 1]).
    """

    def __init__(
        self,
        num_shards: int,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
    ) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        self.num_shards = num_shards
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------ scoring

    def score_assignment(
        self,
        assignments: list[ShardAssignment],
        latency_matrix: NodeLatencyMatrix,
        node_loads: dict[str, float],
    ) -> float:
        """
        Score a full shard assignment (lower is better).

        Parameters
        ----------
        assignments:
            One ``ShardAssignment`` per shard, in any order.
        latency_matrix:
            Pairwise inter-node latencies.
        node_loads:
            Mapping of node_id → current load fraction in [0, 1].

        Returns
        -------
        float
            Composite cost.  Returns ``float('inf')`` for an incomplete
            (wrong number of shards) assignment.
        """
        if len(assignments) != self.num_shards:
            return float("inf")

        ordered = sorted(assignments, key=lambda a: a.shard_index)

        # --- latency component (ms) ------------------------------------------
        pipeline_path = [a.node_id for a in ordered]
        pipeline_latency = latency_matrix.get_avg_path_latency(pipeline_path)

        # --- load imbalance component -----------------------------------------
        # Use std-dev of load fractions across nodes used in this assignment.
        # A perfectly balanced assignment has std-dev 0.
        used_nodes = list({a.node_id for a in assignments})
        loads = [node_loads.get(n, 0.0) for n in used_nodes]
        if len(loads) > 1:
            mean_load = sum(loads) / len(loads)
            variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
            load_imbalance = math.sqrt(variance)
        else:
            load_imbalance = 0.0

        # --- churn component (proportion of shards that changed node) ---------
        # Needs the previous assignment; computed externally and stored on each
        # ShardAssignment.score field.  Here we compare node_ids directly.
        # churn is set to 0 when there is no previous assignment (first run).
        churn_penalty = 0.0  # placeholder — filled in optimize()

        return (
            self.alpha * pipeline_latency
            + self.beta * load_imbalance
            + self.gamma * churn_penalty
        )

    def _score_with_churn(
        self,
        assignments: list[ShardAssignment],
        latency_matrix: NodeLatencyMatrix,
        node_loads: dict[str, float],
        current_assignment: list[ShardAssignment] | None,
    ) -> float:
        """Internal scoring that also accounts for churn against *current_assignment*."""
        if len(assignments) != self.num_shards:
            return float("inf")

        ordered = sorted(assignments, key=lambda a: a.shard_index)

        # latency
        pipeline_path = [a.node_id for a in ordered]
        pipeline_latency = latency_matrix.get_avg_path_latency(pipeline_path)

        # load imbalance
        used_nodes = list({a.node_id for a in assignments})
        loads = [node_loads.get(n, 0.0) for n in used_nodes]
        if len(loads) > 1:
            mean_load = sum(loads) / len(loads)
            variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
            load_imbalance = math.sqrt(variance)
        else:
            load_imbalance = 0.0

        # churn
        if current_assignment:
            cur_map = {a.shard_index: a.node_id for a in current_assignment}
            moves = sum(
                1
                for a in ordered
                if cur_map.get(a.shard_index) != a.node_id
            )
            churn_penalty = moves / self.num_shards
        else:
            churn_penalty = 0.0

        return (
            self.alpha * pipeline_latency
            + self.beta * load_imbalance
            + self.gamma * churn_penalty
        )

    # --------------------------------------------------------------- optimize

    def optimize(
        self,
        candidate_nodes: list[str],
        latency_matrix: NodeLatencyMatrix,
        node_loads: dict[str, float],
        current_assignment: list[ShardAssignment] | None,
        max_iterations: int = 100,
    ) -> list[ShardAssignment]:
        """
        Find a low-cost shard assignment using simulated annealing.

        Starting point
        ~~~~~~~~~~~~~~
        If *current_assignment* covers all shards and only uses nodes from
        *candidate_nodes*, it is used as the initial solution.  Otherwise a
        greedy initial assignment is built (shard 0 → node 0, shard 1 → node 1
        cycling through candidates).

        Neighbourhood move
        ~~~~~~~~~~~~~~~~~~
        A random swap of the node assignments for two randomly chosen shards.

        Acceptance criterion
        ~~~~~~~~~~~~~~~~~~~~
        Always accept improvements.  Accept worse solutions with probability
        ``exp(-Δscore / T)`` where *T* decreases geometrically from
        ``T_start = 10.0`` to ``T_min = 0.01`` over *max_iterations* steps.

        Returns
        -------
        list[ShardAssignment]
            Best assignment found, sorted by shard index.
        """
        if not candidate_nodes:
            raise ValueError("candidate_nodes must not be empty")

        model_id = ""
        if current_assignment:
            model_id = current_assignment[0].model_id

        # Build initial assignment
        if current_assignment and len(current_assignment) == self.num_shards:
            cur_nodes = {a.node_id for a in current_assignment}
            if cur_nodes.issubset(set(candidate_nodes)):
                current_sorted = sorted(current_assignment, key=lambda a: a.shard_index)
                best = [
                    ShardAssignment(
                        shard_index=a.shard_index,
                        node_id=a.node_id,
                        model_id=a.model_id,
                    )
                    for a in current_sorted
                ]
            else:
                best = self._greedy_initial(candidate_nodes, model_id)
        else:
            best = self._greedy_initial(candidate_nodes, model_id)

        best_score = self._score_with_churn(best, latency_matrix, node_loads, current_assignment)

        current = [
            ShardAssignment(
                shard_index=a.shard_index, node_id=a.node_id, model_id=a.model_id
            )
            for a in best
        ]
        current_score = best_score

        # Simulated annealing
        T_start = 10.0
        T_min = 0.01
        cooling = (T_min / T_start) ** (1.0 / max(max_iterations - 1, 1))
        T = T_start

        self.last_exploration_stats: dict[str, int] = {"accepted_worse": 0, "total_moves": 0}

        for _ in range(max_iterations):
            # Generate neighbour by swapping node assignments of two shards
            neighbour = [
                ShardAssignment(
                    shard_index=a.shard_index, node_id=a.node_id, model_id=a.model_id
                )
                for a in current
            ]
            i, j = random.sample(range(self.num_shards), 2) if self.num_shards > 1 else (0, 0)
            neighbour[i].node_id, neighbour[j].node_id = (
                neighbour[j].node_id,
                neighbour[i].node_id,
            )

            # Also occasionally try assigning a random candidate node to a shard
            if random.random() < 0.3:
                k = random.randrange(self.num_shards)
                neighbour[k].node_id = random.choice(candidate_nodes)

            neighbour_score = self._score_with_churn(
                neighbour, latency_matrix, node_loads, current_assignment
            )

            delta = neighbour_score - current_score
            self.last_exploration_stats["total_moves"] += 1
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                if delta > 0:
                    self.last_exploration_stats["accepted_worse"] += 1
                current = neighbour
                current_score = neighbour_score

                if current_score < best_score:
                    best = [
                        ShardAssignment(
                            shard_index=a.shard_index,
                            node_id=a.node_id,
                            model_id=a.model_id,
                        )
                        for a in current
                    ]
                    best_score = current_score

            T *= cooling

        result = sorted(best, key=lambda a: a.shard_index)
        for a in result:
            a.score = best_score
        return result

    # --------------------------------------------------------- utility methods

    def should_rebalance(
        self,
        score_current: float,
        score_proposed: float,
        threshold: float = 0.15,
    ) -> bool:
        """
        Return ``True`` only if the proposed assignment is more than
        *threshold* (default 15 %) better than the current one.

        Guards against thrashing when the improvement is marginal.
        """
        if score_current <= 0:
            # Avoid division by zero; only rebalance if proposed is strictly better
            return score_proposed < score_current
        improvement = (score_current - score_proposed) / score_current
        return improvement > threshold

    def get_pipeline_order(self, assignments: list[ShardAssignment]) -> list[str]:
        """
        Return node IDs in shard order (shard 0 → 1 → 2 → …).

        Parameters
        ----------
        assignments:
            Shard assignments in any order.

        Returns
        -------
        list[str]
            Node IDs ordered by ascending shard index.
        """
        ordered = sorted(assignments, key=lambda a: a.shard_index)
        return [a.node_id for a in ordered]

    def estimate_e2e_latency(
        self,
        assignments: list[ShardAssignment],
        latency_matrix: NodeLatencyMatrix,
    ) -> float:
        """
        Estimate end-to-end inference latency (ms) as the sum of network
        transfer latencies between consecutive shards in the pipeline.

        This is a lower bound — it counts only inter-node network hops and
        ignores compute time.
        """
        pipeline_path = self.get_pipeline_order(assignments)
        return latency_matrix.get_avg_path_latency(pipeline_path)

    # --------------------------------------------------------------- internals

    def _greedy_initial(self, candidate_nodes: list[str], model_id: str) -> list[ShardAssignment]:
        """Assign shards round-robin across candidate nodes as a starting point."""
        return [
            ShardAssignment(
                shard_index=i,
                node_id=candidate_nodes[i % len(candidate_nodes)],
                model_id=model_id,
            )
            for i in range(self.num_shards)
        ]
