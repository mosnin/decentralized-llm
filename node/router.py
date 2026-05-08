import random
from dataclasses import dataclass
from typing import Protocol


class NodeInfo(Protocol):
    node_id: str
    reputation: float  # 0.0-1.0
    current_load: int
    max_load: int
    cost_per_token: int  # lamports per token
    supported_models: list[str]


@dataclass
class RoutingConfig:
    reputation_weight: float = 0.4
    availability_weight: float = 0.3
    cost_weight: float = 0.3
    max_cost_lamports: int | None = None  # if set, excludes nodes above this cost
    min_reputation: float = 0.0  # exclude nodes below this


@dataclass
class RouteDecision:
    node_id: str
    score: float
    reason: str  # "best_score" | "fallback_random" | "only_option"


class InferenceRouter:
    """
    Routes inference requests to the best available node.

    Selection: filter by capability/cost/reputation constraints,
    then score remaining nodes, pick highest scorer (break ties randomly).
    """

    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()

    def _is_eligible(self, node: NodeInfo, model_name: str) -> bool:
        """Check hard constraints."""
        # Must support the model
        model_lower = model_name.lower()
        if not any(model_lower == m.lower() for m in node.supported_models):
            return False
        # Must have capacity
        if node.current_load >= node.max_load:
            return False
        # Reputation floor
        if node.reputation < self.config.min_reputation:
            return False
        # Cost ceiling
        if (
            self.config.max_cost_lamports is not None
            and node.cost_per_token > self.config.max_cost_lamports
        ):
            return False
        return True

    def _score(self, node: NodeInfo, max_cost: int) -> float:
        load_factor = node.current_load / max(node.max_load, 1)
        norm_cost = node.cost_per_token / max(max_cost, 1)
        c = self.config
        return (
            c.reputation_weight * node.reputation
            + c.availability_weight * (1.0 - load_factor)
            + c.cost_weight * (1.0 - norm_cost)
        )

    def route(self, nodes: list, model_name: str) -> RouteDecision | None:
        """
        Select the best node for model_name from the given list.
        Returns None if no eligible node exists.
        """
        eligible = [n for n in nodes if self._is_eligible(n, model_name)]
        if not eligible:
            return None

        if len(eligible) == 1:
            return RouteDecision(node_id=eligible[0].node_id, score=1.0, reason="only_option")

        max_cost = max(n.cost_per_token for n in eligible)
        scores = [(n, self._score(n, max_cost)) for n in eligible]
        best_score = max(s for _, s in scores)
        top_nodes = [n for n, s in scores if s == best_score]
        chosen = random.choice(top_nodes)
        return RouteDecision(node_id=chosen.node_id, score=best_score, reason="best_score")

    def route_multi(self, nodes: list, model_name: str, count: int) -> list[RouteDecision]:
        """
        Route to the top `count` distinct nodes (for redundant requests).
        Returns fewer than count if not enough eligible nodes.
        """
        eligible = [n for n in nodes if self._is_eligible(n, model_name)]
        if not eligible:
            return []

        max_cost = max(n.cost_per_token for n in eligible)
        scored = sorted(eligible, key=lambda n: self._score(n, max_cost), reverse=True)

        return [
            RouteDecision(node_id=n.node_id, score=self._score(n, max_cost), reason="best_score")
            for n in scored[:count]
        ]
