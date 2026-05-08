import random
from dataclasses import dataclass


@dataclass
class ShardNode:
    node_id: str
    shard_index: int  # which shard layer this node hosts
    model_name: str
    current_load: int = 0  # active jobs currently running
    max_load: int = 4  # max concurrent jobs
    reputation: float = 0.5  # 0.0–1.0
    avg_latency_ms: float = 500.0

    @property
    def is_available(self) -> bool:
        return self.current_load < self.max_load

    @property
    def load_factor(self) -> float:
        """0.0 (idle) to 1.0 (full)."""
        if self.max_load == 0:
            return 1.0
        return self.current_load / self.max_load


class ShardBalancer:
    """
    Selects the best available node for each shard using a weighted score:
    score = reputation_weight * reputation
          + availability_weight * (1 - load_factor)
          + latency_weight * (1 - normalized_latency)
    """

    REPUTATION_WEIGHT = 0.4
    AVAILABILITY_WEIGHT = 0.4
    LATENCY_WEIGHT = 0.2
    MAX_LATENCY_MS = 10_000.0  # normalize against this

    def __init__(self):
        self._nodes: dict[str, ShardNode] = {}

    def register(self, node: ShardNode) -> None:
        self._nodes[node.node_id] = node

    def deregister(self, node_id: str) -> bool:
        return self._nodes.pop(node_id, None) is not None

    def _score(self, node: ShardNode) -> float:
        norm_latency = min(1.0, node.avg_latency_ms / self.MAX_LATENCY_MS)
        return (
            self.REPUTATION_WEIGHT * node.reputation
            + self.AVAILABILITY_WEIGHT * (1.0 - node.load_factor)
            + self.LATENCY_WEIGHT * (1.0 - norm_latency)
        )

    def select(self, model_name: str, shard_index: int) -> ShardNode | None:
        """
        Return the best available node for (model_name, shard_index).
        Returns None if no available node exists.
        Breaks ties randomly.
        """
        candidates = [
            n
            for n in self._nodes.values()
            if n.model_name == model_name and n.shard_index == shard_index and n.is_available
        ]
        if not candidates:
            return None
        max_score = max(self._score(n) for n in candidates)
        top = [n for n in candidates if self._score(n) == max_score]
        return random.choice(top)

    def select_pipeline(self, model_name: str, num_shards: int) -> list[ShardNode] | None:
        """
        Select one node per shard index (0..num_shards-1).
        Returns None if any shard index has no available node.
        """
        pipeline = []
        for i in range(num_shards):
            node = self.select(model_name, i)
            if node is None:
                return None
            pipeline.append(node)
        return pipeline

    def update_load(self, node_id: str, delta: int) -> None:
        """Increment (+1 on start) or decrement (-1 on finish) current_load."""
        node = self._nodes.get(node_id)
        if node:
            node.current_load = max(0, node.current_load + delta)

    def update_latency(self, node_id: str, latency_ms: float, alpha: float = 0.2) -> None:
        """Exponential moving average update for avg_latency_ms."""
        node = self._nodes.get(node_id)
        if node:
            node.avg_latency_ms = (1 - alpha) * node.avg_latency_ms + alpha * latency_ms

    def available_nodes(self) -> list[ShardNode]:
        return [n for n in self._nodes.values() if n.is_available]
