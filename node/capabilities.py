from dataclasses import dataclass, field
from enum import Enum


class PrecisionType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class NodeCapabilities:
    node_id: str
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    supported_precisions: list[PrecisionType] = field(default_factory=lambda: [PrecisionType.FP32])
    supported_models: list[str] = field(default_factory=list)
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    network_bandwidth_mbps: float = 100.0

    def can_run_model(self, model_name: str) -> bool:
        """True if model_name is in supported_models (case-insensitive prefix match allowed)."""
        model_lower = model_name.lower()
        return any(
            model_lower == m.lower() or model_lower.startswith(m.lower())
            for m in self.supported_models
        )

    def supports_precision(self, precision: PrecisionType) -> bool:
        return precision in self.supported_precisions


@dataclass
class JobRequirements:
    model_name: str
    min_gpu_memory_gb: float = 0.0
    required_precision: PrecisionType = PrecisionType.FP16
    min_sequence_length: int = 512
    min_batch_size: int = 1


class CapabilityMatcher:
    """Matches job requirements to eligible nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodeCapabilities] = {}

    def register(self, caps: NodeCapabilities) -> None:
        self._nodes[caps.node_id] = caps

    def deregister(self, node_id: str) -> bool:
        return self._nodes.pop(node_id, None) is not None

    def matches(self, caps: NodeCapabilities, req: JobRequirements) -> bool:
        """Return True if this node can satisfy all job requirements."""
        if not caps.can_run_model(req.model_name):
            return False
        if caps.gpu_memory_gb < req.min_gpu_memory_gb:
            return False
        if not caps.supports_precision(req.required_precision):
            return False
        if caps.max_sequence_length < req.min_sequence_length:
            return False
        if caps.max_batch_size < req.min_batch_size:
            return False
        return True

    def find_eligible(self, req: JobRequirements) -> list[NodeCapabilities]:
        """Return all registered nodes that can satisfy req."""
        return [c for c in self._nodes.values() if self.matches(c, req)]

    def best_node(self, req: JobRequirements) -> NodeCapabilities | None:
        """
        Return the single best node by: most GPU memory, then most CPU cores.
        Returns None if no eligible node.
        """
        eligible = self.find_eligible(req)
        if not eligible:
            return None
        return max(eligible, key=lambda c: (c.gpu_memory_gb, c.cpu_cores))
