import pytest

from node.capabilities import CapabilityMatcher, JobRequirements, NodeCapabilities, PrecisionType

# ---------------------------------------------------------------------------
# NodeCapabilities.can_run_model
# ---------------------------------------------------------------------------


def test_can_run_model_exact_match():
    caps = NodeCapabilities(node_id="n1", supported_models=["llama"])
    assert caps.can_run_model("llama") is True


def test_can_run_model_prefix_match():
    caps = NodeCapabilities(node_id="n1", supported_models=["llama"])
    assert caps.can_run_model("llama-3") is True


def test_can_run_model_missing():
    caps = NodeCapabilities(node_id="n1", supported_models=["llama"])
    assert caps.can_run_model("mistral") is False


# ---------------------------------------------------------------------------
# NodeCapabilities.supports_precision
# ---------------------------------------------------------------------------


def test_supports_precision_true():
    caps = NodeCapabilities(
        node_id="n1",
        supported_precisions=[PrecisionType.FP16, PrecisionType.INT8],
    )
    assert caps.supports_precision(PrecisionType.FP16) is True


def test_supports_precision_false():
    caps = NodeCapabilities(
        node_id="n1",
        supported_precisions=[PrecisionType.FP16, PrecisionType.INT8],
    )
    assert caps.supports_precision(PrecisionType.FP32) is False


# ---------------------------------------------------------------------------
# CapabilityMatcher helpers
# ---------------------------------------------------------------------------


def _make_capable_node(node_id: str, gpu_gb: float = 16.0, cpu_cores: int = 8) -> NodeCapabilities:
    return NodeCapabilities(
        node_id=node_id,
        gpu_memory_gb=gpu_gb,
        cpu_cores=cpu_cores,
        supported_precisions=[PrecisionType.FP16, PrecisionType.FP32],
        supported_models=["llama"],
        max_batch_size=4,
        max_sequence_length=4096,
    )


def _make_basic_req() -> JobRequirements:
    return JobRequirements(
        model_name="llama",
        min_gpu_memory_gb=8.0,
        required_precision=PrecisionType.FP16,
        min_sequence_length=1024,
        min_batch_size=2,
    )


# ---------------------------------------------------------------------------
# CapabilityMatcher.find_eligible
# ---------------------------------------------------------------------------


def test_matcher_find_eligible_all_match():
    matcher = CapabilityMatcher()
    node_a = _make_capable_node("a")
    node_b = _make_capable_node("b")
    matcher.register(node_a)
    matcher.register(node_b)

    eligible = matcher.find_eligible(_make_basic_req())
    assert len(eligible) == 2
    ids = {c.node_id for c in eligible}
    assert ids == {"a", "b"}


def test_matcher_find_eligible_none_match():
    matcher = CapabilityMatcher()
    matcher.register(_make_capable_node("a"))

    # Require 100 GB — no node can satisfy this
    req = JobRequirements(
        model_name="llama",
        min_gpu_memory_gb=100.0,
        required_precision=PrecisionType.FP16,
    )
    assert matcher.find_eligible(req) == []


def test_matcher_gpu_filter():
    matcher = CapabilityMatcher()
    weak = _make_capable_node("weak", gpu_gb=4.0)
    strong = _make_capable_node("strong", gpu_gb=24.0)
    matcher.register(weak)
    matcher.register(strong)

    req = JobRequirements(
        model_name="llama",
        min_gpu_memory_gb=16.0,
        required_precision=PrecisionType.FP16,
    )
    eligible = matcher.find_eligible(req)
    assert len(eligible) == 1
    assert eligible[0].node_id == "strong"


def test_matcher_precision_filter():
    matcher = CapabilityMatcher()
    fp16_only = NodeCapabilities(
        node_id="fp16",
        gpu_memory_gb=16.0,
        supported_precisions=[PrecisionType.FP16],
        supported_models=["llama"],
        max_batch_size=4,
        max_sequence_length=4096,
    )
    int4_only = NodeCapabilities(
        node_id="int4",
        gpu_memory_gb=16.0,
        supported_precisions=[PrecisionType.INT4],
        supported_models=["llama"],
        max_batch_size=4,
        max_sequence_length=4096,
    )
    matcher.register(fp16_only)
    matcher.register(int4_only)

    req = JobRequirements(
        model_name="llama",
        required_precision=PrecisionType.FP16,
    )
    eligible = matcher.find_eligible(req)
    assert len(eligible) == 1
    assert eligible[0].node_id == "fp16"


def test_matcher_sequence_length_filter():
    matcher = CapabilityMatcher()
    short_ctx = _make_capable_node("short")
    short_ctx.max_sequence_length = 512
    long_ctx = _make_capable_node("long")
    long_ctx.max_sequence_length = 8192
    matcher.register(short_ctx)
    matcher.register(long_ctx)

    req = JobRequirements(
        model_name="llama",
        required_precision=PrecisionType.FP16,
        min_sequence_length=4096,
    )
    eligible = matcher.find_eligible(req)
    assert len(eligible) == 1
    assert eligible[0].node_id == "long"


# ---------------------------------------------------------------------------
# CapabilityMatcher.best_node
# ---------------------------------------------------------------------------


def test_matcher_best_node():
    matcher = CapabilityMatcher()
    low_gpu = _make_capable_node("low", gpu_gb=8.0, cpu_cores=4)
    high_gpu = _make_capable_node("high", gpu_gb=24.0, cpu_cores=4)
    matcher.register(low_gpu)
    matcher.register(high_gpu)

    best = matcher.best_node(_make_basic_req())
    assert best is not None
    assert best.node_id == "high"


def test_matcher_best_node_none():
    matcher = CapabilityMatcher()
    # No nodes registered
    assert matcher.best_node(_make_basic_req()) is None


# ---------------------------------------------------------------------------
# CapabilityMatcher.deregister
# ---------------------------------------------------------------------------


def test_deregister():
    matcher = CapabilityMatcher()
    node = _make_capable_node("a")
    matcher.register(node)

    removed = matcher.deregister("a")
    assert removed is True
    assert matcher.find_eligible(_make_basic_req()) == []


@pytest.mark.parametrize("node_id", ["nonexistent", "also-not-here"])
def test_deregister_missing(node_id: str):
    matcher = CapabilityMatcher()
    assert matcher.deregister(node_id) is False
