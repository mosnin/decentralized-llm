from node.shard_balancer import ShardBalancer, ShardNode


def make_node(node_id: str, shard_index: int = 0, model_name: str = "llama", **kwargs) -> ShardNode:
    return ShardNode(node_id=node_id, shard_index=shard_index, model_name=model_name, **kwargs)


def test_register_and_select():
    balancer = ShardBalancer()
    node = make_node("n1")
    balancer.register(node)
    result = balancer.select("llama", 0)
    assert result is node


def test_select_returns_none_when_empty():
    balancer = ShardBalancer()
    result = balancer.select("llama", 0)
    assert result is None


def test_select_skips_full_nodes():
    balancer = ShardBalancer()
    node = make_node("n1", current_load=4, max_load=4)
    balancer.register(node)
    result = balancer.select("llama", 0)
    assert result is None


def test_select_prefers_less_loaded():
    balancer = ShardBalancer()
    # Identical nodes except load — same reputation and latency
    heavy = make_node("heavy", current_load=3, max_load=4, reputation=0.5, avg_latency_ms=500.0)
    light = make_node("light", current_load=0, max_load=4, reputation=0.5, avg_latency_ms=500.0)
    balancer.register(heavy)
    balancer.register(light)
    result = balancer.select("llama", 0)
    assert result is light


def test_select_pipeline_full():
    balancer = ShardBalancer()
    n0 = make_node("n0", shard_index=0)
    n1 = make_node("n1", shard_index=1)
    balancer.register(n0)
    balancer.register(n1)
    pipeline = balancer.select_pipeline("llama", 2)
    assert pipeline is not None
    assert len(pipeline) == 2
    assert pipeline[0] is n0
    assert pipeline[1] is n1


def test_select_pipeline_missing_shard():
    balancer = ShardBalancer()
    n0 = make_node("n0", shard_index=0)
    # shard 1 intentionally not registered
    balancer.register(n0)
    result = balancer.select_pipeline("llama", 2)
    assert result is None


def test_update_load_increments():
    balancer = ShardBalancer()
    node = make_node("n1", current_load=1)
    balancer.register(node)
    balancer.update_load("n1", 1)
    assert node.current_load == 2


def test_update_load_floored_at_zero():
    balancer = ShardBalancer()
    node = make_node("n1", current_load=0)
    balancer.register(node)
    balancer.update_load("n1", -10)
    assert node.current_load == 0


def test_update_latency_ema():
    balancer = ShardBalancer()
    node = make_node("n1", avg_latency_ms=500.0)
    balancer.register(node)
    balancer.update_latency("n1", 1000.0, alpha=0.2)
    expected = 0.8 * 500.0 + 0.2 * 1000.0
    assert abs(node.avg_latency_ms - expected) < 1e-9


def test_deregister_removes_node():
    balancer = ShardBalancer()
    node = make_node("n1")
    balancer.register(node)
    removed = balancer.deregister("n1")
    assert removed is True
    assert balancer.select("llama", 0) is None


def test_is_available_false_when_full():
    node = make_node("n1", current_load=4, max_load=4)
    assert node.is_available is False


def test_load_factor_correct():
    node = make_node("n1", current_load=2, max_load=4)
    assert node.load_factor == 0.5
