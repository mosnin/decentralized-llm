"""Tests for the P2P gossip protocol."""

from node.gossip import GossipMessage, GossipMessageType, GossipNode


def make_node(node_id: str, fanout: int = 3) -> GossipNode:
    return GossipNode(node_id=node_id, fanout=fanout)


def connect(a: GossipNode, b: GossipNode) -> None:
    """Bidirectionally connect two nodes."""
    a.add_peer(b)
    b.add_peer(a)


# ---------------------------------------------------------------------------
# Basic node management
# ---------------------------------------------------------------------------


def test_add_peer():
    a = make_node("A")
    b = make_node("B")
    a.add_peer(b)
    assert "B" in a._peers


def test_remove_peer():
    a = make_node("A")
    b = make_node("B")
    a.add_peer(b)
    a.remove_peer("B")
    assert "B" not in a._peers


# ---------------------------------------------------------------------------
# Message creation
# ---------------------------------------------------------------------------


def test_message_create_has_unique_id():
    msg1 = GossipMessage.create("node1", GossipMessageType.NODE_HEALTH, {})
    msg2 = GossipMessage.create("node1", GossipMessageType.NODE_HEALTH, {})
    assert msg1.msg_id != msg2.msg_id


# ---------------------------------------------------------------------------
# Broadcast & delivery
# ---------------------------------------------------------------------------


def test_broadcast_delivers_to_self():
    a = make_node("A")
    msg = a.broadcast(GossipMessageType.NODE_HEALTH, {"status": "ok"})
    assert a.inbox_count() == 1
    assert a.inbox[0].msg_id == msg.msg_id


def test_message_propagates_to_connected():
    a = make_node("A")
    b = make_node("B")
    connect(a, b)
    a.broadcast(GossipMessageType.NODE_HEALTH, {"status": "ok"})
    assert b.inbox_count() == 1


def test_message_propagates_transitively():
    """A → B → C (chain). A broadcasts; C must receive the message."""
    a = make_node("A", fanout=10)
    b = make_node("B", fanout=10)
    c = make_node("C", fanout=10)
    # Only connect A-B and B-C (not A-C)
    a.add_peer(b)
    b.add_peer(a)
    b.add_peer(c)
    c.add_peer(b)
    a.broadcast(GossipMessageType.JOB_ANNOUNCEMENT, {"job_id": "j1"})
    assert c.inbox_count() == 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_not_redelivered():
    a = make_node("A")
    b = make_node("B")
    connect(a, b)
    msg = GossipMessage.create("A", GossipMessageType.NODE_HEALTH, {})
    b.receive(msg)
    b.receive(msg)
    assert b.inbox_count() == 1


# ---------------------------------------------------------------------------
# TTL behaviour
# ---------------------------------------------------------------------------


def test_ttl_zero_not_forwarded():
    a = make_node("A")
    b = make_node("B")
    connect(a, b)
    msg = GossipMessage.create("A", GossipMessageType.NODE_HEALTH, {}, ttl=0)
    accepted = b.receive(msg)
    assert not accepted
    assert b.inbox_count() == 0


def test_ttl_decremented_on_forward():
    """
    A-B-C, ttl=2.
    A sends to B (B sees ttl=2, forwards with ttl=1).
    B forwards to C (C sees ttl=1, which is > 0, so it is accepted).
    """
    a = make_node("A", fanout=10)
    b = make_node("B", fanout=10)
    c = make_node("C", fanout=10)
    a.add_peer(b)
    b.add_peer(a)
    b.add_peer(c)
    c.add_peer(b)

    msg = GossipMessage.create("A", GossipMessageType.NODE_HEALTH, {}, ttl=2)
    b.receive(msg)  # B accepts (ttl=2 > 0), forwards with ttl=1
    # C should have received the forwarded message with ttl=1
    assert c.inbox_count() == 1


# ---------------------------------------------------------------------------
# Seen-message tracking
# ---------------------------------------------------------------------------


def test_seen_count_increments():
    a = make_node("A")
    msg = GossipMessage.create("X", GossipMessageType.NODE_HEALTH, {})
    a.receive(msg)
    assert a.seen_count() > 0


# ---------------------------------------------------------------------------
# Full-mesh propagation
# ---------------------------------------------------------------------------


def test_full_mesh_propagation():
    """5 nodes in a full mesh; one broadcasts; all 5 receive the message."""
    nodes = [make_node(f"N{i}", fanout=10) for i in range(5)]
    # Connect every node to every other node
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if i != j:
                node.add_peer(other)

    nodes[0].broadcast(GossipMessageType.PEER_ANNOUNCEMENT, {"addr": "10.0.0.1:9000"})

    for node in nodes:
        assert node.inbox_count() == 1, f"Node {node.node_id} did not receive the message"
