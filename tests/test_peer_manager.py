import pytest

from node.peer_manager import PeerInfo, PeerManager, PeerStatus


@pytest.fixture
def manager():
    return PeerManager()


def test_add_peer_and_get(manager):
    info = manager.add_peer("p1", "10.0.0.1", 8080, supported_models=["gpt2"], stake_lamports=100)
    assert isinstance(info, PeerInfo)
    assert info.peer_id == "p1"
    assert info.host == "10.0.0.1"
    assert info.port == 8080
    assert info.supported_models == ["gpt2"]
    assert info.stake_lamports == 100
    assert info.missed_heartbeats == 0

    fetched = manager.get("p1")
    assert fetched is info


def test_heartbeat_resets_missed(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    manager.tick()  # missed_heartbeats = 1
    result = manager.heartbeat("p1")
    assert result is True
    assert manager.get("p1").missed_heartbeats == 0
    manager.tick()  # missed_heartbeats = 1
    assert manager.get("p1").missed_heartbeats == 1


def test_tick_increments_missed(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    manager.tick()
    manager.tick()
    manager.tick()
    assert manager.get("p1").missed_heartbeats == 3


def test_status_active(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    assert manager.status("p1") == PeerStatus.ACTIVE


def test_status_suspected(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    for _ in range(PeerManager.SUSPECT_AFTER_MISSED):
        manager.tick()
    assert manager.status("p1") == PeerStatus.SUSPECTED


def test_status_dead(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    for _ in range(PeerManager.DEAD_AFTER_MISSED):
        manager.tick()
    assert manager.status("p1") == PeerStatus.DEAD


def test_tick_returns_newly_dead(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    newly_dead = []
    for _ in range(PeerManager.DEAD_AFTER_MISSED):
        newly_dead = manager.tick()
    assert "p1" in newly_dead


def test_active_peers_filters_dead(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    manager.add_peer("p2", "10.0.0.2", 8080)
    for _ in range(PeerManager.DEAD_AFTER_MISSED):
        manager.tick()
        # keep p2 alive
        manager.heartbeat("p2")
    active_ids = [p.peer_id for p in manager.active_peers()]
    assert "p1" not in active_ids
    assert "p2" in active_ids


def test_peers_for_model(manager):
    manager.add_peer("p1", "10.0.0.1", 8080, supported_models=["llama2"])
    manager.add_peer("p2", "10.0.0.2", 8080, supported_models=["gpt2"])
    manager.add_peer("p3", "10.0.0.3", 8080, supported_models=["llama2"])
    # Kill p3
    for _ in range(PeerManager.DEAD_AFTER_MISSED):
        manager.tick()
        manager.heartbeat("p1")
        manager.heartbeat("p2")
    result = manager.peers_for_model("llama2")
    peer_ids = [p.peer_id for p in result]
    assert "p1" in peer_ids
    assert "p3" not in peer_ids
    assert "p2" not in peer_ids


def test_remove_peer(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    assert manager.remove_peer("p1") is True
    assert manager.get("p1") is None


def test_remove_missing_peer(manager):
    assert manager.remove_peer("nonexistent") is False


def test_peer_address(manager):
    info = manager.add_peer("p1", "192.168.1.5", 9000)
    assert info.address == "192.168.1.5:9000"


def test_add_peer_resets_on_readd(manager):
    manager.add_peer("p1", "10.0.0.1", 8080)
    manager.tick()
    manager.tick()
    assert manager.get("p1").missed_heartbeats == 2
    manager.add_peer("p1", "10.0.0.1", 8080)
    assert manager.get("p1").missed_heartbeats == 0
