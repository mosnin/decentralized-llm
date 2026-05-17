import time
from dataclasses import dataclass, field
from enum import Enum


class PeerStatus(Enum):
    ACTIVE = "active"
    SUSPECTED = "suspected"  # missed recent heartbeat
    DEAD = "dead"  # missed too many heartbeats


@dataclass
class PeerInfo:
    peer_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    missed_heartbeats: int = 0
    supported_models: list[str] = field(default_factory=list)
    stake_lamports: int = 0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class PeerManager:
    """
    Tracks known peers, their health, and supported models.

    Heartbeat-based liveness: peers that miss heartbeats transition
    ACTIVE → SUSPECTED → DEAD.
    """

    SUSPECT_AFTER_MISSED = 2  # missed heartbeats before SUSPECTED
    DEAD_AFTER_MISSED = 5  # missed heartbeats before DEAD

    def __init__(self):
        self._peers: dict[str, PeerInfo] = {}

    def add_peer(
        self,
        peer_id: str,
        host: str,
        port: int,
        supported_models: list[str] | None = None,
        stake_lamports: int = 0,
    ) -> PeerInfo:
        """Register or update a peer. Resets missed_heartbeats on re-add."""
        info = PeerInfo(
            peer_id=peer_id,
            host=host,
            port=port,
            last_seen=time.time(),
            missed_heartbeats=0,
            supported_models=supported_models or [],
            stake_lamports=stake_lamports,
        )
        self._peers[peer_id] = info
        return info

    def heartbeat(self, peer_id: str) -> bool:
        """Record a heartbeat. Returns False if peer is unknown."""
        peer = self._peers.get(peer_id)
        if peer is None:
            return False
        peer.last_seen = time.time()
        peer.missed_heartbeats = 0
        return True

    def tick(self) -> list[str]:
        """
        Called periodically. Increments missed_heartbeats for all peers.
        Returns list of peer_ids that just became DEAD.
        """
        newly_dead = []
        for peer in self._peers.values():
            peer.missed_heartbeats += 1
            if peer.missed_heartbeats == self.DEAD_AFTER_MISSED:
                newly_dead.append(peer.peer_id)
        return newly_dead

    def status(self, peer_id: str) -> PeerStatus | None:
        """Return peer status, or None if unknown."""
        peer = self._peers.get(peer_id)
        if peer is None:
            return None
        if peer.missed_heartbeats >= self.DEAD_AFTER_MISSED:
            return PeerStatus.DEAD
        if peer.missed_heartbeats >= self.SUSPECT_AFTER_MISSED:
            return PeerStatus.SUSPECTED
        return PeerStatus.ACTIVE

    def active_peers(self) -> list[PeerInfo]:
        """Return all peers with ACTIVE status."""
        return [p for p in self._peers.values() if self.status(p.peer_id) == PeerStatus.ACTIVE]

    def peers_for_model(self, model_name: str) -> list[PeerInfo]:
        """Return ACTIVE peers that support the given model."""
        return [p for p in self.active_peers() if model_name in p.supported_models]

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer. Returns True if it existed."""
        return self._peers.pop(peer_id, None) is not None

    def peer_count(self) -> int:
        return len(self._peers)

    def get(self, peer_id: str) -> PeerInfo | None:
        return self._peers.get(peer_id)
