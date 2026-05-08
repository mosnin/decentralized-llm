import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum


class GossipMessageType(Enum):
    PEER_ANNOUNCEMENT = "peer.announcement"
    JOB_ANNOUNCEMENT = "job.announcement"
    NODE_HEALTH = "node.health"
    MODEL_AVAILABILITY = "model.availability"


@dataclass
class GossipMessage:
    msg_id: str  # SHA-256 hex of (sender_id + type.value + timestamp)
    sender_id: str
    msg_type: GossipMessageType
    payload: dict
    timestamp: float
    ttl: int = 5  # hops remaining before message is dropped

    @classmethod
    def create(
        cls,
        sender_id: str,
        msg_type: GossipMessageType,
        payload: dict,
        ttl: int = 5,
    ) -> "GossipMessage":
        ts = time.time()
        raw = f"{sender_id}{msg_type.value}{ts}".encode()
        msg_id = hashlib.sha256(raw).hexdigest()
        return cls(
            msg_id=msg_id,
            sender_id=sender_id,
            msg_type=msg_type,
            payload=payload,
            timestamp=ts,
            ttl=ttl,
        )


@dataclass
class GossipNode:
    """
    A gossip node that maintains a set of seen messages (dedup) and
    a fanout of peers to forward messages to.

    Protocol:
    - receive(): process an incoming message. If not seen and TTL > 0,
      forward to `fanout` randomly-selected peers.
    - broadcast(): create and immediately process a new message.
    - Seen messages are pruned after seen_ttl_seconds.
    """

    node_id: str
    fanout: int = 3
    seen_ttl_seconds: float = 300.0
    _peers: dict = field(default_factory=dict, init=False, repr=False)
    _seen: dict = field(default_factory=dict, init=False, repr=False)
    _inbox: list = field(default_factory=list, init=False, repr=False)

    def add_peer(self, peer: "GossipNode") -> None:
        if peer.node_id != self.node_id:
            self._peers[peer.node_id] = peer

    def remove_peer(self, peer_id: str) -> None:
        self._peers.pop(peer_id, None)

    def receive(self, msg: GossipMessage) -> bool:
        """
        Process a received message.
        Returns True if message was accepted (not a duplicate and TTL > 0).
        Forwards to fanout random peers (excluding sender) if accepted.
        """
        self._prune_seen()

        if msg.msg_id in self._seen:
            return False  # duplicate
        if msg.ttl <= 0:
            return False  # expired

        self._seen[msg.msg_id] = time.time()
        self._inbox.append(msg)

        # Forward to fanout random peers, excluding the original sender
        candidates = [p for pid, p in self._peers.items() if pid != msg.sender_id]
        forward_to = random.sample(candidates, min(self.fanout, len(candidates)))

        # Create a copy with decremented TTL
        forwarded = GossipMessage(
            msg_id=msg.msg_id,
            sender_id=self.node_id,  # we are now the forwarder
            msg_type=msg.msg_type,
            payload=msg.payload,
            timestamp=msg.timestamp,
            ttl=msg.ttl - 1,
        )
        for peer in forward_to:
            peer.receive(forwarded)

        return True

    def broadcast(self, msg_type: GossipMessageType, payload: dict, ttl: int = 5) -> GossipMessage:
        """Create and broadcast a new message from this node."""
        msg = GossipMessage.create(
            sender_id=self.node_id, msg_type=msg_type, payload=payload, ttl=ttl
        )
        self.receive(msg)
        return msg

    def _prune_seen(self) -> None:
        cutoff = time.time() - self.seen_ttl_seconds
        expired = [mid for mid, ts in self._seen.items() if ts < cutoff]
        for mid in expired:
            del self._seen[mid]

    @property
    def inbox(self) -> list[GossipMessage]:
        return list(self._inbox)

    def inbox_count(self) -> int:
        return len(self._inbox)

    def seen_count(self) -> int:
        return len(self._seen)
