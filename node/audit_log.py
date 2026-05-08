import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum


class AuditEventType(Enum):
    JOB_CLAIMED = "job.claimed"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    PAYMENT_RECEIVED = "payment.received"
    SLASHING_APPLIED = "slashing.applied"
    NODE_REGISTERED = "node.registered"
    NODE_DEREGISTERED = "node.deregistered"
    GOVERNANCE_VOTE = "governance.vote"
    CONFIG_CHANGED = "config.changed"


@dataclass
class AuditEntry:
    sequence: int
    event_type: AuditEventType
    actor: str  # pubkey or node_id
    payload: dict
    timestamp: float
    prev_hash: str  # SHA-256 of previous entry (empty string for first entry)
    entry_hash: str = field(default="")

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = json.dumps(
            {
                "sequence": self.sequence,
                "event_type": self.event_type.value,
                "actor": self.actor,
                "payload": self.payload,
                "timestamp": self.timestamp,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(data).hexdigest()

    def verify(self) -> bool:
        """Return True if entry_hash matches recomputed hash."""
        return self.entry_hash == self._compute_hash()


class AuditLog:
    """
    Append-only audit log with hash-chaining (like a mini blockchain).
    Each entry's hash includes the previous entry's hash, making tampering detectable.
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def append(self, event_type: AuditEventType, actor: str, payload: dict) -> AuditEntry:
        """Add a new entry. Returns the new AuditEntry."""
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry = AuditEntry(
            sequence=len(self._entries),
            event_type=event_type,
            actor=actor,
            payload=payload,
            timestamp=time.time(),
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire chain.
        Returns False if any entry's hash is wrong or chain links are broken.
        """
        for i, entry in enumerate(self._entries):
            if not entry.verify():
                return False
            expected_prev = self._entries[i - 1].entry_hash if i > 0 else ""
            if entry.prev_hash != expected_prev:
                return False
        return True

    def entries_by_type(self, event_type: AuditEventType) -> list[AuditEntry]:
        return [e for e in self._entries if e.event_type == event_type]

    def entries_by_actor(self, actor: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.actor == actor]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> AuditEntry:
        return self._entries[idx]
