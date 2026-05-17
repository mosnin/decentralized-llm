from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node.verification_pipeline import InferenceProof


class SettlementStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class PaymentRecord:
    job_id: int
    amount_lamports: int
    client_pubkey: str
    earned_at: float
    proof: InferenceProof | None = field(default=None, repr=False)
    """Optional verification proof attached at record time."""


@dataclass
class SettlementBatch:
    batch_id: str
    records: list[PaymentRecord]
    created_at: float
    status: SettlementStatus = SettlementStatus.PENDING
    tx_signature: str = ""

    @property
    def total_lamports(self) -> int:
        return sum(r.amount_lamports for r in self.records)

    @property
    def job_count(self) -> int:
        return len(self.records)

    def proofs(self) -> list[InferenceProof]:
        """Return all non-None proofs attached to records in this batch."""
        return [r.proof for r in self.records if r.proof is not None]


class SettlementTracker:
    """
    Accumulates payment records and batches them for on-chain settlement.

    Flow: record() → pending_batch → flush() → SettlementBatch(PENDING) → confirm()/fail()
    """

    def __init__(self, min_batch_lamports: int = 10_000, max_batch_size: int = 50):
        self._min_batch_lamports = min_batch_lamports
        self._max_batch_size = max_batch_size
        self._pending: list[PaymentRecord] = []
        self._batches: dict[str, SettlementBatch] = {}
        self._batch_counter = 0

    def record(
        self,
        job_id: int,
        amount_lamports: int,
        client_pubkey: str,
        proof: InferenceProof | None = None,
    ) -> None:
        """Add a payment record to the pending queue.

        Parameters
        ----------
        job_id:
            On-chain job identifier.
        amount_lamports:
            Payment amount in lamports.
        client_pubkey:
            Solana public key of the paying client.
        proof:
            Optional ``InferenceProof`` produced during job execution.  When
            supplied it is stored alongside the payment record so that auditors
            can retrieve proofs from a settled batch.
        """
        self._pending.append(
            PaymentRecord(
                job_id=job_id,
                amount_lamports=amount_lamports,
                client_pubkey=client_pubkey,
                earned_at=time.time(),
                proof=proof,
            )
        )

    def should_flush(self) -> bool:
        """True if pending records meet min_batch_lamports or max_batch_size."""
        if len(self._pending) >= self._max_batch_size:
            return True
        total = sum(r.amount_lamports for r in self._pending)
        return total >= self._min_batch_lamports

    def flush(self) -> SettlementBatch | None:
        """Create a batch from pending records and clear pending. Returns None if empty."""
        if not self._pending:
            return None
        self._batch_counter += 1
        batch_id = f"batch-{self._batch_counter:06d}"
        batch = SettlementBatch(
            batch_id=batch_id,
            records=list(self._pending),
            created_at=time.time(),
        )
        self._batches[batch_id] = batch
        self._pending.clear()
        return batch

    def confirm(self, batch_id: str, tx_signature: str) -> bool:
        """Mark a batch as confirmed. Returns False if batch not found."""
        batch = self._batches.get(batch_id)
        if batch is None:
            return False
        batch.status = SettlementStatus.CONFIRMED
        batch.tx_signature = tx_signature
        return True

    def fail(self, batch_id: str) -> bool:
        """Mark a batch as failed. Returns False if batch not found."""
        batch = self._batches.get(batch_id)
        if batch is None:
            return False
        batch.status = SettlementStatus.FAILED
        return True

    def pending_lamports(self) -> int:
        return sum(r.amount_lamports for r in self._pending)

    def pending_count(self) -> int:
        return len(self._pending)

    def get_batch(self, batch_id: str) -> SettlementBatch | None:
        return self._batches.get(batch_id)

    def batches_by_status(self, status: SettlementStatus) -> list[SettlementBatch]:
        return [b for b in self._batches.values() if b.status == status]

    def total_confirmed_lamports(self) -> int:
        return sum(b.total_lamports for b in self.batches_by_status(SettlementStatus.CONFIRMED))
