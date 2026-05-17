"""
End-to-end devnet tests for the Solana Anchor programs.

All tests are pure Python mocks — no network access, no wallet file, no Solana
packages required. Runnable with:

    pytest tests/test_e2e_devnet.py -v

Programs under test (program IDs from Anchor.toml):
  - inference-market  : 5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW
  - compute-registry  : 8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M
  - governance        : 3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T
"""

from __future__ import annotations

import hashlib
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out Solana / anchor packages so node/blockchain.py can be imported
# ---------------------------------------------------------------------------


def _install_solana_stubs() -> None:
    def _stub(name: str) -> None:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    for mod in (
        "solana",
        "solana.rpc",
        "solana.rpc.async_api",
        "solders",
        "solders.keypair",
        "solders.pubkey",
        "anchorpy",
    ):
        _stub(mod)

    # Provide minimal attribute surfaces used by blockchain.py
    solders_kp = sys.modules["solders.keypair"]
    if not hasattr(solders_kp, "Keypair"):
        solders_kp.Keypair = MagicMock()

    solders_pk = sys.modules["solders.pubkey"]
    if not hasattr(solders_pk, "Pubkey"):
        solders_pk.Pubkey = MagicMock()

    anchorpy = sys.modules["anchorpy"]
    for attr in ("Program", "Provider", "Wallet"):
        if not hasattr(anchorpy, attr):
            setattr(anchorpy, attr, MagicMock())

    rpc_api = sys.modules["solana.rpc.async_api"]
    if not hasattr(rpc_api, "AsyncClient"):
        rpc_api.AsyncClient = MagicMock()


_install_solana_stubs()

# ---------------------------------------------------------------------------
# Constants matching Anchor.toml / programs
# ---------------------------------------------------------------------------

INFERENCE_MARKET_PROGRAM_ID = "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
COMPUTE_REGISTRY_PROGRAM_ID = "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"
GOVERNANCE_PROGRAM_ID = "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"
DEVNET_RPC_URL = "https://api.devnet.solana.com"
LAMPORTS_PER_SOL = 1_000_000_000
PROTOCOL_FEE_BPS = 500  # 5% — mirrors lib.rs

# ---------------------------------------------------------------------------
# MockSolanaClient
# ---------------------------------------------------------------------------


@dataclass
class _AccountInfo:
    """Represents a mock on-chain account."""

    owner: str
    data: bytes = b""
    lamports: int = 0
    executable: bool = False


class MockSolanaClient:
    """
    Simulates the Solana JSON-RPC API (async-style, but synchronous internals).

    Designed to be injected wherever node/blockchain.py uses AsyncClient.
    """

    def __init__(self, default_balance_lamports: int = 10 * LAMPORTS_PER_SOL) -> None:
        self._balances: dict[str, int] = {}
        self._accounts: dict[str, _AccountInfo] = {}
        self._transactions: list[dict[str, Any]] = []
        self._confirmed_sigs: set[str] = set()
        self._program_accounts: dict[str, list[dict[str, Any]]] = {}
        self._default_balance = default_balance_lamports

    # --- balance helpers ---------------------------------------------------

    def set_balance(self, pubkey: str, lamports: int) -> None:
        self._balances[pubkey] = lamports

    def add_lamports(self, pubkey: str, amount: int) -> None:
        self._balances[pubkey] = self._balances.get(pubkey, self._default_balance) + amount

    # --- RPC surface -------------------------------------------------------

    async def get_balance(self, pubkey: Any) -> MagicMock:
        key = str(pubkey)
        balance = self._balances.get(key, self._default_balance)
        resp = MagicMock()
        resp.value = balance
        return resp

    async def get_account_info(self, pubkey: Any) -> MagicMock:
        key = str(pubkey)
        account = self._accounts.get(key)
        resp = MagicMock()
        if account is not None:
            resp.value = MagicMock(
                owner=account.owner,
                data=account.data,
                lamports=account.lamports,
                executable=account.executable,
            )
        else:
            resp.value = None
        return resp

    async def send_transaction(self, tx: Any, *signers: Any, **kwargs: Any) -> MagicMock:
        sig = "sig_" + uuid.uuid4().hex
        record = {"sig": sig, "tx": tx, "signers": signers, "kwargs": kwargs}
        self._transactions.append(record)
        resp = MagicMock()
        resp.value = sig
        return resp

    async def confirm_transaction(self, sig: str, commitment: str = "confirmed") -> MagicMock:
        # Simulate brief confirmation — in tests this is instant
        self._confirmed_sigs.add(sig)
        resp = MagicMock()
        resp.value = MagicMock(err=None)
        return resp

    async def get_program_accounts(
        self, program_id: Any, filters: list | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        key = str(program_id)
        return self._program_accounts.get(key, [])

    async def request_airdrop(self, pubkey: Any, lamports: int) -> MagicMock:
        key = str(pubkey)
        self.add_lamports(key, lamports)
        sig = "airdrop_sig_" + uuid.uuid4().hex
        resp = MagicMock()
        resp.value = sig
        return resp

    async def close(self) -> None:
        pass

    # --- inspection helpers ------------------------------------------------

    @property
    def sent_transactions(self) -> list[dict[str, Any]]:
        return list(self._transactions)

    def register_program_account(self, program_id: str, account: dict[str, Any]) -> None:
        self._program_accounts.setdefault(program_id, []).append(account)


# ---------------------------------------------------------------------------
# MockInferenceMarket
# ---------------------------------------------------------------------------


class JobStatus(Enum):
    OPEN = auto()
    CLAIMED = auto()
    SETTLED = auto()
    DISPUTED = auto()
    REFUNDED = auto()


class InferenceMarketError(Exception):
    """Raised when an on-chain constraint is violated."""


@dataclass
class _Job:
    job_id: int
    client: str
    prompt_hash: bytes
    bounty_lamports: int
    status: JobStatus = JobStatus.OPEN
    node: str | None = None
    result_hash: bytes | None = None
    claimed_at: float = 0.0
    escrow_lamports: int = 0  # tracks locked funds


class MockInferenceMarket:
    """
    In-memory simulation of the inference-market Anchor program.

    Implements the same state machine as programs/inference-market/src/lib.rs.
    """

    PROTOCOL_FEE_BPS: int = PROTOCOL_FEE_BPS

    def __init__(self) -> None:
        self._jobs: dict[int, _Job] = {}
        self._next_id: int = 0
        self._escrow: dict[int, int] = {}  # job_id -> locked lamports
        self._node_balances: dict[str, int] = {}
        self._treasury_balance: int = 0

    # --- instruction handlers ----------------------------------------------

    def post_job(
        self,
        client_pubkey: str,
        prompt_hash: bytes,
        bounty: int,
    ) -> int:
        if bounty <= 0:
            raise InferenceMarketError("ZeroPayment: bounty must be positive")

        job_id = self._next_id
        self._next_id += 1
        job = _Job(
            job_id=job_id,
            client=client_pubkey,
            prompt_hash=prompt_hash,
            bounty_lamports=bounty,
            status=JobStatus.OPEN,
            escrow_lamports=bounty,
        )
        self._jobs[job_id] = job
        self._escrow[job_id] = bounty  # lock in escrow
        return job_id

    def claim_job(self, node_pubkey: str, job_id: int) -> None:
        job = self._get_job(job_id)
        if job.status != JobStatus.OPEN:
            raise InferenceMarketError(f"JobNotOpen: job {job_id} is {job.status.name}")
        job.status = JobStatus.CLAIMED
        job.node = node_pubkey
        job.claimed_at = time.time()

    def settle_job(self, node_pubkey: str, job_id: int, result_hash: bytes) -> int:
        """
        Transitions CLAIMED → SETTLED and releases escrow to the node.

        Returns the lamports credited to node_pubkey (after protocol fee).
        """
        job = self._get_job(job_id)
        if job.status != JobStatus.CLAIMED:
            raise InferenceMarketError(f"JobNotInProgress: job {job_id} is {job.status.name}")
        if job.node != node_pubkey:
            raise InferenceMarketError(f"NotJobNode: expected {job.node!r}, got {node_pubkey!r}")

        locked = self._escrow.get(job_id, 0)
        protocol_fee = (locked * self.PROTOCOL_FEE_BPS) // 10_000
        node_payment = locked - protocol_fee

        self._treasury_balance += protocol_fee
        self._node_balances[node_pubkey] = self._node_balances.get(node_pubkey, 0) + node_payment
        self._escrow[job_id] = 0

        job.status = JobStatus.SETTLED
        job.result_hash = result_hash
        job.escrow_lamports = 0
        return node_payment

    def dispute_job(self, client_pubkey: str, job_id: int) -> None:
        """
        Mark a CLAIMED job as DISPUTED, freezing the escrow.

        Corresponds to dispute_result in lib.rs (called while status is
        PendingAcceptance/CLAIMED in our simplified model).
        """
        job = self._get_job(job_id)
        if job.status not in (JobStatus.CLAIMED, JobStatus.OPEN):
            raise InferenceMarketError(f"ResultNotPending: job {job_id} is {job.status.name}")
        if job.client != client_pubkey:
            raise InferenceMarketError(
                f"NotJobClient: expected {job.client!r}, got {client_pubkey!r}"
            )
        job.status = JobStatus.DISPUTED

    # --- query helpers -----------------------------------------------------

    def get_job(self, job_id: int) -> _Job:
        return self._get_job(job_id)

    def escrow_balance(self, job_id: int) -> int:
        return self._escrow.get(job_id, 0)

    def node_balance(self, node_pubkey: str) -> int:
        return self._node_balances.get(node_pubkey, 0)

    @property
    def treasury_balance(self) -> int:
        return self._treasury_balance

    def open_jobs(self) -> list[_Job]:
        return [j for j in self._jobs.values() if j.status == JobStatus.OPEN]

    # --- private -----------------------------------------------------------

    def _get_job(self, job_id: int) -> _Job:
        if job_id not in self._jobs:
            raise InferenceMarketError(f"JobNotFound: {job_id}")
        return self._jobs[job_id]


# ---------------------------------------------------------------------------
# MockComputeRegistry
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    pass


@dataclass
class _NodeEntry:
    operator: str
    endpoint: str
    vram_gb: int
    gpu_count: int
    model_ids: list[bytes]
    stake_lamports: int
    reputation: int = 0
    active: bool = True


class MockComputeRegistry:
    """In-memory simulation of the compute-registry Anchor program."""

    SLASH_BPS: int = 1000  # 10% slash for Byzantine behaviour

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeEntry] = {}
        self._slashed_stake: int = 0

    def register_node(
        self,
        operator: str,
        endpoint: str,
        vram_gb: int,
        gpu_count: int,
        model_ids: list[bytes],
        stake_lamports: int,
    ) -> None:
        if stake_lamports <= 0:
            raise RegistryError("StakeTooLow: must stake > 0 lamports")
        self._nodes[operator] = _NodeEntry(
            operator=operator,
            endpoint=endpoint,
            vram_gb=vram_gb,
            gpu_count=gpu_count,
            model_ids=list(model_ids),
            stake_lamports=stake_lamports,
        )

    def deregister_node(self, operator: str) -> int:
        """Returns the stake released."""
        entry = self._get_node(operator)
        stake = entry.stake_lamports
        del self._nodes[operator]
        return stake

    def slash_node(self, operator: str, reason: str = "") -> int:
        """Deduct SLASH_BPS from stake. Returns lamports slashed."""
        entry = self._get_node(operator)
        slash = (entry.stake_lamports * self.SLASH_BPS) // 10_000
        entry.stake_lamports -= slash
        self._slashed_stake += slash
        return slash

    def update_reputation(self, operator: str, delta: int) -> int:
        """Adjust reputation score. Returns new score."""
        entry = self._get_node(operator)
        entry.reputation = max(0, entry.reputation + delta)
        return entry.reputation

    def get_node(self, operator: str) -> _NodeEntry:
        return self._get_node(operator)

    def _get_node(self, operator: str) -> _NodeEntry:
        if operator not in self._nodes:
            raise RegistryError(f"NodeNotFound: {operator}")
        return self._nodes[operator]


# ---------------------------------------------------------------------------
# MockGovernance
# ---------------------------------------------------------------------------


class ProposalStatus(Enum):
    ACTIVE = auto()
    PASSED = auto()
    FAILED = auto()
    EXECUTED = auto()
    TIMELOCKED = auto()


class GovernanceError(Exception):
    pass


@dataclass
class _Proposal:
    proposal_id: int
    proposer: str
    title: str
    description: str
    created_at: float
    votes_yes: int = 0
    votes_no: int = 0
    status: ProposalStatus = ProposalStatus.ACTIVE
    voters: set[str] = field(default_factory=set)
    timelock_until: float = 0.0
    executed_at: float = 0.0


class MockGovernance:
    """In-memory simulation of the governance Anchor program."""

    QUORUM_VOTES: int = 3
    TIMELOCK_SECONDS: float = 2.0  # short for tests

    def __init__(self) -> None:
        self._proposals: dict[int, _Proposal] = {}
        self._next_id: int = 0

    def create_proposal(self, proposer: str, title: str, description: str) -> int:
        pid = self._next_id
        self._next_id += 1
        self._proposals[pid] = _Proposal(
            proposal_id=pid,
            proposer=proposer,
            title=title,
            description=description,
            created_at=time.time(),
        )
        return pid

    def vote_yes(self, voter: str, proposal_id: int) -> None:
        proposal = self._get_proposal(proposal_id)
        if proposal.status != ProposalStatus.ACTIVE:
            raise GovernanceError(f"ProposalNotActive: {proposal.status.name}")
        if voter in proposal.voters:
            raise GovernanceError(f"AlreadyVoted: {voter}")
        proposal.voters.add(voter)
        proposal.votes_yes += 1
        self._maybe_pass(proposal)

    def vote_no(self, voter: str, proposal_id: int) -> None:
        proposal = self._get_proposal(proposal_id)
        if proposal.status != ProposalStatus.ACTIVE:
            raise GovernanceError(f"ProposalNotActive: {proposal.status.name}")
        if voter in proposal.voters:
            raise GovernanceError(f"AlreadyVoted: {voter}")
        proposal.voters.add(voter)
        proposal.votes_no += 1

    def check_quorum(self, proposal_id: int) -> bool:
        proposal = self._get_proposal(proposal_id)
        return (proposal.votes_yes + proposal.votes_no) >= self.QUORUM_VOTES

    def execute_proposal(self, executor: str, proposal_id: int) -> None:
        proposal = self._get_proposal(proposal_id)
        if proposal.status == ProposalStatus.TIMELOCKED:
            if time.time() < proposal.timelock_until:
                raise GovernanceError("TimelockNotExpired")
            proposal.status = ProposalStatus.EXECUTED
            proposal.executed_at = time.time()
            return
        if proposal.status != ProposalStatus.PASSED:
            raise GovernanceError(f"CannotExecute: proposal is {proposal.status.name}")
        # Start timelock
        proposal.status = ProposalStatus.TIMELOCKED
        proposal.timelock_until = time.time() + self.TIMELOCK_SECONDS

    def get_proposal(self, proposal_id: int) -> _Proposal:
        return self._get_proposal(proposal_id)

    # --- private -----------------------------------------------------------

    def _get_proposal(self, proposal_id: int) -> _Proposal:
        if proposal_id not in self._proposals:
            raise GovernanceError(f"ProposalNotFound: {proposal_id}")
        return self._proposals[proposal_id]

    def _maybe_pass(self, proposal: _Proposal) -> None:
        if proposal.votes_yes >= self.QUORUM_VOTES and proposal.votes_yes > proposal.votes_no:
            proposal.status = ProposalStatus.PASSED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLIENT_PUBKEY = "ClientPubkey1111111111111111111111111111111"
NODE_PUBKEY = "NodePubkey111111111111111111111111111111111"
OTHER_NODE_PUBKEY = "OtherNodePubkey11111111111111111111111111"
BOUNTY = 1_000_000  # 0.001 SOL in lamports
PROMPT = b"What is the capital of France?"
PROMPT_HASH = hashlib.sha256(PROMPT).digest()
RESULT = b"Paris."
RESULT_HASH = hashlib.sha256(RESULT).digest()


@pytest.fixture
def solana_client() -> MockSolanaClient:
    return MockSolanaClient(default_balance_lamports=10 * LAMPORTS_PER_SOL)


@pytest.fixture
def market() -> MockInferenceMarket:
    return MockInferenceMarket()


@pytest.fixture
def registry() -> MockComputeRegistry:
    return MockComputeRegistry()


@pytest.fixture
def governance() -> MockGovernance:
    return MockGovernance()


# ---------------------------------------------------------------------------
# TestE2EJobLifecycle
# ---------------------------------------------------------------------------


class TestE2EJobLifecycle:
    """Tests for the inference-market job state machine."""

    def test_post_job_creates_escrow(self, market: MockInferenceMarket) -> None:
        """Bounty must be locked in escrow immediately after posting."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        assert market.escrow_balance(job_id) == BOUNTY
        job = market.get_job(job_id)
        assert job.status == JobStatus.OPEN
        assert job.client == CLIENT_PUBKEY
        assert job.bounty_lamports == BOUNTY

    def test_node_claims_job(self, market: MockInferenceMarket) -> None:
        """Claiming an OPEN job transitions it to CLAIMED and records the node."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)
        job = market.get_job(job_id)
        assert job.status == JobStatus.CLAIMED
        assert job.node == NODE_PUBKEY
        assert job.claimed_at > 0

    def test_settlement_releases_escrow(self, market: MockInferenceMarket) -> None:
        """After settling, escrow is zero and node receives payment minus fee."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)
        node_payment = market.settle_job(NODE_PUBKEY, job_id, RESULT_HASH)

        expected_fee = (BOUNTY * PROTOCOL_FEE_BPS) // 10_000
        expected_payment = BOUNTY - expected_fee

        assert node_payment == expected_payment
        assert market.escrow_balance(job_id) == 0
        assert market.node_balance(NODE_PUBKEY) == expected_payment
        assert market.treasury_balance == expected_fee
        assert market.get_job(job_id).status == JobStatus.SETTLED

    def test_dispute_freezes_escrow(self, market: MockInferenceMarket) -> None:
        """Disputing a CLAIMED job marks it DISPUTED; escrow remains locked."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)
        market.dispute_job(CLIENT_PUBKEY, job_id)

        job = market.get_job(job_id)
        assert job.status == JobStatus.DISPUTED
        # Escrow is NOT released — still locked
        assert market.escrow_balance(job_id) == BOUNTY

    def test_full_happy_path(self, market: MockInferenceMarket) -> None:
        """Complete flow: post → claim → settle."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        assert market.get_job(job_id).status == JobStatus.OPEN

        market.claim_job(NODE_PUBKEY, job_id)
        assert market.get_job(job_id).status == JobStatus.CLAIMED

        market.settle_job(NODE_PUBKEY, job_id, RESULT_HASH)
        assert market.get_job(job_id).status == JobStatus.SETTLED
        assert market.node_balance(NODE_PUBKEY) > 0

    def test_double_claim_rejected(self, market: MockInferenceMarket) -> None:
        """A second claim attempt on an already-CLAIMED job must raise."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)

        with pytest.raises(InferenceMarketError, match="JobNotOpen"):
            market.claim_job(OTHER_NODE_PUBKEY, job_id)

        # Original assignment is unchanged
        assert market.get_job(job_id).node == NODE_PUBKEY

    def test_settle_wrong_node_rejected(self, market: MockInferenceMarket) -> None:
        """Only the assigned node may settle; any other pubkey is rejected."""
        job_id = market.post_job(CLIENT_PUBKEY, PROMPT_HASH, BOUNTY)
        market.claim_job(NODE_PUBKEY, job_id)

        with pytest.raises(InferenceMarketError, match="NotJobNode"):
            market.settle_job(OTHER_NODE_PUBKEY, job_id, RESULT_HASH)

        # Escrow still locked
        assert market.escrow_balance(job_id) == BOUNTY
        assert market.get_job(job_id).status == JobStatus.CLAIMED


# ---------------------------------------------------------------------------
# TestE2ENodeRegistration
# ---------------------------------------------------------------------------


class TestE2ENodeRegistration:
    """Tests for the compute-registry node lifecycle."""

    def test_register_node(self, registry: MockComputeRegistry) -> None:
        """Node registers and its entry is stored with the correct stake."""
        stake = 2 * LAMPORTS_PER_SOL
        registry.register_node(
            operator=NODE_PUBKEY,
            endpoint="http://node1.example.com:8080",
            vram_gb=24,
            gpu_count=1,
            model_ids=[b"\x00" * 32],
            stake_lamports=stake,
        )
        entry = registry.get_node(NODE_PUBKEY)
        assert entry.operator == NODE_PUBKEY
        assert entry.stake_lamports == stake
        assert entry.vram_gb == 24
        assert entry.gpu_count == 1
        assert entry.reputation == 0

    def test_deregister_releases_stake(self, registry: MockComputeRegistry) -> None:
        """Deregistering returns the stake amount and removes the node."""
        stake = 3 * LAMPORTS_PER_SOL
        registry.register_node(
            operator=NODE_PUBKEY,
            endpoint="http://node1.example.com:8080",
            vram_gb=16,
            gpu_count=1,
            model_ids=[],
            stake_lamports=stake,
        )
        released = registry.deregister_node(NODE_PUBKEY)
        assert released == stake

        with pytest.raises(RegistryError, match="NodeNotFound"):
            registry.get_node(NODE_PUBKEY)

    def test_slashing_reduces_stake(self, registry: MockComputeRegistry) -> None:
        """Byzantine behaviour results in a stake reduction of SLASH_BPS."""
        stake = LAMPORTS_PER_SOL
        registry.register_node(
            operator=NODE_PUBKEY,
            endpoint="http://node1.example.com:8080",
            vram_gb=8,
            gpu_count=1,
            model_ids=[],
            stake_lamports=stake,
        )
        slashed = registry.slash_node(NODE_PUBKEY, reason="submitted invalid result")
        expected_slash = (stake * MockComputeRegistry.SLASH_BPS) // 10_000
        assert slashed == expected_slash

        entry = registry.get_node(NODE_PUBKEY)
        assert entry.stake_lamports == stake - expected_slash

    def test_reputation_update(self, registry: MockComputeRegistry) -> None:
        """Completing jobs increments reputation; it never goes below zero."""
        registry.register_node(
            operator=NODE_PUBKEY,
            endpoint="http://node1.example.com:8080",
            vram_gb=8,
            gpu_count=1,
            model_ids=[],
            stake_lamports=LAMPORTS_PER_SOL,
        )
        new_rep = registry.update_reputation(NODE_PUBKEY, delta=10)
        assert new_rep == 10

        new_rep = registry.update_reputation(NODE_PUBKEY, delta=5)
        assert new_rep == 15

        # Reputation cannot go negative
        new_rep = registry.update_reputation(NODE_PUBKEY, delta=-100)
        assert new_rep == 0


# ---------------------------------------------------------------------------
# TestE2EGovernance
# ---------------------------------------------------------------------------


class TestE2EGovernance:
    """Tests for the DAO governance program."""

    PROPOSER = "ProposerPubkey111111111111111111111111111"
    VOTER_A = "VoterA_Pubkey1111111111111111111111111111"
    VOTER_B = "VoterB_Pubkey1111111111111111111111111111"
    VOTER_C = "VoterC_Pubkey1111111111111111111111111111"

    def test_create_proposal(self, governance: MockGovernance) -> None:
        """DAO proposal is created with ACTIVE status."""
        pid = governance.create_proposal(
            proposer=self.PROPOSER,
            title="Increase protocol fee",
            description="Raise PROTOCOL_FEE_BPS from 500 to 600",
        )
        proposal = governance.get_proposal(pid)
        assert proposal.status == ProposalStatus.ACTIVE
        assert proposal.proposer == self.PROPOSER
        assert proposal.votes_yes == 0
        assert proposal.votes_no == 0

    def test_vote_yes(self, governance: MockGovernance) -> None:
        """A yes vote is recorded on the proposal."""
        pid = governance.create_proposal(
            proposer=self.PROPOSER,
            title="Add new model support",
            description="Allow nodes to serve Llama-3-70B",
        )
        governance.vote_yes(self.VOTER_A, pid)
        proposal = governance.get_proposal(pid)
        assert proposal.votes_yes == 1
        assert self.VOTER_A in proposal.voters

    def test_quorum_check(self, governance: MockGovernance) -> None:
        """Proposal needs QUORUM_VOTES total votes to reach quorum."""
        pid = governance.create_proposal(
            proposer=self.PROPOSER,
            title="Slash multiplier update",
            description="Raise slash from 10% to 15%",
        )
        # Below quorum
        governance.vote_yes(self.VOTER_A, pid)
        assert not governance.check_quorum(pid)

        governance.vote_yes(self.VOTER_B, pid)
        assert not governance.check_quorum(pid)

        # Reach quorum (3rd vote)
        governance.vote_yes(self.VOTER_C, pid)
        assert governance.check_quorum(pid)

        # Enough yes votes → should auto-pass
        proposal = governance.get_proposal(pid)
        assert proposal.status == ProposalStatus.PASSED

    def test_timelock_execution(self, governance: MockGovernance) -> None:
        """Proposal enters TIMELOCKED state; executes only after the delay."""
        # Give governance a zero-length timelock for the "before expiry" check
        governance.TIMELOCK_SECONDS = 60.0  # 60 s — won't expire during test

        pid = governance.create_proposal(
            proposer=self.PROPOSER,
            title="Zero-fee for genesis nodes",
            description="First 10 nodes pay 0% fee for 30 days",
        )
        # Cast enough yes votes to pass
        for voter in [self.VOTER_A, self.VOTER_B, self.VOTER_C]:
            governance.vote_yes(voter, pid)

        assert governance.get_proposal(pid).status == ProposalStatus.PASSED

        # Begin timelock
        governance.execute_proposal("executor_pubkey", pid)
        assert governance.get_proposal(pid).status == ProposalStatus.TIMELOCKED

        # Attempt to execute before timelock expires → must fail
        with pytest.raises(GovernanceError, match="TimelockNotExpired"):
            governance.execute_proposal("executor_pubkey", pid)

        # Manually advance past the timelock
        governance.TIMELOCK_SECONDS = 0.0
        proposal = governance.get_proposal(pid)
        proposal.timelock_until = time.time() - 1  # force-expire

        governance.execute_proposal("executor_pubkey", pid)
        assert governance.get_proposal(pid).status == ProposalStatus.EXECUTED


# ---------------------------------------------------------------------------
# TestBlockchainClientIntegration
# ---------------------------------------------------------------------------


class _MockConfig:
    """Minimal config object accepted by BlockchainClient."""

    rpc_url: str = DEVNET_RPC_URL
    wallet_path: str = "/tmp/mock_wallet.json"
    inference_market_program: str = INFERENCE_MARKET_PROGRAM_ID
    compute_registry_program: str = COMPUTE_REGISTRY_PROGRAM_ID
    governance_program: str = GOVERNANCE_PROGRAM_ID


class TestBlockchainClientIntegration:
    """
    Wire node/blockchain.py's BlockchainClient to MockSolanaClient.

    BlockchainClient.__init__ raises RuntimeError when SOLANA_AVAILABLE is
    False, so we build it via object.__new__ and inject our mock client
    directly — the same bypass-__init__ pattern used elsewhere in this suite.
    """

    def _make_client(
        self,
        solana_client: MockSolanaClient,
        rpc_url: str = DEVNET_RPC_URL,
    ):
        """Build a BlockchainClient with SOLANA_AVAILABLE forced True."""
        # We need SOLANA_AVAILABLE = True so the class body is accessible.
        with patch("node.blockchain.SOLANA_AVAILABLE", True):
            from node.blockchain import BlockchainClient

        cfg = _MockConfig()
        cfg.rpc_url = rpc_url

        bc = object.__new__(BlockchainClient)
        bc.config = cfg
        bc._client = solana_client
        bc._wallet = MagicMock()
        bc._wallet.public_key = NODE_PUBKEY
        bc._inference_program = None
        bc._registry_program = None
        return bc

    def test_connect_uses_configured_rpc(self, solana_client: MockSolanaClient) -> None:
        """BlockchainClient stores the RPC URL from config."""
        bc = self._make_client(solana_client)
        assert bc.config.rpc_url == DEVNET_RPC_URL

    @pytest.mark.asyncio
    async def test_get_wallet_balance_returns_sol(self, solana_client: MockSolanaClient) -> None:
        """get_wallet_balance() converts lamports to SOL (divides by 1e9)."""
        lamports = 5 * LAMPORTS_PER_SOL
        # bc._client IS solana_client; wallet.public_key == NODE_PUBKEY (set in
        # _make_client), so we register the balance under the same key.
        solana_client.set_balance(NODE_PUBKEY, lamports)

        bc = self._make_client(solana_client)
        # No method reassignment needed — bc._client is already solana_client.
        sol = await bc.get_wallet_balance()
        assert sol == pytest.approx(5.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_airdrop_increases_balance(self, solana_client: MockSolanaClient) -> None:
        """request_airdrop() adds lamports to the mock wallet's balance."""
        initial = 1 * LAMPORTS_PER_SOL
        airdrop_amount = 2 * LAMPORTS_PER_SOL
        # bc._client IS solana_client, so request_airdrop calls through directly.
        solana_client.set_balance(NODE_PUBKEY, initial)

        bc = self._make_client(solana_client)
        success = await bc.request_airdrop(airdrop_amount)
        assert success is True

        # Verify balance increased in the shared MockSolanaClient state.
        new_balance_resp = await solana_client.get_balance(NODE_PUBKEY)
        assert new_balance_resp.value == initial + airdrop_amount

    @pytest.mark.asyncio
    async def test_airdrop_skipped_on_mainnet(self, solana_client: MockSolanaClient) -> None:
        """request_airdrop() is a no-op when the RPC URL contains 'mainnet'."""
        bc = self._make_client(solana_client, rpc_url="https://api.mainnet-beta.solana.com")
        result = await bc.request_airdrop(LAMPORTS_PER_SOL)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_wallet_balance_returns_zero_on_error(
        self, solana_client: MockSolanaClient
    ) -> None:
        """get_wallet_balance() returns 0.0 when the RPC call fails."""
        bc = self._make_client(solana_client)

        async def _failing_get_balance(pubkey):
            raise ConnectionError("RPC unreachable")

        bc._client.get_balance = _failing_get_balance  # type: ignore[method-assign]

        sol = await bc.get_wallet_balance()
        assert sol == 0.0

    @pytest.mark.asyncio
    async def test_send_and_confirm_transaction(self, solana_client: MockSolanaClient) -> None:
        """MockSolanaClient records transactions and confirms them."""
        fake_tx = MagicMock(name="transaction")
        resp = await solana_client.send_transaction(fake_tx)
        sig = resp.value
        assert sig.startswith("sig_")

        confirm_resp = await solana_client.confirm_transaction(sig)
        assert confirm_resp.value.err is None
        assert sig in solana_client._confirmed_sigs

    def test_get_program_accounts_returns_registered(self, solana_client: MockSolanaClient) -> None:
        """get_program_accounts returns only accounts for that program."""
        acct = {"pubkey": "pda123", "data": b"\x00" * 8}
        solana_client.register_program_account(INFERENCE_MARKET_PROGRAM_ID, acct)

        import asyncio

        result = asyncio.run(solana_client.get_program_accounts(INFERENCE_MARKET_PROGRAM_ID))
        assert len(result) == 1
        assert result[0]["pubkey"] == "pda123"

        result_other = asyncio.run(solana_client.get_program_accounts(GOVERNANCE_PROGRAM_ID))
        assert result_other == []
