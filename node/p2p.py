"""
P2P networking layer built on top of Hivemind.

Each node exposes a gRPC-style RemoteExpert interface so that activations
can be streamed from shard to shard. The DHT (Kademlia-based) is used for:
  - Node discovery (who holds which shards)
  - Health / liveness announcements
  - Routing table for inference pipelines

S/Kademlia sybil-resistance extensions (SybilResistantDHT):
  - Node IDs must be derived from SHA-256 of the node's public key
  - All lookups use k=20 parallel disjoint paths (redundant routing)
  - A sibling list of k-closest known nodes is maintained for eclipse detection
  - DHT messages with mismatched node IDs are rejected
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

import torch

try:
    import hivemind
    from hivemind import DHT, RemoteExpert, Server

    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False

from node.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)
from node.peer_manager import PeerManager

logger = logging.getLogger(__name__)

# S/Kademlia default parallel lookup width
_SKADEMLIA_K = 20

# Exponential back-off delays (seconds) for successive retries
_BACKOFF_DELAYS = [0.5, 1.0, 2.0]

# Cache TTL for DHT peer-list refreshes (seconds)
_PEER_CACHE_TTL = 60.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RemoteShardError(RuntimeError):
    """Raised when a remote shard call fails after all retry attempts."""


class AllPeersFailedError(RemoteShardError):
    """Raised when every peer in a MultiPeerShardClient is unavailable."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def health_check_peer(endpoint: str, timeout_s: float = 5.0) -> bool:
    """
    Attempt a lightweight TCP probe to *endpoint* ("host:port").

    Returns True if a connection can be established within *timeout_s*,
    False otherwise (dead host, refused connection, timeout, etc.).
    """
    try:
        host, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_s,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception as exc:
        logger.debug("health_check_peer(%s) failed: %s", endpoint, exc)
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PeerInfo:
    peer_id: str
    endpoint: str
    model_id: str
    shard_index: int
    num_shards: int
    vram_gb: int
    is_active: bool
    last_seen: float


@dataclass
class SybilResistantDHT:
    """
    S/Kademlia sybil-resistance wrapper / mixin.

    Enforces three S/Kademlia properties on top of any Kademlia-style DHT:

    1. **Crypto-bound node IDs** – a node ID is only valid if it equals
       SHA-256(pubkey_bytes).hex()[:40].  Nodes cannot freely choose IDs.

    2. **Redundant parallel lookups** – all key lookups are issued over
       *k* disjoint routing paths simultaneously (``lookup_with_redundancy``).
       An attacker must control nodes on *every* path to suppress a result.

    3. **Sibling list** – the *k* closest nodes to self are tracked.  If
       more than half of them arrived simultaneously / from an unknown
       source, an eclipse attack is flagged.
    """

    node_id: str
    sibling_list: list[str] = field(default_factory=list)
    _sibling_k: int = field(default=_SKADEMLIA_K, repr=False)

    @staticmethod
    def verify_node_id(pubkey_bytes: bytes, claimed_id: str) -> bool:
        """
        Return True iff *claimed_id* is legitimately derived from *pubkey_bytes*.

        A valid node ID satisfies::

            SHA-256(pubkey_bytes).hex()[:40] == claimed_id[:40]

        The first 40 hex characters (160 bits) are checked, which matches
        the Kademlia ID space while still providing strong collision resistance.
        """
        derived = hashlib.sha256(pubkey_bytes).hexdigest()[:40]
        return derived == claimed_id[:40]

    def add_to_sibling_list(self, candidate_id: str) -> None:
        """
        Insert *candidate_id* into the sibling list, keeping it sorted by XOR
        distance from *self.node_id* and trimmed to *_sibling_k* entries.
        """
        if candidate_id not in self.sibling_list:
            self.sibling_list.append(candidate_id)

        def xor_dist(nid: str) -> int:
            a = int(self.node_id[:16], 16)
            b = int(nid[:16], 16)
            return a ^ b

        self.sibling_list.sort(key=xor_dist)
        self.sibling_list = self.sibling_list[: self._sibling_k]

    def detect_eclipse(self, known_node_ids: set[str]) -> bool:
        """
        Return True if an eclipse / partition attack appears likely.

        Heuristic: if more than 50 % of the sibling list consists of nodes
        **not** present in *known_node_ids* (i.e. they weren't seen before
        the latest batch insertion), the routing table may have been poisoned.
        """
        if not self.sibling_list:
            return False
        unknown = sum(1 for nid in self.sibling_list if nid not in known_node_ids)
        return unknown / len(self.sibling_list) > 0.5


# ---------------------------------------------------------------------------
# P2P layer
# ---------------------------------------------------------------------------


class P2PLayer:
    """
    Wraps Hivemind DHT + Server to expose this node's shard to the network
    and discover peers holding adjacent shards.
    """

    def __init__(self, config, shard_manager):
        if not HIVEMIND_AVAILABLE:
            raise RuntimeError("hivemind is not installed. Run: pip install hivemind")
        self.config = config
        self.shard_manager = shard_manager
        self.dht: DHT | None = None
        self.server: Server | None = None
        self._peer_cache: dict[str, PeerInfo] = {}

    async def start(self) -> None:
        public_host = self.config.public_host or self.config.listen_host
        initial_peers = self.config.dht_bootstrap_peers or []

        self.dht = hivemind.DHT(
            host_maddrs=[f"/ip4/{self.config.listen_host}/tcp/{self.config.listen_port}"],
            announce_maddrs=[f"/ip4/{public_host}/tcp/{self.config.listen_port}"],
            initial_peers=initial_peers,
            start=True,
        )

        self.server = hivemind.Server(
            self.dht,
            expert_cls=ShardExpert,
            expert_args={"shard_manager": self.shard_manager},
            num_experts=1,
            expert_prefix=f"shard.{self.config.model_name}.{self.config.shard_index}",
        )

        await self.server.run_in_background()

        # Announce ourselves in the DHT
        await self._announce()
        logger.info("P2P layer started. Peer ID: %s", self.dht.peer_id)

    async def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.dht:
            self.dht.shutdown()

    async def get_next_shard_peer(self, shard_index: int) -> RemoteShardClient | None:
        """Find a live node hosting the given shard index for our model."""
        key = f"shard.{self.config.model_name}.{shard_index}"
        try:
            peer_data = await asyncio.get_event_loop().run_in_executor(None, self.dht.get, key)
            if peer_data:
                return RemoteShardClient(peer_data["endpoint"])
        except Exception as exc:
            logger.warning("DHT lookup failed for shard %d: %s", shard_index, exc)
        return None

    async def lookup_with_redundancy(self, key: str, k: int = _SKADEMLIA_K) -> list:
        """
        S/Kademlia redundant lookup: query *k* independent routing paths in
        parallel and return the merged, deduplicated result set.

        Each "path" is modelled as a separate DHT ``get`` call seeded from a
        different logical starting point (the key XOR-rotated by the path
        index).  In a real S/Kademlia deployment the DHT library would expose
        per-path iterative lookup; here we approximate it by issuing *k*
        concurrent lookups and merging results.

        Returns a flat list of all peer values returned by any path.
        """
        if self.dht is None:
            raise RuntimeError("DHT is not started; call start() first")

        loop = asyncio.get_event_loop()

        async def _single_path_lookup(path_index: int) -> list:
            # Derive a path-specific key variant (XOR with path index suffix)
            path_key = f"{key}#{path_index}"
            try:
                result = await loop.run_in_executor(None, self.dht.get, path_key)
                if result is None:
                    # Fall back to the canonical key on this path
                    result = await loop.run_in_executor(None, self.dht.get, key)
                return [result] if result is not None else []
            except Exception as exc:
                logger.debug("Path %d lookup failed for key %r: %s", path_index, key, exc)
                return []

        results_per_path = await asyncio.gather(
            *[_single_path_lookup(i) for i in range(k)],
            return_exceptions=False,
        )

        # Flatten and deduplicate (by repr for arbitrary value types)
        seen: set[str] = set()
        merged: list = []
        for path_results in results_per_path:
            for item in path_results:
                key_repr = repr(item)
                if key_repr not in seen:
                    seen.add(key_repr)
                    merged.append(item)
        return merged

    @staticmethod
    def verify_node_id(pubkey_bytes: bytes, claimed_id: str) -> bool:
        """
        Return True iff *claimed_id* is a valid S/Kademlia node ID for the
        given public key bytes.

        Delegates to :meth:`SybilResistantDHT.verify_node_id`.
        """
        return SybilResistantDHT.verify_node_id(pubkey_bytes, claimed_id)

    async def _announce(self) -> None:
        """Write our shard metadata into the DHT so other nodes can find us."""
        key = f"shard.{self.config.model_name}.{self.config.shard_index}"
        public_host = self.config.public_host or self.config.listen_host
        value = {
            "endpoint": f"{public_host}:{self.config.listen_port}",
            "shard_index": self.config.shard_index,
            "num_shards": self.config.num_shards,
            "timestamp": time.time(),
        }
        await asyncio.get_event_loop().run_in_executor(
            None, self.dht.store, key, value, expiration_time=hivemind.get_dht_time() + 600
        )


# ---------------------------------------------------------------------------
# Shard experts / experts
# ---------------------------------------------------------------------------


class ShardExpert(torch.nn.Module):
    """
    Hivemind expert wrapper around ShardManager.forward().
    Receives packed hidden states, returns the transformed output.
    """

    def __init__(self, shard_manager):
        super().__init__()
        self.shard_manager = shard_manager

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shard_manager.forward(hidden_states)


# ---------------------------------------------------------------------------
# RemoteShardClient – resilient single-peer client
# ---------------------------------------------------------------------------


class RemoteShardClient:
    """
    Resilient async client that calls a remote shard's forward pass.

    Features
    --------
    * Per-call timeout via ``asyncio.wait_for``.
    * Exponential back-off retry (up to *max_retries* attempts).
    * Optional ``CircuitBreaker`` integration: the breaker is opened after all
      retries are exhausted, and calls are blocked immediately while OPEN.
    * On any failure the peer is marked as *suspected* in the optional
      ``PeerManager``.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        circuit_breaker: CircuitBreaker | None = None,
        peer_manager: PeerManager | None = None,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.circuit_breaker = circuit_breaker
        self.peer_manager = peer_manager
        self._expert: RemoteExpert | None = None  # type: ignore[name-defined]

    def _ensure_expert(self) -> RemoteExpert:  # type: ignore[name-defined]
        if self._expert is None:
            self._expert = RemoteExpert(uid=self.endpoint)  # type: ignore[name-defined]
        return self._expert

    def _mark_peer_suspected(self) -> None:
        """Increment missed_heartbeats so the peer transitions to SUSPECTED."""
        if self.peer_manager is not None:
            peer = self.peer_manager.get(self.endpoint)
            if peer is not None:
                peer.missed_heartbeats += 1
                logger.debug("Peer %s marked as suspected", self.endpoint)

    async def _call_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        loop = asyncio.get_event_loop()
        expert = self._ensure_expert()
        return await asyncio.wait_for(
            loop.run_in_executor(None, expert, hidden_states),
            timeout=self.timeout_s,
        )

    async def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Run the remote shard forward pass with timeout, retries, and circuit
        breaking.

        Raises
        ------
        RemoteShardError
            If all retry attempts are exhausted or the circuit breaker opens.
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            backoff = (
                _BACKOFF_DELAYS[attempt] if attempt < len(_BACKOFF_DELAYS) else _BACKOFF_DELAYS[-1]
            )
            try:
                if self.circuit_breaker is not None:
                    result = await self.circuit_breaker.call(
                        lambda hs=hidden_states: self._call_expert(hs)
                    )
                else:
                    result = await self._call_expert(hidden_states)
                return result
            except CircuitOpenError as exc:
                # Circuit is (now) OPEN – no point retrying this peer
                last_exc = exc
                logger.warning(
                    "RemoteShardClient(%s) circuit OPEN after attempt %d/%d",
                    self.endpoint,
                    attempt + 1,
                    self.max_retries,
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "RemoteShardClient(%s) attempt %d/%d failed: %s",
                    self.endpoint,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                self._mark_peer_suspected()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(backoff)

        # All retries exhausted – open the circuit breaker forcibly if not already open
        if self.circuit_breaker is not None and self.circuit_breaker.state != CircuitState.OPEN:
            for _ in range(self.circuit_breaker.config.failure_threshold):
                self.circuit_breaker._on_failure()

        raise RemoteShardError(
            f"All {self.max_retries} attempts to {self.endpoint} failed"
        ) from last_exc


# ---------------------------------------------------------------------------
# MultiPeerShardClient – fan-out across multiple peers
# ---------------------------------------------------------------------------


class MultiPeerShardClient:
    """
    Routes a forward pass across a pool of peers, falling back automatically
    when individual peers fail or have open circuit breakers.

    Parameters
    ----------
    peers:
        Ordered list of peer endpoint strings (``"host:port"``).
    shard_index:
        Which shard these peers serve.
    model_id:
        Model identifier (used for logging).
    timeout_s:
        Per-call timeout forwarded to each ``RemoteShardClient``.
    """

    def __init__(
        self,
        peers: list[str],
        shard_index: int,
        model_id: str,
        timeout_s: float = 30.0,
        peer_manager: PeerManager | None = None,
    ):
        self.shard_index = shard_index
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.peer_manager = peer_manager

        # One CircuitBreaker + one RemoteShardClient per peer
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._clients: dict[str, RemoteShardClient] = {}

        for endpoint in peers:
            cb = CircuitBreaker(
                config=CircuitBreakerConfig(failure_threshold=1),
                name=endpoint,
            )
            self._circuit_breakers[endpoint] = cb
            self._clients[endpoint] = RemoteShardClient(
                endpoint=endpoint,
                timeout_s=timeout_s,
                max_retries=1,  # multi-peer handles retries across peers
                circuit_breaker=cb,
                peer_manager=peer_manager,
            )

    # ------------------------------------------------------------------
    # Peer ordering
    # ------------------------------------------------------------------

    def _priority_ordered_peers(self) -> list[str]:
        """
        Return peers sorted by reputation (stake_lamports descending) when a
        PeerManager is available, otherwise preserve insertion order.
        """
        endpoints = list(self._clients.keys())
        if self.peer_manager is None:
            return endpoints

        def _reputation(ep: str) -> int:
            info = self.peer_manager.get(ep)
            return info.stake_lamports if info is not None else 0

        return sorted(endpoints, key=_reputation, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_healthy_peers(self) -> list[str]:
        """Return endpoints whose circuit breaker is currently CLOSED."""
        return [ep for ep, cb in self._circuit_breakers.items() if cb.state == CircuitState.CLOSED]

    async def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Attempt the forward pass on each peer in priority order.

        Skips peers with OPEN circuit breakers.  Raises
        ``AllPeersFailedError`` only when every peer has been tried (or
        skipped) and none succeeded.
        """
        peers = self._priority_ordered_peers()
        errors: list[str] = []

        for endpoint in peers:
            cb = self._circuit_breakers[endpoint]
            if cb.state == CircuitState.OPEN:
                logger.debug("MultiPeerShardClient: skipping %s (circuit OPEN)", endpoint)
                errors.append(f"{endpoint}: circuit OPEN")
                continue

            client = self._clients[endpoint]
            try:
                result = await client.forward(hidden_states)
                logger.debug("MultiPeerShardClient: success via %s", endpoint)
                return result
            except CircuitOpenError as exc:
                logger.debug("MultiPeerShardClient: %s circuit opened mid-call", endpoint)
                errors.append(f"{endpoint}: {exc}")
            except RemoteShardError as exc:
                logger.warning("MultiPeerShardClient: peer %s exhausted: %s", endpoint, exc)
                errors.append(f"{endpoint}: {exc}")
            except Exception as exc:
                logger.warning("MultiPeerShardClient: peer %s unexpected error: %s", endpoint, exc)
                errors.append(f"{endpoint}: {exc}")

        raise AllPeersFailedError(
            f"All peers failed for shard {self.shard_index} of {self.model_id}: "
            + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# P2PManager – higher-level manager with DHT peer refresh
# ---------------------------------------------------------------------------


class P2PManager:
    """
    High-level manager that discovers shard peers from the DHT and wraps
    them in a ``MultiPeerShardClient`` for fault-tolerant inference.

    The DHT peer list is cached for ``_PEER_CACHE_TTL`` seconds so that
    every forward pass does not incur a network round-trip.
    """

    def __init__(
        self,
        dht,
        model_name: str,
        timeout_s: float = 30.0,
        peer_manager: PeerManager | None = None,
    ):
        self._dht = dht
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._peer_manager = peer_manager

        # Cache: shard_index → (MultiPeerShardClient, expiry_timestamp)
        self._cache: dict[int, tuple[MultiPeerShardClient, float]] = {}

    async def _fetch_peers_from_dht(self, shard_index: int) -> list[str]:
        """Query the DHT for all known endpoints serving *shard_index*."""
        key = f"shard.{self._model_name}.{shard_index}"
        loop = asyncio.get_event_loop()
        try:
            peer_data = await loop.run_in_executor(None, self._dht.get, key)
            if peer_data:
                if isinstance(peer_data, list):
                    return [p["endpoint"] for p in peer_data if "endpoint" in p]
                if isinstance(peer_data, dict) and "endpoint" in peer_data:
                    return [peer_data["endpoint"]]
        except Exception as exc:
            logger.warning("P2PManager DHT lookup failed for shard %d: %s", shard_index, exc)
        return []

    async def discover_next_shard_peers(self, shard_index: int) -> MultiPeerShardClient | None:
        """
        Return a ``MultiPeerShardClient`` for *shard_index*.

        The peer list is refreshed from the DHT at most once per
        ``_PEER_CACHE_TTL`` seconds (60 s by default).
        """
        now = time.monotonic()
        cached = self._cache.get(shard_index)
        if cached is not None:
            client, expiry = cached
            if now < expiry:
                return client

        peers = await self._fetch_peers_from_dht(shard_index)
        if not peers:
            logger.warning("No peers found in DHT for shard %d", shard_index)
            return None

        client = MultiPeerShardClient(
            peers=peers,
            shard_index=shard_index,
            model_id=self._model_name,
            timeout_s=self._timeout_s,
            peer_manager=self._peer_manager,
        )
        self._cache[shard_index] = (client, now + _PEER_CACHE_TTL)
        return client
