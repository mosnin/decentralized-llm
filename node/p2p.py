"""
P2P networking layer built on top of Hivemind.

Each node exposes a gRPC-style RemoteExpert interface so that activations
can be streamed from shard to shard. The DHT (Kademlia-based) is used for:
  - Node discovery (who holds which shards)
  - Health / liveness announcements
  - Routing table for inference pipelines
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import torch

try:
    import hivemind
    from hivemind import DHT, RemoteExpert, Server

    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False

logger = logging.getLogger(__name__)


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

    async def get_next_shard_peer(self, shard_index: int) -> "RemoteShardClient | None":
        """Find a live node hosting the given shard index for our model."""
        key = f"shard.{self.config.model_name}.{shard_index}"
        try:
            peer_data = await asyncio.get_event_loop().run_in_executor(None, self.dht.get, key)
            if peer_data:
                return RemoteShardClient(peer_data["endpoint"])
        except Exception as exc:
            logger.warning("DHT lookup failed for shard %d: %s", shard_index, exc)
        return None

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


class RemoteShardClient:
    """
    Thin async client that calls a remote shard's forward pass.
    In practice this goes over Hivemind's gRPC transport.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._expert: RemoteExpert | None = None

    def _ensure_expert(self) -> RemoteExpert:
        if self._expert is None:
            self._expert = RemoteExpert(uid=self.endpoint)
        return self._expert

    async def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        loop = asyncio.get_event_loop()
        expert = self._ensure_expert()
        return await loop.run_in_executor(None, expert, hidden_states)
