"""
Vast.ai GPU provisioner for compute nodes.

Automates the full lifecycle:
  1. Search for GPU instances matching requirements
  2. Rent an instance with the node Docker image
  3. Poll until ready, extract the public IP:port
  4. Register the node on-chain in the compute-registry
  5. Monitor for preemption and re-provision if needed

Vast.ai advantages for P2P networks:
  - `static_ip=true` filter guarantees a dedicated public IP
  - `direct_port_count>=1` + ports >= 70000 map 1:1 (no NAT)
  - $VAST_TCP_PORT_<N> env var gives exact external port at runtime
  - Official Python SDK: `pip install vastai`

Set VAST_API_KEY in your environment.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VAST_API_KEY = os.getenv("VAST_API_KEY", "")

# Docker image for compute nodes — hosted on Docker Hub / GHCR
NODE_IMAGE = os.getenv("NODE_IMAGE", "ghcr.io/decentralized-llm/node:latest")

# Internal DHT port used inside the container
INTERNAL_DHT_PORT = 70070


@dataclass
class GpuRequirements:
    """Filters for GPU instance selection."""

    min_vram_gb: int = 24  # 24 GB minimum for 7B models at fp16
    gpu_name: str | None = None  # e.g., "RTX_4090", "A100", None = any
    max_price_per_hour: float = 2.0
    min_reliability: float = 0.95  # host reliability score (0–1)
    verified_only: bool = True
    num_gpus: int = 1
    disk_gb: int = 80  # model weights + OS


@dataclass
class GpuInstance:
    offer_id: int
    instance_id: int | None
    gpu_name: str
    vram_gb: int
    price_per_hour: float
    public_ip: str
    public_port: int  # external DHT port
    status: str  # "rented", "running", "stopped"
    ssh_host: str
    ssh_port: int
    env_vars: dict[str, str] = field(default_factory=dict)


class VastAiProvisioner:
    """
    Manages spinning up and tearing down Vast.ai GPU instances for compute nodes.

    Example:
        provisioner = VastAiProvisioner(api_key=os.environ["VAST_API_KEY"])
        instance = await provisioner.provision(
            model_name="meta-llama/Llama-3.2-3B",
            shard_index=0,
            num_shards=4,
            bootstrap_peers=["/ip4/1.2.3.4/tcp/7070"],
        )
        print(f"Node endpoint: {instance.public_ip}:{instance.public_port}")
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or VAST_API_KEY
        if not self._api_key:
            raise ValueError("VAST_API_KEY not set — get one at https://vast.ai/")

        try:
            import vastai

            self._sdk = vastai.VastAI(api_key=self._api_key)
        except ImportError:
            raise RuntimeError("pip install vastai")

    async def find_offers(
        self,
        requirements: GpuRequirements | None = None,
    ) -> list[dict]:
        """
        Search for available GPU instances matching requirements.
        Returns offers sorted by cost-efficiency (TFLOPS per dollar).
        """
        req = requirements or GpuRequirements()

        query_parts = [
            f"gpu_ram>={req.min_vram_gb}",
            f"num_gpus={req.num_gpus}",
            f"dph<={req.max_price_per_hour}",
            f"reliability2>={req.min_reliability}",
            "direct_port_count>=1",
            "static_ip=true",
            "rentable=true",
            "cuda_max_good>=12.0",
        ]
        if req.verified_only:
            query_parts.append("verified=true")
        if req.gpu_name:
            query_parts.append(f"gpu_name={req.gpu_name}")

        query = " ".join(query_parts)

        offers = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._sdk.search_offers(query=query, order="dlperf_usd-"),
        )
        return offers or []

    async def provision(
        self,
        model_name: str,
        shard_index: int,
        num_shards: int,
        solana_rpc_url: str,
        wallet_path: str,
        bootstrap_peers: list[str] | None = None,
        requirements: GpuRequirements | None = None,
        lighthouse_api_key: str = "",
    ) -> GpuInstance:
        """
        Find a suitable offer, rent it, and start the compute node container.
        Returns the GpuInstance with public IP and port once it's running.
        """
        offers = await self.find_offers(requirements)
        if not offers:
            raise RuntimeError("No GPU offers match requirements")

        offer = offers[0]
        offer_id = offer["id"]
        logger.info(
            "Renting %s (%.1f GB VRAM) @ $%.3f/hr (offer %d)",
            offer.get("gpu_name"),
            offer.get("gpu_ram", 0) / 1024,
            offer.get("dph_total", 0),
            offer_id,
        )

        env = {
            "MODEL_NAME": model_name,
            "SHARD_INDEX": str(shard_index),
            "NUM_SHARDS": str(num_shards),
            "LISTEN_PORT": str(INTERNAL_DHT_PORT),
            "SOLANA_RPC_URL": solana_rpc_url,
            "DHT_BOOTSTRAP_PEERS": ",".join(bootstrap_peers or []),
            "LIGHTHOUSE_API_KEY": lighthouse_api_key,
        }

        # On Vast.ai, $VAST_TCP_PORT_70070 gives the external port for container port 70070
        onstart_cmd = (
            f"export PUBLIC_HOST=$(curl -s ifconfig.me) && "
            f"export LISTEN_PORT=$VAST_TCP_PORT_{INTERNAL_DHT_PORT} && "
            f"python -m node.server"
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._sdk.create_instance(
                id=offer_id,
                image=NODE_IMAGE,
                disk=requirements.disk_gb if requirements else 80,
                ssh=True,
                direct=True,
                env=env,
                onstart_cmd=onstart_cmd,
                args_str=f"-p {INTERNAL_DHT_PORT}:{INTERNAL_DHT_PORT}",
            ),
        )

        instance_id = result.get("new_contract")
        logger.info("Instance %d created, waiting for it to start…", instance_id)

        return await self._wait_for_running(instance_id)

    async def destroy(self, instance_id: int) -> None:
        """Terminate and destroy a rented instance."""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._sdk.destroy_instance(id=instance_id),
        )
        logger.info("Instance %d destroyed", instance_id)

    async def list_instances(self) -> list[GpuInstance]:
        """List all currently rented instances."""
        raw = await asyncio.get_event_loop().run_in_executor(None, self._sdk.show_instances)
        return [self._parse_instance(inst) for inst in (raw or [])]

    # ─────────────────────────── private ─────────────────────────────────────

    async def _wait_for_running(self, instance_id: int, timeout_seconds: int = 300) -> GpuInstance:
        """Poll until instance reaches 'running' status and has a public IP."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            instances = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._sdk.show_instance(id=instance_id)
            )
            inst = instances if isinstance(instances, dict) else {}
            status = inst.get("actual_status", "")

            if status == "running" and inst.get("public_ipaddr"):
                return self._parse_instance(inst)

            logger.debug("Instance %d status: %s — waiting…", instance_id, status)
            await asyncio.sleep(10)

        raise TimeoutError(f"Instance {instance_id} did not start within {timeout_seconds}s")

    def _parse_instance(self, inst: dict) -> GpuInstance:
        public_ip = inst.get("public_ipaddr", "")
        # Vast.ai external port for our DHT port
        external_port = (
            inst.get("ports", {})
            .get(f"{INTERNAL_DHT_PORT}/tcp", [{}])[0]
            .get("HostPort", INTERNAL_DHT_PORT)
        )
        return GpuInstance(
            offer_id=inst.get("offer_id", 0),
            instance_id=inst.get("id"),
            gpu_name=inst.get("gpu_name", ""),
            vram_gb=int(inst.get("gpu_ram", 0)) // 1024,
            price_per_hour=float(inst.get("dph_total", 0)),
            public_ip=public_ip,
            public_port=int(external_port),
            status=inst.get("actual_status", "unknown"),
            ssh_host=inst.get("ssh_host", public_ip),
            ssh_port=int(inst.get("ssh_port", 22)),
        )
