import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NodeConfig:
    # Solana
    rpc_url: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    wallet_path: str = os.getenv("WALLET_PATH", str(Path.home() / ".config/solana/id.json"))
    compute_registry_program: str = os.getenv(
        "COMPUTE_REGISTRY_PROGRAM", "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"
    )
    inference_market_program: str = os.getenv(
        "INFERENCE_MARKET_PROGRAM", "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
    )

    # P2P networking
    # On rented GPU instances, set this to the public IP/port assigned by the provider
    listen_host: str = os.getenv("LISTEN_HOST", "0.0.0.0")
    listen_port: int = int(os.getenv("LISTEN_PORT", "7070"))
    public_host: str | None = os.getenv("PUBLIC_HOST")  # set by GPU rental provider env
    dht_bootstrap_peers: list[str] = field(
        default_factory=lambda: [p for p in os.getenv("DHT_BOOTSTRAP_PEERS", "").split(",") if p]
    )

    # Model serving
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B")
    # List of models this node is willing to serve.  Defaults to [model_name]
    # when not explicitly set so single-model deployments keep working as-is.
    supported_models: list[str] = field(
        default_factory=lambda: [m for m in os.getenv("SUPPORTED_MODELS", "").split(",") if m]
    )
    num_shards: int = int(os.getenv("NUM_SHARDS", "4"))
    shard_index: int = int(os.getenv("SHARD_INDEX", "0"))
    cache_dir: str = os.getenv("MODEL_CACHE_DIR", str(Path.home() / ".cache/decentralized-llm"))
    dtype: str = os.getenv("MODEL_DTYPE", "float16")  # float16, bfloat16, int8, int4

    # Fine-tuning
    finetuning_enabled: bool = os.getenv("FINETUNING_ENABLED", "false").lower() == "true"
    lora_rank: int = int(os.getenv("LORA_RANK", "8"))

    # Storage (Lighthouse IPFS+Filecoin)
    lighthouse_api_key: str = os.getenv("LIGHTHOUSE_API_KEY", "")

    # Pay.sh
    paysh_api_key: str = os.getenv("PAYSH_API_KEY", "")
    paysh_webhook_secret: str = os.getenv("PAYSH_WEBHOOK_SECRET", "")

    # Operational
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
    job_poll_interval_seconds: float = float(os.getenv("JOB_POLL_INTERVAL", "2.0"))

    def wallet_private_key_bytes(self) -> bytes:
        """Return the 32-byte Ed25519 seed from the Solana wallet JSON file."""
        import json

        key_list = json.loads(Path(self.wallet_path).read_text())
        return bytes(key_list[:32])
