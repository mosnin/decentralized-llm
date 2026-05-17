"""
Decentralized storage for inference results and encrypted prompts.

Backend: Lighthouse (IPFS + Filecoin)
  - Python SDK: `pip install lighthouseweb3`
  - Files pinned to IPFS AND backed by Filecoin deals
  - Standard IPFS CIDs — verifiable against on-chain hashes
  - Free tier: 5 GB; Lite plan ($20/mo): 500 GB
  - Fallback read: any IPFS gateway (content-addressed)

Set LIGHTHOUSE_API_KEY in your environment.
"""

import asyncio
import os
import tempfile
from pathlib import Path

LIGHTHOUSE_API_KEY = os.getenv("LIGHTHOUSE_API_KEY", "")

GATEWAYS = [
    "https://gateway.lighthouse.storage/ipfs",
    "https://ipfs.io/ipfs",
    "https://cloudflare-ipfs.com/ipfs",
]


class StorageClient:
    """
    Async wrapper around the Lighthouse SDK for storing and retrieving
    inference results and encrypted prompt blobs.
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or LIGHTHOUSE_API_KEY
        if not self._api_key:
            raise ValueError(
                "LIGHTHOUSE_API_KEY not set. Get a free key at https://files.lighthouse.storage/"
            )

        try:
            from lighthouseweb3 import Lighthouse

            self._lh = Lighthouse(token=self._api_key)
        except ImportError:
            raise RuntimeError("pip install lighthouseweb3")

    async def upload(self, content: bytes, filename: str = "data.bin") -> str:
        """Upload bytes and return the IPFS CID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._upload_sync, content, filename
        )

    async def download(self, cid: str) -> bytes:
        """Fetch content by CID, trying multiple gateways."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            for gateway in GATEWAYS:
                try:
                    async with session.get(
                        f"{gateway}/{cid}",
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            return await resp.read()
                except Exception:
                    continue
        raise RuntimeError(f"Could not fetch CID {cid} from any gateway")

    async def upload_encrypted_prompt(self, blob: bytes, job_id: int) -> str:
        return await self.upload(blob, filename=f"prompt_{job_id}.bin")

    async def upload_result(self, result_text: str, job_id: int) -> str:
        return await self.upload(result_text.encode("utf-8"), filename=f"result_{job_id}.txt")

    # ─────────────────────────── sync internals ───────────────────────────────

    def _upload_sync(self, content: bytes, filename: str) -> str:
        """Run in a thread pool — Lighthouse SDK is synchronous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            path.write_bytes(content)
            response = self._lh.upload(source=str(path))
            return response["data"]["Hash"]


class ArweaveStorageClient:
    """
    Async wrapper around the arweave-python-client SDK for permanent
    model weight storage on the Arweave permaweb.

    Install the SDK with: pip install arweave-python-client
    """

    def __init__(self, wallet_path: str) -> None:
        wallet_file = Path(wallet_path)
        if not wallet_file.exists():
            raise FileNotFoundError(f"Arweave wallet file not found: {wallet_path}")

        try:
            import arweave  # noqa: F401
        except ImportError:
            raise RuntimeError("pip install arweave-python-client")

        import arweave as _arweave

        self._wallet = _arweave.Wallet(wallet_path)

    async def upload_model_weights(self, model_dir: str | Path, model_id: str) -> str:
        """
        Archive *model_dir* as a tar file and upload to Arweave.

        Returns the Arweave transaction ID.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._upload_weights_sync, Path(model_dir), model_id
        )

    async def get_model_url(self, tx_id: str) -> str:
        """Return the Arweave gateway URL for a given transaction ID."""
        return f"https://arweave.net/{tx_id}"

    async def download_model_weights(self, tx_id: str, dest_dir: str | Path) -> None:
        """
        Download a model archive from the Arweave gateway and extract it
        to *dest_dir*.
        """
        import aiohttp

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        url = await self.get_model_url(tx_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Failed to download tx {tx_id} from Arweave gateway (HTTP {resp.status})"
                    )
                data = await resp.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "model.tar"
            archive_path.write_bytes(data)
            import tarfile

            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=dest)

    # ─────────────────────────── sync internals ───────────────────────────────

    def _upload_weights_sync(self, model_dir: Path, model_id: str) -> str:
        """Run in a thread pool — Arweave SDK is synchronous."""
        import tarfile

        import arweave

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "model.tar"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(model_dir, arcname=model_dir.name)

            transaction = arweave.Transaction(
                self._wallet,
                data=archive_path.read_bytes(),
            )
            transaction.add_tag("App-Name", "decentralized-llm")
            transaction.add_tag("Model-ID", model_id)
            transaction.add_tag("Content-Type", "application/x-tar")
            transaction.sign()
            transaction.send()
            return transaction.id
