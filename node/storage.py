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
import hashlib
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
                "LIGHTHOUSE_API_KEY not set. "
                "Get a free key at https://files.lighthouse.storage/"
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
        return await self.upload(
            result_text.encode("utf-8"), filename=f"result_{job_id}.txt"
        )

    # ─────────────────────────── sync internals ───────────────────────────────

    def _upload_sync(self, content: bytes, filename: str) -> str:
        """Run in a thread pool — Lighthouse SDK is synchronous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            path.write_bytes(content)
            response = self._lh.upload(source=str(path))
            return response["data"]["Hash"]
