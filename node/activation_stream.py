"""
Direct peer-to-peer activation streaming for pipeline-parallel shards.

Replaces the DHT-based _push_activations / _receive_activations round-trip
with a low-latency TCP stream (wire-compatible with a future QUIC/aioquic
upgrade).

Wire protocol (big-endian):
  [4 bytes]  magic      = 0xAC71AC71
  [8 bytes]  job_id     uint64
  [4 bytes]  payload_length  uint32
  [N bytes]  msgpack-encoded dict:
               "shape": list[int]
               "dtype": str  ("float32", "float16", …)
               "data":  bytes  (raw tensor bytes, CPU contiguous)

Importing this module does NOT require torch; torch is imported lazily
inside the methods that actually handle tensors.
"""

from __future__ import annotations

import asyncio
import logging
import struct

logger = logging.getLogger(__name__)

MAGIC = 0xAC71AC71
_HEADER_FMT = ">IQI"  # magic(4) + job_id(8) + payload_length(4)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 16 bytes

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

try:
    import msgpack as _msgpack

    def _pack(obj: dict) -> bytes:
        return _msgpack.packb(obj, use_bin_type=True)

    def _unpack(data: bytes) -> dict:
        return _msgpack.unpackb(data, raw=False)

except ModuleNotFoundError:  # pragma: no cover – msgpack always present in CI
    import base64
    import json

    def _pack(obj: dict) -> bytes:  # type: ignore[misc]
        encoded = dict(obj)
        if isinstance(encoded.get("data"), (bytes, bytearray)):
            encoded["data"] = base64.b64encode(encoded["data"]).decode()
            encoded["_b64"] = True
        return json.dumps(encoded).encode()

    def _unpack(data: bytes) -> dict:  # type: ignore[misc]
        obj = json.loads(data.decode())
        if obj.pop("_b64", False):
            obj["data"] = base64.b64decode(obj["data"])
        return obj


# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------


async def _read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly *n* bytes, handling partial reads correctly."""
    buf = bytearray()
    while len(buf) < n:
        chunk = await reader.read(n - len(buf))
        if not chunk:
            raise EOFError(f"Connection closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# ActivationServer
# ---------------------------------------------------------------------------


class ActivationServer:
    """
    Listens for incoming activation tensors from the previous shard.

    Uses asyncio.start_server for TCP transport (upgradeable to QUIC/aioquic).

    Wire protocol:
      [4 bytes: magic 0xAC71AC71]
      [8 bytes: job_id uint64 big-endian]
      [4 bytes: payload_length uint32 big-endian]
      [payload_length bytes: msgpack-encoded dict with keys:
         "shape": list of ints
         "dtype": str  ("float32", "float16", etc.)
         "data": bytes (raw tensor bytes)
      ]
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9090) -> None:
        self.host = host
        self.port = port
        self._server: asyncio.Server | None = None
        # Maps job_id → asyncio.Future that resolves to a torch.Tensor
        self._pending: dict[int, asyncio.Future] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening for incoming activation streams."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self.host,
            port=self.port,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("ActivationServer listening on %s:%s", addr[0], addr[1])
        await self._server.start_serving()

    async def stop(self) -> None:
        """Stop the server and cancel all pending futures."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def receive(self, job_id: int, timeout: float = 60.0) -> torch.Tensor:  # noqa: F821
        """Block until activations for *job_id* arrive, or raise TimeoutError."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[job_id] = fut
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"Timed out waiting for activations for job {job_id}")
        finally:
            self._pending.pop(job_id, None)

    # ------------------------------------------------------------------
    # Internal connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            await self._read_frame(reader)
        except Exception as exc:
            logger.warning("ActivationServer: error from %s: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_frame(self, reader: asyncio.StreamReader) -> None:
        """Parse one activation frame and resolve the corresponding future."""
        header_bytes = await _read_exactly(reader, _HEADER_SIZE)
        magic, job_id, payload_length = struct.unpack(_HEADER_FMT, header_bytes)

        if magic != MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08X}; expected 0x{MAGIC:08X}")

        payload = await _read_exactly(reader, payload_length)
        obj = _unpack(payload)

        import torch  # lazy import — keep module importable without torch

        shape = obj["shape"]
        dtype = getattr(torch, obj["dtype"])
        data: bytes = obj["data"]
        tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).clone()

        fut = self._pending.get(job_id)
        if fut is not None and not fut.done():
            fut.set_result(tensor)
        else:
            logger.warning("ActivationServer: received job_id=%d but no waiter registered", job_id)


# ---------------------------------------------------------------------------
# ActivationClient
# ---------------------------------------------------------------------------


class ActivationClient:
    """
    Connects to the next shard's ActivationServer and streams activations.

    Each :meth:`send` call opens a new TCP connection, writes the frame,
    and closes the connection.  This keeps the implementation stateless
    and avoids connection-reuse complexity for the initial version.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def send(self, job_id: int, tensor: torch.Tensor) -> None:  # noqa: F821
        """Serialize and send activations to the remote shard."""
        import torch  # lazy import

        # Ensure CPU-side, contiguous, detached
        cpu_tensor: torch.Tensor = tensor.detach().cpu().contiguous()
        dtype_str = str(cpu_tensor.dtype).replace("torch.", "")
        # Use untyped_storage for numpy-free raw bytes extraction
        raw_bytes: bytes = bytes(cpu_tensor.untyped_storage())
        payload = _pack(
            {
                "shape": list(cpu_tensor.shape),
                "dtype": dtype_str,
                "data": raw_bytes,
            }
        )

        payload_length = len(payload)
        header = struct.pack(_HEADER_FMT, MAGIC, job_id, payload_length)
        frame = header + payload

        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            writer.write(frame)
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
