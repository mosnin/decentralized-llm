"""
Tests for node/activation_stream.py — direct P2P activation streaming.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

torch = pytest.importorskip("torch")

from node.activation_stream import (  # noqa: E402
    _HEADER_FMT,
    ActivationClient,
    ActivationServer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _start_server(port: int) -> ActivationServer:
    """Create and start an ActivationServer on the given port."""
    server = ActivationServer(host="127.0.0.1", port=port)
    await server.start()
    return server


def _rand_tensor(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape).to(dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSendReceiveFloat32:
    """test_send_receive_float32: send a float32 tensor and verify it matches."""

    async def test_send_receive_float32(self):
        server = await _start_server(19100)
        client = ActivationClient("127.0.0.1", 19100)
        original = _rand_tensor((2, 4, 8), torch.float32)

        receive_task = asyncio.create_task(server.receive(job_id=1))
        await asyncio.sleep(0)  # let the future get registered

        await client.send(job_id=1, tensor=original)
        received = await asyncio.wait_for(receive_task, timeout=5.0)

        assert received.shape == original.shape
        assert received.dtype == original.dtype
        assert torch.allclose(original, received)

        await server.stop()


class TestSendReceiveFloat16:
    """test_send_receive_float16: send a float16 tensor and verify it matches."""

    async def test_send_receive_float16(self):
        server = await _start_server(19101)
        client = ActivationClient("127.0.0.1", 19101)
        original = _rand_tensor((3, 16), torch.float16)

        receive_task = asyncio.create_task(server.receive(job_id=2))
        await asyncio.sleep(0)

        await client.send(job_id=2, tensor=original)
        received = await asyncio.wait_for(receive_task, timeout=5.0)

        assert received.shape == original.shape
        assert received.dtype == torch.float16
        assert torch.allclose(original, received)

        await server.stop()


class TestTimeoutRaises:
    """test_timeout_raises: no sender within 0.05 s → TimeoutError."""

    async def test_timeout_raises(self):
        server = await _start_server(19102)

        with pytest.raises(TimeoutError):
            await server.receive(job_id=999, timeout=0.05)

        await server.stop()


class TestMultipleJobsConcurrent:
    """test_multiple_jobs_concurrent: two jobs in parallel, no cross-contamination."""

    async def test_multiple_jobs_concurrent(self):
        server = await _start_server(19103)
        client = ActivationClient("127.0.0.1", 19103)

        t_a = _rand_tensor((2, 4), torch.float32)
        t_b = _rand_tensor((5, 3), torch.float32)

        receive_a = asyncio.create_task(server.receive(job_id=10))
        receive_b = asyncio.create_task(server.receive(job_id=20))
        await asyncio.sleep(0)  # let both futures be registered

        await asyncio.gather(
            client.send(job_id=10, tensor=t_a),
            client.send(job_id=20, tensor=t_b),
        )

        got_a, got_b = await asyncio.gather(
            asyncio.wait_for(receive_a, timeout=5.0),
            asyncio.wait_for(receive_b, timeout=5.0),
        )

        # Shapes must match their respective senders — no cross-contamination
        assert got_a.shape == t_a.shape
        assert got_b.shape == t_b.shape
        assert torch.allclose(t_a, got_a)
        assert torch.allclose(t_b, got_b)

        await server.stop()


class TestMagicHeaderValidation:
    """test_magic_header_validation: corrupt magic bytes → server rejects gracefully."""

    async def test_magic_header_validation(self):
        server = await _start_server(19104)

        # Register a waiter — it should NOT be resolved by the bad frame.
        receive_task = asyncio.create_task(server.receive(job_id=42, timeout=0.3))
        await asyncio.sleep(0)

        # Manually open a connection and send a frame with corrupted magic.
        reader, writer = await asyncio.open_connection("127.0.0.1", 19104)
        bad_magic = 0xDEADBEEF
        bad_header = struct.pack(_HEADER_FMT, bad_magic, 42, 0)
        writer.write(bad_header)
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        # The server should log a warning and close the connection without
        # resolving the future — which means we get a TimeoutError.
        with pytest.raises(TimeoutError):
            await receive_task

        await server.stop()
