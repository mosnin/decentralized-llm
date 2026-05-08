"""
Token streaming support for SSE-based inference endpoints.

TokenStream holds an asyncio.Queue that the inference engine writes tokens
into and HTTP clients drain via async iteration.  TokenStreamRegistry maps
job_id → active TokenStream so the API gateway can look streams up by ID.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class TokenStream:
    job_id: int
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _done: bool = False

    async def push(self, token: str) -> None:
        """Called by the inference engine to emit a token."""
        await self._queue.put(token)

    async def finish(self, error: str | None = None) -> None:
        """Signal end of stream."""
        self._done = True
        await self._queue.put(None)  # sentinel

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            token = await self._queue.get()
            if token is None:
                break
            yield token


class TokenStreamRegistry:
    """Maps job_id → active TokenStream.  Thread-safe via asyncio."""

    def __init__(self) -> None:
        self._streams: dict[int, TokenStream] = {}

    def create(self, job_id: int) -> TokenStream:
        stream = TokenStream(job_id=job_id)
        self._streams[job_id] = stream
        return stream

    def get(self, job_id: int) -> TokenStream | None:
        return self._streams.get(job_id)

    def remove(self, job_id: int) -> None:
        self._streams.pop(job_id, None)
