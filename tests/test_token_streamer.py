"""
Unit tests for node.token_streamer — TokenStream and TokenStreamRegistry.
"""

import asyncio

from node.token_streamer import TokenStream, TokenStreamRegistry

# ---------------------------------------------------------------------------
# TokenStream tests
# ---------------------------------------------------------------------------


class TestTokenStream:
    def test_token_stream_push_and_iterate(self):
        """Push 3 tokens + finish; iterating yields exactly those 3 tokens."""

        async def _run():
            stream = TokenStream(job_id=1)
            await stream.push("hello")
            await stream.push(" world")
            await stream.push("!")
            await stream.finish()

            tokens = []
            async for token in stream:
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_run())
        assert tokens == ["hello", " world", "!"]

    def test_token_stream_finish_closes_iterator(self):
        """After finish(), the async iterator stops immediately."""

        async def _run():
            stream = TokenStream(job_id=2)
            await stream.finish()

            tokens = []
            async for token in stream:
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_run())
        assert tokens == []

    def test_token_stream_finish_with_error_still_closes(self):
        """finish(error=...) still sends the sentinel so the iterator stops."""

        async def _run():
            stream = TokenStream(job_id=3)
            await stream.push("partial")
            await stream.finish(error="something went wrong")

            tokens = []
            async for token in stream:
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_run())
        assert tokens == ["partial"]


# ---------------------------------------------------------------------------
# TokenStreamRegistry tests
# ---------------------------------------------------------------------------


class TestTokenStreamRegistry:
    def test_token_stream_registry_create_and_get(self):
        """create() returns a stream and get() returns the same object."""
        registry = TokenStreamRegistry()
        stream = registry.create(job_id=10)
        assert stream is not None
        assert stream.job_id == 10
        assert registry.get(10) is stream

    def test_token_stream_registry_get_missing_returns_none(self):
        """get() on an unknown id returns None."""
        registry = TokenStreamRegistry()
        assert registry.get(999) is None

    def test_token_stream_registry_remove(self):
        """After remove(), get() returns None."""
        registry = TokenStreamRegistry()
        registry.create(job_id=20)
        assert registry.get(20) is not None
        registry.remove(20)
        assert registry.get(20) is None

    def test_token_stream_registry_remove_missing_is_noop(self):
        """remove() on a non-existent id does not raise."""
        registry = TokenStreamRegistry()
        registry.remove(404)  # should not raise

    def test_token_stream_multiple_streams_independent(self):
        """Two streams with different job_ids do not interfere."""

        async def _run():
            registry = TokenStreamRegistry()
            s1 = registry.create(job_id=1)
            s2 = registry.create(job_id=2)

            await s1.push("a")
            await s2.push("x")
            await s1.push("b")
            await s2.push("y")
            await s1.finish()
            await s2.finish()

            tokens1 = [t async for t in s1]
            tokens2 = [t async for t in s2]
            return tokens1, tokens2

        t1, t2 = asyncio.run(_run())
        assert t1 == ["a", "b"]
        assert t2 == ["x", "y"]
