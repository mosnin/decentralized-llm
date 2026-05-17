"""Exponential backoff retry utilities for the client SDK."""

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.exponential_base = exponential_base
        self.jitter = jitter


class RetryableError(Exception):
    """Wrap a transient error to signal the retry loop should retry."""


async def with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    config: RetryConfig | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (TimeoutError, OSError),
) -> Any:
    """
    Call coro_factory() up to config.max_attempts times, with exponential backoff
    between attempts. Raises the last exception if all attempts fail.

    coro_factory is called fresh each attempt (not the same coroutine object).
    delay = min(base_delay * exponential_base**(attempt-1), max_delay)
    with ±20% jitter if config.jitter is True.
    """
    if config is None:
        config = RetryConfig()

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await coro_factory()
        except RetryableError as exc:
            last_exc = exc.__cause__ or exc
        except retryable_exceptions as exc:  # type: ignore[misc]
            last_exc = exc
        except Exception:
            raise

        if attempt < config.max_attempts:
            delay = min(
                config.base_delay_s * (config.exponential_base ** (attempt - 1)),
                config.max_delay_s,
            )
            if config.jitter:
                delay *= 1.0 + random.uniform(-0.2, 0.2)
            logger.debug(
                "Attempt %d/%d failed (%s). Retrying in %.2fs.",
                attempt,
                config.max_attempts,
                last_exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_exc
