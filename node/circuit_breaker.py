"""Async circuit breaker for fault-tolerant service calls (stdlib only)."""

import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is blocked because the circuit is OPEN."""


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # failures before opening
    success_threshold: int = 2  # successes in HALF_OPEN before closing
    timeout_seconds: float = 60.0  # how long to stay OPEN before trying HALF_OPEN
    # which exceptions count as failures
    exceptions: tuple = field(default_factory=lambda: (Exception,))


class CircuitBreaker:
    """
    Async circuit breaker. Wrap async calls with .call(coro_factory).

    Usage:
        cb = CircuitBreaker(config)
        result = await cb.call(lambda: some_async_func(args))
    """

    def __init__(self, config: CircuitBreakerConfig | None = None, name: str = ""):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Return current state, transitioning OPEN→HALF_OPEN if timeout elapsed."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.config.timeout_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state

    async def call(self, coro_factory: Callable[[], Coroutine]) -> Any:
        """
        Execute coro_factory() if circuit allows it.
        - OPEN → raise CircuitOpenError immediately
        - CLOSED / HALF_OPEN → execute; on success record success; on failure record failure
        """
        current = self.state
        if current == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")

        try:
            result = await coro_factory()
            self._on_success()
            return result
        except self.config.exceptions as exc:
            self._on_failure()
            raise exc

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually close the circuit (e.g., after maintenance)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }
