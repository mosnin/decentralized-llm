"""Structured JSON logging for the decentralized-LLM node."""

import contextvars
import json
import logging
import uuid
from datetime import UTC, datetime

# Context var so correlation ID flows across async task boundaries
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


class CorrelationFilter(logging.Filter):
    """Injects correlation_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id.get("")
        if not cid:
            cid = f"auto-{uuid.uuid4().hex[:8]}"
        record.correlation_id = cid  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON.

    Fields emitted: timestamp, level, logger, message, correlation_id, **extra.
    All extra kwargs passed to logger.info(..., extra={...}) appear as
    top-level JSON fields.
    """

    # Keys that are standard LogRecord attributes and should *not* be promoted
    # as user-supplied extra fields.
    _RESERVED = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "taskName",
            "correlation_id",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "correlation_id": getattr(record, "correlation_id", ""),
        }

        # Promote any extra fields the caller attached
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure root logger with CorrelationFilter + JSONFormatter.

    Falls back to standard text format if json_output=False (useful for tests).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationFilter())

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )

    root.addHandler(handler)


def set_correlation_id(cid: str) -> None:
    """Set the current correlation ID in the context var."""
    correlation_id.set(cid)


def get_correlation_id() -> str:
    """Get current correlation ID, generating a short UUID if unset."""
    cid = correlation_id.get("")
    if not cid:
        cid = f"auto-{uuid.uuid4().hex[:8]}"
        correlation_id.set(cid)
    return cid
