"""
Tests for node/logging_config.py — structured JSON logging with correlation IDs.

All tests are pure-Python / asyncio; no external dependencies required.
"""

import asyncio
import json
import logging

import pytest

from node.logging_config import (
    CorrelationFilter,
    JSONFormatter,
    configure_logging,
    correlation_id,
    get_correlation_id,
    set_correlation_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(msg: str = "hello", level: int = logging.INFO, **kwargs) -> logging.LogRecord:
    """Create a minimal LogRecord for testing."""
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="test_logging_config.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# Test: JSON formatter produces valid JSON
# ---------------------------------------------------------------------------


class TestJSONFormatterProducesValidJSON:
    def test_json_formatter_produces_valid_json(self):
        formatter = JSONFormatter()
        record = _make_record("test message", correlation_id="abc-123")
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert "timestamp" in parsed
        assert "correlation_id" in parsed

    def test_json_formatter_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = _make_record("msg with extra", correlation_id="x1")
        record.job_id = 42
        record.node_id = "node-5"
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["job_id"] == 42
        assert parsed["node_id"] == "node-5"

    def test_json_formatter_single_line(self):
        formatter = JSONFormatter()
        record = _make_record("single line test", correlation_id="")
        output = formatter.format(record)
        assert "\n" not in output


# ---------------------------------------------------------------------------
# Test: correlation filter injects ID
# ---------------------------------------------------------------------------


class TestCorrelationFilterInjectsID:
    def test_correlation_filter_injects_id(self):
        set_correlation_id("filter-test-id")
        filt = CorrelationFilter()
        record = _make_record("checking filter")
        result = filt.filter(record)

        assert result is True
        assert record.correlation_id == "filter-test-id"

    def test_correlation_filter_returns_true(self):
        """filter() must always return True so records are not dropped."""
        set_correlation_id("")
        filt = CorrelationFilter()
        record = _make_record("any message")
        assert filt.filter(record) is True


# ---------------------------------------------------------------------------
# Test: default correlation ID when unset
# ---------------------------------------------------------------------------


class TestCorrelationIDDefaultWhenUnset:
    def test_correlation_id_default_when_unset(self):
        """When correlation_id context var is empty, filter injects an auto-generated ID."""
        # Reset to empty
        correlation_id.set("")
        filt = CorrelationFilter()
        record = _make_record("default id test")
        filt.filter(record)

        assert record.correlation_id != ""
        # auto-generated IDs start with "auto-"
        assert record.correlation_id.startswith("auto-")


# ---------------------------------------------------------------------------
# Test: set and get correlation ID
# ---------------------------------------------------------------------------


class TestSetAndGetCorrelationID:
    def test_set_and_get_correlation_id(self):
        set_correlation_id("my-custom-id")
        assert get_correlation_id() == "my-custom-id"

    def test_get_correlation_id_generates_when_empty(self):
        correlation_id.set("")
        cid = get_correlation_id()
        assert cid != ""
        assert cid.startswith("auto-")

    def test_get_correlation_id_consistent_after_set(self):
        set_correlation_id("stable-id-123")
        assert get_correlation_id() == "stable-id-123"
        assert get_correlation_id() == "stable-id-123"


# ---------------------------------------------------------------------------
# Test: configure_logging text mode
# ---------------------------------------------------------------------------


class TestConfigureLoggingTextMode:
    def test_configure_logging_text_mode(self):
        configure_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        # In text mode the formatter is a plain Formatter, not JSONFormatter
        assert not isinstance(handler.formatter, JSONFormatter)
        assert root.level == logging.DEBUG

    def test_configure_logging_text_mode_handler_has_filter(self):
        configure_logging(level="WARNING", json_output=False)
        root = logging.getLogger()
        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert CorrelationFilter in filter_types


# ---------------------------------------------------------------------------
# Test: configure_logging JSON mode
# ---------------------------------------------------------------------------


class TestConfigureLoggingJSONMode:
    def test_configure_logging_json_mode(self):
        configure_logging(level="INFO", json_output=True)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)
        assert root.level == logging.INFO

    def test_configure_logging_json_mode_produces_json(self, caplog):
        configure_logging(level="INFO", json_output=True)
        set_correlation_id("json-mode-test")
        logger = logging.getLogger("test.json_mode")

        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(CorrelationFilter())
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

        logger.info("structured output test")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["message"] == "structured output test"
        assert parsed["correlation_id"] == "json-mode-test"

        # Cleanup
        logger.removeHandler(handler)
        logger.propagate = True


# ---------------------------------------------------------------------------
# Test: correlation ID isolated between async tasks
# ---------------------------------------------------------------------------


class TestCorrelationIDIsolatedBetweenAsyncTasks:
    @pytest.mark.asyncio
    async def test_correlation_id_isolated_between_async_tasks(self):
        """Two concurrent tasks must each see their own correlation ID."""
        results: dict[str, str] = {}

        async def task_a():
            set_correlation_id("task-A-id")
            await asyncio.sleep(0)  # yield so task_b can run
            results["a"] = get_correlation_id()

        async def task_b():
            set_correlation_id("task-B-id")
            await asyncio.sleep(0)
            results["b"] = get_correlation_id()

        await asyncio.gather(task_a(), task_b())

        assert results["a"] == "task-A-id", f"Task A saw: {results['a']}"
        assert results["b"] == "task-B-id", f"Task B saw: {results['b']}"

    @pytest.mark.asyncio
    async def test_correlation_id_not_shared_across_tasks(self):
        """Setting a correlation ID in one task must not bleed into another."""
        seen_in_b: list[str] = []

        async def setter():
            set_correlation_id("bleeder-id")
            await asyncio.sleep(0.01)

        async def observer():
            # Reset so we start clean; any bleed would show up here
            correlation_id.set("")
            await asyncio.sleep(0)
            cid = get_correlation_id()
            seen_in_b.append(cid)

        await asyncio.gather(setter(), observer())

        # observer started with empty ID so it should have gotten an auto-ID, not "bleeder-id"
        assert seen_in_b[0] != "bleeder-id", (
            "Correlation ID bled from setter task into observer task"
        )
