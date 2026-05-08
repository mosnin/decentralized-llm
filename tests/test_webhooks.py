import asyncio
from unittest.mock import MagicMock, patch

from node.webhooks import WebhookDelivery, WebhookEvent, WebhookRegistration, WebhookRegistry

# ---------------------------------------------------------------------------
# sign / verify
# ---------------------------------------------------------------------------


def test_sign_produces_hex_string():
    sig = WebhookDelivery.sign(b"hello", "secret")
    assert isinstance(sig, str)
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)


def test_verify_valid_signature():
    payload = b"test payload"
    secret = "mysecret"
    sig = WebhookDelivery.sign(payload, secret)
    assert WebhookDelivery.verify(payload, secret, sig) is True


def test_verify_invalid_signature():
    payload = b"test payload"
    sig = WebhookDelivery.sign(payload, "correct_secret")
    assert WebhookDelivery.verify(payload, "wrong_secret", sig) is False


# ---------------------------------------------------------------------------
# WebhookRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    registry = WebhookRegistry()
    reg = registry.register(job_id=1, url="http://example.com/hook", secret="s")
    result = registry.get(1)
    assert len(result) == 1
    assert result[0] is reg
    assert isinstance(reg, WebhookRegistration)


def test_registry_get_unknown_returns_empty():
    registry = WebhookRegistry()
    assert registry.get(999) == []


def test_registry_remove():
    registry = WebhookRegistry()
    registry.register(job_id=5, url="http://example.com/hook", secret="s")
    registry.remove(5)
    assert registry.get(5) == []


def test_registry_len():
    registry = WebhookRegistry()
    registry.register(job_id=1, url="http://a.com/1", secret="s1")
    registry.register(job_id=1, url="http://a.com/2", secret="s2")
    registry.register(job_id=2, url="http://b.com/1", secret="s3")
    assert len(registry) == 3


# ---------------------------------------------------------------------------
# WebhookDelivery.deliver
# ---------------------------------------------------------------------------


def _make_mock_response(status: int) -> MagicMock:
    """Return a context-manager mock that reports the given HTTP status."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_registration(url: str = "http://example.com/hook", max_retries: int = 3):
    return WebhookRegistration(url=url, secret="secret", job_id=42, max_retries=max_retries)


def _make_event():
    return WebhookEvent(event_type="job.completed", job_id=42, payload={"result": "ok"})


def test_deliver_success():
    reg = _make_registration()
    event = _make_event()

    async def _fake_sleep(_delay):
        pass

    mock_resp = _make_mock_response(200)
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        with patch("asyncio.sleep", side_effect=_fake_sleep):
            result = asyncio.run(WebhookDelivery.deliver(reg, event))

    assert result is True
    assert mock_urlopen.call_count == 1


def test_deliver_retries_on_failure():
    reg = _make_registration(max_retries=3)
    event = _make_event()

    call_count = 0

    def _urlopen_side_effect(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OSError("connection refused")
        return _make_mock_response(200)

    async def _fake_sleep(_delay):
        pass

    with patch("urllib.request.urlopen", side_effect=_urlopen_side_effect):
        with patch("asyncio.sleep", side_effect=_fake_sleep):
            result = asyncio.run(WebhookDelivery.deliver(reg, event))

    assert result is True
    assert call_count == 3


def test_deliver_exhausts_retries():
    reg = _make_registration(max_retries=3)
    event = _make_event()

    async def _fake_sleep(_delay):
        pass

    with patch("urllib.request.urlopen", side_effect=OSError("always fails")):
        with patch("asyncio.sleep", side_effect=_fake_sleep):
            result = asyncio.run(WebhookDelivery.deliver(reg, event))

    assert result is False
