import asyncio
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class WebhookRegistration:
    url: str
    secret: str  # HMAC-SHA256 signing secret
    job_id: int
    registered_at: float = field(default_factory=time.time)
    max_retries: int = 3


@dataclass
class WebhookEvent:
    event_type: str  # "job.completed" | "job.failed"
    job_id: int
    payload: dict
    timestamp: float = field(default_factory=time.time)


class WebhookDelivery:
    """Handles signing and delivery of webhook events."""

    @staticmethod
    def sign(payload_bytes: bytes, secret: str) -> str:
        """Return HMAC-SHA256 hex signature."""
        return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    @staticmethod
    def verify(payload_bytes: bytes, secret: str, signature: str) -> bool:
        """Constant-time signature verification."""
        expected = WebhookDelivery.sign(payload_bytes, secret)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    async def deliver(registration: WebhookRegistration, event: WebhookEvent) -> bool:
        """
        POST the event to registration.url with HMAC signature header.
        Retries up to max_retries times with 1s, 2s, 4s backoff.
        Returns True on success, False after exhausting retries.
        Uses urllib (stdlib) to make the HTTP request in a thread executor.
        """
        payload = json.dumps(
            {
                "event_type": event.event_type,
                "job_id": event.job_id,
                "timestamp": event.timestamp,
                **event.payload,
            }
        ).encode("utf-8")

        signature = WebhookDelivery.sign(payload, registration.secret)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Job-Id": str(event.job_id),
        }

        def _post():
            req = urllib.request.Request(
                registration.url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status

        loop = asyncio.get_event_loop()
        for attempt in range(registration.max_retries):
            try:
                status = await loop.run_in_executor(None, _post)
                if 200 <= status < 300:
                    return True
            except Exception:
                pass
            if attempt < registration.max_retries - 1:
                await asyncio.sleep(2**attempt)
        return False


class WebhookRegistry:
    """Tracks active webhook registrations per job."""

    def __init__(self):
        self._registrations: dict[int, list[WebhookRegistration]] = {}

    def register(
        self, job_id: int, url: str, secret: str, max_retries: int = 3
    ) -> WebhookRegistration:
        reg = WebhookRegistration(url=url, secret=secret, job_id=job_id, max_retries=max_retries)
        self._registrations.setdefault(job_id, []).append(reg)
        return reg

    def get(self, job_id: int) -> list[WebhookRegistration]:
        return list(self._registrations.get(job_id, []))

    def remove(self, job_id: int) -> None:
        self._registrations.pop(job_id, None)

    def __len__(self) -> int:
        return sum(len(v) for v in self._registrations.values())
