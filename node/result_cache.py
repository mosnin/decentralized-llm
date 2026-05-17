import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CachedResult:
    result_text: str
    created_at: float
    hit_count: int = field(default=0)


class ResultCache:
    """
    LRU cache for inference results, keyed by (model_name, prompt_hash).

    Eviction policy: LRU when max_size is reached; TTL-based expiry.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()

    @staticmethod
    def make_key(model_name: str, prompt: str) -> str:
        """SHA-256 hex of 'model_name:prompt'."""
        raw = f"{model_name}:{prompt}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, model_name: str, prompt: str) -> str | None:
        """Return cached result or None if missing/expired."""
        key = self.make_key(model_name, prompt)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        return entry.result_text

    def put(self, model_name: str, prompt: str, result_text: str) -> None:
        """Store a result. Evicts LRU entry if at capacity."""
        key = self.make_key(model_name, prompt)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].result_text = result_text
            self._cache[key].created_at = time.time()
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # evict LRU (first item)
        self._cache[key] = CachedResult(result_text=result_text, created_at=time.time())

    def invalidate(self, model_name: str, prompt: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        key = self.make_key(model_name, prompt)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "total_hits": total_hits,
        }
