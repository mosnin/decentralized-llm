import time

from node.result_cache import ResultCache


def test_get_miss_returns_none():
    cache = ResultCache()
    assert cache.get("gpt2", "hello world") is None


def test_put_and_get_returns_result():
    cache = ResultCache()
    cache.put("gpt2", "hello world", "the response")
    assert cache.get("gpt2", "hello world") == "the response"


def test_expired_entry_returns_none():
    cache = ResultCache(ttl_seconds=0.1)
    cache.put("gpt2", "hello", "response")
    time.sleep(0.15)
    assert cache.get("gpt2", "hello") is None


def test_lru_eviction():
    cache = ResultCache(max_size=2)
    cache.put("model", "prompt1", "result1")
    cache.put("model", "prompt2", "result2")
    cache.put("model", "prompt3", "result3")  # should evict prompt1
    assert cache.get("model", "prompt1") is None
    assert cache.get("model", "prompt2") == "result2"
    assert cache.get("model", "prompt3") == "result3"


def test_lru_order_updated_on_get():
    cache = ResultCache(max_size=2)
    cache.put("model", "prompt1", "result1")
    cache.put("model", "prompt2", "result2")
    # Access prompt1 so it becomes most recently used
    cache.get("model", "prompt1")
    # Adding prompt3 should evict prompt2 (LRU), not prompt1
    cache.put("model", "prompt3", "result3")
    assert cache.get("model", "prompt1") == "result1"
    assert cache.get("model", "prompt2") is None
    assert cache.get("model", "prompt3") == "result3"


def test_hit_count_increments():
    cache = ResultCache()
    cache.put("gpt2", "hello", "response")
    cache.get("gpt2", "hello")
    cache.get("gpt2", "hello")
    key = ResultCache.make_key("gpt2", "hello")
    assert cache._cache[key].hit_count == 2


def test_invalidate_existing():
    cache = ResultCache()
    cache.put("gpt2", "hello", "response")
    assert cache.invalidate("gpt2", "hello") is True
    assert cache.get("gpt2", "hello") is None


def test_invalidate_missing():
    cache = ResultCache()
    assert cache.invalidate("gpt2", "nonexistent") is False


def test_clear_empties_cache():
    cache = ResultCache()
    cache.put("gpt2", "prompt1", "result1")
    cache.put("gpt2", "prompt2", "result2")
    cache.clear()
    assert cache.stats()["size"] == 0


def test_stats_returns_dict():
    cache = ResultCache(max_size=50, ttl_seconds=120.0)
    cache.put("gpt2", "hello", "response")
    cache.get("gpt2", "hello")
    s = cache.stats()
    assert "size" in s
    assert "max_size" in s
    assert "ttl_seconds" in s
    assert "total_hits" in s
    assert s["size"] == 1
    assert s["max_size"] == 50
    assert s["ttl_seconds"] == 120.0
    assert s["total_hits"] == 1


def test_same_model_different_prompt():
    cache = ResultCache()
    cache.put("gpt2", "prompt_a", "result_a")
    cache.put("gpt2", "prompt_b", "result_b")
    assert cache.get("gpt2", "prompt_a") == "result_a"
    assert cache.get("gpt2", "prompt_b") == "result_b"
    assert cache.stats()["size"] == 2


def test_same_prompt_different_model():
    cache = ResultCache()
    cache.put("model_a", "hello", "result_a")
    cache.put("model_b", "hello", "result_b")
    assert cache.get("model_a", "hello") == "result_a"
    assert cache.get("model_b", "hello") == "result_b"
    assert cache.stats()["size"] == 2
