"""Tests for ResultVerifier — result integrity and format checks."""

import hashlib

from node.verifier import ResultVerifier


def _make_hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# ─────────────────────────── verify_result ───────────────────────────────────


class TestVerifyResult:
    def test_valid_result_passes(self):
        result_bytes = b"The capital of France is Paris."
        claimed_hash = _make_hash(result_bytes)
        cid = "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"

        ok, reason = ResultVerifier.verify_result(result_bytes, claimed_hash, cid, max_tokens=100)

        assert ok is True
        assert reason == ""

    def test_wrong_hash_fails(self):
        result_bytes = b"Correct result text."
        wrong_hash = _make_hash(b"completely different content")
        cid = "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"

        ok, reason = ResultVerifier.verify_result(result_bytes, wrong_hash, cid, max_tokens=100)

        assert ok is False
        assert "hash" in reason.lower()

    def test_empty_result_fails(self):
        ok, reason = ResultVerifier.verify_result(
            b"",
            _make_hash(b""),
            "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354",
            max_tokens=100,
        )

        assert ok is False
        assert "empty" in reason.lower()

    def test_result_too_long_fails(self):
        # 100-word sentence, estimate_tokens → ceil(100 * 1.3) = 130 > max_tokens=50
        words = " ".join(f"word{i}" for i in range(100))
        result_bytes = words.encode()
        claimed_hash = _make_hash(result_bytes)
        cid = "bafkreibm6jg3ux5qumhcn36dbhygngv6wzlzx6lxzxvfbwmhbkf2x6aee"

        ok, reason = ResultVerifier.verify_result(result_bytes, claimed_hash, cid, max_tokens=50)

        assert ok is False
        assert "max_tokens" in reason or "token" in reason.lower()

    def test_invalid_cid_format_fails(self):
        result_bytes = b"Some valid inference output."
        claimed_hash = _make_hash(result_bytes)
        bad_cid = "not-a-real-cid-12345"

        ok, reason = ResultVerifier.verify_result(
            result_bytes, claimed_hash, bad_cid, max_tokens=100
        )

        assert ok is False
        assert "cid" in reason.lower()

    def test_invalid_utf8_fails(self):
        result_bytes = b"\xff\xfe invalid utf-8 sequence"
        claimed_hash = _make_hash(result_bytes)
        cid = "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"

        ok, reason = ResultVerifier.verify_result(result_bytes, claimed_hash, cid, max_tokens=100)

        assert ok is False
        assert "utf" in reason.lower() or "unicode" in reason.lower()


# ─────────────────────────── verify_cid_format ───────────────────────────────


class TestVerifyCidFormat:
    def test_valid_cid_formats(self):
        """bafy..., bafk..., and Qm... prefixes are all accepted."""
        assert ResultVerifier.verify_cid_format(
            "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
        )
        assert ResultVerifier.verify_cid_format(
            "bafkreibm6jg3ux5qumhcn36dbhygngv6wzlzx6lxzxvfbwmhbkf2x6aee"
        )
        assert ResultVerifier.verify_cid_format("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_empty_cid_rejected(self):
        assert ResultVerifier.verify_cid_format("") is False

    def test_random_string_rejected(self):
        assert ResultVerifier.verify_cid_format("not-a-cid") is False
        assert ResultVerifier.verify_cid_format("http://example.com/hash") is False

    def test_wrong_prefix_rejected(self):
        # starts with "bafc" — not one of the accepted prefixes
        assert ResultVerifier.verify_cid_format("bafcfoo123") is False


# ─────────────────────────── estimate_tokens ─────────────────────────────────


class TestEstimateTokens:
    def test_estimate_tokens_approximation(self):
        """A 10-word sentence should give approximately 13 tokens (within ±3)."""
        sentence = "The quick brown fox jumps over the lazy dog now"
        assert sentence.split().__len__() == 10  # sanity-check fixture

        estimate = ResultVerifier.estimate_tokens(sentence)

        assert abs(estimate - 13) <= 3, f"expected ~13, got {estimate}"

    def test_empty_string_gives_zero(self):
        assert ResultVerifier.estimate_tokens("") == 0

    def test_single_word(self):
        # ceil(1 * 1.3) = 2
        assert ResultVerifier.estimate_tokens("hello") == 2

    def test_scaling(self):
        """More words → proportionally higher estimate."""
        short = ResultVerifier.estimate_tokens("one two three")
        long_ = ResultVerifier.estimate_tokens("one two three four five six")
        assert long_ > short
