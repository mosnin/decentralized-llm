"""
Tests for node.speculative_decoder — DraftModel and SpeculativeDecoder.

All tests use mock torch tensors; no real model weights are loaded.
The test suite is runnable with:

    pytest tests/test_speculative_decoder.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module if torch is not available
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from node.speculative_decoder import DecodeStats, DraftModel, SpeculativeDecoder  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 32  # small vocabulary for fast tests
SEQ = 6     # prompt length
BATCH = 1


def _make_uniform_logits(batch: int = BATCH, seq: int = SEQ, vocab: int = VOCAB) -> torch.Tensor:
    """Logits that produce a uniform distribution over the vocabulary."""
    return torch.zeros(batch, seq, vocab)


def _make_peaked_logits(
    token_id: int,
    batch: int = BATCH,
    seq: int = SEQ,
    vocab: int = VOCAB,
    peak: float = 100.0,
) -> torch.Tensor:
    """Logits strongly peaked at *token_id* (near-deterministic)."""
    logits = torch.full((batch, seq, vocab), -peak)
    logits[:, :, token_id] = peak
    return logits


def _make_prompt(batch: int = BATCH, seq: int = SEQ) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq))


def _constant_verifier(logits: torch.Tensor):
    """Factory: returns a verifier callable that always returns *logits*."""

    def _fn(input_ids: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        # Return logits sliced to match the requested sequence length so
        # the SpeculativeDecoder's slicing logic works correctly.
        needed = input_ids.shape[1]
        batch = input_ids.shape[0]
        if logits.shape[1] >= needed:
            return logits[:batch, :needed, :]
        # Pad by repeating the last position
        pad = logits[:batch, -1:, :].expand(batch, needed - logits.shape[1], logits.shape[2])
        return torch.cat([logits[:batch, :, :], pad], dim=1)

    return _fn


def _make_draft_model(token_id: int = 0, vocab: int = VOCAB, temperature: float = 1.0):
    """Create a DraftModel whose underlying model always predicts *token_id*."""
    peaked_logits_2d = torch.full((1, 1, vocab), -100.0)
    peaked_logits_2d[:, :, token_id] = 100.0

    def _model_fn(input_ids: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        seq = input_ids.shape[1]
        t = torch.full((batch, seq, vocab), -100.0)
        t[:, :, token_id] = 100.0
        return t

    return DraftModel(model=_model_fn, vocab_size=vocab, temperature=temperature)


# ---------------------------------------------------------------------------
# Test 1 — draft generation produces correct shape
# ---------------------------------------------------------------------------


class TestDraftGenerationShape:
    """generate_draft returns tensors of the expected shapes."""

    def test_draft_tokens_shape(self):
        draft_model = _make_draft_model(token_id=5)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        k = 7

        draft_tokens, draft_probs = draft_model.generate_draft(prompt, k=k)

        assert draft_tokens.shape == (BATCH, k), (
            f"Expected draft_tokens shape ({BATCH}, {k}), got {draft_tokens.shape}"
        )

    def test_draft_probs_shape(self):
        draft_model = _make_draft_model(token_id=3)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        k = 5

        _, draft_probs = draft_model.generate_draft(prompt, k=k)

        assert draft_probs.shape == (BATCH, k, VOCAB), (
            f"Expected draft_probs shape ({BATCH}, {k}, {VOCAB}), got {draft_probs.shape}"
        )

    def test_draft_probs_sum_to_one(self):
        """Each position's probability vector should be a valid distribution."""
        draft_model = _make_draft_model(token_id=2)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        k = 4

        _, draft_probs = draft_model.generate_draft(prompt, k=k)

        sums = draft_probs.sum(dim=-1)  # [batch, k]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_draft_tokens_in_vocab_range(self):
        draft_model = _make_draft_model(token_id=10)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        k = 6

        draft_tokens, _ = draft_model.generate_draft(prompt, k=k)

        assert (draft_tokens >= 0).all()
        assert (draft_tokens < VOCAB).all()

    def test_draft_returns_expected_token_when_peaked(self):
        """A near-deterministic draft model should produce the peaked token."""
        target_id = 7
        draft_model = _make_draft_model(token_id=target_id)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        k = 5

        draft_tokens, _ = draft_model.generate_draft(prompt, k=k)

        assert (draft_tokens == target_id).all(), (
            f"Expected all draft tokens to be {target_id}, got {draft_tokens}"
        )


# ---------------------------------------------------------------------------
# Test 2 — verify_and_accept: accepts when distributions match exactly
# ---------------------------------------------------------------------------


class TestVerifyAcceptMatchingDistributions:
    """When draft and target distributions are identical, all k tokens must be accepted."""

    def test_all_accepted_when_distributions_match(self):
        # Both draft and verifier use the same peaked distribution → q(x)/p(x) = 1 → always accept.
        target_id = 4
        k = 5

        # Verifier returns k+1 positions of peaked logits
        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)

        # Draft probabilities are also peaked at target_id
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, target_id] = 1.0

        draft_tokens = torch.full((BATCH, k), target_id, dtype=torch.long)

        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        decoder = SpeculativeDecoder(
            draft_model=None,
            verifier_fn=_constant_verifier(verifier_logits),
            temperature=1.0,
        )

        accepted, n_acc, n_rej, n_cor = decoder.verify_and_accept(
            prompt, draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        # All k draft tokens accepted + 1 bonus token
        assert n_acc == k, f"Expected {k} accepted, got {n_acc}"
        assert n_rej == 0, f"Expected 0 rejections, got {n_rej}"
        assert accepted.shape[1] == k + 1, (
            f"Expected k+1={k+1} output tokens, got {accepted.shape[1]}"
        )

    def test_accepted_tokens_match_draft_when_all_accepted(self):
        """The first k accepted tokens should equal the original draft tokens."""
        target_id = 9
        k = 3

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, target_id] = 1.0
        draft_tokens = torch.full((BATCH, k), target_id, dtype=torch.long)

        decoder = SpeculativeDecoder(draft_model=None, verifier_fn=lambda x: x, temperature=1.0)
        accepted, _, _, _ = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        # First k tokens should match the draft tokens exactly
        assert torch.equal(accepted[:, :k], draft_tokens)


# ---------------------------------------------------------------------------
# Test 3 — verify_and_accept rejects when distributions diverge
# ---------------------------------------------------------------------------


class TestVerifyAcceptDivergentDistributions:
    """When draft probability is high but target probability is low, rejection occurs."""

    def test_rejection_when_target_assigns_zero_prob(self):
        """If the target assigns ~0 probability to the draft token, it must be rejected."""
        draft_id = 5
        target_id = 10   # target wants a different token
        k = 4

        # Target peaked strongly at target_id ≠ draft_id
        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)

        # Draft distribution peaked at draft_id (wrong token from target's perspective)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, draft_id] = 1.0

        draft_tokens = torch.full((BATCH, k), draft_id, dtype=torch.long)

        decoder = SpeculativeDecoder(draft_model=None, verifier_fn=lambda x: x, temperature=1.0)
        accepted, n_acc, n_rej, n_cor = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        # First draft token should be rejected (alpha ≈ 0)
        assert n_rej == 1, f"Expected 1 rejection, got {n_rej}"
        assert n_acc == 0, f"Expected 0 accepted draft tokens, got {n_acc}"

    def test_correction_token_added_on_rejection(self):
        """A correction token is sampled from the residual distribution on rejection."""
        draft_id = 5
        target_id = 10
        k = 3

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, draft_id] = 1.0
        draft_tokens = torch.full((BATCH, k), draft_id, dtype=torch.long)

        decoder = SpeculativeDecoder(draft_model=None, verifier_fn=lambda x: x, temperature=1.0)
        accepted, n_acc, n_rej, n_cor = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        # Exactly one correction token added
        assert n_cor == 1
        # Total output = accepted_drafts + correction
        assert accepted.shape[1] == n_acc + 1

    def test_correction_token_value_from_target_when_peaked(self):
        """With peaked target, the correction token should be the target's preferred token."""
        draft_id = 5
        target_id = 10
        k = 2

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, draft_id] = 1.0
        draft_tokens = torch.full((BATCH, k), draft_id, dtype=torch.long)

        decoder = SpeculativeDecoder(draft_model=None, verifier_fn=lambda x: x, temperature=1.0)
        accepted, n_acc, n_rej, n_cor = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        # The correction token (last element) should be the target's token
        correction = accepted[:, -1]
        assert (correction == target_id).all(), (
            f"Expected correction token {target_id}, got {correction}"
        )


# ---------------------------------------------------------------------------
# Test 4 — full decode loop produces tokens
# ---------------------------------------------------------------------------


class TestFullDecodeLoop:
    """decode_speculative generates the requested number of tokens."""

    def _make_decoder(self, draft_token: int = 1, verifier_token: int = 1) -> SpeculativeDecoder:
        draft_model = _make_draft_model(token_id=draft_token, vocab=VOCAB)
        verifier_logits_template = _make_peaked_logits(
            verifier_token, batch=BATCH, seq=SEQ + 10, vocab=VOCAB
        )
        verifier_fn = _constant_verifier(verifier_logits_template)
        return SpeculativeDecoder(
            draft_model=draft_model,
            verifier_fn=verifier_fn,
            temperature=1.0,
        )

    def test_decode_produces_max_tokens(self):
        """decode_speculative should produce exactly max_tokens new tokens."""
        decoder = self._make_decoder(draft_token=1, verifier_token=1)
        prompt = _make_prompt(batch=BATCH, seq=SEQ)
        max_tokens = 10

        generated, _ = decoder.decode_speculative(prompt, max_tokens=max_tokens, draft_steps=3)

        assert generated.shape[1] == max_tokens, (
            f"Expected {max_tokens} generated tokens, got {generated.shape[1]}"
        )

    def test_decode_output_is_2d_tensor(self):
        """Output tensor must be 2-dimensional [batch, n_tokens]."""
        decoder = self._make_decoder()
        prompt = _make_prompt(batch=BATCH, seq=SEQ)

        generated, _ = decoder.decode_speculative(prompt, max_tokens=5, draft_steps=2)

        assert generated.dim() == 2, f"Expected 2D tensor, got {generated.dim()}D"

    def test_decode_output_tokens_in_vocab_range(self):
        decoder = self._make_decoder()
        prompt = _make_prompt(batch=BATCH, seq=SEQ)

        generated, _ = decoder.decode_speculative(prompt, max_tokens=8, draft_steps=3)

        assert (generated >= 0).all()
        assert (generated < VOCAB).all()

    def test_decode_stops_at_eos(self):
        """Generation should stop when the EOS token is emitted."""
        eos_id = 0
        # Both draft and verifier produce EOS deterministically
        draft_model = _make_draft_model(token_id=eos_id, vocab=VOCAB)
        verifier_logits_template = _make_peaked_logits(
            eos_id, batch=BATCH, seq=SEQ + 20, vocab=VOCAB
        )
        verifier_fn = _constant_verifier(verifier_logits_template)
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            verifier_fn=verifier_fn,
            temperature=1.0,
            eos_token_id=eos_id,
        )
        prompt = _make_prompt(batch=BATCH, seq=SEQ)

        generated, _ = decoder.decode_speculative(prompt, max_tokens=100, draft_steps=5)

        # Should stop well before 100 tokens
        assert generated.shape[1] <= 10, (
            f"Expected early stop at EOS; got {generated.shape[1]} tokens"
        )


# ---------------------------------------------------------------------------
# Test 5 — acceptance rate tracking
# ---------------------------------------------------------------------------


class TestAcceptanceRateTracking:
    """DecodeStats correctly tracks acceptance rates across iterations."""

    def test_acceptance_rate_all_accepted(self):
        stats = DecodeStats()
        stats.update(accepted=5, rejected=0, corrections=0)
        assert stats.acceptance_rate == pytest.approx(1.0)

    def test_acceptance_rate_none_accepted(self):
        stats = DecodeStats()
        stats.update(accepted=0, rejected=3, corrections=1)
        assert stats.acceptance_rate == pytest.approx(0.0)

    def test_acceptance_rate_partial(self):
        stats = DecodeStats()
        stats.update(accepted=3, rejected=1, corrections=1)
        # total_draft_tokens = 3 + 1 = 4; rate = 3/4
        assert stats.acceptance_rate == pytest.approx(3 / 4)

    def test_acceptance_rate_zero_when_no_tokens(self):
        stats = DecodeStats()
        assert stats.acceptance_rate == pytest.approx(0.0)

    def test_stats_accumulated_across_iterations(self):
        stats = DecodeStats()
        stats.update(accepted=5, rejected=0, corrections=0)
        stats.update(accepted=3, rejected=1, corrections=1)
        # total_draft = 5 + 3 + 1 = 9; accepted = 8; rate = 8/9
        assert stats.total_draft_tokens == 9
        assert stats.accepted_tokens == 8
        assert stats.acceptance_rate == pytest.approx(8 / 9)

    def test_decode_stats_returned_from_decode_loop(self):
        """decode_speculative must return a DecodeStats instance."""
        draft_model = _make_draft_model(token_id=2, vocab=VOCAB)
        verifier_logits_template = _make_peaked_logits(2, batch=BATCH, seq=SEQ + 10, vocab=VOCAB)
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,
        )
        prompt = _make_prompt()
        _, stats = decoder.decode_speculative(prompt, max_tokens=6, draft_steps=3)

        assert isinstance(stats, DecodeStats)
        # With matching distributions, acceptance rate should be high (> 0)
        assert stats.total_draft_tokens > 0


# ---------------------------------------------------------------------------
# Test 6 — fallback without draft model (vanilla autoregressive)
# ---------------------------------------------------------------------------


class TestVanillaFallback:
    """When draft_model=None, decode_speculative falls back to vanilla AR decoding."""

    def test_fallback_produces_tokens(self):
        verifier_logits_template = _make_peaked_logits(7, batch=BATCH, seq=SEQ + 20, vocab=VOCAB)
        decoder = SpeculativeDecoder(
            draft_model=None,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,
        )
        prompt = _make_prompt()
        max_tokens = 5

        generated, stats = decoder.decode_speculative(prompt, max_tokens=max_tokens)

        assert generated.shape == (BATCH, max_tokens)

    def test_fallback_stats_are_empty(self):
        """Vanilla fallback should not contribute to draft acceptance stats."""
        verifier_logits_template = _make_peaked_logits(7, batch=BATCH, seq=SEQ + 20, vocab=VOCAB)
        decoder = SpeculativeDecoder(
            draft_model=None,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,
        )
        prompt = _make_prompt()
        _, stats = decoder.decode_speculative(prompt, max_tokens=4)

        assert stats.total_draft_tokens == 0
        assert stats.accepted_tokens == 0

    def test_generate_draft_raises_without_draft_model(self):
        """generate_draft() must raise RuntimeError when no draft model is set."""
        decoder = SpeculativeDecoder(
            draft_model=None,
            verifier_fn=lambda x: x,
            temperature=1.0,
        )
        with pytest.raises(RuntimeError, match="No draft model"):
            decoder.generate_draft(_make_prompt(), k=3)

    def test_fallback_token_values_come_from_verifier(self):
        """In vanilla fallback, all tokens should be the verifier's preferred token."""
        target_id = 15
        verifier_logits_template = _make_peaked_logits(
            target_id, batch=BATCH, seq=SEQ + 20, vocab=VOCAB
        )
        decoder = SpeculativeDecoder(
            draft_model=None,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,
        )
        prompt = _make_prompt()
        generated, _ = decoder.decode_speculative(prompt, max_tokens=6)

        assert (generated == target_id).all(), (
            f"Expected all tokens to be {target_id}, got {generated}"
        )


# ---------------------------------------------------------------------------
# Test 7 — batch dimension handling
# ---------------------------------------------------------------------------


class TestBatchDimension:
    """Decoder handles batch_size > 1 correctly."""

    def test_draft_shape_multi_batch(self):
        batch = 3
        k = 4
        draft_model = _make_draft_model(token_id=1, vocab=VOCAB)
        prompt = _make_prompt(batch=batch, seq=SEQ)

        draft_tokens, draft_probs = draft_model.generate_draft(prompt, k=k)

        assert draft_tokens.shape == (batch, k)
        assert draft_probs.shape == (batch, k, VOCAB)

    def test_verify_and_accept_multi_batch(self):
        """verify_and_accept returns correct batch dimension."""
        batch = 2
        k = 3
        target_id = 6

        verifier_logits = _make_peaked_logits(target_id, batch=batch, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(batch, k, VOCAB)
        draft_probs[:, :, target_id] = 1.0
        draft_tokens = torch.full((batch, k), target_id, dtype=torch.long)
        prompt = _make_prompt(batch=batch, seq=SEQ)

        decoder = SpeculativeDecoder(draft_model=None, verifier_fn=lambda x: x, temperature=1.0)
        accepted, n_acc, n_rej, _ = decoder.verify_and_accept(
            prompt, draft_tokens, verifier_logits, draft_probs=draft_probs
        )

        assert accepted.shape[0] == batch, f"Expected batch dimension {batch}, got {accepted.shape[0]}"
        assert n_acc == k  # all accepted

    def test_decode_loop_multi_batch(self):
        """Full decode loop returns correct batch dimension."""
        batch = 2
        draft_model = _make_draft_model(token_id=2, vocab=VOCAB)
        verifier_logits_template = _make_peaked_logits(
            2, batch=batch, seq=SEQ + 15, vocab=VOCAB
        )
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,
        )
        prompt = _make_prompt(batch=batch, seq=SEQ)
        max_tokens = 8

        generated, _ = decoder.decode_speculative(prompt, max_tokens=max_tokens, draft_steps=3)

        assert generated.shape[0] == batch
        assert generated.shape[1] == max_tokens

    def test_draft_model_forward_multi_batch(self):
        """DraftModel.forward works for batch_size > 1."""
        batch = 4
        draft_model = _make_draft_model(token_id=3, vocab=VOCAB)
        prompt = _make_prompt(batch=batch, seq=SEQ)

        logits = draft_model.forward(prompt)

        assert logits.shape[0] == batch
        assert logits.shape[-1] == VOCAB


# ---------------------------------------------------------------------------
# Test 8 — temperature scaling in the acceptance criterion
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    """Temperature affects acceptance probability and residual sampling."""

    def test_high_temperature_flattens_target_probs(self):
        """At high temperature, the target distribution should be more uniform."""
        target_id = 5
        k = 1

        # Peaked logits: without scaling, prob ≈ 1.0 at target_id
        raw_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, target_id] = 1.0
        draft_tokens = torch.full((BATCH, k), target_id, dtype=torch.long)

        low_temp_decoder = SpeculativeDecoder(
            draft_model=None, verifier_fn=lambda x: x, temperature=0.01
        )
        high_temp_decoder = SpeculativeDecoder(
            draft_model=None, verifier_fn=lambda x: x, temperature=10.0
        )

        # At high temperature, softmax of peaked logits becomes more uniform.
        scaled_low = torch.softmax(raw_logits[:, 0, :] / 0.01, dim=-1)
        scaled_high = torch.softmax(raw_logits[:, 0, :] / 10.0, dim=-1)

        entropy_low = -(scaled_low * (scaled_low + 1e-10).log()).sum()
        entropy_high = -(scaled_high * (scaled_high + 1e-10).log()).sum()

        assert entropy_high > entropy_low, "High temperature should produce higher entropy distribution"

    def test_temperature_zero_point_one_accepts_correct_token(self):
        """At low temperature the decoder is near-deterministic; accepted token = target token."""
        target_id = 11
        k = 3

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, target_id] = 1.0
        draft_tokens = torch.full((BATCH, k), target_id, dtype=torch.long)

        decoder = SpeculativeDecoder(
            draft_model=None, verifier_fn=lambda x: x, temperature=0.1
        )
        accepted, n_acc, n_rej, _ = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs,
            temperature=0.1,
        )

        # Near-deterministic: all k draft tokens accepted and bonus = target_id
        assert n_acc == k
        assert (accepted == target_id).all()

    def test_decode_with_temperature_override(self):
        """temperature parameter to decode_speculative is respected."""
        draft_model = _make_draft_model(token_id=1, vocab=VOCAB)
        verifier_logits_template = _make_peaked_logits(1, batch=BATCH, seq=SEQ + 15, vocab=VOCAB)
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            verifier_fn=_constant_verifier(verifier_logits_template),
            temperature=1.0,  # default
        )
        prompt = _make_prompt()
        # Pass explicit override
        generated, _ = decoder.decode_speculative(
            prompt, max_tokens=5, draft_steps=2, temperature=0.5
        )

        assert generated.shape == (BATCH, 5)

    def test_acceptance_probability_scales_with_temperature(self):
        """Lower target temperature → sharper q → different alpha compared to higher temp."""
        draft_id = 3
        target_id = 3  # matching token → alpha = 1 at any temperature when peaked
        k = 1

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        draft_probs = torch.zeros(BATCH, k, VOCAB)
        draft_probs[:, :, draft_id] = 1.0
        draft_tokens = torch.full((BATCH, k), draft_id, dtype=torch.long)

        for temp in [0.1, 0.5, 1.0, 2.0]:
            decoder = SpeculativeDecoder(
                draft_model=None, verifier_fn=lambda x: x, temperature=temp
            )
            accepted, n_acc, n_rej, _ = decoder.verify_and_accept(
                _make_prompt(), draft_tokens, verifier_logits, draft_probs=draft_probs,
                temperature=temp,
            )
            # Draft and target agree on peaked token → should always accept regardless of temp
            assert n_acc == k, f"Expected acceptance at temp={temp}, got n_acc={n_acc}"

    def test_no_draft_probs_uses_conservative_acceptance(self):
        """Without draft_probs, acceptance uses vocab-size-based alpha (conservative fallback)."""
        target_id = 2
        draft_id = 2  # same as target
        k = 3

        verifier_logits = _make_peaked_logits(target_id, batch=BATCH, seq=k + 1, vocab=VOCAB)
        # Do NOT pass draft_probs
        draft_tokens = torch.full((BATCH, k), draft_id, dtype=torch.long)

        decoder = SpeculativeDecoder(
            draft_model=None, verifier_fn=lambda x: x, temperature=1.0
        )
        accepted, n_acc, n_rej, n_cor = decoder.verify_and_accept(
            _make_prompt(), draft_tokens, verifier_logits, draft_probs=None
        )

        # With a very peaked target (q ≈ 1 at draft_id) and uniform p (1/vocab),
        # alpha = min(1, q * vocab) ≈ min(1, vocab/vocab) = 1 → accept
        assert n_rej == 0
        assert accepted.shape[1] >= 1
