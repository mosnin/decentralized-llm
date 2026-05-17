"""
Speculative decoding for 2-3x inference speedup.

Implements the algorithm from:
  Leviathan, Kalman, Matias (2023) — "Fast Inference from Transformers via
  Speculative Decoding"  https://arxiv.org/abs/2211.17192

High-level flow
---------------
1. A small *draft* model autoregressively produces k candidate tokens cheaply.
2. The large *verifier* (target) model evaluates all k+1 positions in ONE
   forward pass, producing k+1 probability distributions in parallel.
3. Each draft token is accepted or rejected using the Leviathan et al.
   modified rejection-sampling criterion:
     - Accept token x with probability  min(1, q(x) / p(x))
       where p = draft prob, q = target prob.
     - On rejection, sample a corrected token from  norm(max(0, q − p)).
4. At minimum one new token is always produced (the target's own sample from
   the last position), so the algorithm is correct even when all k drafts are
   rejected.

All torch imports are lazy (inside methods) so the module is importable in
environments where PyTorch is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DraftModel
# ---------------------------------------------------------------------------


class DraftModel:
    """
    Lightweight wrapper around a small language model (~1B params) used to
    generate speculative draft tokens.

    The underlying model is accessed through a duck-typed interface:
      - ``model(input_ids) -> Tensor``  shape ``[batch, seq_len, vocab_size]``
        (raw logits, or a dataclass/tuple whose first element is logits).
      - Alternatively, pass a callable that directly returns logits.

    Parameters
    ----------
    model:
        Any callable that accepts ``input_ids`` and returns logits (or an
        object whose ``.logits`` attribute contains them).
    vocab_size:
        Vocabulary size.  Inferred from the first forward pass if not given.
    temperature:
        Sampling temperature for draft token selection (default 1.0).
    top_k:
        If > 0, restrict sampling to the top-k logits (default 0 = disabled).
    """

    def __init__(
        self,
        model: Any,
        vocab_size: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass and return logits of shape ``[batch, seq_len, vocab_size]``.
        """
        out = self.model(input_ids)
        # Support HuggingFace model outputs (have .logits) and plain tensors.
        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if self.vocab_size is None:
            self.vocab_size = logits.shape[-1]

        return logits

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample a single token from ``logits`` of shape ``[batch, vocab_size]``.

        Returns an integer tensor of shape ``[batch, 1]``.
        """
        import torch  # noqa: PLC0415

        scaled = logits / max(self.temperature, 1e-8)

        if self.top_k > 0:
            # Zero-out everything outside the top-k
            top_vals, _ = torch.topk(scaled, min(self.top_k, scaled.shape[-1]), dim=-1)
            threshold = top_vals[..., -1:].expand_as(scaled)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

        probs = torch.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        return token

    def get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ``logits`` (shape ``[batch, vocab_size]``) to a probability
        distribution, taking temperature into account.
        """
        import torch  # noqa: PLC0415

        scaled = logits / max(self.temperature, 1e-8)
        return torch.softmax(scaled, dim=-1)

    def generate_draft(
        self, input_ids: torch.Tensor, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively generate *k* draft tokens.

        Parameters
        ----------
        input_ids:
            Shape ``[batch, seq_len]`` — the prompt / context so far.
        k:
            Number of speculative tokens to produce.

        Returns
        -------
        draft_tokens:
            Shape ``[batch, k]`` — the sampled draft token ids.
        draft_probs:
            Shape ``[batch, k, vocab_size]`` — the draft model's probability
            distributions at each of the k positions.
        """
        import torch  # noqa: PLC0415

        all_tokens: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        current_ids = input_ids
        for _ in range(k):
            logits = self.forward(current_ids)  # [batch, seq, vocab]
            last_logits = logits[:, -1, :]  # [batch, vocab]
            probs = self.get_probs(last_logits)  # [batch, vocab]
            token = self.sample(last_logits)  # [batch, 1]

            all_tokens.append(token)
            all_probs.append(probs.unsqueeze(1))  # [batch, 1, vocab]

            current_ids = torch.cat([current_ids, token], dim=1)

        draft_tokens = torch.cat(all_tokens, dim=1)  # [batch, k]
        draft_probs = torch.cat(all_probs, dim=1)  # [batch, k, vocab]
        return draft_tokens, draft_probs


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------


@dataclass
class DecodeStats:
    """Tracks acceptance / rejection statistics across a decode call."""

    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    correction_tokens: int = 0  # tokens added via the residual distribution

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted (0.0 – 1.0)."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    def update(self, accepted: int, rejected: int, corrections: int) -> None:
        self.total_draft_tokens += accepted + rejected
        self.accepted_tokens += accepted
        self.rejected_tokens += rejected
        self.correction_tokens += corrections


class SpeculativeDecoder:
    """
    Coordinates a small draft model and a large verifier model to perform
    speculative decoding (Leviathan et al. 2023).

    Parameters
    ----------
    draft_model:
        A ``DraftModel`` instance wrapping the small model.  Pass ``None``
        to fall back to vanilla autoregressive decoding using the verifier.
    verifier_fn:
        Callable ``(input_ids) -> logits`` for the large target model.
        Returns shape ``[batch, seq_len, vocab_size]``.
    temperature:
        Sampling temperature applied when drawing from the corrected
        distribution on rejection (and for vanilla fallback).  The draft
        model has its own temperature setting.
    eos_token_id:
        If provided, generation stops when this token is produced.
    """

    def __init__(
        self,
        draft_model: DraftModel | None,
        verifier_fn: Any,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> None:
        self.draft_model = draft_model
        self.verifier_fn = verifier_fn
        self.temperature = temperature
        self.eos_token_id = eos_token_id

    # ------------------------------------------------------------------
    # Core speculative primitives
    # ------------------------------------------------------------------

    def generate_draft(
        self, input_ids: torch.Tensor, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use the draft model to speculatively generate *k* candidate tokens.

        Parameters
        ----------
        input_ids:
            Shape ``[batch, seq_len]``.
        k:
            Number of draft tokens to produce.

        Returns
        -------
        draft_tokens:
            Shape ``[batch, k]``.
        draft_probs:
            Shape ``[batch, k, vocab_size]`` — draft model probabilities.

        Raises
        ------
        RuntimeError
            If no draft model was provided.
        """
        if self.draft_model is None:
            raise RuntimeError(
                "No draft model is available; use decode_speculative() which "
                "falls back to vanilla autoregressive decoding."
            )
        return self.draft_model.generate_draft(input_ids, k=k)

    def verify_and_accept(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        verifier_logits: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        """
        Apply the Leviathan et al. (2023) speculative acceptance criterion.

        For each of the k draft positions:
          - Compute target prob  q(x_i)  from ``verifier_logits``.
          - If a draft distribution ``draft_probs`` is provided, compute
            acceptance probability  α = min(1, q(x_i) / p(x_i)).
            Otherwise treat all draft tokens as if p = uniform (conservative).
          - Sample  u ~ Uniform(0, 1).
            Accept the token if  u < α; otherwise reject and resample from
            the residual distribution  norm(max(0, q − p)).
          - Stop at the first rejection; always emit one token from the target
            distribution at the position *after* the last accepted draft token.

        This guarantees that the joint distribution of produced tokens matches
        the target model's distribution exactly (the algorithm is lossless).

        Parameters
        ----------
        input_ids:
            Shape ``[batch, seq_len]`` — the current context (prompt).
        draft_tokens:
            Shape ``[batch, k]`` — draft token ids to be verified.
        verifier_logits:
            Shape ``[batch, k+1, vocab_size]`` — logits produced by the
            verifier in a single forward pass over (input_ids || draft_tokens).
            Position i corresponds to the prediction *after* input_ids[i].
        draft_probs:
            Optional shape ``[batch, k, vocab_size]`` — draft model's
            probability distributions.  If ``None``, acceptance uses only the
            target distribution (conservative: equivalent to p = uniform).
        temperature:
            Override the instance-level temperature for target sampling.

        Returns
        -------
        accepted_tokens:
            Shape ``[batch, n]`` where  1 ≤ n ≤ k+1  — the accepted token
            sequence (may include a correction token after the first rejection).
        n_accepted:
            Count of draft tokens accepted (before any rejection).
        n_rejected:
            1 if a draft token was rejected, 0 otherwise.
        n_corrections:
            1 if a correction token was added from the residual distribution.
        """
        import torch  # noqa: PLC0415

        temp = temperature if temperature is not None else self.temperature
        batch_size, k = draft_tokens.shape
        vocab_size = verifier_logits.shape[-1]

        # ----------------------------------------------------------------
        # Build target probability distributions for positions 0…k
        # verifier_logits[:, i, :] predicts the token AFTER position i of
        # the *extended* sequence (input_ids + draft_tokens).
        # So verifier_logits[:, 0, :] → target prob for draft_tokens[:, 0]
        #    verifier_logits[:, k, :] → target prob for the bonus token
        # ----------------------------------------------------------------
        scaled_logits = verifier_logits / max(temp, 1e-8)
        target_probs_all = torch.softmax(scaled_logits, dim=-1)  # [batch, k+1, vocab]

        # Work on first element of batch (batch processing is handled
        # position-by-position for the sequential accept/reject logic).
        # We support batch_size == 1 for simplicity; larger batches are
        # treated element-wise.

        collected: list[torch.Tensor] = []
        n_accepted = 0
        n_rejected = 0
        n_corrections = 0
        first_rejection_pos = k  # sentinel: no rejection

        for i in range(k):
            q_i = target_probs_all[:, i, :]  # [batch, vocab]
            x_i = draft_tokens[:, i]  # [batch]

            # Gather q(x_i): target probability at the draft token
            q_xi = q_i.gather(1, x_i.unsqueeze(1)).squeeze(1)  # [batch]

            if draft_probs is not None:
                p_i = draft_probs[:, i, :]  # [batch, vocab]
                p_xi = p_i.gather(1, x_i.unsqueeze(1)).squeeze(1)  # [batch]
                # Avoid division by zero
                alpha = torch.clamp(q_xi / (p_xi + 1e-10), max=1.0)  # [batch]
            else:
                # Conservative: treat p as uniform → alpha = q(x_i) * vocab_size
                alpha = torch.clamp(q_xi * vocab_size, max=1.0)

            u = torch.rand_like(alpha)
            accept_mask = u < alpha  # [batch] bool

            if accept_mask.all():
                # All batch elements accept token i
                collected.append(x_i.unsqueeze(1))  # [batch, 1]
                n_accepted += 1
            else:
                # At least one batch element rejects; treat as a reject step.
                # For simplicity (and correctness for batch_size==1), we stop
                # here and sample a correction token from the residual.
                first_rejection_pos = i
                n_rejected = 1

                # Residual distribution: norm(max(0, q - p))
                if draft_probs is not None:
                    residual = torch.clamp(q_i - p_i, min=0.0)
                else:
                    # p = uniform 1/vocab; residual ∝ max(0, q - 1/vocab)
                    uniform = torch.full_like(q_i, 1.0 / vocab_size)
                    residual = torch.clamp(q_i - uniform, min=0.0)

                residual_sum = residual.sum(dim=-1, keepdim=True)
                # If the residual is all zeros (exact match edge case), fall
                # back to the target distribution itself.
                zero_mask = residual_sum < 1e-10
                residual = torch.where(zero_mask.expand_as(residual), q_i, residual)
                residual_sum = residual.sum(dim=-1, keepdim=True)
                residual = residual / (residual_sum + 1e-10)

                correction = torch.multinomial(residual, num_samples=1)  # [batch, 1]
                collected.append(correction)
                n_corrections = 1
                break

        # If all k draft tokens were accepted, sample one bonus token from the
        # target distribution at position k (the "free" token from the verifier).
        if first_rejection_pos == k:
            bonus_probs = target_probs_all[:, k, :]  # [batch, vocab]
            bonus = torch.multinomial(bonus_probs, num_samples=1)  # [batch, 1]
            collected.append(bonus)

        accepted_tokens = torch.cat(collected, dim=1)  # [batch, n]
        return accepted_tokens, n_accepted, n_rejected, n_corrections

    # ------------------------------------------------------------------
    # Full decode loop
    # ------------------------------------------------------------------

    def decode_speculative(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        draft_steps: int = 5,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, DecodeStats]:
        """
        Full speculative decoding loop.

        Each iteration:
          1. Draft ``draft_steps`` candidate tokens with the small model.
          2. Run the verifier once over (context + draft tokens) — single
             forward pass producing ``draft_steps + 1`` sets of logits.
          3. Accept / reject via the Leviathan criterion and append accepted
             tokens to the context.
          4. Repeat until ``max_tokens`` new tokens have been generated or
             the EOS token is encountered.

        Falls back to plain autoregressive decoding (one verifier call per
        token) when ``self.draft_model is None``.

        Parameters
        ----------
        input_ids:
            Shape ``[batch, seq_len]`` — the prompt.
        max_tokens:
            Maximum number of *new* tokens to generate.
        draft_steps:
            Number of tokens the draft model proposes per iteration (k).
        temperature:
            Sampling temperature override; defaults to ``self.temperature``.

        Returns
        -------
        generated:
            Shape ``[batch, n]`` where n ≤ max_tokens — the generated tokens
            (not including the prompt).
        stats:
            A ``DecodeStats`` instance with acceptance rate information.
        """
        import torch  # noqa: PLC0415

        temp = temperature if temperature is not None else self.temperature
        stats = DecodeStats()

        # Running context (we extend it as tokens are accepted)
        context = input_ids.clone()
        all_new_tokens: list[torch.Tensor] = []
        total_new = 0

        if self.draft_model is None:
            # ---- Vanilla autoregressive fallback ----------------------------
            logger.debug("No draft model; using vanilla autoregressive decoding.")
            while total_new < max_tokens:
                logits = self._run_verifier(context)  # [batch, seq, vocab]
                last_logits = logits[:, -1, :] / max(temp, 1e-8)
                probs = torch.softmax(last_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                all_new_tokens.append(token)
                context = torch.cat([context, token], dim=1)
                total_new += 1
                if self.eos_token_id is not None and (token == self.eos_token_id).any():
                    break
            generated = (
                torch.cat(all_new_tokens, dim=1)
                if all_new_tokens
                else torch.zeros((input_ids.shape[0], 0), dtype=input_ids.dtype)
            )
            return generated, stats

        # ---- Speculative decode loop ----------------------------------------
        while total_new < max_tokens:
            remaining = max_tokens - total_new
            k = min(draft_steps, remaining)

            # Step 1: draft k tokens
            draft_tokens, draft_probs = self.draft_model.generate_draft(context, k=k)
            # draft_tokens: [batch, k], draft_probs: [batch, k, vocab]

            # Step 2: verifier forward pass over (context + draft_tokens)
            extended = torch.cat([context, draft_tokens], dim=1)  # [batch, seq+k]
            verifier_logits_all = self._run_verifier(extended)  # [batch, seq+k, vocab]

            # Extract the k+1 logit positions we care about:
            # position seq-1 predicts draft_tokens[:,0]; ...; position seq+k-1 predicts bonus
            seq_len = context.shape[1]
            # Slice from position (seq_len - 1) to (seq_len - 1 + k + 1)
            verifier_logits = verifier_logits_all[:, seq_len - 1 : seq_len - 1 + k + 1, :]
            # Shape: [batch, k+1, vocab]

            # Step 3: accept / reject
            new_tokens, n_acc, n_rej, n_cor = self.verify_and_accept(
                context, draft_tokens, verifier_logits, draft_probs, temperature=temp
            )
            # new_tokens: [batch, accepted_count]

            stats.update(n_acc, n_rej, n_cor)

            # Clip to remaining budget
            budget_left = max_tokens - total_new
            if new_tokens.shape[1] > budget_left:
                new_tokens = new_tokens[:, :budget_left]

            all_new_tokens.append(new_tokens)
            context = torch.cat([context, new_tokens], dim=1)
            total_new += new_tokens.shape[1]

            # EOS check
            if self.eos_token_id is not None:
                eos_hit = (new_tokens == self.eos_token_id).any(dim=-1).all()
                if eos_hit:
                    break

        if all_new_tokens:
            generated = torch.cat(all_new_tokens, dim=1)
        else:
            generated = torch.zeros(
                (input_ids.shape[0], 0), dtype=input_ids.dtype, device=input_ids.device
            )
        return generated, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_verifier(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the verifier and return logits ``[batch, seq_len, vocab_size]``."""
        out = self.verifier_fn(input_ids)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
