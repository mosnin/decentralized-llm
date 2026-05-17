"""
Tests for the Byzantine gradient defense in the federated trainer.

These tests run entirely on CPU tensors — no GPU, no Hivemind required.
The defense mechanism is a standalone method that can be exercised in isolation.
"""

import pytest

torch = pytest.importorskip("torch")


def _make_trainer():
    """Create a FederatedTrainer with mocked heavy deps."""
    import sys
    import types

    # Stub out peft and hivemind so FinetuneConfig can be instantiated
    for mod in ("peft", "hivemind"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # FinetuneConfig is a dataclass — no heavy deps needed for construction
    from finetuning.trainer import FederatedTrainer, FinetuneConfig

    cfg = FinetuneConfig()

    # Bypass __init__ checks by constructing directly
    trainer = object.__new__(FederatedTrainer)
    trainer.config = cfg
    trainer.model = None
    trainer.optimizer = None
    trainer.dht = None
    trainer.tokenizer = None
    return trainer


class TestByzantineFilter:
    def test_first_call_initializes_ema(self):
        trainer = _make_trainer()
        from finetuning.trainer import FederatedTrainer

        params = [torch.nn.Parameter(torch.randn(10))]
        for p in params:
            p.grad = torch.randn(10)

        assert not hasattr(trainer, "_grad_ema")
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)
        assert trainer._grad_ema is not None
        assert trainer._grad_ema.shape == (10,)

    def test_aligned_gradient_is_kept(self):
        trainer = _make_trainer()
        from finetuning.trainer import FederatedTrainer

        grad = torch.ones(10)
        params = [torch.nn.Parameter(torch.randn(10))]
        params[0].grad = grad.clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)

        # Second call with same direction — cosine = 1.0, above threshold 0.0
        params[0].grad = grad.clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)
        assert params[0].grad is not None
        assert params[0].grad.abs().sum().item() > 0

    def test_sign_flipped_gradient_is_zeroed(self):
        trainer = _make_trainer()
        from finetuning.trainer import FederatedTrainer

        base = torch.ones(10)
        params = [torch.nn.Parameter(torch.randn(10))]

        # Initialize EMA with positive gradient
        params[0].grad = base.clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)

        # Now pass a perfectly sign-flipped gradient (cosine = -1.0 < 0.0)
        params[0].grad = -base.clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params, threshold=0.0)
        assert params[0].grad.abs().sum().item() == 0.0

    def test_ema_updates_toward_new_gradient(self):
        trainer = _make_trainer()
        from finetuning.trainer import FederatedTrainer

        params = [torch.nn.Parameter(torch.randn(10))]
        grad = torch.ones(10)
        params[0].grad = grad.clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)
        ema_after_first = trainer._grad_ema.clone()

        params[0].grad = (grad * 2).clone()
        FederatedTrainer._filter_gradient_by_cosine(trainer, params)

        # EMA should have moved toward the new gradient (α=0.1)
        assert not torch.allclose(trainer._grad_ema, ema_after_first)

    def test_filter_handles_multiple_param_groups(self):
        trainer = _make_trainer()
        from finetuning.trainer import FederatedTrainer

        params = [torch.nn.Parameter(torch.randn(5)), torch.nn.Parameter(torch.randn(5))]
        for p in params:
            p.grad = torch.ones_like(p)

        FederatedTrainer._filter_gradient_by_cosine(trainer, params)
        assert trainer._grad_ema.shape == (10,)  # concatenated across both params
