"""Tests for node.economics — reward/cost calculations."""

import math

import pytest

from node.economics import (
    MIN_STAKE_TOKENS,
    PROTOCOL_FEE_BPS,
    REPUTATION_DECAY,
    SLASH_RATE_BPS,
    CostEstimator,
    RewardCalculator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calc() -> RewardCalculator:
    return RewardCalculator()


@pytest.fixture()
def estimator() -> CostEstimator:
    return CostEstimator()


# ---------------------------------------------------------------------------
# RewardCalculator tests
# ---------------------------------------------------------------------------


def test_node_reward_after_protocol_fee(calc: RewardCalculator) -> None:
    """Node reward should equal payment minus the 5% protocol fee."""
    payment = 10_000
    expected = payment - (payment * PROTOCOL_FEE_BPS // 10_000)
    assert calc.node_reward(payment) == expected
    assert calc.node_reward(payment) == 9_500


def test_protocol_fee_is_5_percent(calc: RewardCalculator) -> None:
    """Protocol fee should be exactly 5% of the payment amount."""
    payment = 20_000
    assert calc.protocol_fee(payment) == 1_000
    # Also verify PROTOCOL_FEE_BPS represents 5%
    assert PROTOCOL_FEE_BPS == 500


def test_slash_amount_is_10_percent(calc: RewardCalculator) -> None:
    """Slash amount should be 10% of staked tokens."""
    staked = 500_000
    assert calc.slash_amount(staked) == 50_000
    assert SLASH_RATE_BPS == 1000


def test_estimate_job_cost_scales_with_model_size(calc: RewardCalculator) -> None:
    """Job cost should scale linearly with model size in billions of params."""
    tokens = 1_000
    small_cost = calc.estimate_job_cost(tokens, 7.0)
    large_cost = calc.estimate_job_cost(tokens, 70.0)
    assert large_cost == small_cost * 10
    assert small_cost == 7_000
    assert large_cost == 70_000


def test_unstake_cooldown_7_days(calc: RewardCalculator) -> None:
    """Cooldown end timestamp should be exactly 7 days after unstaking."""
    unstake_ts = 1_700_000_000.0
    seven_days = 7 * 24 * 60 * 60
    assert calc.unstake_cooldown_ends(unstake_ts) == pytest.approx(unstake_ts + seven_days)


def test_reputation_increases_on_success(calc: RewardCalculator) -> None:
    """Reputation should increase by 1 on a successful job."""
    assert calc.reputation_after_job(500, success=True) == 501
    assert calc.reputation_after_job(0, success=True) == 1


def test_reputation_decays_on_failure(calc: RewardCalculator) -> None:
    """Reputation should decay by REPUTATION_DECAY factor on failure."""
    current = 200
    expected = max(0, int(current * REPUTATION_DECAY))
    assert calc.reputation_after_job(current, success=False) == expected
    assert calc.reputation_after_job(current, success=False) == 190


def test_reputation_floor_at_zero(calc: RewardCalculator) -> None:
    """Reputation should never fall below 0 on failure."""
    assert calc.reputation_after_job(0, success=False) == 0
    assert calc.reputation_after_job(1, success=False) == 0


def test_reputation_ceiling_at_1000(calc: RewardCalculator) -> None:
    """Reputation should never exceed 1000 on success."""
    assert calc.reputation_after_job(1000, success=True) == 1000
    assert calc.reputation_after_job(999, success=True) == 1000


def test_is_eligible_with_sufficient_stake(calc: RewardCalculator) -> None:
    """Node with stake >= MIN_STAKE_TOKENS and non-negative reputation should be eligible."""
    assert calc.is_eligible_to_register(MIN_STAKE_TOKENS, 0) is True
    assert calc.is_eligible_to_register(MIN_STAKE_TOKENS + 1, 500) is True


def test_is_not_eligible_with_insufficient_stake(calc: RewardCalculator) -> None:
    """Node with stake below MIN_STAKE_TOKENS should not be eligible."""
    assert calc.is_eligible_to_register(MIN_STAKE_TOKENS - 1, 500) is False
    assert calc.is_eligible_to_register(0, 500) is False


# ---------------------------------------------------------------------------
# CostEstimator tests
# ---------------------------------------------------------------------------


def test_cost_estimator_recommended_is_1_5x_min(estimator: CostEstimator) -> None:
    """Recommended cost should be 1.5x the minimum cost (rounded up)."""
    result = estimator.estimate(500, 500, "llama-7b")
    assert result["recommended_cost"] == math.ceil(result["min_cost"] * 1.5)


def test_cost_estimator_max_is_3x_min(estimator: CostEstimator) -> None:
    """Max cost should be 3x the minimum cost."""
    result = estimator.estimate(100, 100, "llama-7b")
    assert result["max_cost"] == result["min_cost"] * 3


def test_cost_estimator_returns_all_keys(estimator: CostEstimator) -> None:
    """estimate() should return all three cost keys."""
    result = estimator.estimate(200, 300, "llama-13b")
    assert "min_cost" in result
    assert "recommended_cost" in result
    assert "max_cost" in result


def test_token_count_rough_estimate(estimator: CostEstimator) -> None:
    """token_count should return ceil(word_count * 1.3)."""
    text = "hello world foo bar"  # 4 words
    expected = math.ceil(4 * 1.3)
    assert estimator.token_count(text) == expected


def test_token_count_empty_string(estimator: CostEstimator) -> None:
    """token_count of an empty string should be 0."""
    assert estimator.token_count("") == 0
