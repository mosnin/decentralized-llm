"""Token economics calculations for the decentralized LLM network."""

import math

PROTOCOL_FEE_BPS = 500  # 5% to DAO treasury (basis points)
MIN_STAKE_TOKENS = 100_000  # minimum node stake
SLASH_RATE_BPS = 1000  # 10% slashed on dispute loss
REPUTATION_DECAY = 0.95  # per-job decay factor

# Model size lookup (billions of parameters) for known model families
_MODEL_SIZE_LOOKUP: dict[str, float] = {
    "7b": 7.0,
    "13b": 13.0,
    "30b": 30.0,
    "34b": 34.0,
    "70b": 70.0,
    "3b": 3.0,
    "1b": 1.0,
}

_COOLDOWN_SECONDS = 7 * 24 * 60 * 60  # 7 days in seconds


def _parse_model_size(model: str) -> float:
    """Return model size in billions of parameters from a model name string."""
    model_lower = model.lower()
    for key, size in sorted(_MODEL_SIZE_LOOKUP.items(), key=lambda x: -x[1]):
        if key in model_lower:
            return size
    # Default to 7B if unknown
    return 7.0


class RewardCalculator:
    """Calculate rewards, fees, and costs for network participants."""

    def node_reward(self, payment_amount: int) -> int:
        """Return the node's take after the 5% protocol fee."""
        fee = self.protocol_fee(payment_amount)
        return payment_amount - fee

    def protocol_fee(self, payment_amount: int) -> int:
        """Return the 5% protocol fee amount."""
        return (payment_amount * PROTOCOL_FEE_BPS) // 10_000

    def slash_amount(self, staked: int) -> int:
        """Return 10% of stake (slashed on dispute loss)."""
        return (staked * SLASH_RATE_BPS) // 10_000

    def estimate_job_cost(self, max_tokens: int, model_size_b: float) -> int:
        """
        Heuristic cost estimate for a job.

        Cost = 1 token-unit per token per billion params.
        """
        return math.ceil(max_tokens * model_size_b)

    def unstake_cooldown_ends(self, unstake_timestamp: float) -> float:
        """Return the timestamp when the 7-day unstaking cooldown ends."""
        return unstake_timestamp + _COOLDOWN_SECONDS

    def reputation_after_job(self, current: int, success: bool) -> int:
        """
        Return updated reputation score after a job.

        On success: min(1000, current + 1)
        On failure: max(0, int(current * REPUTATION_DECAY))
        """
        if success:
            return min(1000, current + 1)
        return max(0, int(current * REPUTATION_DECAY))

    def is_eligible_to_register(self, staked: int, reputation: int) -> bool:
        """Return True if the node meets the minimum stake and reputation requirements."""
        return staked >= MIN_STAKE_TOKENS and reputation >= 0


class CostEstimator:
    """Estimate costs for clients before posting a job."""

    def estimate(self, prompt_tokens: int, max_output_tokens: int, model: str) -> dict:
        """
        Return cost estimates for a job.

        Returns {"min_cost": int, "recommended_cost": int, "max_cost": int}
        where recommended = 1.5x min and max = 3x min.
        """
        model_size_b = _parse_model_size(model)
        calculator = RewardCalculator()
        total_tokens = prompt_tokens + max_output_tokens
        min_cost = calculator.estimate_job_cost(total_tokens, model_size_b)
        recommended_cost = math.ceil(min_cost * 1.5)
        max_cost = min_cost * 3
        return {
            "min_cost": min_cost,
            "recommended_cost": recommended_cost,
            "max_cost": max_cost,
        }

    def token_count(self, text: str) -> int:
        """Rough estimate: len(text.split()) * 1.3 rounded up."""
        return math.ceil(len(text.split()) * 1.3)
