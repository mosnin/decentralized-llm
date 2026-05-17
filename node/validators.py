import re
from dataclasses import dataclass

# Validation limits
MAX_PROMPT_LENGTH = 32_768  # characters
MAX_MODEL_NAME_LENGTH = 256
MAX_TOKENS_LIMIT = 4096
MIN_TOKENS = 1
ALLOWED_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9/_\-\.]+$")


class ValidationError(ValueError):
    """Raised when input fails validation."""

    pass


def validate_prompt(prompt: str) -> str:
    """
    Validate and sanitize a prompt string.
    - Must be a non-empty string
    - Must not exceed MAX_PROMPT_LENGTH characters
    - Strips leading/trailing whitespace
    Returns sanitized prompt or raises ValidationError.
    """
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValidationError(f"prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
    return prompt


def validate_model_name(name: str) -> str:
    """
    Validate a model name.
    - Must be a non-empty string
    - Must not exceed MAX_MODEL_NAME_LENGTH
    - Must match ALLOWED_MODEL_NAME_RE (alphanumeric, /, _, -, . only)
    Returns name or raises ValidationError.
    """
    if not isinstance(name, str):
        raise ValidationError("model_name must be a string")
    if not name:
        raise ValidationError("model_name must not be empty")
    if len(name) > MAX_MODEL_NAME_LENGTH:
        raise ValidationError(
            f"model_name exceeds maximum length of {MAX_MODEL_NAME_LENGTH} characters"
        )
    if not ALLOWED_MODEL_NAME_RE.match(name):
        raise ValidationError(
            "model_name contains invalid characters; "
            "only alphanumeric characters and /, _, -, . are allowed"
        )
    return name


def validate_max_tokens(value: int) -> int:
    """
    Validate max_tokens parameter.
    - Must be an integer
    - Must be between MIN_TOKENS and MAX_TOKENS_LIMIT (inclusive)
    Returns value or raises ValidationError.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError("max_tokens must be an integer")
    if value < MIN_TOKENS:
        raise ValidationError(f"max_tokens must be at least {MIN_TOKENS}")
    if value > MAX_TOKENS_LIMIT:
        raise ValidationError(f"max_tokens must not exceed {MAX_TOKENS_LIMIT}")
    return value


def validate_payment_amount(amount: int) -> int:
    """
    Validate payment amount in lamports.
    - Must be a positive integer (> 0)
    Returns amount or raises ValidationError.
    """
    if not isinstance(amount, int) or isinstance(amount, bool):
        raise ValidationError("payment_amount must be an integer")
    if amount <= 0:
        raise ValidationError("payment_amount must be a positive integer greater than 0")
    return amount


@dataclass
class InferenceRequest:
    """Validated inference request parameters."""

    prompt: str
    model_name: str
    max_tokens: int
    payment_amount: int

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceRequest":
        """Parse and validate all fields from a raw dict."""
        return cls(
            prompt=validate_prompt(data.get("prompt", "")),
            model_name=validate_model_name(data.get("model_name", "")),
            max_tokens=validate_max_tokens(data.get("max_tokens", 256)),
            payment_amount=validate_payment_amount(data.get("payment_amount", 0)),
        )
