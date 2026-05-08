import pytest

from node.validators import (
    MAX_MODEL_NAME_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_TOKENS_LIMIT,
    InferenceRequest,
    ValidationError,
    validate_max_tokens,
    validate_model_name,
    validate_payment_amount,
    validate_prompt,
)


class TestValidatePrompt:
    def test_validate_prompt_valid(self):
        result = validate_prompt("Hello, world!")
        assert result == "Hello, world!"

    def test_validate_prompt_empty_raises(self):
        with pytest.raises(ValidationError):
            validate_prompt("")

    def test_validate_prompt_too_long_raises(self):
        with pytest.raises(ValidationError):
            validate_prompt("x" * (MAX_PROMPT_LENGTH + 1))

    def test_validate_prompt_strips_whitespace(self):
        result = validate_prompt("  hello  ")
        assert result == "hello"

    def test_validate_prompt_exactly_max_length(self):
        prompt = "a" * MAX_PROMPT_LENGTH
        assert validate_prompt(prompt) == prompt

    def test_validate_prompt_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            validate_prompt("   ")


class TestValidateModelName:
    def test_validate_model_name_valid(self):
        name = "meta-llama/Llama-3.2-3B"
        assert validate_model_name(name) == name

    def test_validate_model_name_empty_raises(self):
        with pytest.raises(ValidationError):
            validate_model_name("")

    def test_validate_model_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            validate_model_name("a" * (MAX_MODEL_NAME_LENGTH + 1))

    def test_validate_model_name_invalid_chars(self):
        with pytest.raises(ValidationError):
            validate_model_name("my model;rm -rf")

    def test_validate_model_name_with_dots_and_underscores(self):
        name = "my_model.v1"
        assert validate_model_name(name) == name

    def test_validate_model_name_exactly_max_length(self):
        name = "a" * MAX_MODEL_NAME_LENGTH
        assert validate_model_name(name) == name

    def test_validate_model_name_dollar_sign_raises(self):
        with pytest.raises(ValidationError):
            validate_model_name("model$name")


class TestValidateMaxTokens:
    def test_validate_max_tokens_valid(self):
        assert validate_max_tokens(256) == 256

    def test_validate_max_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_max_tokens(0)

    def test_validate_max_tokens_too_large_raises(self):
        with pytest.raises(ValidationError):
            validate_max_tokens(9999)

    def test_validate_max_tokens_minimum(self):
        assert validate_max_tokens(1) == 1

    def test_validate_max_tokens_maximum(self):
        assert validate_max_tokens(MAX_TOKENS_LIMIT) == MAX_TOKENS_LIMIT

    def test_validate_max_tokens_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_max_tokens(-1)

    def test_validate_max_tokens_bool_raises(self):
        with pytest.raises(ValidationError):
            validate_max_tokens(True)


class TestValidatePaymentAmount:
    def test_validate_payment_amount_valid(self):
        assert validate_payment_amount(1000) == 1000

    def test_validate_payment_amount_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_payment_amount(0)

    def test_validate_payment_amount_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_payment_amount(-1)

    def test_validate_payment_amount_one(self):
        assert validate_payment_amount(1) == 1

    def test_validate_payment_amount_bool_raises(self):
        with pytest.raises(ValidationError):
            validate_payment_amount(True)


class TestInferenceRequest:
    def test_inference_request_from_dict_valid(self):
        data = {
            "prompt": "What is the capital of France?",
            "model_name": "meta-llama/Llama-3.2-3B",
            "max_tokens": 256,
            "payment_amount": 1000,
        }
        req = InferenceRequest.from_dict(data)
        assert req.prompt == "What is the capital of France?"
        assert req.model_name == "meta-llama/Llama-3.2-3B"
        assert req.max_tokens == 256
        assert req.payment_amount == 1000

    def test_inference_request_from_dict_missing_field(self):
        data = {
            "model_name": "meta-llama/Llama-3.2-3B",
            "max_tokens": 256,
            "payment_amount": 1000,
        }
        with pytest.raises(ValidationError):
            InferenceRequest.from_dict(data)

    def test_inference_request_strips_prompt_whitespace(self):
        data = {
            "prompt": "  hello  ",
            "model_name": "gpt2",
            "max_tokens": 64,
            "payment_amount": 500,
        }
        req = InferenceRequest.from_dict(data)
        assert req.prompt == "hello"

    def test_inference_request_invalid_model_name_raises(self):
        data = {
            "prompt": "test prompt",
            "model_name": "bad model name!",
            "max_tokens": 64,
            "payment_amount": 500,
        }
        with pytest.raises(ValidationError):
            InferenceRequest.from_dict(data)

    def test_inference_request_invalid_max_tokens_raises(self):
        data = {
            "prompt": "test prompt",
            "model_name": "gpt2",
            "max_tokens": 0,
            "payment_amount": 500,
        }
        with pytest.raises(ValidationError):
            InferenceRequest.from_dict(data)

    def test_inference_request_invalid_payment_raises(self):
        data = {
            "prompt": "test prompt",
            "model_name": "gpt2",
            "max_tokens": 64,
            "payment_amount": -100,
        }
        with pytest.raises(ValidationError):
            InferenceRequest.from_dict(data)
