"""Tests for detecting provider-truncated LLM responses (finish_reason length)."""

from unittest.mock import MagicMock

from litellm import ResponsesAPIResponse
from litellm.types.utils import Choices, Message as LiteLLMMessage, ModelResponse

from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage


def _completion(finish_reason: str) -> ModelResponse:
    return ModelResponse(
        id="r1",
        choices=[
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=LiteLLMMessage(content="partial", role="assistant"),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _responses(status: str, incomplete_details=None) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="r1",
        created_at=0,
        output=[],
        status=status,
        incomplete_details=incomplete_details,
    )


def test_finish_reason_length_is_truncated():
    assert LLMResponse.is_truncated_response(_completion("length")) is True


def test_finish_reason_stop_is_not_truncated():
    assert LLMResponse.is_truncated_response(_completion("stop")) is False


def test_responses_incomplete_is_truncated():
    resp = _responses("incomplete", incomplete_details={"reason": "max_output_tokens"})

    assert LLMResponse.is_truncated_response(resp) is True


def test_responses_completed_is_not_truncated():
    assert LLMResponse.is_truncated_response(_responses("completed")) is False


def test_responses_stand_in_without_status_is_not_truncated():
    """A spec'd stand-in that omits ``status`` must not raise.

    ``isinstance`` matches anything shaped like a ``ResponsesAPIResponse``, so a
    stand-in missing the field has to read as "not truncated" rather than
    turning an ordinary reply into an ``AttributeError``.
    """
    resp = MagicMock(spec=ResponsesAPIResponse)
    del resp.status

    assert LLMResponse.is_truncated_response(resp) is False


def test_truncated_defaults_to_false():
    """Existing construction sites keep working without the new flag."""
    response = LLMResponse(
        message=Message(role="assistant", content=[TextContent(text="hi")]),
        metrics=MetricsSnapshot(
            model_name="test",
            accumulated_cost=0.0,
            max_budget_per_task=0.0,
            accumulated_token_usage=TokenUsage(model="test"),
        ),
        raw_response=MagicMock(spec=ModelResponse, id="r1"),
    )

    assert response.truncated is False
