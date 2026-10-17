from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openhands.sdk.llm.call_context import (
    apply_llm_call_context,
    resolve_llm_call_context,
)
from openhands.sdk.llm.options.common import (
    apply_defaults_if_absent,
    apply_extra_body,
    apply_extra_headers,
)
from openhands.sdk.llm.utils.model_features import model_matches


if TYPE_CHECKING:
    from openhands.sdk.llm.call_context import LLMCallContext


# DeepSeek V4/V4.1 dual-mode (Thinking / Non-Thinking) models take a
# request-level ``thinking`` toggle. Matched on the bare token so prefixed model
# strings (``openhands/...``, ``openai/...`` on a custom base URL) are covered —
# LiteLLM only translates ``reasoning_effort`` for a route it recognizes as
# DeepSeek, so the SDK sends the toggle itself.
DEEPSEEK_THINKING_MODELS: list[str] = ["deepseek-v4"]


def select_chat_options(
    llm,
    user_kwargs: dict[str, Any],
    has_tools: bool,
    call_context: LLMCallContext | None = None,
) -> dict[str, Any]:
    """Behavior-preserving extraction of _normalize_call_kwargs.

    This keeps the exact provider-aware mappings and precedence.
    """
    # First pass: apply simple defaults without touching user-supplied values
    max_output_tokens = llm.effective_max_output_tokens
    defaults: dict[str, Any] = {
        "top_k": llm.top_k,
        "top_p": llm.top_p,
        "temperature": llm.temperature,
        # OpenAI-compatible param is `max_completion_tokens`
        "max_completion_tokens": max_output_tokens,
    }
    out = apply_defaults_if_absent(user_kwargs, defaults)

    # Azure -> uses max_tokens instead
    if llm.model.startswith("azure"):
        if "max_completion_tokens" in out:
            out["max_tokens"] = out.pop("max_completion_tokens")

    out = apply_extra_headers(out, llm)

    model_features = llm._model_features()
    supports_reasoning_effort = model_features.supports_reasoning_effort
    if supports_reasoning_effort:
        if llm.reasoning_effort is not None:
            out["reasoning_effort"] = llm.reasoning_effort

    model_name = llm._model_name_for_capabilities()
    if model_features.supports_sampling_params is False or (
        model_features.supports_sampling_params is None
        and supports_reasoning_effort
        and "gemini" not in model_name.lower()
    ):
        out.pop("temperature", None)
        out.pop("top_p", None)
        out.pop("top_k", None)

    # DeepSeek V4/V4.1 dual-mode models: forward the reasoning_effort toggle as
    # DeepSeek's native ``thinking`` parameter. LiteLLM only performs this
    # translation when it recognizes the route as DeepSeek, so a prefixed model
    # string (``openhands/deepseek-v4-flash``, custom base URL) would silently
    # drop it and leave thinking on. Sending it here keeps `none` authoritative.
    if model_matches(model_name, DEEPSEEK_THINKING_MODELS):
        out.pop("reasoning_effort", None)
        if llm.reasoning_effort is not None:
            out["thinking"] = {
                "type": "disabled" if llm.reasoning_effort == "none" else "enabled"
            }

    if model_features.thinking_mode == "manual":
        if llm.extended_thinking_budget and max_output_tokens:
            budget_tokens = min(
                llm.extended_thinking_budget,
                max_output_tokens - 1,
            )
            out["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
            existing = out.get("extra_headers") or {}
            out["extra_headers"] = {
                "anthropic-beta": "interleaved-thinking-2025-05-14",
                **existing,
            }
            out["max_tokens"] = max_output_tokens
        out.pop("temperature", None)
        out.pop("top_p", None)
        out.pop("top_k", None)

    # Tools: if not using native, strip tool_choice so we don't confuse providers
    if not has_tools:
        out.pop("tools", None)
        out.pop("tool_choice", None)

    # Send prompt_cache_retention only if model supports it
    if model_features.supports_prompt_cache_retention and llm.prompt_cache_retention:
        out["prompt_cache_retention"] = llm.prompt_cache_retention

    out = apply_extra_body(out, llm)
    context = resolve_llm_call_context(call_context)
    apply_llm_call_context(
        out,
        context,
        supports_prompt_cache_key=model_features.supports_prompt_cache_key,
    )

    return out
