"""Classify LLM responses and dispatch to type-specific handlers.

Contains:
  - ``LLMResponseType`` — enum for response classification.
  - ``classify_response`` — pure classifier function (no side effects).
  - ``ResponseDispatchMixin`` — handler methods mixed into ``Agent``.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from openhands.sdk.agent.stream_context import StreamContext
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import LLMResponse, Message, TextContent
from openhands.sdk.llm.utils.dsml import reconcile_dsml_message
from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    from openhands.sdk.conversation import (
        ConversationCallbackType,
        ConversationState,
        LocalConversation,
    )
    from openhands.sdk.critic.base import CriticBase, CriticResult
    from openhands.sdk.event import ActionEvent
    from openhands.sdk.llm import (
        MessageToolCall,
        ReasoningItemModel,
        RedactedThinkingBlock,
        ThinkingBlock,
    )
    from openhands.sdk.security.analyzer import SecurityAnalyzerBase

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def is_complete_tool_call(tool_call: MessageToolCall) -> bool:
    """Return True when ``tool_call``'s arguments are complete JSON.

    A reply cut off at the output token limit can still contain a whole tool
    call (the truncation happened in the prose before or after it). Only calls
    whose arguments are valid JSON are safe to run; a half-written call would
    either fail validation or act on truncated input.
    """
    arguments = tool_call.arguments
    if not arguments or not arguments.strip():
        # A no-argument call serializes to an empty object; an empty string is
        # also treated as complete by ``parse_tool_call_arguments``.
        return True
    try:
        json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def has_only_complete_tool_calls(message: Message) -> bool:
    """True when ``message`` carries tool calls and every one is complete."""
    calls = message.tool_calls
    return bool(calls) and all(is_complete_tool_call(call) for call in calls)


class LLMResponseType(StrEnum):
    """Mutually exclusive classification of an LLM response."""

    TOOL_CALLS = "tool_calls"
    CONTENT = "content"
    REASONING_ONLY = "reasoning_only"
    EMPTY = "empty"


def classify_response(message: Message) -> LLMResponseType:
    """Classify an LLM response message into exactly one type.

    Decision priority (first match wins):
      1. TOOL_CALLS  — message contains tool calls
      2. CONTENT     — message contains non-blank TextContent
      3. REASONING_ONLY — message has reasoning but no visible content
      4. EMPTY       — nothing useful

    This function is pure: no side effects, no logging, no mutation.
    """
    if message.tool_calls:
        return LLMResponseType.TOOL_CALLS

    if any(isinstance(c, TextContent) and c.text.strip() for c in message.content):
        return LLMResponseType.CONTENT

    if (
        message.responses_reasoning_item is not None
        or message.reasoning_content is not None
        or message.thinking_blocks
    ):
        return LLMResponseType.REASONING_ONLY

    return LLMResponseType.EMPTY


# ---------------------------------------------------------------------------
# Announced-action detection
# ---------------------------------------------------------------------------

# A text-only reply whose final sentence promises an action ("Let me check the
# anchors output.") but carries no tool call is not a final answer: the model
# stalled mid-plan. Treating it as final ends the run with the promise as the
# last message, which is the "ending short" bug. The phrases below are the
# openers models actually emit before a call; they are matched at the start of
# the *last* sentence so ordinary prose that merely contains "Let me" earlier
# is not misread as a stall.
_ANNOUNCED_ACTION_PREFIXES = (
    "let me",
    "i'll",
    "i will",
    "now i",
    "now let me",
    "next",
    "first",
    "then",
    "finally",
    "let's",
    "lets",
    "i'm going to",
    "i am going to",
    "i need to",
    "i'm now",
    "i am now",
)

# An announcement that trails off with one of these is incomplete by
# construction — the model meant to continue (usually with a tool call) and
# stopped instead.
_ANNOUNCED_ACTION_SUFFIXES = (":", "...", "\u2026")

# Sentence boundary: end punctuation followed by whitespace, or a newline. The
# whitespace requirement keeps decimals ("1.5") and abbreviations ("e.g.") from
# splitting mid-token.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")

# A nudge that produced no tool call may repeat; cap consecutive attempts so a
# model that keeps announcing without acting cannot spin forever.
MAX_CONSECUTIVE_ANNOUNCED_ACTION_NUDGES = 3
ANNOUNCED_ACTION_NUDGE_COUNT_KEY = "announced_action_nudge_count"


def last_sentence(text: str) -> str:
    """Return the trailing sentence of ``text``, trimmed.

    Falls back to the whole (trimmed) string when no boundary is found.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped)]
    for part in reversed(parts):
        if part:
            return part
    return stripped


def announces_action_without_tool_call(message: Message) -> bool:
    """True when a text-only reply promises an action but makes no tool call.

    A reply that already carries tool calls is never an announcement (the model
    acted); the caller only reaches this for content-only responses, but the
    guard keeps the helper safe to call anywhere.
    """
    if message.tool_calls:
        return False
    text = "\n".join(
        part.text for part in message.content if isinstance(part, TextContent)
    )
    if not text.strip():
        return False
    candidate = last_sentence(text)
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.endswith(_ANNOUNCED_ACTION_SUFFIXES):
        return True
    # Compare on the bare opener: strip a leading bullet/numbering the model may
    # have added ("- Let me ...", "2. Now I ...").
    bare = lowered.lstrip("-*\u2022 \t0123456789.)")
    return bare.startswith(_ANNOUNCED_ACTION_PREFIXES)


def _finish_reason(llm_response: LLMResponse) -> str | None:
    """Best-effort read of the provider's finish reason for logging."""
    raw = llm_response.raw_response
    choices = getattr(raw, "choices", None)
    if not choices:
        return getattr(raw, "status", None)
    reason = getattr(choices[0], "finish_reason", None)
    return reason if reason is None else str(reason)


# ---------------------------------------------------------------------------
# Dispatch mixin
# ---------------------------------------------------------------------------


class ResponseDispatchMixin:
    """Handler methods for each ``LLMResponseType``. Mixed into ``Agent``.

    Expects the host class (``Agent``) to provide the members declared in the
    ``TYPE_CHECKING`` block below.
    """

    # Declared for pyright — the actual implementations live on Agent.
    if TYPE_CHECKING:
        critic: CriticBase | None

        def _get_action_event(
            self,
            tool_call: MessageToolCall,
            conversation: LocalConversation,
            llm_response_id: str,
            on_event: ConversationCallbackType,
            security_analyzer: SecurityAnalyzerBase | None = None,
            thought: list[TextContent] | None = None,
            reasoning_content: str | None = None,
            thinking_blocks: (
                list[ThinkingBlock | RedactedThinkingBlock] | None
            ) = None,
            responses_reasoning_item: ReasoningItemModel | None = None,
            stream: StreamContext | None = None,
        ) -> ActionEvent | None: ...

        def _execute_actions(
            self,
            conversation: LocalConversation,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType,
        ) -> None: ...

        async def _aexecute_actions(
            self,
            conversation: LocalConversation,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType,
        ) -> None: ...

        def _requires_user_confirmation(
            self,
            state: ConversationState,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType | None = None,
        ) -> bool: ...

        async def _arequires_user_confirmation(
            self,
            conversation: LocalConversation,
            state: ConversationState,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType,
        ) -> bool: ...

        def _maybe_emit_vllm_tokens(
            self,
            llm_response: LLMResponse,
            on_event: ConversationCallbackType,
        ) -> None: ...

        def _evaluate_with_critic(
            self,
            conversation: LocalConversation,
            event: ActionEvent | MessageEvent,
        ) -> CriticResult | None: ...

    def _handle_tool_calls(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
    ) -> None:
        """Handle LLM response containing tool calls."""
        self._reset_announced_action(state)
        if not all(isinstance(c, TextContent) for c in message.content):
            logger.warning(
                "LLM returned tool calls but message content is not all "
                "TextContent - ignoring non-text content"
            )

        thought_content = [c for c in message.content if isinstance(c, TextContent)]

        action_events: list[ActionEvent] = []
        assert message.tool_calls, "classify_response guarantees tool_calls"
        for i, tool_call in enumerate(message.tool_calls):
            action_event = self._get_action_event(
                tool_call,
                conversation=conversation,
                llm_response_id=llm_response.id,
                on_event=on_event,
                security_analyzer=state.security_analyzer,
                thought=thought_content if i == 0 else [],
                reasoning_content=(message.reasoning_content if i == 0 else None),
                thinking_blocks=(list(message.thinking_blocks) if i == 0 else []),
                responses_reasoning_item=(
                    message.responses_reasoning_item if i == 0 else None
                ),
                # The streamed text is this action's thought, so the first
                # action event is what retires the slot.
                stream=stream if i == 0 else None,
            )
            if action_event is None:
                continue
            action_events.append(action_event)

        if self._requires_user_confirmation(state, action_events, on_event):
            return

        if action_events:
            self._execute_actions(conversation, action_events, on_event)

        self._maybe_emit_vllm_tokens(llm_response, on_event)

    async def _ahandle_tool_calls(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
    ) -> None:
        """Async variant of :meth:`_handle_tool_calls`.

        Delegates tool execution to :meth:`_aexecute_actions` so each
        tool call runs in its own thread and multiple calls are scheduled
        concurrently via :func:`asyncio.gather`.
        """
        self._reset_announced_action(state)
        if not all(isinstance(c, TextContent) for c in message.content):
            logger.warning(
                "LLM returned tool calls but message content is not all "
                "TextContent - ignoring non-text content"
            )

        thought_content = [c for c in message.content if isinstance(c, TextContent)]

        action_events: list[ActionEvent] = []
        assert message.tool_calls, "classify_response guarantees tool_calls"
        for i, tool_call in enumerate(message.tool_calls):
            action_event = self._get_action_event(
                tool_call,
                conversation=conversation,
                llm_response_id=llm_response.id,
                on_event=on_event,
                security_analyzer=state.security_analyzer,
                thought=thought_content if i == 0 else [],
                reasoning_content=(message.reasoning_content if i == 0 else None),
                thinking_blocks=(list(message.thinking_blocks) if i == 0 else []),
                responses_reasoning_item=(
                    message.responses_reasoning_item if i == 0 else None
                ),
                # The streamed text is this action's thought, so the first
                # action event is what retires the slot.
                stream=stream if i == 0 else None,
            )
            if action_event is None:
                continue
            action_events.append(action_event)

        if await self._arequires_user_confirmation(
            conversation, state, action_events, on_event
        ):
            return

        if action_events:
            await self._aexecute_actions(conversation, action_events, on_event)

        self._maybe_emit_vllm_tokens(llm_response, on_event)

    def _handle_content_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        mask_secrets: bool = True,
    ) -> None:
        """Handle LLM response with text content — finishes conversation.

        A reply whose "text" is DeepSeek DSML tool-call markup is reclassified
        here: the markup is parsed into real tool calls and the run continues,
        instead of ending with raw markup as the final message.
        """
        message, unparsed_dsml = reconcile_dsml_message(message)
        if message.tool_calls:
            self._handle_tool_calls(
                message, llm_response, conversation, state, on_event, stream
            )
            return
        if unparsed_dsml:
            self._handle_dsml_failure(
                message,
                llm_response,
                conversation,
                on_event,
                stream,
                mask_secrets=mask_secrets,
            )
            return

        self._emit_message_event(
            message,
            llm_response,
            conversation,
            on_event,
            stream,
            mask_secrets=mask_secrets,
        )
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        if not llm_response.truncated and announces_action_without_tool_call(message):
            self._handle_announced_action_stall(llm_response, state, on_event)
            return
        logger.debug("LLM produced a message response - awaits user input")
        state.execution_status = ConversationExecutionStatus.FINISHED

    async def _ahandle_content_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        mask_secrets: bool = True,
    ) -> None:
        """Async variant of :meth:`_handle_content_response`."""
        message, unparsed_dsml = reconcile_dsml_message(message)
        if message.tool_calls:
            await self._ahandle_tool_calls(
                message, llm_response, conversation, state, on_event, stream
            )
            return
        if unparsed_dsml:
            self._handle_dsml_failure(
                message,
                llm_response,
                conversation,
                on_event,
                stream,
                mask_secrets=mask_secrets,
            )
            return

        self._emit_message_event(
            message,
            llm_response,
            conversation,
            on_event,
            stream,
            mask_secrets=mask_secrets,
        )
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        if not llm_response.truncated and announces_action_without_tool_call(message):
            self._handle_announced_action_stall(llm_response, state, on_event)
            return
        logger.debug("LLM produced a message response - awaits user input")
        state.execution_status = ConversationExecutionStatus.FINISHED

    def _handle_announced_action_stall(
        self,
        llm_response: LLMResponse,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        """Nudge a model that announced an action but made no tool call.

        The reply is already emitted; this decides whether the run continues.
        Consecutive nudges are capped so a model that keeps promising without
        acting cannot loop forever — past the cap the reply is accepted as the
        final answer, which is the pre-existing behaviour. The counter resets
        whenever a tool call actually runs (see ``_reset_announced_action``).
        """
        count = self._announced_action_nudge_count(state)
        if count >= MAX_CONSECUTIVE_ANNOUNCED_ACTION_NUDGES:
            logger.warning(
                "LLM announced an action without a tool call %d times in a row "
                "(finish_reason=%s) - accepting the reply as final",
                count,
                _finish_reason(llm_response),
            )
            state.execution_status = ConversationExecutionStatus.FINISHED
            return

        state.agent_state = {
            **state.agent_state,
            ANNOUNCED_ACTION_NUDGE_COUNT_KEY: count + 1,
        }
        usage = llm_response.metrics.accumulated_token_usage
        logger.warning(
            "LLM announced an action but made no tool call (attempt %d/%d, "
            "finish_reason=%s, prompt_tokens=%s, completion_tokens=%s) - "
            "requesting the tool call",
            count + 1,
            MAX_CONSECUTIVE_ANNOUNCED_ACTION_NUDGES,
            _finish_reason(llm_response),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        self._send_corrective_nudge(
            on_event,
            text=(
                "You said you would act but made no tool call. Make the tool "
                "call now, or call finish if the task is complete."
            ),
        )

    @staticmethod
    def _announced_action_nudge_count(state: ConversationState) -> int:
        """Consecutive announced-action nudges sent without an intervening call."""
        value = state.agent_state.get(ANNOUNCED_ACTION_NUDGE_COUNT_KEY, 0)
        return value if isinstance(value, int) and value > 0 else 0

    @staticmethod
    def _reset_announced_action(state: ConversationState) -> None:
        """Clear the stall counter after a tool call actually runs."""
        if ANNOUNCED_ACTION_NUDGE_COUNT_KEY in state.agent_state:
            state.agent_state = {
                key: value
                for key, value in state.agent_state.items()
                if key != ANNOUNCED_ACTION_NUDGE_COUNT_KEY
            }

    def _handle_dsml_failure(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        mask_secrets: bool = True,
    ) -> None:
        """Emit the reply and nudge the model after unparseable DSML markup."""
        self._emit_message_event(
            message,
            llm_response,
            conversation,
            on_event,
            stream,
            mask_secrets=mask_secrets,
        )
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        self._send_corrective_nudge(
            on_event,
            text=(
                "Your last response used DSML tool-call markup that could not "
                "be parsed into a function call. Please call the tool again "
                "using a proper function call."
            ),
        )

    def _handle_truncated_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,  # noqa: ARG002
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        mask_secrets: bool = True,
    ) -> None:
        """Handle a reply cut off by the output token limit.

        Treating a length-truncated reply as final silently ends the run with a
        half-written answer; instead the partial reply is emitted and the model
        is nudged to continue. Only *incomplete* tool calls are dropped: their
        arguments are truncated, so running them would fail (or, worse, act on
        truncated input), and storing an unexecuted call confuses the history. A
        tool call that survived the cut-off intact is left in place for the
        caller to run.
        """
        logger.warning(
            "LLM response was truncated at the output token limit - "
            "requesting continuation"
        )
        if message.tool_calls and not all(
            is_complete_tool_call(call) for call in message.tool_calls
        ):
            message = message.model_copy(update={"tool_calls": None})
        self._emit_message_event(
            message,
            llm_response,
            conversation,
            on_event,
            stream,
            mask_secrets=mask_secrets,
        )
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        self._send_corrective_nudge(
            on_event,
            text=(
                "Your last response was cut off because it exceeded the output "
                "token limit. Continue from where you stopped, or break the "
                "task into smaller steps."
            ),
        )

    def _handle_no_content_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,  # noqa: ARG002
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        response_type: LLMResponseType,
        mask_secrets: bool = True,
    ) -> None:
        """Handle LLM response with no user-facing content.

        Covers both reasoning-only and empty responses. Emits the message
        event and sends corrective feedback so the model knows it must
        produce a tool call or user-facing content.
        """
        if response_type is LLMResponseType.EMPTY:
            logger.warning("LLM produced empty response - continuing agent loop")
        self._emit_message_event(
            message,
            llm_response,
            conversation,
            on_event,
            stream,
            mask_secrets=mask_secrets,
        )
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        self._send_corrective_nudge(on_event)

    def _emit_message_event(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        stream: StreamContext | None = None,
        *,
        mask_secrets: bool = True,
    ) -> MessageEvent:
        """Create and emit a MessageEvent, running critic if configured.

        The message carries the id its own stream minted, so a client holding
        an open slot retires it on ``frame.event.id == slot.item_id``.
        """
        minted: dict[str, Any] = {}
        if stream is not None and (item_id := stream.claim()):
            minted["id"] = item_id
        msg_event = MessageEvent(
            **minted,
            source="agent",
            llm_message=self._mask_secrets(message, conversation)
            if mask_secrets
            else message,
            llm_response_id=llm_response.id,
        )
        if self.critic is not None and self.critic.mode == "finish_and_message":
            critic_result = self._evaluate_with_critic(conversation, msg_event)
            if critic_result is not None:
                msg_event = msg_event.model_copy(
                    update={"critic_result": critic_result}
                )
        on_event(msg_event)
        # Retired only now: on_event can raise while persisting, and a slot
        # retired before that leaves the client holding it open forever.
        if stream is not None and minted:
            stream.commit()
        return msg_event

    @staticmethod
    def _mask_secrets(message: Message, conversation: LocalConversation) -> Message:
        """Return ``message`` with registered secret values masked in its text.

        ``thinking_blocks`` and ``responses_reasoning_item`` are left alone:
        they are signed provider payloads, and rewriting them invalidates the
        signature replayed on the next request.
        """
        mask = conversation.state.secret_registry.mask_secrets_in_output
        reasoning = message.reasoning_content
        return message.model_copy(
            update={
                "content": [
                    part.model_copy(update={"text": mask(part.text)})
                    if isinstance(part, TextContent)
                    else part
                    for part in message.content
                ],
                "reasoning_content": mask(reasoning) if reasoning else reasoning,
            }
        )

    def _send_corrective_nudge(
        self,
        on_event: ConversationCallbackType,
        text: str | None = None,
    ) -> None:
        """Inject corrective feedback when no tool call and no content.

        The model still receives this as a user-role message, but the event
        source marks that it came from the framework rather than the human.
        ``text`` overrides the default message for specific failure modes
        (unparseable DSML markup, output-limit truncation, an announced action
        that was never made).
        """
        if text is None:
            logger.warning(
                "LLM response contained no tool call and no content"
                " - sending corrective feedback"
            )
        else:
            logger.warning("Sending corrective feedback: %s", text)
        nudge = MessageEvent(
            source="environment",
            llm_message=Message(
                role="user",
                content=[
                    TextContent(
                        text=text
                        or (
                            "Your last response did not include a "
                            "function call or a message. Please "
                            "use a tool to proceed with the task."
                        )
                    )
                ],
            ),
        )
        on_event(nudge)
