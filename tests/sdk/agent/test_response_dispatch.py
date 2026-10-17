"""Unit tests for LLM response classification and dispatch."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from litellm.types.utils import (
    ModelResponse,
)

from openhands.sdk.agent import Agent
from openhands.sdk.agent.response_dispatch import (
    ANNOUNCED_ACTION_NUDGE_COUNT_KEY,
    MAX_CONSECUTIVE_ANNOUNCED_ACTION_NUDGES,
    LLMResponseType,
    announces_action_without_tool_call,
    classify_response,
)
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.conversation.stuck_detector import StuckDetector
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.llm import (
    LLM,
    LLMResponse,
    Message,
    MessageToolCall,
    ReasoningItemModel,
    RedactedThinkingBlock,
    TextContent,
    ThinkingBlock,
)
from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage
from openhands.sdk.tool import Action, Observation, Tool, ToolExecutor, register_tool
from openhands.sdk.tool.tool import ToolDefinition


class _ReadAction(Action):
    path: str


class _ReadObservation(Observation):
    path: str


class _ReadExecutor(ToolExecutor[_ReadAction, _ReadObservation]):
    def __call__(self, action: _ReadAction, conversation=None) -> _ReadObservation:
        return _ReadObservation(path=action.path)


class _ReadTool(ToolDefinition[_ReadAction, _ReadObservation]):
    name = "read_file"

    @classmethod
    def create(cls, conv_state=None):
        return [
            cls(
                description="Read a file",
                action_type=_ReadAction,
                observation_type=_ReadObservation,
                executor=_ReadExecutor(),
            )
        ]


register_tool("ReadTool", _ReadTool)


class _LoopAction(Action):
    command: str


class _LoopObservation(Observation):
    command: str
    exit_code: int


def _msg(**kwargs) -> Message:
    """Shorthand to build a Message with defaults."""
    return Message(role="assistant", **kwargs)


def _tool_call() -> MessageToolCall:
    return MessageToolCall(id="tc1", name="bash", arguments="{}", origin="completion")


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(
                tool_calls=[_tool_call()],
                content=[TextContent(text="Let me run this")],
                reasoning_content="I should use bash",
            ),
            id="row1-tools+content+reasoning",
        ),
        pytest.param(
            dict(
                tool_calls=[_tool_call()],
                content=[TextContent(text="Running command")],
            ),
            id="row2-tools+content",
        ),
        pytest.param(
            dict(tool_calls=[_tool_call()], reasoning_content="Thinking about it..."),
            id="row3-tools+reasoning",
        ),
        pytest.param(
            dict(tool_calls=[_tool_call()]),
            id="row4-tools-only",
        ),
        pytest.param(
            dict(tool_calls=[_tool_call()], content=[]),
            id="tools-with-empty-content",
        ),
    ],
)
def test_tool_calls_response(kwargs):
    """Any message with tool_calls classifies as TOOL_CALLS."""
    assert classify_response(_msg(**kwargs)) == LLMResponseType.TOOL_CALLS


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(
                content=[TextContent(text="The answer is 42")],
                reasoning_content="Let me calculate...",
            ),
            id="row5-content+reasoning",
        ),
        pytest.param(
            dict(content=[TextContent(text="Hello world")]),
            id="row6-content-only",
        ),
        pytest.param(
            dict(
                content=[TextContent(text="Here is my answer")],
                thinking_blocks=[
                    ThinkingBlock(thinking="Let me think", signature="sig")
                ],
            ),
            id="content-with-thinking-blocks",
        ),
    ],
)
def test_content_response(kwargs):
    """No tool_calls + non-blank TextContent classifies as CONTENT."""
    assert classify_response(_msg(**kwargs)) == LLMResponseType.CONTENT


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(reasoning_content="Let me think about this..."),
            id="row7a-reasoning-content",
        ),
        pytest.param(
            dict(
                content=[],
                thinking_blocks=[
                    ThinkingBlock(thinking="The answer is 2", signature="sig-1")
                ],
            ),
            id="row7b-thinking-blocks",
        ),
        pytest.param(
            dict(
                content=[],
                thinking_blocks=[RedactedThinkingBlock(data="encrypted")],
            ),
            id="row7c-redacted-thinking",
        ),
        pytest.param(
            dict(
                content=[],
                responses_reasoning_item=ReasoningItemModel(
                    id="ri-1", summary=["thinking"]
                ),
            ),
            id="row7d-responses-reasoning-item",
        ),
    ],
)
def test_reasoning_only_response(kwargs):
    """No tool_calls, no visible content, but reasoning classifies as REASONING_ONLY."""
    assert classify_response(_msg(**kwargs)) == LLMResponseType.REASONING_ONLY


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(content=[]), id="row8-empty-content"),
        pytest.param(
            dict(content=[TextContent(text="   \n  ")]),
            id="whitespace-only-content",
        ),
        pytest.param(
            dict(content=[], thinking_blocks=[]),
            id="empty-content-and-thinking-blocks",
        ),
    ],
)
def test_empty_response(kwargs):
    """No tool_calls, no content, no reasoning classifies as EMPTY."""
    assert classify_response(_msg(**kwargs)) == LLMResponseType.EMPTY


# ---------------------------------------------------------------------------
# ResponseDispatchMixin (via Agent integration)
# ---------------------------------------------------------------------------


def _make_metrics() -> MetricsSnapshot:
    return MetricsSnapshot(
        model_name="test",
        accumulated_cost=0.0,
        max_budget_per_task=0.0,
        accumulated_token_usage=TokenUsage(model="test"),
    )


def _make_llm_response(message: Message) -> LLMResponse:
    return LLMResponse(
        message=message,
        metrics=_make_metrics(),
        raw_response=MagicMock(spec=ModelResponse, id="r1"),
    )


def _run_single_step(
    llm_response: LLMResponse,
    *,
    seed_events: list[Event] | None = None,
    record_emitted_events_in_state: bool = False,
    secrets: dict[str, str] | None = None,
    prepare: Callable[[LocalConversation], None] | None = None,
    tools: list[str] | None = None,
) -> tuple[list[Event], LocalConversation]:
    """Run one agent step with a canned LLM response."""
    from pydantic import PrivateAttr

    class SingleShotLLM(LLM):
        _response: LLMResponse = PrivateAttr()

        def __init__(self, response: LLMResponse):
            super().__init__(model="test-model")
            self._response = response

        def completion(  # type: ignore[override]
            self, *, messages, tools=None, **kwargs
        ) -> LLMResponse:
            return self._response

    llm = SingleShotLLM(llm_response)
    agent = Agent(llm=llm, tools=[Tool(name=n) for n in (tools or [])])
    conversation = Conversation(agent=agent)
    conversation._ensure_agent_ready()
    if secrets is not None:
        conversation.update_secrets(secrets)
    if seed_events is not None:
        for event in seed_events:
            conversation.state.append_event(event)
    if prepare is not None:
        prepare(conversation)

    events: list[Event] = []

    def on_event(e: Event) -> None:
        events.append(e)
        if record_emitted_events_in_state:
            conversation.state.append_event(e)

    agent.step(conversation, on_event=on_event)
    return events, conversation


async def _run_single_astep(
    llm_response: LLMResponse,
    *,
    tools: list[str] | None = None,
) -> tuple[list[Event], LocalConversation]:
    """Run one agent async step with a canned LLM response."""
    from pydantic import PrivateAttr

    class SingleShotLLM(LLM):
        _response: LLMResponse = PrivateAttr()

        def __init__(self, response: LLMResponse):
            super().__init__(model="test-model")
            self._response = response

        def completion(  # type: ignore[override]
            self, *, messages, tools=None, **kwargs
        ) -> LLMResponse:
            return self._response

        async def acompletion(  # type: ignore[override]
            self, *, messages, tools=None, **kwargs
        ) -> LLMResponse:
            return self._response

    llm = SingleShotLLM(llm_response)
    agent = Agent(llm=llm, tools=[Tool(name=n) for n in (tools or [])])
    conversation = Conversation(agent=agent)
    conversation._ensure_agent_ready()

    events: list[Event] = []
    await agent.astep(conversation, on_event=events.append)
    return events, conversation


def _user_message(text: str) -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


def _repeating_terminal_loop_events(repeat_count: int) -> list[Event]:
    loop_events: list[Event] = []
    for i in range(repeat_count):
        action = ActionEvent(
            source="agent",
            thought=[TextContent(text="I need to run ls command")],
            action=_LoopAction(command="ls"),
            tool_name="terminal",
            tool_call_id=f"call_{i}",
            tool_call=MessageToolCall(
                id=f"call_{i}",
                name="terminal",
                arguments='{"command": "ls"}',
                origin="completion",
            ),
            llm_response_id=f"response_{i}",
        )
        loop_events.append(action)
        loop_events.append(
            ObservationEvent(
                source="environment",
                observation=_LoopObservation.from_text(
                    text="file1.txt\nfile2.txt",
                    command="ls",
                    exit_code=0,
                ),
                action_id=action.id,
                tool_name="terminal",
                tool_call_id=f"call_{i}",
            )
        )
    return loop_events


def test_content_response_sets_finished():
    """_handle_content_response sets execution status to FINISHED."""
    msg = Message(role="assistant", content=[TextContent(text="Done!")])
    events, convo = _run_single_step(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status == ConversationExecutionStatus.FINISHED
    assert len(msg_events) == 1
    assert msg_events[0].source == "agent"


def test_empty_response_sends_nudge():
    """_handle_no_content_response emits agent message + corrective nudge."""
    msg = Message(role="assistant", content=[])
    events, convo = _run_single_step(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert len(msg_events) == 2
    assert msg_events[0].source == "agent"
    assert msg_events[1].source == "environment"
    assert msg_events[1].llm_message.role == "user"
    nudge_content = msg_events[1].llm_message.content[0]
    assert isinstance(nudge_content, TextContent)
    assert "function call" in nudge_content.text


def test_reasoning_only_sends_nudge():
    """_handle_no_content_response sends corrective nudge for reasoning-only."""
    msg = Message(role="assistant", reasoning_content="Let me think...")
    events, convo = _run_single_step(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert len(msg_events) == 2
    assert msg_events[0].source == "agent"
    assert msg_events[1].source == "environment"
    assert msg_events[1].llm_message.role == "user"


def test_corrective_nudge_does_not_reset_stuck_detection_window():
    """Framework feedback must not count as a new human turn for stuck detection."""
    seed_events = [
        _user_message("Please keep trying ls"),
        *_repeating_terminal_loop_events(repeat_count=4),
    ]
    msg = Message(role="assistant", content=[])
    events, convo = _run_single_step(
        _make_llm_response(msg),
        seed_events=seed_events,
        record_emitted_events_in_state=True,
    )
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert len(msg_events) == 2
    assert msg_events[1].source == "environment"
    assert msg_events[1].llm_message.role == "user"
    assert StuckDetector(convo.state).is_stuck() is True


def test_tool_calls_response_executes_actions():
    """_handle_tool_calls creates and executes action events."""
    tool_call = MessageToolCall(
        id="tc-finish",
        name="finish",
        arguments='{"message": "All done"}',
        origin="completion",
    )
    msg = Message(
        role="assistant",
        tool_calls=[tool_call],
        content=[TextContent(text="Finishing up")],
    )
    events, convo = _run_single_step(_make_llm_response(msg))
    action_events = [e for e in events if isinstance(e, ActionEvent)]

    assert len(action_events) == 1
    assert action_events[0].tool_call_id == "tc-finish"
    assert convo.state.execution_status == ConversationExecutionStatus.FINISHED


def test_batch_action_events_are_emitted_consecutively():
    """Pin the invariant the metrics visualizer relies on (#4189).

    Parallel tool calls from one LLM response are emitted as a consecutive run
    of ActionEvents that share one llm_response_id, in tool-call order, and only
    the first carries the turn's thought. ``DefaultConversationVisualizer.
    _claim_batch_primary`` tracks only the *previous* response id to spot batch
    siblings in O(1); if emission ever stops being consecutive, that shortcut
    would silently mislabel metrics -- this test should catch it first.
    """
    tool_calls = [
        MessageToolCall(
            id="tc-1",
            name="think",
            arguments='{"thought": "first"}',
            origin="completion",
        ),
        MessageToolCall(
            id="tc-2",
            name="think",
            arguments='{"thought": "second"}',
            origin="completion",
        ),
    ]
    msg = Message(
        role="assistant",
        tool_calls=tool_calls,
        content=[TextContent(text="Thinking in parallel")],
    )
    events, _ = _run_single_step(_make_llm_response(msg))

    action_events = [e for e in events if isinstance(e, ActionEvent)]
    assert len(action_events) == 2
    # One LLM response -> one shared llm_response_id for the whole batch.
    assert len({a.llm_response_id for a in action_events}) == 1
    # Emitted as a consecutive run (nothing of another response interleaves),
    # in tool-call order.
    assert [a.tool_call_id for a in action_events] == ["tc-1", "tc-2"]
    # Only the first event of the batch carries the turn's thought.
    assert action_events[0].thought and not action_events[1].thought


def _agent_message_text(events: list[Event]) -> str:
    """Text of the single agent MessageEvent emitted by a step."""
    msg_event = next(
        e for e in events if isinstance(e, MessageEvent) and e.source == "agent"
    )
    part = msg_event.llm_message.content[0]
    assert isinstance(part, TextContent)
    return part.text


def test_content_response_masks_registered_secret():
    """The model's own words are masked in the durable MessageEvent.

    Only tool output was masked before, so a secret the model echoed back was
    persisted verbatim into the copy that is stored, replayed to the LLM,
    rendered, exported and POSTed to webhooks.
    """
    msg = Message(
        role="assistant", content=[TextContent(text="the value is supersecret now")]
    )
    events, _ = _run_single_step(
        _make_llm_response(msg), secrets={"TOKEN": "supersecret"}
    )

    text = _agent_message_text(events)
    assert "supersecret" not in text
    assert text == "the value is <secret-hidden> now"


def test_content_response_masks_secret_joined_from_stream_chunks():
    """A secret straddling two deltas is masked in the durable message.

    The ``Agent`` path accumulates deltas before it builds the ``Message``, so
    neither chunk matches on its own — only the reassembled text does, and that
    is what the durable copy is built from.
    """
    chunks = ["the value is super", "secret now"]
    msg = Message(role="assistant", content=[TextContent(text="".join(chunks))])
    events, _ = _run_single_step(
        _make_llm_response(msg), secrets={"TOKEN": "supersecret"}
    )

    assert _agent_message_text(events) == "the value is <secret-hidden> now"


def test_content_response_without_secrets_is_left_alone():
    """With nothing registered, the message text is passed through unchanged."""
    msg = Message(role="assistant", content=[TextContent(text="nothing to hide")])
    events, _ = _run_single_step(_make_llm_response(msg))

    assert _agent_message_text(events) == "nothing to hide"


def test_reasoning_only_response_masks_registered_secret():
    """The nudge path routes through the same chokepoint.

    A reasoning-only message carries no ``content``, so masking that alone
    would be a no-op here — ``reasoning_content`` is the text that leaks.
    """
    msg = Message(role="assistant", reasoning_content="the value is supersecret now")
    events, _ = _run_single_step(
        _make_llm_response(msg), secrets={"TOKEN": "supersecret"}
    )

    msg_event = next(
        e for e in events if isinstance(e, MessageEvent) and e.source == "agent"
    )
    assert msg_event.llm_message.reasoning_content == "the value is <secret-hidden> now"


def test_content_response_masks_value_dropped_from_secret_sources():
    """Masking follows the tracked values, not the registry's source list.

    ``EventService.activate_credential_binding`` pops a name from
    ``secret_sources`` while the resolved value stays tracked for masking, so
    keying off ``secret_sources`` would leak a live credential here.
    """

    def drop_source(convo: LocalConversation) -> None:
        registry = convo.state.secret_registry
        registry.get_secret_value("TOKEN")  # resolve -> tracked for masking
        with convo.state:
            convo.state.secret_registry = registry.model_copy(
                update={"secret_sources": {}}
            )

    msg = Message(
        role="assistant", content=[TextContent(text="the value is supersecret now")]
    )
    events, convo = _run_single_step(
        _make_llm_response(msg),
        secrets={"TOKEN": "supersecret"},
        prepare=drop_source,
    )

    assert not convo.state.secret_registry.secret_sources
    assert _agent_message_text(events) == "the value is <secret-hidden> now"


# ---------------------------------------------------------------------------
# DeepSeek DSML tool-call markup
# ---------------------------------------------------------------------------


_DSML = "\uff5cDSML\uff5c"


def _dsml_block(name: str, param: str, value: str) -> str:
    return (
        f"<{_DSML}tool_calls>\n"
        f'<{_DSML}invoke name="{name}">\n'
        f'<{_DSML}parameter name="{param}" string="true">{value}'
        f"</{_DSML}parameter>\n"
        f"</{_DSML}invoke>\n"
        f"</{_DSML}tool_calls>"
    )


def test_dsml_markup_is_parsed_into_a_tool_call():
    """A reply whose text is DSML markup runs the tool instead of finishing.

    Without recovery the run ends and the raw markup becomes the final message.
    """
    msg = Message(
        role="assistant",
        content=[
            TextContent(
                text="I will read it. " + _dsml_block("read_file", "path", "README.md")
            )
        ],
    )
    events, convo = _run_single_step(_make_llm_response(msg), tools=["ReadTool"])
    action_events = [e for e in events if isinstance(e, ActionEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert len(action_events) == 1
    assert action_events[0].tool_name == "read_file"
    action = action_events[0].action
    assert isinstance(action, _ReadAction)
    assert action.path == "README.md"


def test_dsml_markup_never_reaches_the_user_as_the_final_message():
    """Raw DSML is stripped from the emitted message even on recovery."""
    msg = Message(
        role="assistant",
        content=[TextContent(text=_dsml_block("read_file", "path", "a.py"))],
    )
    events, _ = _run_single_step(_make_llm_response(msg), tools=["ReadTool"])

    agent_messages = [
        e for e in events if isinstance(e, MessageEvent) and e.source == "agent"
    ]
    for event in agent_messages:
        for part in event.llm_message.content:
            if isinstance(part, TextContent):
                assert _DSML not in part.text


def test_unparseable_dsml_sends_a_corrective_nudge():
    """Malformed DSML nudges the model instead of finishing the run."""
    malformed = (
        f"<{_DSML}tool_calls>\n"
        f'<{_DSML}parameter name="path" string="true">x.ts</{_DSML}parameter>\n'
        f"</{_DSML}tool_calls>"
    )
    msg = Message(role="assistant", content=[TextContent(text=malformed)])
    events, convo = _run_single_step(_make_llm_response(msg), tools=["ReadTool"])
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert msg_events[-1].source == "environment"
    assert msg_events[-1].llm_message.role == "user"


def test_plain_content_still_finishes():
    """Ordinary prose is untouched by DSML recovery."""
    msg = Message(role="assistant", content=[TextContent(text="All done!")])
    events, convo = _run_single_step(_make_llm_response(msg))

    assert convo.state.execution_status == ConversationExecutionStatus.FINISHED


# ---------------------------------------------------------------------------
# finish_reason == "length"
# ---------------------------------------------------------------------------


def _make_truncated_response(message: Message) -> LLMResponse:
    """A response the provider cut off at the output token limit."""
    return LLMResponse(
        message=message,
        metrics=_make_metrics(),
        raw_response=MagicMock(spec=ModelResponse, id="r1"),
        truncated=True,
    )


def test_truncated_content_response_does_not_finish():
    """A reply cut off at the token limit continues instead of finishing."""
    msg = Message(role="assistant", content=[TextContent(text="Here is the first")])
    events, convo = _run_single_step(_make_truncated_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert msg_events[-1].source == "environment"
    nudge = msg_events[-1].llm_message.content[0]
    assert isinstance(nudge, TextContent)
    assert "cut off" in nudge.text


def test_truncated_tool_call_response_still_executes():
    """A truncated reply that did produce a valid tool call is not re-routed."""
    tool_call = MessageToolCall(
        id="tc-1",
        name="finish",
        arguments='{"message": "done"}',
        origin="completion",
    )
    msg = Message(role="assistant", tool_calls=[tool_call])
    events, _ = _run_single_step(_make_truncated_response(msg))
    action_events = [e for e in events if isinstance(e, ActionEvent)]

    assert len(action_events) == 1


# ---------------------------------------------------------------------------
# Announced action with no tool call ("ending short")
# ---------------------------------------------------------------------------


def test_announces_action_detects_promise_without_call():
    """A trailing promise of an action is detected; a plain answer is not."""
    assert announces_action_without_tool_call(
        _msg(content=[TextContent(text="Let me check the anchors output.")])
    )
    assert announces_action_without_tool_call(
        _msg(content=[TextContent(text="Now I'll run the tests:")])
    )
    assert announces_action_without_tool_call(
        _msg(content=[TextContent(text="I'll inspect the file...")])
    )
    # A tool call means the model acted.
    assert not announces_action_without_tool_call(
        _msg(tool_calls=[_tool_call()], content=[TextContent(text="Let me run this.")])
    )
    # A completed answer must still finish.
    assert not announces_action_without_tool_call(
        _msg(content=[TextContent(text="The task is complete. All tests pass.")])
    )
    # "Let me" earlier in the message does not make the final answer a stall.
    assert not announces_action_without_tool_call(
        _msg(
            content=[
                TextContent(
                    text="Let me explain. The anchors output shows 3 matches. Done."
                )
            ]
        )
    )


def test_announced_action_stall_does_not_finish_and_nudges():
    """The reported bug: a promise with no call must not end the run."""
    msg = Message(
        role="assistant", content=[TextContent(text="Let me check the anchors output.")]
    )
    events, convo = _run_single_step(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    # The reply is still emitted, then the corrective nudge.
    assert msg_events[0].source == "agent"
    assert msg_events[-1].source == "environment"
    nudge = msg_events[-1].llm_message.content[0]
    assert isinstance(nudge, TextContent)
    assert "made no tool call" in nudge.text


def test_announced_action_nudges_are_capped():
    """Past the cap the reply is accepted as final rather than nudging forever."""
    msg = Message(role="assistant", content=[TextContent(text="Let me check.")])
    _, convo = _run_single_step(_make_llm_response(msg))
    convo.state.agent_state = {
        **convo.state.agent_state,
        ANNOUNCED_ACTION_NUDGE_COUNT_KEY: MAX_CONSECUTIVE_ANNOUNCED_ACTION_NUDGES,
    }

    events: list[Event] = []
    convo.agent.step(convo, on_event=events.append)
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status == ConversationExecutionStatus.FINISHED
    assert len(msg_events) == 1
    assert msg_events[0].source == "agent"


def test_tool_call_resets_announced_action_counter():
    """A real tool call clears the stall counter so a later stall starts fresh."""
    _, convo = _run_single_step(
        _make_llm_response(Message(role="assistant", content=[TextContent(text="Hi.")]))
    )
    convo.state.agent_state = {
        **convo.state.agent_state,
        ANNOUNCED_ACTION_NUDGE_COUNT_KEY: 2,
    }

    tool_call = MessageToolCall(
        id="tc-1", name="finish", arguments='{"message": "done"}', origin="completion"
    )
    tool_msg = Message(role="assistant", tool_calls=[tool_call])
    convo.agent._handle_tool_calls(
        tool_msg,
        _make_llm_response(tool_msg),
        convo,
        convo.state,
        lambda e: None,
    )

    assert ANNOUNCED_ACTION_NUDGE_COUNT_KEY not in convo.state.agent_state


@pytest.mark.asyncio
async def test_announced_action_stall_nudges_on_async_path():
    """The async twin behaves the same as the sync path."""
    msg = Message(
        role="assistant", content=[TextContent(text="Let me check the anchors output.")]
    )
    events, convo = await _run_single_astep(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status != ConversationExecutionStatus.FINISHED
    assert msg_events[0].source == "agent"
    assert msg_events[-1].source == "environment"
    nudge = msg_events[-1].llm_message.content[0]
    assert isinstance(nudge, TextContent)
    assert "made no tool call" in nudge.text


@pytest.mark.asyncio
async def test_completed_answer_still_finishes_on_async_path():
    """A genuine final answer on the async path still ends the run."""
    msg = Message(role="assistant", content=[TextContent(text="The task is complete.")])
    events, convo = await _run_single_astep(_make_llm_response(msg))
    msg_events = [e for e in events if isinstance(e, MessageEvent)]

    assert convo.state.execution_status == ConversationExecutionStatus.FINISHED
    assert len(msg_events) == 1
