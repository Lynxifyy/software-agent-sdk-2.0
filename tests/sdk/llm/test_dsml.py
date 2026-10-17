"""Tests for DeepSeek DSML tool-call markup parsing."""

import json

import pytest

from openhands.sdk.llm import Message, TextContent
from openhands.sdk.llm.utils.dsml import (
    DSML_TOKEN,
    contains_dsml,
    parse_dsml_tool_calls,
    reconcile_dsml_message,
)


def _tool_calls_block(*invokes: str) -> str:
    body = "\n".join(invokes)
    return f"<{DSML_TOKEN}tool_calls>\n{body}\n</{DSML_TOKEN}tool_calls>"


def _invoke(name: str, *params: str) -> str:
    body = "\n".join(params)
    return f'<{DSML_TOKEN}invoke name="{name}">\n{body}\n</{DSML_TOKEN}invoke>'


def _param(name: str, value: str, *, string: bool) -> str:
    flag = "true" if string else "false"
    return (
        f'<{DSML_TOKEN}parameter name="{name}" string="{flag}">'
        f"{value}</{DSML_TOKEN}parameter>"
    )


def test_contains_dsml():
    assert contains_dsml(f"text <{DSML_TOKEN}tool_calls>") is True
    assert contains_dsml("plain answer with <function=..> markup") is False


def test_parse_string_and_json_parameters():
    text = "Searching now.\n" + _tool_calls_block(
        _invoke(
            "read_file",
            _param("path", "README.md", string=True),
            _param("limit", "5", string=False),
            _param("tags", '["a", "b"]', string=False),
        )
    )

    calls, cleaned = parse_dsml_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].origin == "completion"
    assert json.loads(calls[0].arguments) == {
        "path": "README.md",
        "limit": 5,
        "tags": ["a", "b"],
    }
    assert cleaned.strip() == "Searching now."


def test_parse_multiple_invokes_get_distinct_ids():
    text = _tool_calls_block(
        _invoke("read_file", _param("path", "a.py", string=True)),
        _invoke("read_file", _param("path", "b.py", string=True)),
    )

    calls, _ = parse_dsml_tool_calls(text)

    assert [c.name for c in calls] == ["read_file", "read_file"]
    assert len({c.id for c in calls}) == 2


def test_malformed_invoke_raises():
    # Dropped opening invoke tag: DeepSeek V4 Flash emits this intermittently.
    text = (
        f"<{DSML_TOKEN}tool_calls>\n"
        + _param("path", "src/lib/gateway.ts", string=True)
        + f"\n</{DSML_TOKEN}tool_calls>"
    )

    with pytest.raises(ValueError):
        parse_dsml_tool_calls(text)


def test_reconcile_turns_markup_into_tool_calls():
    message = Message(
        role="assistant",
        content=[
            TextContent(
                text="I will read it. "
                + _tool_calls_block(
                    _invoke("read_file", _param("path", "README.md", string=True))
                )
            )
        ],
    )

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is False
    assert reconciled.tool_calls is not None
    assert reconciled.tool_calls[0].name == "read_file"
    assert json.loads(reconciled.tool_calls[0].arguments) == {"path": "README.md"}
    text = "".join(p.text for p in reconciled.content if isinstance(p, TextContent))
    assert DSML_TOKEN not in text
    assert "I will read it." in text


def test_reconcile_flags_unparseable_markup_and_drops_it():
    message = Message(
        role="assistant",
        content=[
            TextContent(
                text=(
                    f"<{DSML_TOKEN}tool_calls>"
                    + _param("path", "x.ts", string=True)
                    + f"</{DSML_TOKEN}tool_calls>"
                )
            )
        ],
    )

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is True
    assert reconciled.tool_calls is None
    visible = "".join(p.text for p in reconciled.content if isinstance(p, TextContent))
    assert DSML_TOKEN not in visible


def test_reconcile_noop_without_markup():
    message = Message(role="assistant", content=[TextContent(text="All done.")])

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is False
    assert reconciled == message


def test_reconcile_noop_when_tool_calls_present():
    from openhands.sdk.llm import MessageToolCall

    message = Message(
        role="assistant",
        content=[TextContent(text="ok")],
        tool_calls=[
            MessageToolCall(id="tc1", name="bash", arguments="{}", origin="completion")
        ],
    )

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is False
    assert reconciled is message


def test_reconcile_strips_stray_markup_when_tool_calls_present():
    """A real call plus leftover markup keeps the call but loses the markup."""
    from openhands.sdk.llm import MessageToolCall

    message = Message(
        role="assistant",
        content=[
            TextContent(
                text="running "
                + _tool_calls_block(
                    _invoke("bash", _param("command", "ls", string=True))
                )
            )
        ],
        tool_calls=[
            MessageToolCall(id="tc1", name="bash", arguments="{}", origin="completion")
        ],
    )

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is False
    assert reconciled.tool_calls == message.tool_calls
    visible = "".join(p.text for p in reconciled.content if isinstance(p, TextContent))
    assert DSML_TOKEN not in visible
    assert "running" in visible


def test_parse_tolerates_invoke_attributes_and_omitted_string():
    """Extra invoke attributes and a missing ``string`` must not drop a param."""
    text = _tool_calls_block(
        f'<{DSML_TOKEN}invoke name="terminal" extra="x">'
        + f'<{DSML_TOKEN}parameter name="command">ls -la</{DSML_TOKEN}parameter>'
        + f'<{DSML_TOKEN}parameter name="timeout">30</{DSML_TOKEN}parameter>'
        + f"</{DSML_TOKEN}invoke>"
    )

    calls, cleaned = parse_dsml_tool_calls(text)

    assert len(calls) == 1
    assert calls[0].name == "terminal"
    # ``ls -la`` is not valid JSON, so it stays text; ``30`` parses as a number.
    assert json.loads(calls[0].arguments) == {"command": "ls -la", "timeout": 30}
    assert cleaned.strip() == ""


def test_strip_removes_orphan_parameter_values():
    """An unwrapped parameter must be dropped whole, not left as loose text."""
    message = Message(
        role="assistant",
        content=[
            TextContent(
                text=(
                    "done\n"
                    + _param("path", "src/lib/gateway.ts", string=True)
                    + f"</{DSML_TOKEN}tool_calls>"
                )
            )
        ],
    )

    reconciled, unparsed = reconcile_dsml_message(message)

    assert unparsed is True
    visible = "".join(p.text for p in reconciled.content if isinstance(p, TextContent))
    assert DSML_TOKEN not in visible
    assert "gateway.ts" not in visible
