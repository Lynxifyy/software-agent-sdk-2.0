"""Parse DeepSeek DSML tool-call markup into real tool calls.

DeepSeek V4 / V4.1 models normally return structured ``tool_calls``. When the
serving stack fails to convert them, the DSML markup lands verbatim in the
assistant ``content`` and the run otherwise looks like a plain final answer::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="read_file">
    <｜DSML｜parameter name="path" string="true">README.md</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

``DSML_TOKEN`` is U+FF5C ``DSML`` U+FF5C. ``string="true"`` marks a raw string
value; ``string="false"`` marks a JSON value (number, boolean, array, object).
"""

import json
import re
from collections.abc import Sequence

from openhands.sdk.llm.message import Message, MessageToolCall, TextContent


DSML_TOKEN = "\uff5cDSML\uff5c"

_DSML = re.escape(DSML_TOKEN)
_TOOL_CALLS_BLOCK = re.compile(
    rf"<{_DSML}tool_calls>(.*?)</{_DSML}tool_calls>", re.DOTALL
)
_INVOKE = re.compile(
    rf'<{_DSML}invoke\s+name="([^"]*)">(.*?)</{_DSML}invoke>', re.DOTALL
)
_PARAMETER = re.compile(
    rf'<{_DSML}parameter\s+name="([^"]*)"\s+string="(true|false)">'
    rf"(.*?)</{_DSML}parameter>",
    re.DOTALL,
)


class DsmlParseError(ValueError):
    """A DSML tool-call block was present but could not be parsed."""


def contains_dsml(text: str) -> bool:
    """Return True when ``text`` carries the DSML markup token."""
    return DSML_TOKEN in text


def _parse_parameters(body: str) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for match in _PARAMETER.finditer(body):
        name, is_string, raw = match.group(1), match.group(2), match.group(3)
        if is_string == "true":
            parameters[name] = raw
            continue
        try:
            parameters[name] = json.loads(raw)
        except json.JSONDecodeError:
            # ``string="false"`` promises JSON; pass a truncated value through
            # as text so the tool can report the real validation error.
            parameters[name] = raw
    return parameters


def parse_dsml_tool_calls(text: str) -> tuple[list[MessageToolCall], str]:
    """Parse DSML tool-call blocks out of ``text``.

    Returns the recovered tool calls and ``text`` with the DSML blocks removed.

    Raises:
        DsmlParseError: A ``<tool_calls>`` block was found but contained no
            usable ``<invoke>`` entry (for example a dropped opening invoke tag,
            which DeepSeek V4 Flash emits intermittently).
    """
    calls: list[MessageToolCall] = []
    for block in _TOOL_CALLS_BLOCK.finditer(text):
        invokes = list(_INVOKE.finditer(block.group(1)))
        if not invokes:
            raise DsmlParseError("DSML tool_calls block without a valid invoke")
        for invoke in invokes:
            name = invoke.group(1).strip()
            if not name:
                raise DsmlParseError("DSML invoke without a tool name")
            calls.append(
                MessageToolCall(
                    id=f"dsml_{len(calls)}",
                    name=name,
                    arguments=json.dumps(_parse_parameters(invoke.group(2))),
                    origin="completion",
                )
            )
    return calls, _TOOL_CALLS_BLOCK.sub("", text)


def reconcile_dsml_message(message: Message) -> tuple[Message, bool]:
    """Convert DeepSeek DSML tool-call markup in ``message`` into real tool calls.

    Returns the rewritten message and a flag set when DSML markup was present
    but unparseable, so the caller can nudge instead of treating the reply as
    the final answer.
    """
    if message.tool_calls:
        return message, False

    text = "".join(
        part.text for part in message.content if isinstance(part, TextContent)
    )
    if not contains_dsml(text):
        return message, False

    non_text: Sequence[object] = [
        part for part in message.content if not isinstance(part, TextContent)
    ]
    try:
        calls, cleaned = parse_dsml_tool_calls(text)
    except DsmlParseError:
        # A malformed wrapper defeated the block regex; drop the visible text so
        # raw markup never reaches the user as the final answer.
        return _rebuild(message, non_text, "", tool_calls=None), True

    if not calls:
        return _rebuild(message, non_text, cleaned, tool_calls=None), True

    return _rebuild(message, non_text, cleaned, tool_calls=calls), False


def _rebuild(
    message: Message,
    non_text: Sequence[object],
    text: str,
    tool_calls: list[MessageToolCall] | None,
) -> Message:
    content: list[object] = list(non_text)
    if text.strip():
        content.append(TextContent(text=text))
    return message.model_copy(update={"content": content, "tool_calls": tool_calls})
