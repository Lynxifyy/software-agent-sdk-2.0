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
import uuid
from collections.abc import Sequence

from openhands.sdk.llm.message import Message, MessageToolCall, TextContent


DSML_TOKEN = "\uff5cDSML\uff5c"

_DSML = re.escape(DSML_TOKEN)
_TOOL_CALLS_BLOCK = re.compile(
    rf"<{_DSML}tool_calls>(.*?)</{_DSML}tool_calls>", re.DOTALL
)
_INVOKE = re.compile(rf"<{_DSML}invoke\s+([^>]*?)>(.*?)</{_DSML}invoke>", re.DOTALL)
# Attributes are parsed separately from the element body so a parameter is not
# dropped when the model omits ``string`` or orders the attributes differently.
_PARAMETER = re.compile(
    rf"<{_DSML}parameter\s+([^>]*?)>(.*?)</{_DSML}parameter>", re.DOTALL
)
_ATTRIBUTE = re.compile(r'([A-Za-z_][\w-]*)\s*=\s*"([^"]*)"')
# An ``invoke``/``parameter`` element whose ``tool_calls`` wrapper is missing:
# DeepSeek V4 Flash emits these intermittently, so the element is dropped whole
# (value included) instead of leaving the raw value behind in the message text.
_ORPHAN_ELEMENT = re.compile(
    rf"<{_DSML}(?:invoke|parameter)\b[^>]*>.*?</{_DSML}(?:invoke|parameter)>",
    re.DOTALL,
)
# Any DSML tag left over, matched or not, so no markup reaches the user.
_DSML_FRAGMENT = re.compile(rf"</?{_DSML}[^>]*>")


class DsmlParseError(ValueError):
    """A DSML tool-call block was present but could not be parsed."""


def contains_dsml(text: str) -> bool:
    """Return True when ``text`` carries the DSML markup token."""
    return DSML_TOKEN in text


def _strip_dsml(text: str) -> str:
    """Remove DSML markup from ``text``.

    Complete ``tool_calls`` blocks and orphaned ``invoke``/``parameter``
    elements are dropped whole, then any remaining tag is removed. Without the
    orphan pass a malformed block leaves its parameter *values* in the message,
    and without the final pass unmatched closing tags survive.
    """
    text = _TOOL_CALLS_BLOCK.sub("", text)
    text = _ORPHAN_ELEMENT.sub("", text)
    return _DSML_FRAGMENT.sub("", text)


def _parse_parameters(body: str) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for match in _PARAMETER.finditer(body):
        attributes = dict(_ATTRIBUTE.findall(match.group(1)))
        name = attributes.get("name")
        if not name:
            continue
        raw = match.group(2)
        is_string = attributes.get("string")
        if is_string == "true":
            parameters[name] = raw
            continue
        if is_string is None:
            # DeepSeek omits ``string`` on some calls. Assume a JSON value when
            # it parses as one and fall back to the raw text, so the parameter
            # is never dropped.
            try:
                parameters[name] = json.loads(raw)
            except json.JSONDecodeError:
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

    Returns the recovered tool calls and ``text`` with the DSML markup removed.

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
            name = dict(_ATTRIBUTE.findall(invoke.group(1))).get("name", "").strip()
            if not name:
                raise DsmlParseError("DSML invoke without a tool name")
            calls.append(
                MessageToolCall(
                    # A fresh id per parse: the previous positional ``dsml_N``
                    # scheme repeated across turns, and duplicate ids in history
                    # break tool-result matching and are rejected by some APIs.
                    id=f"dsml_{uuid.uuid4().hex}",
                    name=name,
                    arguments=json.dumps(_parse_parameters(invoke.group(2))),
                    origin="completion",
                )
            )
    return calls, _strip_dsml(text)


def reconcile_dsml_message(message: Message) -> tuple[Message, bool]:
    """Convert DeepSeek DSML tool-call markup in ``message`` into real tool calls.

    Returns the rewritten message and a flag set when DSML markup was present
    but unparseable, so the caller can nudge instead of treating the reply as
    the final answer.
    """
    text = "".join(
        part.text for part in message.content if isinstance(part, TextContent)
    )
    if not contains_dsml(text):
        return message, False

    non_text: Sequence[object] = [
        part for part in message.content if not isinstance(part, TextContent)
    ]

    if message.tool_calls:
        # Real tool calls are already present, so nothing to recover — but the
        # stray markup still has to go, or it reaches the user and the next
        # request verbatim.
        return _rebuild(message, non_text, _strip_dsml(text), message.tool_calls), False

    try:
        calls, cleaned = parse_dsml_tool_calls(text)
    except DsmlParseError:
        # A malformed wrapper defeated the block regex; drop the visible markup
        # so raw DSML never reaches the user as the final answer.
        return _rebuild(message, non_text, _strip_dsml(text), None), True

    if not calls:
        return _rebuild(message, non_text, cleaned, None), True

    return _rebuild(message, non_text, cleaned, calls), False


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
