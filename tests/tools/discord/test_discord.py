"""Tests for the Discord tool: request building, executor behaviour, secrets."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openhands.tools.discord.definition import (
    DISCORD_API_BASE_URL,
    DISCORD_BOT_TOKEN_SECRET_NAME,
    DiscordAction,
    DiscordTool,
)
from openhands.tools.discord.impl import DiscordExecutor


def _response(status: int, body: Any, headers: dict[str, str] | None = None):
    """A minimal stand-in for the stdlib HTTP response context manager."""
    response = MagicMock()
    response.status = status
    response.read.return_value = json.dumps(body).encode()
    response.headers = headers or {}
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestActionSchema:
    def test_method_is_required_and_uppercased(self) -> None:
        action = DiscordAction(
            method="post",
            path="/guilds/123/channels",
            body={"name": "general"},
        )
        assert action.method == "POST"
        assert action.path == "/guilds/123/channels"
        assert action.body == {"name": "general"}

    def test_rejects_an_unsupported_method(self) -> None:
        with pytest.raises(ValueError):
            DiscordAction(method="TRACE", path="/guilds/1")

    def test_rejects_a_path_without_a_leading_slash(self) -> None:
        with pytest.raises(ValueError):
            DiscordAction(method="GET", path="guilds/1")

    def test_defaults(self) -> None:
        action = DiscordAction(method="GET", path="/users/@me")
        assert action.body is None
        assert action.query is None


class TestToolDefinition:
    def test_registers_under_the_expected_name(self) -> None:
        assert DiscordTool.name == "discord"

    def test_create_returns_a_single_tool(self) -> None:
        tools = DiscordTool.create()
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "discord"
        assert "discord.com/api" in tool.description
        assert DISCORD_BOT_TOKEN_SECRET_NAME in tool.description

    def test_annotations_mark_it_as_a_mutating_network_tool(self) -> None:
        tool = DiscordTool.create()[0]
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.openWorldHint is True


class TestExecutor:
    def _conversation(self, token: str | None) -> MagicMock:
        conversation = MagicMock()
        conversation.state.secret_registry.get_secret_value.return_value = token
        return conversation

    def test_requires_the_bot_token_secret(self) -> None:
        executor = DiscordExecutor()
        observation = executor(
            DiscordAction(method="GET", path="/users/@me"),
            self._conversation(None),
        )
        assert observation.is_error is True
        assert DISCORD_BOT_TOKEN_SECRET_NAME in observation.text

    def test_builds_the_request_with_bot_authorization(self) -> None:
        executor = DiscordExecutor()
        with patch(
            "openhands.tools.discord.impl.urllib.request.urlopen",
            return_value=_response(200, {"id": "1", "username": "bot"}),
        ) as urlopen:
            observation = executor(
                DiscordAction(method="GET", path="/users/@me"),
                self._conversation("bot-token"),
            )

        request = urlopen.call_args.args[0]
        assert request.full_url == f"{DISCORD_API_BASE_URL}/users/@me"
        assert request.get_header("Authorization") == "Bot bot-token"
        assert observation.is_error is False
        assert json.loads(observation.text) == {"id": "1", "username": "bot"}

    def test_serializes_the_body_and_query(self) -> None:
        executor = DiscordExecutor()
        with patch(
            "openhands.tools.discord.impl.urllib.request.urlopen",
            return_value=_response(201, {"id": "9"}),
        ) as urlopen:
            executor(
                DiscordAction(
                    method="POST",
                    path="/guilds/1/channels",
                    body={"name": "general"},
                    query={"reason": "setup"},
                ),
                self._conversation("bot-token"),
            )

        request = urlopen.call_args.args[0]
        assert request.data == json.dumps({"name": "general"}).encode()
        assert request.full_url.endswith("/guilds/1/channels?reason=setup")

    def test_maps_a_discord_error_payload_to_an_error_observation(self) -> None:
        executor = DiscordExecutor()
        with patch(
            "openhands.tools.discord.impl.urllib.request.urlopen",
            return_value=_response(
                403, {"message": "Missing Permissions", "code": 50013}
            ),
        ):
            observation = executor(
                DiscordAction(method="POST", path="/guilds/1/channels"),
                self._conversation("bot-token"),
            )

        assert observation.is_error is True
        assert "403" in observation.text
        assert "Missing Permissions" in observation.text

    def test_reports_a_network_failure_without_raising(self) -> None:
        executor = DiscordExecutor()
        with patch(
            "openhands.tools.discord.impl.urllib.request.urlopen",
            side_effect=OSError("connection reset"),
        ):
            observation = executor(
                DiscordAction(method="GET", path="/users/@me"),
                self._conversation("bot-token"),
            )

        assert observation.is_error is True
        assert "connection reset" in observation.text

    def test_refuses_an_absolute_url_path(self) -> None:
        with pytest.raises(ValueError):
            DiscordAction(method="GET", path="/https://evil.example.com")
