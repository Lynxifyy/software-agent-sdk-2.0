"""Discord tool: act on a server over Discord's REST API with the bot token.

The FreeHands Discord bot is added to a server through the Agent Canvas
``connect_discord`` client tool; once it is in, this tool lets the agent create
and organise channels, roles and messages. The bot token is read from the
conversation's secret registry under ``DISCORD_BOT_TOKEN`` — it is never an
action argument and never appears in an observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Self

from pydantic import Field, field_validator, model_validator
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


DISCORD_API_BASE_URL: Final[str] = "https://discord.com/api/v10"
DISCORD_BOT_TOKEN_SECRET_NAME: Final[str] = "DISCORD_BOT_TOKEN"
DISCORD_API_VERSION: Final[str] = "10"

DiscordMethod = Literal["GET", "POST", "PATCH", "PUT", "DELETE"]


class DiscordAction(Action):
    """A single call to Discord's REST API."""

    method: DiscordMethod = Field(
        description=(
            "HTTP method. Use GET to read, POST to create, PATCH to update, "
            "DELETE to remove."
        )
    )
    path: str = Field(
        description=(
            "API path after the version, beginning with a slash — for example "
            '"/users/@me", "/guilds/{guild_id}/channels", or '
            '"/channels/{channel_id}/messages". Never a full URL.'
        )
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description="JSON body for POST/PATCH/PUT requests.",
    )
    query: dict[str, str] | None = Field(
        default=None,
        description='Optional query-string parameters, e.g. {"limit": "50"}.',
    )

    @field_validator("method", mode="before")
    @classmethod
    def _normalize_method(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_path(self) -> Self:
        if not self.path.startswith("/"):
            raise ValueError('path must begin with a slash, e.g. "/users/@me"')
        if "://" in self.path:
            raise ValueError(
                "path must be an API path such as /guilds/<id>/channels, "
                "not an absolute URL"
            )
        return self

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append(f"Discord {self.method} ", style="bold cyan")
        content.append(self.path)
        return content


class DiscordObservation(Observation):
    """Result of a Discord API call, as JSON text or an error message."""

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Discord request failed", style="bold red")
        else:
            content.append("Discord request succeeded", style="bold green")
        if self.text:
            content.append("\n")
            content.append(self.text)
        return content


_DESCRIPTION = (
    "Call Discord's REST API as the FreeHands bot, to build and manage a "
    "Discord server.\n\n"
    "The bot must already be in the server. If it is not, ask the user to run "
    "the connect step first (the Agent Canvas connect menu); you cannot add the "
    "bot yourself.\n\n"
    "Send one request per call. `path` is the API path after "
    "https://discord.com/api/v10 and begins with a slash (e.g. "
    "`/guilds/<guild_id>/channels`); `method` is the "
    "HTTP verb; `body` is the JSON payload for writes. Common operations:\n"
    "- List channels: GET /guilds/<guild_id>/channels\n"
    "- Create a channel: POST /guilds/<guild_id>/channels "
    '{"name": "...", "type": 0}\n'
    '- Create a role: POST /guilds/<guild_id>/roles {"name": "...", "color": 0}\n'
    '- Post a message: POST /channels/<channel_id>/messages {"content": "..."}\n'
    '- Edit a channel: PATCH /channels/<channel_id> {"topic": "..."}\n'
    "- Delete a channel: DELETE /channels/<channel_id>\n\n"
    "Authentication is automatic: the tool sends the bot token from the "
    f"{DISCORD_BOT_TOKEN_SECRET_NAME} secret as `Authorization: Bot ...`. Never "
    "ask the user for the token and never include it in `body` or `path`.\n\n"
    "Rate limits: on HTTP 429 the call is retried after Discord's advertised "
    "delay. A 403 means the bot lacks the permission for that action; report it "
    "to the user instead of retrying."
)


class DiscordTool(ToolDefinition[DiscordAction, DiscordObservation]):
    """Tool for acting on a Discord server over the REST API."""

    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("DiscordTool does not accept parameters")

        # Import here to keep module import light and avoid any import cycles.
        from openhands.tools.discord.impl import DiscordExecutor

        return [
            cls(
                description=_DESCRIPTION,
                action_type=DiscordAction,
                observation_type=DiscordObservation,
                executor=DiscordExecutor(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


# Automatically register when this module is imported.
register_tool(DiscordTool.name, DiscordTool)
