"""Discord tool package.

Provides a ``discord`` tool that acts on a Discord server over the REST API as
the FreeHands bot. The bot is added to a server by the user through Agent
Canvas's connect menu; the token is read from the conversation secret registry
under ``DISCORD_BOT_TOKEN``.

Usage:
    from openhands.tools.discord import DiscordTool

    agent = Agent(
        llm=llm,
        tools=[Tool(name=DiscordTool.name)],
    )
"""

from openhands.tools.discord.definition import (
    DISCORD_API_BASE_URL,
    DISCORD_BOT_TOKEN_SECRET_NAME,
    DiscordAction,
    DiscordObservation,
    DiscordTool,
)
from openhands.tools.discord.impl import DiscordExecutor


__all__ = [
    "DISCORD_API_BASE_URL",
    "DISCORD_BOT_TOKEN_SECRET_NAME",
    "DiscordAction",
    "DiscordObservation",
    "DiscordTool",
    "DiscordExecutor",
]
