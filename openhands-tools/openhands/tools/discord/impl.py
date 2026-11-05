"""Discord tool executor: authenticated REST calls with rate-limit retries."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Final

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.discord.definition import (
    DISCORD_API_BASE_URL,
    DISCORD_API_VERSION,
    DISCORD_BOT_TOKEN_SECRET_NAME,
    DiscordAction,
    DiscordObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
# Discord 429s carry `retry_after` seconds; retry a bounded number of times.
MAX_RATE_LIMIT_RETRIES: Final[int] = 3
MAX_RETRY_WAIT_SECONDS: Final[float] = 30.0


def _read_token(conversation: LocalConversation | None) -> str | None:
    """Resolve the bot token from the conversation's secret registry."""
    if conversation is None:
        return None
    try:
        registry = conversation.state.secret_registry
        return registry.get_secret_value(DISCORD_BOT_TOKEN_SECRET_NAME)
    except Exception as error:  # noqa: BLE001 - never let secret lookup crash a tool
        logger.warning(f"Discord token lookup failed: {error}")
        return None


def _retry_after_seconds(response: Any, payload: Any) -> float:
    """Delay to wait after a 429, from the body or the Retry-After header."""
    seconds: float | None = None
    if isinstance(payload, dict):
        value = payload.get("retry_after")
        if isinstance(value, (int, float)):
            seconds = float(value)
    if seconds is None:
        header = response.headers.get("Retry-After") if response.headers else None
        if header:
            try:
                seconds = float(header)
            except (TypeError, ValueError):
                seconds = None
    if seconds is None:
        seconds = 1.0
    return min(max(seconds, 0.0), MAX_RETRY_WAIT_SECONDS)


class DiscordExecutor(ToolExecutor[DiscordAction, DiscordObservation]):
    """Execute Discord REST calls with the conversation's bot token."""

    def __init__(self, base_url: str = DISCORD_API_BASE_URL) -> None:
        self.base_url: str = base_url.rstrip("/")

    def _build_url(self, action: DiscordAction) -> str:
        url = f"{self.base_url}{action.path}"
        if action.query:
            url = f"{url}?{urllib.parse.urlencode(action.query)}"
        return url

    def _send(self, request: urllib.request.Request) -> tuple[int, Any, Any]:
        """Send one request; return (status, decoded body, raw response)."""
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
            body: Any
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw.decode(errors="replace")
            return resp.status, body, resp

    def __call__(
        self,
        action: DiscordAction,
        conversation: LocalConversation | None = None,
    ) -> DiscordObservation:
        token = _read_token(conversation)
        if not token:
            return DiscordObservation.from_text(
                text=(
                    f"The {DISCORD_BOT_TOKEN_SECRET_NAME} secret is not available "
                    "to this conversation. Ask the user to add it in Settings > "
                    "Secrets, or run the connect step, then try again."
                ),
                is_error=True,
            )

        data = json.dumps(action.body).encode() if action.body is not None else None
        headers = {
            "Authorization": f"Bot {token}",
            "Accept": "application/json",
            "User-Agent": f"DiscordBot (openhands-agent-canvas, {DISCORD_API_VERSION})",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self._build_url(action),
            data=data,
            headers=headers,
            method=action.method,
        )

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                status, body, response = self._send(request)
            except urllib.error.HTTPError as error:
                raw = error.read()
                try:
                    error_body: Any = json.loads(raw) if raw else None
                except json.JSONDecodeError:
                    error_body = raw.decode(errors="replace")

                if error.code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                    delay = _retry_after_seconds(error, error_body)
                    logger.debug(f"Discord rate limited; retrying in {delay}s")
                    time.sleep(delay)
                    continue

                message = (
                    error_body.get("message")
                    if isinstance(error_body, dict)
                    else error_body
                )
                return DiscordObservation.from_text(
                    text=(
                        f"Discord API returned HTTP {error.code} for "
                        f"{action.method} {action.path}: {message or error.reason}"
                    ),
                    is_error=True,
                )
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                return DiscordObservation.from_text(
                    text=(
                        f"Discord request {action.method} {action.path} failed: {error}"
                    ),
                    is_error=True,
                )

            # A 429 can also arrive with a normal response object.
            if status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                delay = _retry_after_seconds(response, body)
                logger.debug(f"Discord rate limited; retrying in {delay}s")
                time.sleep(delay)
                continue

            text = json.dumps(body, ensure_ascii=False) if body is not None else ""
            if status >= 400:
                message = body.get("message") if isinstance(body, dict) else body
                return DiscordObservation.from_text(
                    text=(
                        f"Discord API returned HTTP {status} for "
                        f"{action.method} {action.path}: {message or ''}"
                    ).rstrip(),
                    is_error=True,
                )
            return DiscordObservation.from_text(text=text, is_error=False)

        return DiscordObservation.from_text(
            text=(
                f"Discord rate limited {action.method} {action.path} beyond "
                f"{MAX_RATE_LIMIT_RETRIES} retries. Try again shortly."
            ),
            is_error=True,
        )
