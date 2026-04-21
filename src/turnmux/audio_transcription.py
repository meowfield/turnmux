from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import TurnmuxConfig


logger = logging.getLogger(__name__)


class AudioTranscriptionError(RuntimeError):
    """Raised when OpenAI audio transcription fails."""


class AudioTranscriptionNotConfiguredError(AudioTranscriptionError):
    """Raised when voice transcription is requested without OpenAI credentials."""


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def transcribe_audio(
    config: TurnmuxConfig,
    *,
    filename: str,
    content_type: str,
    payload: bytes,
) -> str:
    if not config.openai_api_key:
        raise AudioTranscriptionNotConfiguredError(
            "Voice transcription is not configured. Set `openai_api_key` in `~/.turnmux/config.toml` "
            "or `OPENAI_API_KEY` in the TurnMux environment, then restart the bot."
        )

    url = f"{config.openai_base_url.rstrip('/')}/audio/transcriptions"
    client = _get_client()
    try:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {config.openai_api_key}"},
            files={"file": (filename, payload, content_type)},
            data={"model": config.openai_transcription_model},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AudioTranscriptionError(_http_error_message(exc)) from exc
    except httpx.HTTPError as exc:
        raise AudioTranscriptionError(f"OpenAI transcription request failed: {exc}") from exc

    try:
        body: Any = response.json()
    except json.JSONDecodeError as exc:
        raise AudioTranscriptionError("OpenAI transcription returned a non-JSON response.") from exc

    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise AudioTranscriptionError("OpenAI transcription returned empty text.")
    return text.strip()


async def close_transcription_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code
    detail = _extract_error_message(exc.response)
    if status_code == 401:
        return f"OpenAI transcription returned 401 Unauthorized. {detail or 'Check openai_api_key.'}"
    if status_code == 429:
        return f"OpenAI transcription is rate-limited right now. {detail or 'Try again in a moment.'}"
    if detail:
        return f"OpenAI transcription failed with HTTP {status_code}: {detail}"
    return f"OpenAI transcription failed with HTTP {status_code}."


def _extract_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        text = response.text.strip()
        return text[:200] if text else None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    return None
