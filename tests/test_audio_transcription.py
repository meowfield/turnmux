from __future__ import annotations

from pathlib import Path
import unittest

import httpx

from turnmux.audio_transcription import (
    AudioTranscriptionError,
    AudioTranscriptionNotConfiguredError,
    transcribe_audio,
)
from turnmux.config import TurnmuxConfig


class AudioTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_audio_posts_file_to_openai(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {"text": "transcribed text"}

        class FakeClient:
            async def post(self, url, *, headers, files, data):
                captured["url"] = url
                captured["headers"] = headers
                captured["files"] = files
                captured["data"] = data
                return FakeResponse()

        config = TurnmuxConfig(
            telegram_bot_token="token",
            allowed_user_ids=(1,),
            allowed_roots=(Path.cwd(),),
            tmux_session_name="turnmux",
            claude_command=("claude",),
            codex_command=("codex",),
            opencode_command=None,
            opencode_model=None,
            config_path=Path("/tmp/turnmux.toml"),
            relay_claude_thinking=False,
            openai_api_key="sk-test",
            openai_base_url="https://api.openai.com/v1/",
            openai_transcription_model="gpt-4o-transcribe",
        )

        with unittest.mock.patch("turnmux.audio_transcription._get_client", return_value=FakeClient()):
            result = await transcribe_audio(
                config,
                filename="voice.ogg",
                content_type="audio/ogg",
                payload=b"ogg-data",
            )

        self.assertEqual(result, "transcribed text")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/audio/transcriptions")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer sk-test"})
        self.assertEqual(captured["data"], {"model": "gpt-4o-transcribe"})
        self.assertEqual(captured["files"], {"file": ("voice.ogg", b"ogg-data", "audio/ogg")})

    async def test_transcribe_audio_requires_api_key(self) -> None:
        config = TurnmuxConfig(
            telegram_bot_token="token",
            allowed_user_ids=(1,),
            allowed_roots=(Path.cwd(),),
            tmux_session_name="turnmux",
            claude_command=("claude",),
            codex_command=("codex",),
            opencode_command=None,
            opencode_model=None,
            config_path=Path("/tmp/turnmux.toml"),
            relay_claude_thinking=False,
        )

        with self.assertRaises(AudioTranscriptionNotConfiguredError):
            await transcribe_audio(
                config,
                filename="voice.ogg",
                content_type="audio/ogg",
                payload=b"ogg-data",
            )

    async def test_transcribe_audio_surfaces_http_errors(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
        response = httpx.Response(
            429,
            request=request,
            json={"error": {"message": "Too many requests"}},
        )

        class FakeClient:
            async def post(self, url, *, headers, files, data):
                return response

        config = TurnmuxConfig(
            telegram_bot_token="token",
            allowed_user_ids=(1,),
            allowed_roots=(Path.cwd(),),
            tmux_session_name="turnmux",
            claude_command=("claude",),
            codex_command=("codex",),
            opencode_command=None,
            opencode_model=None,
            config_path=Path("/tmp/turnmux.toml"),
            relay_claude_thinking=False,
            openai_api_key="sk-test",
        )

        with unittest.mock.patch("turnmux.audio_transcription._get_client", return_value=FakeClient()):
            with self.assertRaises(AudioTranscriptionError) as cm:
                await transcribe_audio(
                    config,
                    filename="voice.ogg",
                    content_type="audio/ogg",
                    payload=b"ogg-data",
                )

        self.assertIn("rate-limited", str(cm.exception))
