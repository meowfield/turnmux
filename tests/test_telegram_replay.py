from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from turnmux.config import TurnmuxConfig
from turnmux.providers.base import ParseBatch, ProviderSession, ProviderTranscriptEvent
from turnmux.providers.codex import CodexAdapter
from turnmux.state.db import bootstrap_database
from turnmux.state.models import BindingStatus, ProviderName
from turnmux.state.repository import StateRepository
from turnmux.transport.telegram_bot import TurnmuxTelegramBot, send_thread_message, split_text
from telegram.error import TimedOut


def make_config(base_dir: Path) -> TurnmuxConfig:
    return TurnmuxConfig(
        telegram_bot_token="token",
        allowed_user_ids=(1,),
        allowed_roots=(base_dir,),
        tmux_session_name="turnmux",
        claude_command=None,
        codex_command=("codex",),
        opencode_command=None,
        opencode_model=None,
        config_path=base_dir / "config.toml",
        relay_claude_thinking=False,
        openai_api_key="sk-test",
    )


@dataclass(frozen=True, slots=True)
class SentTelegramMessage:
    chat_id: int
    thread_id: int | None
    text: str


class FakeTelegramClient:
    def __init__(self) -> None:
        self.sent_messages: list[SentTelegramMessage] = []
        self.topic_edits: list[tuple[int, int, str]] = []

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        message_thread_id: int | None = None,
        reply_markup=None,
    ) -> None:
        self.sent_messages.append(
            SentTelegramMessage(
                chat_id=chat_id,
                thread_id=message_thread_id,
                text=text,
            )
        )

    async def edit_forum_topic(self, *, chat_id: int, message_thread_id: int, name: str) -> None:
        self.topic_edits.append((chat_id, message_thread_id, name))


class FailOnceTelegramClient(FakeTelegramClient):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        message_thread_id: int | None = None,
        reply_markup=None,
    ) -> None:
        if not self.failed:
            self.failed = True
            raise TimedOut("timed out")
        await super().send_message(
            chat_id=chat_id,
            text=text,
            message_thread_id=message_thread_id,
            reply_markup=reply_markup,
        )


class ScriptedCodexAdapter:
    def __init__(self, *, repo_path: Path, transcript_path: Path) -> None:
        self._repo_path = repo_path
        self._transcript_path = transcript_path
        self._session = ProviderSession(
            session_id="session-123",
            display_name="Replay Session",
            updated_at="2026-04-22T13:00:00Z",
            repo_path=repo_path,
            transcript_path=transcript_path,
        )
        self.discovery_enabled = False
        self.events: tuple[ProviderTranscriptEvent, ...] = ()

    def runtime_env(self) -> dict[str, str]:
        return {}

    def initial_monitor_offset(self, session: ProviderSession) -> int:
        return 0

    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        command = ["codex", "--no-alt-screen", str(repo_path)]
        if initial_prompt:
            command.append(initial_prompt)
        return command

    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        return ["codex", "resume", session_id, "--cd", str(repo_path)]

    def list_resumable_sessions(self, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        return [self._session]

    def discover_session(
        self,
        repo_path: Path,
        *,
        started_after: str,
        requested_session_id: str | None = None,
        tmux_session_name: str | None = None,
        tmux_window_id: str | None = None,
    ) -> ProviderSession | None:
        if not self.discovery_enabled:
            return None
        if requested_session_id and requested_session_id != self._session.session_id:
            return None
        return self._session

    def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
        if transcript_path != self._transcript_path:
            return ParseBatch(events=(), new_offset=offset, last_event_ts=None, last_message_hash=None)
        if offset >= len(self.events):
            return ParseBatch(events=(), new_offset=offset, last_event_ts=None, last_message_hash=None)
        new_events = self.events[offset:]
        return ParseBatch(
            events=new_events,
            new_offset=len(self.events),
            last_event_ts=new_events[-1].timestamp if new_events else None,
            last_message_hash="replay-hash" if new_events else None,
        )

    def history(self, transcript_path: Path, *, session_id: str | None = None, limit: int = 10):
        return list(self.events)[-limit:]


class SingleProviderRegistry:
    def __init__(self, adapter: ScriptedCodexAdapter) -> None:
        self.adapter = adapter

    def get(self, provider: ProviderName) -> ScriptedCodexAdapter:
        if provider != ProviderName.CODEX:
            raise KeyError(provider)
        return self.adapter

    def available_providers(self) -> tuple[ProviderName, ...]:
        return (ProviderName.CODEX,)


def make_update(
    bot: FakeTelegramClient,
    *,
    thread_id: int,
    text: str | None = None,
    voice=None,
    audio=None,
    video_note=None,
    extra_message_fields: dict[str, object] | None = None,
    user_id: int = 1,
    user_is_bot: bool = False,
):
    message_fields = {
        "message_thread_id": thread_id,
        "text": text,
        "voice": voice,
        "audio": audio,
        "video_note": video_note,
    }
    if extra_message_fields:
        message_fields.update(extra_message_fields)
    message = SimpleNamespace(**message_fields)
    message.get_bot = lambda: bot
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id, is_bot=user_is_bot),
        effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
        effective_message=message,
    )


def make_voice_update(bot: FakeTelegramClient, *, thread_id: int):
    telegram_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"voice-data")))
    voice = SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))
    return make_update(bot, thread_id=thread_id, voice=voice)


async def flush_outbound(bot: FakeTelegramClient, outbound_messages) -> None:
    for outbound in outbound_messages:
        await send_thread_message(
            bot,
            chat_id=outbound.chat_id,
            thread_id=outbound.thread_id,
            text=outbound.text,
        )


def collect_texts(bot: FakeTelegramClient) -> list[str]:
    return [message.text for message in bot.sent_messages]


def write_codex_rollout(
    codex_root: Path,
    *,
    repo_path: Path,
    session_id: str,
    updated_at: str,
    thread_name: str,
    assistant_messages: list[tuple[str, str, str]],
) -> Path:
    session_index = codex_root / "session_index.jsonl"
    session_index.parent.mkdir(parents=True, exist_ok=True)
    session_index.write_text(
        json.dumps(
            {
                "id": session_id,
                "thread_name": thread_name,
                "updated_at": updated_at,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    sessions_dir = codex_root / "sessions" / "2026" / "04" / "22"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = sessions_dir / f"rollout-2026-04-22T13-00-00-{session_id}.jsonl"

    records = [
        {
            "type": "session_meta",
            "timestamp": updated_at,
            "payload": {
                "id": session_id,
                "cwd": str(repo_path.resolve(strict=False)),
                "timestamp": updated_at,
            },
        }
    ]
    for timestamp, phase, text in assistant_messages:
        records.append(
            {
                "type": "response_item",
                "timestamp": timestamp,
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": phase,
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    rollout_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return rollout_path


class TelegramReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_new_topic_voice_flow_stays_clean_through_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            transcript_path = root / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            bootstrap_database(root / "state.db")

            adapter = ScriptedCodexAdapter(repo_path=repo_path, transcript_path=transcript_path)
            adapter.events = (
                ProviderTranscriptEvent(
                    role="assistant",
                    content_type="text",
                    text="Checking the local diff first.",
                    timestamp="2026-04-22T13:00:01Z",
                    is_final=False,
                ),
                ProviderTranscriptEvent(
                    role="assistant",
                    content_type="text",
                    text="`codex` adapter keeps literal `<oai-mem-citation>` mentions visible.\nNext step: run the tests.",
                    timestamp="2026-04-22T13:00:02Z",
                    is_final=True,
                ),
            )

            repository = StateRepository(root / "state.db")
            fake_bot = FakeTelegramClient()
            bot = TurnmuxTelegramBot(
                config=make_config(root),
                repository=repository,
                providers=SingleProviderRegistry(adapter),
            )

            service_update = make_update(
                fake_bot,
                thread_id=42,
                extra_message_fields={"forum_topic_created": object()},
            )
            voice_update = make_voice_update(fake_bot, thread_id=42)
            topic_update = make_update(fake_bot, thread_id=42)

            with (
                patch("turnmux.transport.telegram_bot.transcribe_audio", new=AsyncMock(return_value="Please inspect the Telegram fixes.")),
                patch("turnmux.app.service.ensure_provider_trust"),
                patch("turnmux.app.service.tmux.ensure_session"),
                patch("turnmux.app.service.tmux.create_window", return_value=SimpleNamespace(window_id="@21", name="codex:turnmux")),
                patch("turnmux.app.service.tmux.launch_command"),
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value=""),
            ):
                await bot._handle_message(service_update, None)
                await bot._handle_message(voice_update, None)
                await bot._launch_from_onboarding(
                    topic_update,
                    provider=ProviderName.CODEX,
                    repo_path=repo_path,
                    mode="fresh",
                )
                adapter.discovery_enabled = True
                await flush_outbound(fake_bot, bot.service.refresh_pending_and_active_bindings())

            texts = collect_texts(fake_bot)
            self.assertEqual(
                texts,
                [
                    "Transcribed audio:\nPlease inspect the Telegram fixes.",
                    'Saved first message: "Please inspect the Telegram fixes."\n\nChoose a provider for this topic.\nIt will be sent after setup.',
                    'Started codex for `{repo}`.\nSent to codex: "Please inspect the Telegram fixes."'.format(repo=f"{root.name}/turnmux"),
                    "Checking the local diff first.",
                    "`codex` adapter keeps literal `<oai-mem-citation>` mentions visible.\nNext step: run the tests.",
                ],
            )
            self.assertFalse(any("Unsupported attachment type" in text for text in texts))
            self.assertFalse(any("Session is active:" in text for text in texts))
            self.assertFalse(any("You are not allowed" in text for text in texts))
            self.assertFalse(any("Working update:" in text for text in texts))

    async def test_replay_active_binding_voice_flow_preserves_long_final_answer_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            transcript_path = root / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            bootstrap_database(root / "state.db")

            long_sections = [
                "Final summary starts here.",
                "A" * 3400,
                "Literal marker `<oai-mem-citation>` must stay in the visible answer.",
                "Tail section after the literal marker must also survive.",
                "B" * 600,
            ]
            long_final = "\n".join(long_sections)
            expected_chunks = split_text(long_final)

            adapter = ScriptedCodexAdapter(repo_path=repo_path, transcript_path=transcript_path)
            adapter.events = (
                ProviderTranscriptEvent(
                    role="assistant",
                    content_type="text",
                    text="Checking the current diff.",
                    timestamp="2026-04-22T14:00:01Z",
                    is_final=False,
                ),
                ProviderTranscriptEvent(
                    role="assistant",
                    content_type="text",
                    text=long_final,
                    timestamp="2026-04-22T14:00:02Z",
                    is_final=True,
                ),
            )

            repository = StateRepository(root / "state.db")
            repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@22",
                tmux_window_name="codex:turnmux",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )

            fake_bot = FakeTelegramClient()
            bot = TurnmuxTelegramBot(
                config=make_config(root),
                repository=repository,
                providers=SingleProviderRegistry(adapter),
            )
            voice_update = make_voice_update(fake_bot, thread_id=42)

            with (
                patch("turnmux.transport.telegram_bot.transcribe_audio", new=AsyncMock(return_value="Check what is uncommitted and explain it.")),
                patch("turnmux.app.service.tmux.paste_text"),
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value=""),
            ):
                await bot._handle_message(voice_update, None)
                await flush_outbound(fake_bot, bot.service.refresh_pending_and_active_bindings())

            texts = collect_texts(fake_bot)
            self.assertEqual(texts[:3], [
                "Transcribed audio:\nCheck what is uncommitted and explain it.",
                "Sent to codex. I will post progress updates here.",
                "Checking the current diff.",
            ])
            self.assertEqual(texts[3:], expected_chunks)
            self.assertTrue(any("Tail section after the literal marker must also survive." in chunk for chunk in texts[3:]))
            self.assertFalse(any("Command failed (" in text for text in texts))
            self.assertFalse(any("Session is active:" in text for text in texts))
            self.assertFalse(any("Working update:" in text for text in texts))


class TelegramRealCodexReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_codex_rollout_replay_discovers_session_and_relays_clean_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            bootstrap_database(root / "state.db")
            codex_root = root / ".codex"

            updated_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
            write_codex_rollout(
                codex_root,
                repo_path=repo_path,
                session_id="session-real-1",
                updated_at=updated_at,
                thread_name="Replay Session",
                assistant_messages=[
                    (
                        updated_at,
                        "commentary",
                        "Проверяю diff.\n\n<oai-mem-citation>\n<citation_entries>\nMEMORY.md:1-2|note=[noise]\n</citation_entries>\n<rollout_ids>\nabc\n</rollout_ids>\n</oai-mem-citation>\n::git-stage{cwd=\"/tmp/repo\"}",
                    ),
                    (
                        updated_at,
                        "final_answer",
                        "Нашел 4 измененных файла. Literal `<oai-mem-citation>` должен остаться видимым.\n::git-stage{cwd=\"/tmp/repo\"}",
                    ),
                ],
            )

            repository = StateRepository(root / "state.db")
            fake_bot = FakeTelegramClient()
            adapter = CodexAdapter(make_config(root), codex_home=codex_root)
            bot = TurnmuxTelegramBot(
                config=make_config(root),
                repository=repository,
                providers=SingleProviderRegistry(adapter),
            )

            service_update = make_update(
                fake_bot,
                thread_id=42,
                extra_message_fields={"forum_topic_created": object()},
            )
            voice_update = make_voice_update(fake_bot, thread_id=42)
            topic_update = make_update(fake_bot, thread_id=42)

            with (
                patch("turnmux.transport.telegram_bot.transcribe_audio", new=AsyncMock(return_value="Проверь реальный Codex replay.")),
                patch("turnmux.app.service.ensure_provider_trust"),
                patch("turnmux.app.service.tmux.ensure_session"),
                patch("turnmux.app.service.tmux.create_window", return_value=SimpleNamespace(window_id="@31", name="codex:turnmux")),
                patch("turnmux.app.service.tmux.launch_command"),
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value=""),
            ):
                await bot._handle_message(service_update, None)
                await bot._handle_message(voice_update, None)
                await bot._launch_from_onboarding(
                    topic_update,
                    provider=ProviderName.CODEX,
                    repo_path=repo_path,
                    mode="fresh",
                )
                await flush_outbound(fake_bot, bot.service.refresh_pending_and_active_bindings())

            texts = collect_texts(fake_bot)
            self.assertEqual(
                texts,
                [
                    "Transcribed audio:\nПроверь реальный Codex replay.",
                    'Saved first message: "Проверь реальный Codex replay."\n\nChoose a provider for this topic.\nIt will be sent after setup.',
                    'Started codex for `{repo}`.\nSent to codex: "Проверь реальный Codex replay."'.format(repo=f"{root.name}/turnmux"),
                    "Проверяю diff.",
                    "Нашел 4 измененных файла. Literal `<oai-mem-citation>` должен остаться видимым.",
                ],
            )
            self.assertFalse(any("Session is active:" in text for text in texts))
            self.assertFalse(any("Working update:" in text for text in texts))
            self.assertFalse(any("You are not allowed" in text for text in texts))

    async def test_real_codex_rollout_replay_preserves_long_chunked_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            bootstrap_database(root / "state.db")
            codex_root = root / ".codex"

            updated_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
            long_final = "\n".join(
                [
                    "Финальный ответ начинается.",
                    "A" * 3400,
                    "Literal `<oai-mem-citation>` должен сохраниться в середине.",
                    "Хвост ответа после literal marker тоже должен доехать.",
                    "B" * 700,
                ]
            )
            rollout_path = write_codex_rollout(
                codex_root,
                repo_path=repo_path,
                session_id="session-real-2",
                updated_at=updated_at,
                thread_name="Replay Session",
                assistant_messages=[
                    (updated_at, "commentary", "Смотрю, что изменено."),
                    (updated_at, "final_answer", long_final),
                ],
            )

            repository = StateRepository(root / "state.db")
            repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@32",
                tmux_window_name="codex:turnmux",
                provider_session_id="session-real-2",
                transcript_path=rollout_path,
                status=BindingStatus.ACTIVE,
            )

            fake_bot = FakeTelegramClient()
            adapter = CodexAdapter(make_config(root), codex_home=codex_root)
            bot = TurnmuxTelegramBot(
                config=make_config(root),
                repository=repository,
                providers=SingleProviderRegistry(adapter),
            )
            voice_update = make_voice_update(fake_bot, thread_id=42)
            expected_chunks = split_text(
                "Финальный ответ начинается.\n"
                + "A" * 3400
                + "\nLiteral `<oai-mem-citation>` должен сохраниться в середине.\n"
                + "Хвост ответа после literal marker тоже должен доехать.\n"
                + "B" * 700
            )

            with (
                patch("turnmux.transport.telegram_bot.transcribe_audio", new=AsyncMock(return_value="Проверь длинный ответ.")),
                patch("turnmux.app.service.tmux.paste_text"),
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value=""),
            ):
                await bot._handle_message(voice_update, None)
                await flush_outbound(fake_bot, bot.service.refresh_pending_and_active_bindings())

            texts = collect_texts(fake_bot)
            self.assertEqual(
                texts[:3],
                [
                    "Transcribed audio:\nПроверь длинный ответ.",
                    "Sent to codex. I will post progress updates here.",
                    "Смотрю, что изменено.",
                ],
            )
            self.assertEqual(texts[3:], expected_chunks)
            self.assertTrue(any("Хвост ответа после literal marker тоже должен доехать." in chunk for chunk in texts[3:]))
            self.assertFalse(any("Working update:" in text for text in texts))

    async def test_real_codex_rollout_retries_after_send_timeout_without_losing_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            bootstrap_database(root / "state.db")
            codex_root = root / ".codex"

            updated_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
            rollout_path = write_codex_rollout(
                codex_root,
                repo_path=repo_path,
                session_id="session-real-3",
                updated_at=updated_at,
                thread_name="Replay Session",
                assistant_messages=[
                    (updated_at, "commentary", "Проверяю diff."),
                    (updated_at, "final_answer", "Финальный ответ должен доехать даже после timeout."),
                ],
            )

            repository = StateRepository(root / "state.db")
            binding = repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@33",
                tmux_window_name="codex:turnmux",
                provider_session_id="session-real-3",
                transcript_path=rollout_path,
                status=BindingStatus.ACTIVE,
            )

            adapter = CodexAdapter(make_config(root), codex_home=codex_root)
            bot = TurnmuxTelegramBot(
                config=make_config(root),
                repository=repository,
                providers=SingleProviderRegistry(adapter),
            )
            flaky_bot = FailOnceTelegramClient()

            with (
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value=""),
            ):
                outbound = bot.service.refresh_pending_and_active_bindings()
                with self.assertRaises(TimedOut):
                    for message in outbound:
                        await send_thread_message(
                            flaky_bot,
                            chat_id=message.chat_id,
                            thread_id=message.thread_id,
                            text=message.text,
                        )
                        bot.service.mark_outbound_delivered(message)

                offset = repository.get_monitor_offset(binding.id)
                self.assertIsNone(offset)

                retry_outbound = bot.service.refresh_pending_and_active_bindings()
                for message in retry_outbound:
                    await send_thread_message(
                        flaky_bot,
                        chat_id=message.chat_id,
                        thread_id=message.thread_id,
                        text=message.text,
                    )
                    bot.service.mark_outbound_delivered(message)

            self.assertEqual(
                collect_texts(flaky_bot),
                [
                    "Проверяю diff.",
                    "Финальный ответ должен доехать даже после timeout.",
                ],
            )
            offset = repository.get_monitor_offset(binding.id)
            assert offset is not None
            self.assertGreater(offset.byte_offset, 0)
