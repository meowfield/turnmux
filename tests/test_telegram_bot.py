from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from turnmux.config import TurnmuxConfig
from turnmux.providers import ProviderRegistry
from turnmux.state.repository import StateRepository
from turnmux.state.models import BindingStatus, ProviderName
from turnmux.transport.telegram_bot import (
    REPO_BROWSER_PAGE_SIZE,
    _bot_commands,
    _build_approval_keyboard,
    _build_launch_keyboard,
    _browser_parent,
    _decode_repo_browser_state,
    _encode_repo_browser_state,
    _encode_pending_state,
    _extract_seed_text,
    _format_topic_name,
    _format_topic_setup_name,
    _is_forum_lobby,
    _is_named_topic,
    _format_launch_prompt,
    _format_repo_prompt,
    _make_repo_browser_state,
    topic_key,
    TurnmuxTelegramBot,
)
from turnmux.providers.base import ProviderSession
from turnmux.state.db import bootstrap_database


class RepoBrowserTests(unittest.TestCase):
    def test_make_repo_browser_state_lists_repos_before_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "repo-a"
            plain_dir = root / "folder-b"
            hidden_dir = root / ".hidden"
            repo_dir.mkdir()
            plain_dir.mkdir()
            hidden_dir.mkdir()
            (repo_dir / ".git").mkdir()

            state = _make_repo_browser_state(
                recent_repos=[repo_dir],
                allowed_roots=(root,),
                browse_dir=root,
                browse_page=0,
            )

            self.assertEqual(state.recent_repos, (repo_dir.resolve(strict=False),))
            self.assertEqual([entry.label for entry in state.browse_entries], ["repo-a", "folder-b"])
            self.assertEqual([entry.kind for entry in state.browse_entries], ["repo", "dir"])

    def test_repo_browser_state_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "repo-a"
            repo_dir.mkdir()
            (repo_dir / ".git").mkdir()

            state = _make_repo_browser_state(
                recent_repos=[repo_dir],
                allowed_roots=(root,),
                browse_dir=root,
                browse_page=0,
            )

            decoded = _decode_repo_browser_state(_encode_repo_browser_state(state))
            self.assertEqual(decoded.recent_repos, state.recent_repos)
            self.assertEqual(decoded.browse_dir, state.browse_dir)
            self.assertEqual(decoded.browse_page, state.browse_page)
            self.assertEqual(tuple((entry.path, entry.kind, entry.label) for entry in decoded.browse_entries), tuple((entry.path, entry.kind, entry.label) for entry in state.browse_entries))

    def test_browser_parent_stops_at_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            child = root / "child"
            child.mkdir()

            self.assertIsNone(_browser_parent(root, (root,)))
            self.assertEqual(_browser_parent(child, (root,)), root.resolve(strict=False))

    def test_repo_prompt_is_concise_and_launch_prompt_skips_candidate_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "repo-a"
            repo_dir.mkdir()
            (repo_dir / ".git").mkdir()
            extra_dirs = []
            for index in range(REPO_BROWSER_PAGE_SIZE + 2):
                path = root / f"folder-{index}"
                path.mkdir()
                extra_dirs.append(path)

            state = _make_repo_browser_state(
                recent_repos=[repo_dir],
                allowed_roots=(root,),
                browse_dir=root,
                browse_page=0,
            )

            repo_prompt = _format_repo_prompt(ProviderName.CLAUDE, "resume", state)
            self.assertIn("Browsing", repo_prompt)
            self.assertIn("Showing 1-", repo_prompt)
            self.assertNotIn("Recent repos:", repo_prompt)
            self.assertNotIn(str(repo_dir), repo_prompt)

            launch_prompt = _format_launch_prompt(
                ProviderName.CODEX,
                repo_dir,
                [
                    {"session_id": "session-123", "display_name": "Latest session", "updated_at": "2026-04-20T10:00:00Z"},
                ],
                "resume",
            )
            self.assertIn("Choose a resumable session below or start fresh.", launch_prompt)
            self.assertIn("exact session ID", launch_prompt)
            self.assertNotIn("Latest session", launch_prompt)
            self.assertNotIn("session-123", launch_prompt)

    def test_topic_helpers_treat_general_topic_as_lobby(self) -> None:
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
            effective_message=SimpleNamespace(message_thread_id=1),
        )
        self.assertEqual(topic_key(update), (-100123, 0))
        self.assertFalse(_is_named_topic(update))
        self.assertTrue(_is_forum_lobby(update))

    def test_format_topic_name_is_short_and_provider_scoped(self) -> None:
        repo_path = Path("/tmp") / ("very-long-repo-name-" * 20)
        topic_name = _format_topic_name(ProviderName.CODEX, repo_path)
        self.assertTrue(topic_name.startswith("🔵 codex"))
        self.assertLessEqual(len(topic_name), 128)

    def test_format_topic_setup_name_has_visual_provider_badge(self) -> None:
        self.assertEqual(_format_topic_setup_name(ProviderName.CLAUDE), "🟠 claude · setup")
        self.assertEqual(_format_topic_setup_name(ProviderName.CODEX), "🔵 codex · setup")
        self.assertEqual(_format_topic_setup_name(ProviderName.OPENCODE), "🟢 opencode · setup")

    def test_launch_keyboard_builds_resume_buttons_from_provider_sessions(self) -> None:
        candidate = ProviderSession(
            session_id="session-123",
            display_name="Latest session",
            updated_at="2026-04-20T10:00:00Z",
            repo_path=Path("/tmp/repo"),
            transcript_path=Path("/tmp/repo/session.jsonl"),
        )
        keyboard = _build_launch_keyboard([candidate], preferred_mode="resume")
        labels = [button.text for row in keyboard.inline_keyboard for button in row]
        self.assertIn("Start Fresh", labels[0])
        self.assertTrue(any(label.startswith("Resume 1: Latest session") for label in labels))

    def test_pending_state_keeps_seed_text_alongside_repo_browser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "repo-a"
            repo_dir.mkdir()
            (repo_dir / ".git").mkdir()
            state = _make_repo_browser_state(
                recent_repos=[repo_dir],
                allowed_roots=(root,),
                browse_dir=root,
                browse_page=0,
            )
            payload = _encode_pending_state(seed_text="hello from lobby", repo_browser_state=state)
            self.assertEqual(_extract_seed_text(payload), "hello from lobby")
            decoded = _decode_repo_browser_state(payload)
            self.assertEqual(decoded.recent_repos, state.recent_repos)
            self.assertEqual(decoded.browse_dir, state.browse_dir)

    def test_bot_commands_include_kill_and_interrupt(self) -> None:
        commands = _bot_commands((ProviderName.CLAUDE, ProviderName.CODEX, ProviderName.OPENCODE))
        names = [command.command for command in commands]
        self.assertEqual(names[:3], ["start", "new", "resume"])
        self.assertIn("interrupt", names)
        self.assertIn("kill", names)

    def test_approval_keyboard_can_hide_deny_button(self) -> None:
        keyboard = _build_approval_keyboard(has_deny=False)
        labels = [button.text for row in keyboard.inline_keyboard for button in row]
        self.assertEqual(labels, ["Approve"])


class TelegramBotKillTests(unittest.IsolatedAsyncioTestCase):
    async def test_kill_deletes_unbound_named_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bootstrap_database(root / "state.db")
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(root,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=None,
                opencode_model=None,
                config_path=root / "config.toml",
                relay_claude_thinking=False,
            )
            repository = StateRepository(root / "state.db")
            bot = TurnmuxTelegramBot(config=config, repository=repository, providers=ProviderRegistry(config))
            bot._ensure_allowed = AsyncMock(return_value=True)  # type: ignore[method-assign]
            bot._maybe_delete_topic = AsyncMock(return_value=True)  # type: ignore[method-assign]
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42),
            )

            await bot._handle_kill(update, None)

            bot._maybe_delete_topic.assert_awaited_once_with(update)
            bot._reply.assert_not_called()


class TelegramBotApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_approval_callback_routes_decision_to_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bootstrap_database(root / "state.db")
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(root,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=None,
                opencode_model=None,
                config_path=root / "config.toml",
                relay_claude_thinking=False,
            )
            repository = StateRepository(root / "state.db")
            repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=root,
                tmux_session_name="turnmux",
                tmux_window_id="@1",
                tmux_window_name="codex:repo",
                status=BindingStatus.ACTIVE,
            )
            bot = TurnmuxTelegramBot(config=config, repository=repository, providers=ProviderRegistry(config))
            bot._ensure_allowed = AsyncMock(return_value=True)  # type: ignore[method-assign]
            bot._reply = AsyncMock()  # type: ignore[method-assign]
            bot.service.resolve_pending_approval = unittest.mock.Mock(return_value="Approval sent.")  # type: ignore[method-assign]

            query_message = SimpleNamespace(edit_reply_markup=AsyncMock())
            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42),
                callback_query=SimpleNamespace(
                    data="ap:approve",
                    answer=AsyncMock(),
                    message=query_message,
                ),
            )

            await bot._handle_approval_callback(update, None)

            bot.service.resolve_pending_approval.assert_called_once()
            query_message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
            bot._reply.assert_awaited_once_with(update, "Approval sent.")


class TelegramBotAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_audio_transcribes_voice_and_routes_like_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bootstrap_database(root / "state.db")
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(root,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=None,
                opencode_model=None,
                config_path=root / "config.toml",
                relay_claude_thinking=False,
                openai_api_key="sk-test",
            )
            repository = StateRepository(root / "state.db")
            bot = TurnmuxTelegramBot(config=config, repository=repository, providers=ProviderRegistry(config))
            bot._ensure_allowed = AsyncMock(return_value=True)  # type: ignore[method-assign]
            bot._route_incoming_text = AsyncMock()  # type: ignore[method-assign]
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            telegram_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"ogg-data")))
            voice = SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))
            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42, text=None, voice=voice, audio=None, video_note=None),
            )

            with unittest.mock.patch(
                "turnmux.transport.telegram_bot.transcribe_audio",
                new=AsyncMock(return_value="hello from voice"),
            ) as transcribe:
                await bot._handle_message(update, None)

            transcribe.assert_awaited_once()
            bot._route_incoming_text.assert_awaited_once_with(update, "hello from voice")
            bot._reply.assert_not_called()

    async def test_handle_attachment_replies_for_unsupported_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bootstrap_database(root / "state.db")
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(root,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=None,
                opencode_model=None,
                config_path=root / "config.toml",
                openai_api_key="sk-test",
            )
            repository = StateRepository(root / "state.db")
            bot = TurnmuxTelegramBot(config=config, repository=repository, providers=ProviderRegistry(config))
            bot._ensure_allowed = AsyncMock(return_value=True)  # type: ignore[method-assign]
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42, text=None, photo=[object()], voice=None, audio=None, video_note=None),
            )

            await bot._handle_message(update, None)

            bot._reply.assert_awaited_once_with(
                update,
                "Unsupported attachment type: photo. Send text, voice, or audio.",
            )
