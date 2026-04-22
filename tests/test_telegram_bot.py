from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from turnmux.config import TurnmuxConfig
from turnmux.providers import ProviderRegistry
from turnmux.state.repository import StateRepository
from turnmux.state.models import BindingStatus, OnboardingStep, ProviderName
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
            bot._route_incoming_text.assert_awaited_once_with(update, "hello from voice", from_audio=True)
            bot._reply.assert_awaited_once_with(update, "Transcribed audio:\nhello from voice")

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

    async def test_handle_message_ignores_forum_topic_service_updates(self) -> None:
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
            bot._route_incoming_text = AsyncMock()  # type: ignore[method-assign]

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(
                    message_thread_id=42,
                    text=None,
                    voice=None,
                    audio=None,
                    video_note=None,
                    forum_topic_created=object(),
                ),
            )

            await bot._handle_message(update, None)

            bot._reply.assert_not_called()
            bot._route_incoming_text.assert_not_called()

    async def test_handle_attachment_uses_effective_attachment_name_when_available(self) -> None:
        class Contact:
            pass

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
                effective_message=SimpleNamespace(
                    message_thread_id=42,
                    text=None,
                    voice=None,
                    audio=None,
                    video_note=None,
                    effective_attachment=Contact(),
                ),
            )

            await bot._handle_message(update, None)

            bot._reply.assert_awaited_once_with(
                update,
                "Unsupported attachment type: contact. Send text, voice, or audio.",
            )

    async def test_handle_message_ignores_bot_authored_updates_without_unauthorized_reply(self) -> None:
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
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=999, is_bot=True),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(
                    message_thread_id=42,
                    text=None,
                    voice=None,
                    audio=None,
                    video_note=None,
                    forum_topic_edited=object(),
                ),
            )

            await bot._handle_message(update, None)

            bot._reply.assert_not_called()

    async def test_new_topic_voice_flow_ignores_service_message_and_starts_setup(self) -> None:
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

            service_update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(
                    message_thread_id=42,
                    text=None,
                    voice=None,
                    audio=None,
                    video_note=None,
                    forum_topic_created=object(),
                ),
            )
            telegram_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"voice-data")))
            voice = SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))
            voice_update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42, text=None, voice=voice, audio=None, video_note=None),
            )

            with unittest.mock.patch(
                "turnmux.transport.telegram_bot.transcribe_audio",
                new=AsyncMock(return_value="hello from first voice"),
            ):
                await bot._handle_message(service_update, None)
                await bot._handle_message(voice_update, None)

            onboarding = repository.get_onboarding_state(-100123, 42)
            self.assertIsNotNone(onboarding)
            self.assertEqual(onboarding.step, OnboardingStep.CHOOSE_PROVIDER)
            self.assertEqual(onboarding.mode, "fresh")
            self.assertEqual(_extract_seed_text(onboarding.pending_user_text), "hello from first voice")
            self.assertEqual(bot._reply.await_count, 2)
            transcript_text = bot._reply.await_args_list[0].args[1]
            self.assertEqual(transcript_text, "Transcribed audio:\nhello from first voice")
            reply_text = bot._reply.await_args_list[1].args[1]
            self.assertIn('Saved first message: "hello from first voice"', reply_text)
            self.assertIn("Choose a provider for this topic.", reply_text)
            self.assertIn("It will be sent after setup.", reply_text)

    async def test_supported_audio_variants_start_setup_in_new_topic(self) -> None:
        cases = (
            (
                "voice",
                lambda telegram_file: {"voice": SimpleNamespace(get_file=AsyncMock(return_value=telegram_file)), "audio": None, "video_note": None},
                "voice.ogg",
                "audio/ogg",
            ),
            (
                "audio",
                lambda telegram_file: {
                    "voice": None,
                    "audio": SimpleNamespace(
                        get_file=AsyncMock(return_value=telegram_file),
                        file_name="clip.mp3",
                        mime_type="audio/mpeg",
                    ),
                    "video_note": None,
                },
                "clip.mp3",
                "audio/mpeg",
            ),
            (
                "video_note",
                lambda telegram_file: {"voice": None, "audio": None, "video_note": SimpleNamespace(get_file=AsyncMock(return_value=telegram_file))},
                "video_note.mp4",
                "video/mp4",
            ),
        )

        for thread_id, (kind, builder, expected_filename, expected_content_type) in enumerate(cases, start=41):
            with self.subTest(kind=kind):
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

                    telegram_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"media-data")))
                    message_payload = builder(telegram_file)
                    update = SimpleNamespace(
                        effective_user=SimpleNamespace(id=1),
                        effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                        effective_message=SimpleNamespace(message_thread_id=thread_id, text=None, **message_payload),
                    )
                    transcribe = AsyncMock(return_value=f"seed from {kind}")

                    with unittest.mock.patch("turnmux.transport.telegram_bot.transcribe_audio", new=transcribe):
                        await bot._handle_message(update, None)

                    transcribe.assert_awaited_once()
                    self.assertEqual(transcribe.await_args.kwargs["filename"], expected_filename)
                    self.assertEqual(transcribe.await_args.kwargs["content_type"], expected_content_type)
                    self.assertEqual(transcribe.await_args.kwargs["payload"], b"media-data")

                    onboarding = repository.get_onboarding_state(-100123, thread_id)
                    self.assertIsNotNone(onboarding)
                    self.assertEqual(onboarding.step, OnboardingStep.CHOOSE_PROVIDER)
                    self.assertEqual(onboarding.mode, "fresh")
                    self.assertEqual(_extract_seed_text(onboarding.pending_user_text), f"seed from {kind}")
                    self.assertEqual(bot._reply.await_count, 2)
                    transcript_text = bot._reply.await_args_list[0].args[1]
                    self.assertEqual(transcript_text, f"Transcribed audio:\nseed from {kind}")
                    reply_text = bot._reply.await_args_list[1].args[1]
                    self.assertIn(f'Saved first message: "seed from {kind}"', reply_text)
                    self.assertIn("Choose a provider for this topic.", reply_text)
                    self.assertIn("It will be sent after setup.", reply_text)

    async def test_audio_forward_acknowledges_progress_for_active_binding(self) -> None:
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
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            binding = repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=root,
                tmux_session_name="turnmux",
                tmux_window_id="@9",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=root / "transcript.jsonl",
                status=BindingStatus.ACTIVE,
            )
            service = Mock()
            service.send_user_text = Mock()
            bot.service = service

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42, text=None),
            )

            await bot._route_incoming_text(update, "hello from voice", from_audio=True)

            service.send_user_text.assert_called_once_with(binding, "hello from voice")
            self.assertEqual(
                [call.args[1] for call in bot._reply.await_args_list],
                ["Sent to codex. I will post progress updates here."],
            )

    async def test_long_text_forward_acknowledges_progress(self) -> None:
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
            bot._reply = AsyncMock()  # type: ignore[method-assign]

            binding = repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=root,
                tmux_session_name="turnmux",
                tmux_window_id="@9",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=root / "transcript.jsonl",
                status=BindingStatus.ACTIVE,
            )
            service = Mock()
            service.send_user_text = Mock()
            bot.service = service

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42, text=None),
            )

            await bot._route_incoming_text(update, "x" * 260)

            service.send_user_text.assert_called_once_with(binding, "x" * 260)
            self.assertEqual(
                [call.args[1] for call in bot._reply.await_args_list],
                ["Sent to codex. I will post progress updates here."],
            )

    async def test_launch_from_onboarding_uses_user_facing_status_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_path = root / "turnmux"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
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
            bot._reply = AsyncMock()  # type: ignore[method-assign]
            bot._maybe_name_topic = AsyncMock()  # type: ignore[method-assign]

            repository.save_onboarding_state(
                chat_id=-100123,
                thread_id=42,
                step=OnboardingStep.CHOOSE_RESUME,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                mode="fresh",
                pending_user_text=_encode_pending_state(seed_text="hello from voice"),
            )

            binding = repository.save_binding(
                chat_id=-100123,
                thread_id=42,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@9",
                tmux_window_name="codex:turnmux",
                status=BindingStatus.PENDING_START,
                provider_session_id=None,
                transcript_path=None,
            )
            service = Mock()
            service.launch_binding = AsyncMock(return_value=binding)
            service.send_user_text = Mock()
            bot.service = service

            update = SimpleNamespace(
                effective_user=SimpleNamespace(id=1),
                effective_chat=SimpleNamespace(id=-100123, type="supergroup", is_forum=True),
                effective_message=SimpleNamespace(message_thread_id=42),
            )

            await bot._launch_from_onboarding(
                update,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                mode="fresh",
            )

            service.send_user_text.assert_called_once_with(binding, "hello from voice")
            bot._reply.assert_awaited_once()
            reply_text = bot._reply.await_args.args[1]
            self.assertEqual(
                reply_text,
                f'Started codex for `{root.name}/turnmux`.\nSent to codex: "hello from voice"',
            )
