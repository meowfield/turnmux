from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turnmux.attachments import AttachmentStore
from turnmux.app.service import AppService
from turnmux.config import TurnmuxConfig
from turnmux.input_types import UserTurn
from turnmux.providers.base import ParseBatch, ProviderTranscriptEvent
from turnmux.state.db import bootstrap_database
from turnmux.state.models import BindingStatus, ProviderName
from turnmux.state.repository import StateRepository


def make_config(base_dir: Path) -> TurnmuxConfig:
    return TurnmuxConfig(
        telegram_bot_token="123456:valid-looking-token-value",
        allowed_user_ids=(1,),
        allowed_roots=(base_dir,),
        tmux_session_name="turnmux",
        claude_command=("claude", "--dangerously-skip-permissions"),
        codex_command=("codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"),
        opencode_command=None,
        opencode_model=None,
        config_path=base_dir / "config.toml",
        relay_claude_thinking=False,
    )


class FakeAdapter:
    def runtime_env(self) -> dict[str, str]:
        return {}

    def initial_monitor_offset(self, session) -> int:
        return 0

    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        command = ["fake-provider", str(repo_path)]
        if initial_prompt:
            command.append(initial_prompt)
        return command

    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        return ["fake-provider", "--resume", session_id, str(repo_path)]

    def discover_session(
        self,
        repo_path: Path,
        *,
        started_after: str,
        requested_session_id: str | None = None,
        tmux_session_name: str | None = None,
        tmux_window_id: str | None = None,
    ):
        return None

    def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
        return ParseBatch(events=(), new_offset=offset, last_event_ts=None, last_message_hash=None)

    def history(self, transcript_path: Path, *, session_id: str | None = None, limit: int = 10):
        return []


class FakeRegistry:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter

    def get(self, provider: ProviderName) -> FakeAdapter:
        return self.adapter


class AppServiceTests(unittest.TestCase):
    def test_suggest_repos_prefers_current_repo_and_existing_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            repo_a = base_dir / "repo-a"
            repo_b = base_dir / "repo-b"
            repo_a.mkdir()
            repo_b.mkdir()
            (repo_a / ".git").mkdir()
            (repo_b / ".git").mkdir()

            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=repo_b,
                tmux_session_name="turnmux",
                tmux_window_id="@1",
                tmux_window_name="codex:repo-b",
                status=BindingStatus.ACTIVE,
            )
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(repo_a)
                suggestions = service.suggest_repos(limit=4)
            finally:
                os.chdir(original_cwd)

            self.assertGreaterEqual(len(suggestions), 2)
            self.assertEqual(suggestions[0], repo_a.resolve(strict=False))
            self.assertIn(repo_b.resolve(strict=False), suggestions)

    def test_kill_binding_deletes_binding_and_pending_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@9",
                tmux_window_name="codex:tmp",
                status=BindingStatus.PENDING_START,
            )
            repository.save_pending_launch(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                discovery_deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
            )

            with patch("turnmux.app.service.tmux.kill_window") as kill_window:
                service.kill_binding(binding)

            kill_window.assert_called_once()
            self.assertEqual(repository.list_pending_launches(), [])
            self.assertIsNone(repository.get_binding_by_id(binding.id))

    def test_refresh_marks_expired_pending_launch_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CLAUDE,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@10",
                tmux_window_name="claude:tmp",
                status=BindingStatus.PENDING_START,
            )
            repository.save_pending_launch(
                binding_id=binding.id,
                provider=ProviderName.CLAUDE,
                repo_path=base_dir,
                discovery_deadline_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            )

            outbound = service.refresh_pending_and_active_bindings()
            self.assertEqual(len(outbound), 1)
            self.assertIn("discovery timeout", outbound[0].text)
            refreshed = repository.get_binding_by_id(binding.id)
            assert refreshed is not None
            self.assertEqual(refreshed.status, BindingStatus.MISSING)
            self.assertEqual(repository.list_pending_launches(), [])

    def test_send_user_text_launches_pending_fresh_binding_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            runtime_home = base_dir / "runtime-home"
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(
                config=make_config(base_dir),
                repository=repository,
                providers=FakeRegistry(FakeAdapter()),
                runtime_home=runtime_home,
            )

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@11",
                tmux_window_name="codex:tmp",
                status=BindingStatus.PENDING_START,
            )

            with (
                patch("turnmux.app.service.ensure_provider_trust") as ensure_trust,
                patch("turnmux.app.service.tmux.launch_command") as launch_command,
            ):
                service.send_user_text(binding, "hello from telegram")

            ensure_trust.assert_called_once_with(
                ProviderName.CODEX,
                base_dir,
                runtime_home=runtime_home.resolve(strict=False),
            )
            launch_command.assert_called_once_with("@11", ["fake-provider", str(base_dir), "hello from telegram"])
            pending = repository.get_pending_launch(binding.id)
            self.assertIsNotNone(pending)

            with self.assertRaisesRegex(RuntimeError, "still starting"):
                service.send_user_text(binding, "second message")

    def test_send_user_turn_projects_attachment_into_repo_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            runtime_home = base_dir / "runtime-home"
            repo_path = base_dir / "repo"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            attachment_store = AttachmentStore(runtime_home)
            service = AppService(
                config=make_config(base_dir),
                repository=repository,
                providers=FakeRegistry(FakeAdapter()),
                runtime_home=runtime_home,
                attachment_store=attachment_store,
            )
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:repo",
                status=BindingStatus.ACTIVE,
            )
            attachment = attachment_store.store_attachment(
                1,
                10,
                original_name="notes.txt",
                mime_type="text/plain",
                payload=b"hello from file",
                source_message_id=1,
                source_kind="document",
            )

            with patch("turnmux.app.service.tmux.paste_text") as paste_text:
                service.send_user_turn(binding, UserTurn(text="check file", attachments=(attachment,)))

            pasted_prompt = paste_text.call_args.args[1]
            self.assertIn("check file", pasted_prompt)
            self.assertIn(".turnmux-tmp/turnmux-1-10", pasted_prompt)
            self.assertTrue((repo_path / ".turnmux-tmp" / "turnmux-1-10").exists())

    def test_kill_binding_clears_topic_tmp_and_repo_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            runtime_home = base_dir / "runtime-home"
            repo_path = base_dir / "repo"
            repo_path.mkdir()
            (repo_path / ".git").mkdir()
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            attachment_store = AttachmentStore(runtime_home)
            service = AppService(
                config=make_config(base_dir),
                repository=repository,
                providers=FakeRegistry(FakeAdapter()),
                runtime_home=runtime_home,
                attachment_store=attachment_store,
            )
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=repo_path,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:repo",
                status=BindingStatus.ACTIVE,
            )
            attachment = attachment_store.store_attachment(
                1,
                10,
                original_name="notes.txt",
                mime_type="text/plain",
                payload=b"hello from file",
                source_message_id=1,
                source_kind="document",
            )
            attachment_store.project_attachment(repo_path, chat_id=1, thread_id=10, attachment=attachment)
            self.assertTrue(attachment_store.topic_dir(1, 10).exists())
            self.assertTrue((repo_path / ".turnmux-tmp" / "turnmux-1-10").exists())

            with patch("turnmux.app.service.tmux.kill_window") as kill_window:
                service.kill_binding(binding)

            kill_window.assert_called_once()
            self.assertFalse(attachment_store.topic_dir(1, 10).exists())
            self.assertFalse((repo_path / ".turnmux-tmp" / "turnmux-1-10").exists())

    def test_refresh_emits_pending_approval_once_and_resolves_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CLAUDE,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="claude:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )

            with (
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch(
                    "turnmux.app.service.tmux.capture_pane",
                    return_value=(
                        "WARNING: Claude Code running in Bypass Permissions mode\n"
                        "1. No, exit\n"
                        "2. Yes, I accept\n"
                    ),
                ),
                patch("turnmux.app.service.tmux.send_keys") as send_keys,
            ):
                outbound = service.refresh_pending_and_active_bindings()
                self.assertEqual(len(outbound), 1)
                self.assertEqual(outbound[0].markup_kind, "approval")
                self.assertTrue(outbound[0].markup_has_deny)
                pending = repository.get_pending_approval(binding.id)
                self.assertIsNotNone(pending)
                assert pending is not None
                self.assertEqual(pending.approve_keys, ("2", "Enter"))
                self.assertEqual(pending.deny_keys, ("1", "Enter"))

                repeated = service.refresh_pending_and_active_bindings()
                self.assertEqual(repeated, [])

                result = service.resolve_pending_approval(binding, approve=True)

            self.assertEqual(result, "Approval sent.")
            send_keys.assert_called_once_with("@12", "2", "Enter")
            self.assertIsNone(repository.get_pending_approval(binding.id))

    def test_refresh_auto_skips_codex_update_prompt_without_approval_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )
            repository.save_pending_approval(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                fingerprint="old-false-positive",
                prompt_text="2. Skip\n3. Skip until next version\nPress enter to continue",
                approve_keys=("Enter",),
            )

            with (
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch(
                    "turnmux.app.service.tmux.capture_pane",
                    return_value=(
                        "approval policy: on-request\n"
                        "Update available! 0.122.0 -> 0.124.0\n"
                        "Release notes: https://github.com/openai/codex/releases/latest\n"
                        "› 1. Update now (runs `npm install -g @openai/codex`)\n"
                        "2. Skip\n"
                        "3. Skip until next version\n"
                        "Press enter to continue\n"
                    ),
                ),
                patch("turnmux.app.service.tmux.send_keys") as send_keys,
            ):
                outbound = service.refresh_pending_and_active_bindings()

            self.assertEqual(outbound, [])
            send_keys.assert_called_once_with("@12", "Down", "Enter")
            self.assertIsNone(repository.get_pending_approval(binding.id))

    def test_refresh_ignores_stale_pending_launch_for_active_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )
            repository.save_pending_launch(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                started_at=(datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
                discovery_deadline_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            )

            with (
                patch("turnmux.app.service.tmux.window_exists", return_value=True),
                patch("turnmux.app.service.tmux.capture_pane", return_value="Codex ready"),
            ):
                outbound = service.refresh_pending_and_active_bindings()

            self.assertEqual(outbound, [])
            self.assertIsNone(repository.get_pending_launch(binding.id))
            rebound = repository.get_binding_by_id(binding.id)
            self.assertIsNotNone(rebound)
            assert rebound is not None
            self.assertEqual(rebound.status, BindingStatus.ACTIVE)

    def test_send_user_text_rejects_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(FakeAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )
            repository.save_pending_approval(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                fingerprint="abc",
                prompt_text="Approve this command?",
                approve_keys=("y", "Enter"),
                deny_keys=("n", "Enter"),
            )

            with self.assertRaisesRegex(RuntimeError, "waiting for an approval decision"):
                service.send_user_text(binding, "next command")

    def test_refresh_formats_non_final_transcript_events_as_progress_updates(self) -> None:
        class ProgressAdapter(FakeAdapter):
            def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
                return ParseBatch(
                    events=(
                        ProviderTranscriptEvent(
                            role="assistant",
                            content_type="text",
                            text="Checking git diff",
                            timestamp="2026-04-22T12:00:00Z",
                            is_final=False,
                        ),
                        ProviderTranscriptEvent(
                            role="assistant",
                            content_type="text",
                            text="Final verdict",
                            timestamp="2026-04-22T12:01:00Z",
                            is_final=True,
                        ),
                    ),
                    new_offset=10,
                    last_event_ts="2026-04-22T12:01:00Z",
                    last_message_hash="hash",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(ProgressAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )

            with patch("turnmux.app.service.tmux.window_exists", return_value=True):
                outbound = service.refresh_pending_and_active_bindings()

            self.assertEqual([message.text for message in outbound], ["Checking git diff", "Final verdict"])
            self.assertIsNone(repository.get_monitor_offset(binding.id))
            service.mark_outbound_delivered(outbound[0])
            self.assertIsNone(repository.get_monitor_offset(binding.id))
            service.mark_outbound_delivered(outbound[1])
            offset = repository.get_monitor_offset(binding.id)
            assert offset is not None
            self.assertEqual(offset.byte_offset, 10)
            self.assertEqual(offset.last_event_ts, "2026-04-22T12:01:00Z")
            self.assertEqual(offset.last_message_hash, "hash")

    def test_history_text_labels_non_final_events_as_progress(self) -> None:
        class HistoryAdapter(FakeAdapter):
            def history(self, transcript_path: Path, *, session_id: str | None = None, limit: int = 10):
                return [
                    ProviderTranscriptEvent(
                        role="assistant",
                        content_type="text",
                        text="Checking git diff",
                        timestamp="2026-04-22T12:00:00Z",
                        is_final=False,
                    ),
                    ProviderTranscriptEvent(
                        role="assistant",
                        content_type="text",
                        text="Final verdict",
                        timestamp="2026-04-22T12:01:00Z",
                        is_final=True,
                    ),
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            service = AppService(config=make_config(base_dir), repository=repository, providers=FakeRegistry(HistoryAdapter()))

            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                provider_session_id="session-123",
                transcript_path=transcript_path,
                status=BindingStatus.ACTIVE,
            )

            history = service.history_text(binding)
            self.assertEqual(history, "[progress] Checking git diff\n\n[assistant] Final verdict")

    def test_refresh_promotes_pending_launch_without_activation_message(self) -> None:
        class DiscoveringAdapter(FakeAdapter):
            def __init__(self, transcript_path: Path) -> None:
                self._transcript_path = transcript_path

            def discover_session(
                self,
                repo_path: Path,
                *,
                started_after: str,
                requested_session_id: str | None = None,
                tmux_session_name: str | None = None,
                tmux_window_id: str | None = None,
            ):
                return type(
                    "Session",
                    (),
                    {
                        "session_id": "session-123",
                        "display_name": "Latest task",
                        "updated_at": "2026-04-22T12:00:00Z",
                        "repo_path": repo_path,
                        "transcript_path": self._transcript_path,
                    },
                )()

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            db_path = base_dir / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)
            transcript_path = base_dir / "transcript.jsonl"
            transcript_path.write_text("", encoding="utf-8")
            service = AppService(
                config=make_config(base_dir),
                repository=repository,
                providers=FakeRegistry(DiscoveringAdapter(transcript_path)),
            )

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                tmux_session_name="turnmux",
                tmux_window_id="@12",
                tmux_window_name="codex:tmp",
                status=BindingStatus.PENDING_START,
            )
            repository.save_pending_launch(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                repo_path=base_dir,
                discovery_deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
            )

            with patch("turnmux.app.service.tmux.window_exists", return_value=True):
                outbound = service.refresh_pending_and_active_bindings()

            self.assertEqual(outbound, [])
            rebound = repository.get_binding_by_id(binding.id)
            assert rebound is not None
            self.assertEqual(rebound.status, BindingStatus.ACTIVE)
            self.assertEqual(rebound.provider_session_id, "session-123")
