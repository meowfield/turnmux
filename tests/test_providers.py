from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from turnmux.config import TurnmuxConfig
from turnmux.providers import ProviderRegistry
from turnmux.providers.claude import ClaudeAdapter
from turnmux.providers.claude_session_hook import ClaudeSessionMapEntry
from turnmux.providers.codex import CodexAdapter
from turnmux.providers.opencode import OpenCodeAdapter
from turnmux.state.models import ProviderName


def make_config(base_dir: Path, *, relay_claude_thinking: bool = False) -> TurnmuxConfig:
    return TurnmuxConfig(
        telegram_bot_token="token",
        allowed_user_ids=(1,),
        allowed_roots=(base_dir,),
        tmux_session_name="turnmux",
        claude_command=("claude", "--dangerously-skip-permissions"),
        codex_command=("codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access"),
        opencode_command=None,
        opencode_model=None,
        config_path=base_dir / "config.toml",
        relay_claude_thinking=relay_claude_thinking,
    )


class ProviderRegistryTests(unittest.TestCase):
    def test_registry_only_exposes_configured_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(tmp_path,),
                tmux_session_name="turnmux",
                claude_command=None,
                codex_command=("codex", "--no-alt-screen"),
                opencode_command=None,
                opencode_model=None,
                config_path=tmp_path / "config.toml",
                relay_claude_thinking=False,
            )

            registry = ProviderRegistry(config)
            self.assertEqual(registry.available_providers(), (ProviderName.CODEX,))

    def test_registry_passes_runtime_home_to_claude_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runtime_home = tmp_path / "runtime-home"

            registry = ProviderRegistry(make_config(tmp_path), runtime_home=runtime_home)

            claude = registry.get(ProviderName.CLAUDE)
            self.assertIsInstance(claude, ClaudeAdapter)
            self.assertEqual(claude.runtime_home, runtime_home.resolve(strict=False))


class ClaudeAdapterTests(unittest.TestCase):
    def test_project_dir_name_matches_claude_path_normalization(self) -> None:
        repo_path = Path("/opt/example-workspace/turnmux_Æ repo")
        self.assertEqual(
            ClaudeAdapter._project_dir_name(repo_path),
            "-opt-example-workspace-turnmux---repo",
        )

    def test_lists_resumable_sessions_and_parses_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            transcript_path = project_dir / "session-claude.jsonl"
            transcript_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"user","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T09:59:00Z","message":{{"role":"user","content":[{{"type":"text","text":"Investigate login failure"}}]}}}}
                    {{"type":"assistant","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"internal scratchpad"}},{{"type":"text","text":"I am checking the login flow."}}]}}}}
                    {{"type":"last-prompt","sessionId":"session-claude","lastPrompt":"Investigate login failure"}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path), claude_home=claude_root)
            sessions = adapter.list_resumable_sessions(repo_path)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].session_id, "session-claude")
            self.assertEqual(sessions[0].display_name, "Investigate login failure")

            batch = adapter.parse_new_events(transcript_path, 0)
            self.assertEqual([event.text for event in batch.events], ["I am checking the login flow."])

    def test_history_excludes_thinking_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            transcript_path = project_dir / "session-claude.jsonl"
            transcript_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"user","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T09:59:00Z","message":{{"role":"user","content":[{{"type":"text","text":"Investigate login failure"}}]}}}}
                    {{"type":"assistant","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"internal scratchpad"}},{{"type":"text","text":"Final answer only"}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path), claude_home=claude_root)
            history = adapter.history(transcript_path, limit=10)

            self.assertEqual([event.role for event in history], ["user", "assistant"])
            self.assertEqual(history[-1].text, "Final answer only")

    def test_relay_thinking_flag_restores_thinking_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            transcript_path = project_dir / "session-claude.jsonl"
            transcript_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"assistant","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"internal scratchpad"}},{{"type":"text","text":"Final answer only"}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path, relay_claude_thinking=True), claude_home=claude_root)
            batch = adapter.parse_new_events(transcript_path, 0)
            self.assertEqual([event.content_type for event in batch.events], ["thinking", "text"])

            history = adapter.history(transcript_path, limit=10)
            self.assertEqual(history[0].text, "internal scratchpad\nFinal answer only")

    def test_discovers_recent_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            transcript_path = project_dir / "session-claude.jsonl"
            transcript_path.write_text(
                f'{{"type":"user","sessionId":"session-claude","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:01:00Z","message":{{"role":"user","content":"hello"}}}}\n',
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path), claude_home=claude_root)
            discovered = adapter.discover_session(repo_path, started_after="2026-04-20T10:00:00+00:00")
            self.assertIsNotNone(discovered)
            assert discovered is not None
            self.assertEqual(discovered.session_id, "session-claude")

    def test_discover_session_uses_session_map_entry_for_exact_window_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()
            runtime_home = tmp_path / "runtime-home"

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            transcript_path = project_dir / "session-hooked.jsonl"
            transcript_path.write_text(
                f'{{"type":"assistant","sessionId":"session-hooked","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:01:00Z","message":{{"role":"assistant","content":[{{"type":"text","text":"hello"}}]}}}}\n',
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path), claude_home=claude_root, runtime_home=runtime_home)
            entry = ClaudeSessionMapEntry(
                session_id="session-hooked",
                cwd=repo_path.resolve(),
                window_name="claude:repo",
                source_path=tmp_path / "claude_session_map.json",
            )
            with patch("turnmux.providers.claude.find_claude_session_map_entry", return_value=entry) as find_entry:
                discovered = adapter.discover_session(
                    repo_path,
                    started_after="2026-04-20T10:00:00+00:00",
                    tmux_session_name="turnmux",
                    tmux_window_id="@12",
                )

            self.assertIsNotNone(discovered)
            assert discovered is not None
            self.assertEqual(discovered.session_id, "session-hooked")
            find_entry.assert_called_once_with("turnmux", "@12", runtime_home=runtime_home.resolve(strict=False))

    def test_list_resumable_sessions_uses_sessions_index_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            claude_root = tmp_path / ".claude"
            project_dir = claude_root / "projects" / str(repo_path.resolve()).replace("/", "-")
            project_dir.mkdir(parents=True)
            nested_dir = tmp_path / "nested"
            nested_dir.mkdir()
            transcript_path = nested_dir / "indexed-session.jsonl"
            transcript_path.write_text(
                f'{{"type":"assistant","sessionId":"indexed-session","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:01:00Z","message":{{"role":"assistant","content":[{{"type":"text","text":"hello"}}]}}}}\n',
                encoding="utf-8",
            )
            (project_dir / "sessions-index.json").write_text(
                textwrap.dedent(
                    f"""
                    {{
                      "entries": [
                        {{
                          "sessionId": "indexed-session",
                          "fullPath": "{transcript_path.resolve()}",
                          "projectPath": "{repo_path.resolve()}"
                        }}
                      ]
                    }}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = ClaudeAdapter(make_config(tmp_path), claude_home=claude_root)
            sessions = adapter.list_resumable_sessions(repo_path)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].session_id, "indexed-session")


class CodexAdapterTests(unittest.TestCase):
    def test_lists_resumable_sessions_and_parses_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            codex_root = tmp_path / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "20"
            sessions_dir.mkdir(parents=True)
            session_index = codex_root / "session_index.jsonl"
            session_index.parent.mkdir(parents=True, exist_ok=True)
            session_index.write_text(
                '{"id":"session-codex","thread_name":"Implement auth","updated_at":"2026-04-20T10:05:00Z"}\n',
                encoding="utf-8",
            )

            rollout_path = sessions_dir / "rollout-2026-04-20T10-00-00-session-codex.jsonl"
            rollout_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"session_meta","timestamp":"2026-04-20T10:00:00Z","payload":{{"id":"session-codex","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z"}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:01:00Z","payload":{{"type":"message","role":"assistant","phase":"commentary","content":[{{"type":"output_text","text":"Reviewing auth flow\\n\\n<oai-mem-citation>\\n<citation_entries>\\nMEMORY.md:1-2|note=[noise]\\n</citation_entries>\\n<rollout_ids>\\nabc\\n</rollout_ids>\\n</oai-mem-citation>\\n::git-stage{{cwd=\\"/tmp/repo\\"}}"}}]}}}}
                    {{"type":"event_msg","timestamp":"2026-04-20T10:02:00Z","payload":{{"type":"exec_command_end","command":["bash","-lc","pytest"],"exit_code":1,"aggregated_output":"tests failed"}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:03:00Z","payload":{{"type":"message","role":"assistant","phase":"final_answer","content":[{{"type":"output_text","text":"Auth flow is fixed.\\n::git-stage{{cwd=\\"/tmp/repo\\"}}"}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = CodexAdapter(make_config(tmp_path), codex_home=codex_root)
            sessions = adapter.list_resumable_sessions(repo_path)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].display_name, "Implement auth")

            batch = adapter.parse_new_events(rollout_path, 0)
            self.assertEqual(len(batch.events), 2)
            self.assertEqual(batch.events[0].text, "Reviewing auth flow")
            self.assertFalse(batch.events[0].is_final)
            self.assertEqual(batch.events[1].text, "Auth flow is fixed.")
            self.assertTrue(batch.events[1].is_final)

    def test_history_includes_codex_commentary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            codex_root = tmp_path / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "20"
            sessions_dir.mkdir(parents=True)
            rollout_path = sessions_dir / "rollout-2026-04-20T10-00-00-session-codex.jsonl"
            rollout_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"session_meta","timestamp":"2026-04-20T10:00:00Z","payload":{{"id":"session-codex","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z"}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:01Z","payload":{{"type":"message","role":"assistant","phase":"commentary","content":[{{"type":"output_text","text":"Thinking out loud"}}]}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:02Z","payload":{{"type":"message","role":"assistant","phase":"final_answer","content":[{{"type":"output_text","text":"Final answer only"}}]}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:03Z","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"Investigate flaky auth redirect"}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = CodexAdapter(make_config(tmp_path), codex_home=codex_root)
            history = adapter.history(rollout_path, limit=10)
            self.assertEqual([event.role for event in history], ["assistant", "assistant", "user"])
            self.assertEqual([event.text for event in history], ["Thinking out loud", "Final answer only", "Investigate flaky auth redirect"])
            self.assertEqual([event.is_final for event in history], [False, True, True])

    def test_sanitize_preserves_literal_citation_tag_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            codex_root = tmp_path / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "20"
            sessions_dir.mkdir(parents=True)
            rollout_path = sessions_dir / "rollout-2026-04-20T10-00-00-session-codex.jsonl"
            rollout_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"session_meta","timestamp":"2026-04-20T10:00:00Z","payload":{{"id":"session-codex","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z"}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:02Z","payload":{{"type":"message","role":"assistant","phase":"final_answer","content":[{{"type":"output_text","text":"`codex` adapter now cleans literal `<oai-mem-citation>` mentions safely."}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = CodexAdapter(make_config(tmp_path), codex_home=codex_root)
            batch = adapter.parse_new_events(rollout_path, 0)
            self.assertEqual(len(batch.events), 1)
            self.assertEqual(
                batch.events[0].text,
                "`codex` adapter now cleans literal `<oai-mem-citation>` mentions safely.",
            )

    def test_build_resume_command_uses_explicit_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = CodexAdapter(make_config(Path(tmp_dir)), codex_home=Path(tmp_dir) / ".codex")
            repo_path = Path(tmp_dir) / "repo"
            command = adapter.build_resume_command(repo_path, "session-codex")
            self.assertEqual(command[:6], ["codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"])
            self.assertIn("resume", command)
            self.assertIn("session-codex", command)

    def test_falls_back_to_first_user_prompt_when_thread_name_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()

            codex_root = tmp_path / ".codex"
            sessions_dir = codex_root / "sessions" / "2026" / "04" / "20"
            sessions_dir.mkdir(parents=True)
            session_index = codex_root / "session_index.jsonl"
            session_index.parent.mkdir(parents=True, exist_ok=True)
            session_index.write_text(
                '{"id":"session-codex","updated_at":"2026-04-20T10:05:00Z"}\n',
                encoding="utf-8",
            )

            rollout_path = sessions_dir / "rollout-2026-04-20T10-00-00-session-codex.jsonl"
            rollout_path.write_text(
                textwrap.dedent(
                    f"""
                    {{"type":"session_meta","timestamp":"2026-04-20T10:00:00Z","payload":{{"id":"session-codex","cwd":"{repo_path.resolve()}","timestamp":"2026-04-20T10:00:00Z"}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:01Z","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"<environment_context>ignore me</environment_context>"}}]}}}}
                    {{"type":"response_item","timestamp":"2026-04-20T10:00:02Z","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"Investigate flaky auth redirect"}}]}}}}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            adapter = CodexAdapter(make_config(tmp_path), codex_home=codex_root)
            sessions = adapter.list_resumable_sessions(repo_path)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].display_name, "Investigate flaky auth redirect")


class OpenCodeAdapterTests(unittest.TestCase):
    def test_builds_start_and_resume_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(tmp_path,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=("/opt/opencode/bin/opencode",),
                opencode_model="zai/glm-5.1",
                config_path=tmp_path / "config.toml",
                relay_claude_thinking=False,
            )
            adapter = OpenCodeAdapter(config, data_home=tmp_path / ".local" / "share" / "opencode")
            repo_path = tmp_path / "repo"
            command = adapter.build_start_command(repo_path, initial_prompt="hello")
            self.assertEqual(command[:3], ["/opt/opencode/bin/opencode", "--model", "zai/glm-5.1"])
            self.assertIn("--prompt", command)
            self.assertEqual(command[-1], str(repo_path))

            resume_command = adapter.build_resume_command(repo_path, "ses_123")
            self.assertEqual(resume_command, ["/opt/opencode/bin/opencode", "--session", "ses_123", str(repo_path)])

    def test_lists_sessions_parses_history_and_tracks_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()
            db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
            db_path.parent.mkdir(parents=True)
            _create_opencode_db(
                db_path,
                repo_path=repo_path,
                session_id="ses_123",
                title="New session - 2026-04-20T17:00:00.000Z",
                messages=[
                    {"id": "msg_user_1", "role": "user", "text_parts": ["Investigate flaky deploy"], "created_ms": 1000},
                    {"id": "msg_asst_1", "role": "assistant", "text_parts": ["Looking into the deploy now."], "created_ms": 2000},
                    {"id": "msg_asst_2", "role": "assistant", "text_parts": ["I found the issue.", "It is a bad env var."], "created_ms": 3000},
                ],
            )

            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(tmp_path,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=("/opt/opencode/bin/opencode",),
                opencode_model=None,
                config_path=tmp_path / "config.toml",
                relay_claude_thinking=False,
            )
            adapter = OpenCodeAdapter(config, data_home=db_path.parent)

            sessions = adapter.list_resumable_sessions(repo_path)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].session_id, "ses_123")
            self.assertEqual(sessions[0].display_name, "Investigate flaky deploy")

            history = adapter.history(db_path, session_id="ses_123", limit=10)
            self.assertEqual([event.role for event in history], ["user", "assistant", "assistant"])
            self.assertEqual(history[-1].text, "I found the issue.\nIt is a bad env var.")

            self.assertEqual(adapter.initial_monitor_offset(sessions[0]), 4)

            batch = adapter.parse_new_events(db_path, 1, session_id="ses_123")
            self.assertEqual(batch.new_offset, 4)
            self.assertEqual([event.text for event in batch.events], ["Looking into the deploy now.", "I found the issue.\nIt is a bad env var."])

    def test_discovers_requested_or_recent_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_path = tmp_path / "repo"
            repo_path.mkdir()
            db_path = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
            db_path.parent.mkdir(parents=True)
            _create_opencode_db(
                db_path,
                repo_path=repo_path,
                session_id="ses_older",
                title="Earlier task",
                messages=[{"id": "msg1", "role": "user", "text_parts": ["hello"], "created_ms": 1000}],
                session_created_ms=1000,
                session_updated_ms=1500,
            )
            _append_opencode_session(
                db_path,
                repo_path=repo_path,
                session_id="ses_newer",
                title="Latest task",
                messages=[{"id": "msg2", "role": "user", "text_parts": ["world"], "created_ms": 5000}],
                session_created_ms=5000,
                session_updated_ms=6000,
            )

            config = TurnmuxConfig(
                telegram_bot_token="token",
                allowed_user_ids=(1,),
                allowed_roots=(tmp_path,),
                tmux_session_name="turnmux",
                claude_command=("claude",),
                codex_command=("codex",),
                opencode_command=("/opt/opencode/bin/opencode",),
                opencode_model=None,
                config_path=tmp_path / "config.toml",
                relay_claude_thinking=False,
            )
            adapter = OpenCodeAdapter(config, data_home=db_path.parent)

            discovered = adapter.discover_session(repo_path, started_after="1970-01-01T00:00:05+00:00")
            self.assertIsNotNone(discovered)
            assert discovered is not None
            self.assertEqual(discovered.session_id, "ses_newer")

            requested = adapter.discover_session(repo_path, started_after="1970-01-01T00:00:00+00:00", requested_session_id="ses_older")
            self.assertIsNotNone(requested)
            assert requested is not None
            self.assertEqual(requested.session_id, "ses_older")


def _create_opencode_db(
    db_path: Path,
    *,
    repo_path: Path,
    session_id: str,
    title: str,
    messages: list[dict[str, object]],
    session_created_ms: int = 1000,
    session_updated_ms: int = 6000,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                parent_id TEXT,
                slug TEXT NOT NULL,
                directory TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT NOT NULL,
                share_url TEXT,
                summary_additions INTEGER,
                summary_deletions INTEGER,
                summary_files INTEGER,
                summary_diffs TEXT,
                revert TEXT,
                permission TEXT,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                time_compacting INTEGER,
                time_archived INTEGER,
                workspace_id TEXT
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        _append_opencode_session(
            db_path,
            repo_path=repo_path,
            session_id=session_id,
            title=title,
            messages=messages,
            session_created_ms=session_created_ms,
            session_updated_ms=session_updated_ms,
            connection=connection,
        )
    finally:
        connection.close()


def _append_opencode_session(
    db_path: Path,
    *,
    repo_path: Path,
    session_id: str,
    title: str,
    messages: list[dict[str, object]],
    session_created_ms: int,
    session_updated_ms: int,
    connection: sqlite3.Connection | None = None,
) -> None:
    owned = connection is None
    connection = connection or sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO session (
                id, project_id, parent_id, slug, directory, title, version,
                share_url, summary_additions, summary_deletions, summary_files,
                summary_diffs, revert, permission, time_created, time_updated,
                time_compacting, time_archived, workspace_id
            ) VALUES (?, 'project', NULL, ?, ?, ?, '1.14.19', NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL)
            """,
            (session_id, session_id, str(repo_path.resolve()), title, session_created_ms, session_updated_ms),
        )
        for index, message in enumerate(messages, start=1):
            message_id = str(message["id"])
            created_ms = int(message["created_ms"])
            role = str(message["role"])
            connection.execute(
                "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    created_ms,
                    created_ms,
                    json_dumps(
                        {
                            "role": role,
                            "time": {"created": created_ms, "completed": created_ms},
                        }
                    ),
                ),
            )
            for part_index, text in enumerate(message["text_parts"], start=1):
                part_id = f"prt_{message_id}_{part_index}"
                part_created = created_ms + part_index
                connection.execute(
                    "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        part_id,
                        message_id,
                        session_id,
                        part_created,
                        part_created,
                        json_dumps({"type": "text", "text": str(text), "time": {"start": part_created, "end": part_created}}),
                    ),
                )
        connection.commit()
    finally:
        if owned:
            connection.close()


def json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
