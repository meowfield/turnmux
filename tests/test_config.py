from __future__ import annotations

from pathlib import Path
import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from turnmux.config import (
    ConfigError,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    load_config,
)


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_parses_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111, 222, 111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = "claude --dangerously-skip-permissions"
                    codex_command = ["codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.telegram_bot_token, "bot-token")
            self.assertEqual(config.allowed_user_ids, (111, 222))
            self.assertEqual(config.allowed_roots, (repo_root.resolve(),))
            self.assertEqual(config.tmux_session_name, "turnmux")
            self.assertEqual(config.claude_command, ("claude", "--dangerously-skip-permissions"))
            self.assertEqual(
                config.codex_command,
                ("codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"),
            )
            self.assertIsNone(config.opencode_command)
            self.assertIsNone(config.opencode_model)
            self.assertFalse(config.relay_claude_thinking)
            self.assertIsNone(config.openai_api_key)
            self.assertEqual(config.openai_base_url, DEFAULT_OPENAI_BASE_URL)
            self.assertEqual(config.openai_transcription_model, DEFAULT_OPENAI_TRANSCRIPTION_MODEL)

    def test_load_config_parses_optional_opencode_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    opencode_command = ["/opt/opencode/bin/opencode"]
                    opencode_model = "zai/glm-5.1"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(config.opencode_command, ("/opt/opencode/bin/opencode",))
            self.assertEqual(config.opencode_model, "zai/glm-5.1")

    def test_load_config_allows_single_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    codex_command = ["codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertIsNone(config.claude_command)
            self.assertEqual(
                config.codex_command,
                ("codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"),
            )

    def test_load_config_parses_optional_claude_thinking_relay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    relay_claude_thinking = true
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertTrue(config.relay_claude_thinking)

    def test_load_config_reads_openai_voice_transcription_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    openai_api_key = "sk-openai"
                    openai_base_url = "https://proxy.example.com/v1"
                    openai_transcription_model = "gpt-4o-mini-transcribe"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(config.openai_api_key, "sk-openai")
            self.assertEqual(config.openai_base_url, "https://proxy.example.com/v1")
            self.assertEqual(config.openai_transcription_model, "gpt-4o-mini-transcribe")

    def test_load_config_falls_back_to_openai_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-env",
                    "OPENAI_BASE_URL": "https://env.example.com/v1",
                    "OPENAI_TRANSCRIPTION_MODEL": "gpt-4o-mini-transcribe",
                },
                clear=False,
            ):
                config = load_config(config_path)

            self.assertEqual(config.openai_api_key, "sk-env")
            self.assertEqual(config.openai_base_url, "https://env.example.com/v1")
            self.assertEqual(config.openai_transcription_model, "gpt-4o-mini-transcribe")

    def test_allowed_roots_are_resolved_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            canonical_root = tmp_path / "workspace"
            canonical_root.mkdir()
            alternate_spelling = canonical_root.parent / canonical_root.name / ".." / canonical_root.name

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{canonical_root}", "{alternate_spelling}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.allowed_roots, (canonical_root.resolve(),))

    def test_missing_required_field_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    claude_command = ["claude"]
                    codex_command = ["codex", "--no-alt-screen"]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(config_path)

    def test_missing_all_provider_commands_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "workspace"
            repo_root.mkdir()

            config_path = tmp_path / "config.toml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    telegram_bot_token = "bot-token"
                    allowed_user_ids = [111]
                    allowed_roots = ["{repo_root}"]
                    tmux_session_name = "turnmux"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(config_path)
