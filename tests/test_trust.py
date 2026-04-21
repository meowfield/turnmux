from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from turnmux.providers.trust import (
    ensure_claude_skip_dangerous_prompt,
    ensure_claude_project_trusted,
    ensure_codex_project_trusted,
    is_claude_skip_dangerous_prompt_enabled,
    is_claude_project_trusted,
    is_codex_project_trusted,
)


class ProviderTrustTests(unittest.TestCase):
    def test_ensure_claude_project_trusted_sets_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            repo_path.mkdir()
            state_path = Path(tmp_dir) / ".claude.json"
            state_path.write_text('{"projects":{}}', encoding="utf-8")

            ensure_claude_project_trusted(repo_path, state_path=state_path)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            project_state = payload["projects"][str(repo_path)]
            self.assertTrue(project_state["hasTrustDialogAccepted"])
            self.assertTrue(is_claude_project_trusted(repo_path, state_path=state_path))

    def test_ensure_codex_project_trusted_upserts_exact_project_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            repo_path.mkdir()
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text('model = "gpt-5.4"\n', encoding="utf-8")

            ensure_codex_project_trusted(repo_path, config_path=config_path)

            text = config_path.read_text(encoding="utf-8")
            self.assertIn(f'[projects."{repo_path}"]', text)
            self.assertIn('trust_level = "trusted"', text)
            self.assertTrue(is_codex_project_trusted(repo_path, config_path=config_path))

    def test_ensure_claude_skip_dangerous_prompt_updates_user_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            settings_path.write_text('{"permissions":{"allow":["Read"]}}', encoding="utf-8")

            ensure_claude_skip_dangerous_prompt(settings_path=settings_path)

            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["permissions"]["skipDangerousModePermissionPrompt"])
            self.assertTrue(is_claude_skip_dangerous_prompt_enabled(settings_path=settings_path))

    def test_invalid_claude_settings_are_backed_up_and_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = Path(tmp_dir) / "settings.json"
            settings_path.write_text('{"permissions":', encoding="utf-8")

            ensure_claude_skip_dangerous_prompt(settings_path=settings_path)

            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["permissions"]["skipDangerousModePermissionPrompt"])
            backups = list(settings_path.parent.glob("settings.json.turnmux-invalid-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), '{"permissions":')

    def test_invalid_claude_project_state_does_not_break_trust_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            repo_path.mkdir()
            state_path = Path(tmp_dir) / ".claude.json"
            state_path.write_text('{"projects":', encoding="utf-8")

            self.assertFalse(is_claude_project_trusted(repo_path, state_path=state_path))

            ensure_claude_project_trusted(repo_path, state_path=state_path)

            self.assertTrue(is_claude_project_trusted(repo_path, state_path=state_path))
            backups = list(state_path.parent.glob(".claude.json.turnmux-invalid-*.bak"))
            self.assertEqual(len(backups), 1)

    def test_invalid_codex_config_is_backed_up_and_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            repo_path.mkdir()
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text('model = "gpt-5.4"\n[projects."\n', encoding="utf-8")

            self.assertFalse(is_codex_project_trusted(repo_path, config_path=config_path))

            ensure_codex_project_trusted(repo_path, config_path=config_path)

            self.assertTrue(is_codex_project_trusted(repo_path, config_path=config_path))
            backups = list(config_path.parent.glob("config.toml.turnmux-invalid-*.bak"))
            self.assertEqual(len(backups), 1)
