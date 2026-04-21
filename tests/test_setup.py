from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turnmux.doctor import TOKEN_PLACEHOLDER, USER_ID_PLACEHOLDER, render_sample_config, write_sample_config
from turnmux.runtime.home import initialize_runtime_home


class SetupHelpersTests(unittest.TestCase):
    def test_render_sample_config_uses_detected_working_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            working_root = Path(tmp_dir) / "workspace"
            working_root.mkdir()
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")

            rendered = render_sample_config(runtime_paths, working_dir=working_root)

            self.assertIn(f'allowed_roots = ["{working_root.resolve(strict=False)}"]', rendered)
            self.assertIn(TOKEN_PLACEHOLDER, rendered)
            self.assertIn(str(USER_ID_PLACEHOLDER), rendered)
            self.assertIn("--dangerously-skip-permissions", rendered)
            self.assertIn("--ask-for-approval", rendered)
            self.assertIn("on-request", rendered)
            self.assertIn("danger-full-access", rendered)
            self.assertIn("--no-alt-screen", rendered)
            self.assertIn('openai_base_url = "https://api.openai.com/v1"', rendered)
            self.assertIn('openai_transcription_model = "gpt-4o-transcribe"', rendered)

    def test_render_sample_config_prefers_git_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "workspace" / "repo"
            nested_dir = repo_root / "src" / "turnmux"
            nested_dir.mkdir(parents=True)
            (repo_root / ".git").mkdir()
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")

            rendered = render_sample_config(runtime_paths, working_dir=nested_dir)

            self.assertIn(f'allowed_roots = ["{repo_root.resolve(strict=False)}"]', rendered)

    def test_write_sample_config_writes_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            written_path = write_sample_config(runtime_paths, working_dir=Path(tmp_dir))

            self.assertTrue(written_path.exists())
            self.assertIn(TOKEN_PLACEHOLDER, written_path.read_text(encoding="utf-8"))

            with self.assertRaises(FileExistsError):
                write_sample_config(runtime_paths, working_dir=Path(tmp_dir))

            if os.name == "posix":
                self.assertEqual(runtime_paths.home.stat().st_mode & 0o777, 0o700)
                self.assertEqual(written_path.stat().st_mode & 0o777, 0o600)

    def test_render_sample_config_comments_unavailable_providers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            working_root = Path(tmp_dir) / "workspace"
            working_root.mkdir()
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")

            with patch(
                "turnmux.doctor._detect_binary",
                side_effect=lambda name: {"claude": None, "codex": "/usr/local/bin/codex", "opencode": None}[name],
            ):
                rendered = render_sample_config(runtime_paths, working_dir=working_root)

            self.assertIn("# claude_command = [", rendered)
            self.assertIn("codex_command = [", rendered)
            self.assertIn('"/usr/local/bin/codex"', rendered)
            self.assertIn("# opencode_command = [", rendered)
