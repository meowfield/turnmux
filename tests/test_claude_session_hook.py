from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turnmux.providers.claude_session_hook import (
    default_claude_session_map_path,
    find_claude_session_map_entry,
    process_claude_session_start_hook,
)


class ClaudeSessionHookTests(unittest.TestCase):
    def test_process_session_start_hook_writes_turnmux_session_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / ".turnmux"
            payload = {
                "hook_event_name": "SessionStart",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "cwd": str((Path(tmp_dir) / "repo").resolve()),
            }

            exit_code = process_claude_session_start_hook(
                runtime_home=runtime_home,
                payload=payload,
                env={"TMUX_PANE": "%42"},
                tmux_display_message=lambda pane_id: "turnmux:@12:claude:repo",
            )

            self.assertEqual(exit_code, 0)
            map_path = default_claude_session_map_path(runtime_home)
            stored = json.loads(map_path.read_text(encoding="utf-8"))
            self.assertEqual(
                stored["turnmux:@12"]["session_id"],
                "550e8400-e29b-41d4-a716-446655440000",
            )

    def test_process_session_start_hook_ignores_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / ".turnmux"
            exit_code = process_claude_session_start_hook(
                runtime_home=runtime_home,
                payload={"hook_event_name": "SessionStart", "session_id": "not-a-uuid"},
                env={"TMUX_PANE": "%42"},
                tmux_display_message=lambda pane_id: "turnmux:@12:claude:repo",
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse(default_claude_session_map_path(runtime_home).exists())

    def test_find_session_map_entry_falls_back_to_legacy_ccbot_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "session_map.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "turnmux:@12": {
                            "session_id": "550e8400-e29b-41d4-a716-446655440000",
                            "cwd": "/tmp/repo",
                            "window_name": "claude:repo",
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "turnmux.providers.claude_session_hook.LEGACY_CCBOT_SESSION_MAP_PATH",
                legacy_path,
            ):
                entry = find_claude_session_map_entry("turnmux", "@12", runtime_home=Path(tmp_dir) / ".turnmux")

            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.session_id, "550e8400-e29b-41d4-a716-446655440000")
            self.assertEqual(entry.window_name, "claude:repo")
