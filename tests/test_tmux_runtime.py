from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from turnmux.runtime import tmux


class TmuxRuntimeUnitTests(unittest.TestCase):
    def test_paste_text_preserves_multiline_payload_as_bracketed_paste(self) -> None:
        calls = []

        def fake_run_tmux(args, *, check=True, input_text=None):
            calls.append((list(args), input_text))
            if args[:3] == ["display-message", "-p", "-t"]:
                return SimpleNamespace(returncode=0, stdout="%7\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("turnmux.runtime.tmux._run_tmux", side_effect=fake_run_tmux):
            tmux.paste_text("@7", "first line\nsecond line", enter=True, enter_delay_seconds=0.0)

        load_call = calls[1]
        paste_call = calls[2]
        send_enter_call = calls[3]

        self.assertEqual(load_call[0][0], "load-buffer")
        self.assertEqual(load_call[1], "first line\nsecond line")
        self.assertEqual(paste_call[0][:3], ["paste-buffer", "-p", "-r"])
        self.assertIn("-d", paste_call[0])
        self.assertEqual(send_enter_call[0], ["send-keys", "-t", "%7", "Enter"])


def _tmux_integration_available() -> bool:
    tmux_binary = shutil.which("tmux")
    if not tmux_binary:
        return False

    probe = subprocess.run(
        [tmux_binary, "display-message", "-p", "#{pid}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "Operation not permitted" not in (probe.stderr or "")


@unittest.skipUnless(_tmux_integration_available(), "tmux socket access is required for integration tests")
class TmuxRuntimeIntegrationTests(unittest.TestCase):
    def test_kill_window_terminates_process_inside_window(self) -> None:
        session_name = f"turnmux-test-{uuid4().hex[:8]}"
        pane_pid: int | None = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir)
            window = tmux.create_window(session_name, repo_path, window_name="kill-check")
            try:
                tmux.paste_text(window.window_id, "exec sleep 1000", enter=True, enter_delay_seconds=0.0)
                time.sleep(0.4)

                pane_pid = int(
                    subprocess.run(
                        ["tmux", "display-message", "-p", "-t", window.window_id, "#{pane_pid}"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )

                self.assertTrue(tmux.window_exists(session_name, window.window_id))
                os.kill(pane_pid, 0)

                tmux.kill_window(session_name, window.window_id)
                time.sleep(0.4)

                self.assertFalse(tmux.window_exists(session_name, window.window_id))
                with self.assertRaises(OSError):
                    os.kill(pane_pid, 0)
            finally:
                subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True, text=True)
