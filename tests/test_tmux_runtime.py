from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from uuid import uuid4

from turnmux.runtime import tmux


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
