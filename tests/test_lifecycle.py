from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from turnmux.runtime.lifecycle import HeartbeatWriter, read_heartbeat


class LifecycleTests(unittest.TestCase):
    def test_heartbeat_writer_persists_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            heartbeat_path = Path(tmp_dir) / "heartbeat.json"
            writer = HeartbeatWriter(heartbeat_path, started_at="2026-04-20T19:00:00+00:00")

            writer.write(status="running", note="test")
            payload = read_heartbeat(heartbeat_path)

            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["note"], "test")
            self.assertEqual(payload["started_at"], "2026-04-20T19:00:00+00:00")
            self.assertIn("last_heartbeat_at", payload)
            self.assertEqual(payload["pid"], writer.pid)
            if os.name == "posix":
                self.assertEqual(heartbeat_path.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(heartbeat_path.stat().st_mode & 0o777, 0o600)
