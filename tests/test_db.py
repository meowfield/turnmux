from __future__ import annotations

from contextlib import closing
import os
from pathlib import Path
import tempfile
import unittest

from turnmux.state.db import bootstrap_database, connect


class DatabaseBootstrapTests(unittest.TestCase):
    def test_bootstrap_creates_database_and_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime" / "state.db"

            created_path = bootstrap_database(db_path)

            self.assertEqual(created_path, db_path.resolve())
            self.assertTrue(created_path.exists())
            if os.name == "posix":
                self.assertEqual(created_path.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(created_path.stat().st_mode & 0o777, 0o600)

            with closing(connect(created_path)) as connection:
                table_names = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table';"
                    )
                }

            self.assertTrue(
                {"bindings", "monitor_offsets", "pending_launches", "pending_approvals", "onboarding_states", "settings"}.issubset(
                    table_names
                )
            )

    def test_bindings_table_contains_mvp_metadata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)

            with closing(connect(db_path)) as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(bindings);")
                }

            self.assertTrue(
                {
                    "chat_id",
                    "thread_id",
                    "provider",
                    "repo_path",
                    "tmux_session_name",
                    "tmux_window_id",
                    "tmux_window_name",
                    "provider_session_id",
                    "transcript_path",
                    "status",
                }.issubset(columns)
            )

    def test_pending_launches_table_supports_requested_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)

            with closing(connect(db_path)) as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(pending_launches);")
                }

            self.assertIn("requested_session_id", columns)

    def test_pending_approvals_table_supports_key_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)

            with closing(connect(db_path)) as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(pending_approvals);")
                }

            self.assertTrue({"approve_keys_json", "deny_keys_json", "fingerprint", "prompt_text"}.issubset(columns))
