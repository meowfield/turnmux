from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

from ..runtime.home import ensure_private_directory, set_private_file_permissions


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS bindings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    tmux_session_name TEXT NOT NULL,
    tmux_window_id TEXT,
    tmux_window_name TEXT,
    provider_session_id TEXT,
    transcript_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending_start', 'active', 'stopped', 'missing')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, thread_id)
);

CREATE TABLE IF NOT EXISTS monitor_offsets (
    binding_id INTEGER PRIMARY KEY,
    byte_offset INTEGER NOT NULL DEFAULT 0 CHECK (byte_offset >= 0),
    last_event_ts TEXT,
    last_message_hash TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(binding_id) REFERENCES bindings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pending_launches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binding_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    discovery_deadline_at TEXT NOT NULL,
    requested_session_id TEXT,
    FOREIGN KEY(binding_id) REFERENCES bindings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binding_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    approve_keys_json TEXT NOT NULL,
    deny_keys_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(binding_id) REFERENCES bindings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS onboarding_states (
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL,
    step TEXT NOT NULL,
    provider TEXT,
    repo_path TEXT,
    mode TEXT CHECK (mode IN ('fresh', 'resume')),
    pending_user_text TEXT,
    resume_candidates_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(chat_id, thread_id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bindings_status ON bindings(status);
CREATE INDEX IF NOT EXISTS idx_pending_launches_binding_id ON pending_launches(binding_id);
CREATE INDEX IF NOT EXISTS idx_pending_approvals_binding_id ON pending_approvals(binding_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    resolved_db_path = db_path.expanduser().resolve(strict=False)
    ensure_private_directory(resolved_db_path.parent)

    connection = sqlite3.connect(resolved_db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def bootstrap_database(db_path: Path) -> Path:
    resolved_db_path = db_path.expanduser().resolve(strict=False)

    with closing(connect(resolved_db_path)) as connection:
        connection.executescript(SCHEMA)
        _ensure_column(connection, "pending_launches", "requested_session_id", "TEXT")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_launches_binding_id_unique ON pending_launches(binding_id);"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_approvals_binding_id_unique ON pending_approvals(binding_id);"
        )
        connection.commit()
    set_private_file_permissions(resolved_db_path)

    return resolved_db_path


def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name});")
    }
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql};")
