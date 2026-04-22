from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .base import ParseBatch, ProviderAdapter, ProviderError, ProviderSession, ProviderTranscriptEvent, compute_message_hash, parse_timestamp, shorten_text
from ..state.models import ProviderName


class OpenCodeAdapter(ProviderAdapter):
    name = ProviderName.OPENCODE

    def __init__(self, config, data_home: Path | None = None) -> None:
        super().__init__(config)
        self.data_home = data_home or (Path.home() / ".local" / "share" / "opencode")
        self.db_path = self.data_home / "opencode.db"

    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        command = list(self._required_command())
        if self.config.opencode_model:
            command.extend(["--model", self.config.opencode_model])
        if initial_prompt:
            command.extend(["--prompt", initial_prompt])
        command.append(str(repo_path))
        return command

    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        return [*self._required_command(), "--session", session_id, str(repo_path)]

    def list_resumable_sessions(self, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        rows = self._session_rows(repo_path=repo_path, limit=limit)
        return [self._provider_session_from_row(repo_path, row) for row in rows]

    def discover_session(
        self,
        repo_path: Path,
        *,
        started_after: str,
        requested_session_id: str | None = None,
        tmux_session_name: str | None = None,
        tmux_window_id: str | None = None,
    ) -> ProviderSession | None:
        started_after_dt = parse_timestamp(started_after)
        started_after_ms = _datetime_to_epoch_ms(started_after_dt) if started_after_dt else None
        rows = self._session_rows(
            repo_path=repo_path,
            requested_session_id=requested_session_id,
            started_after_ms=started_after_ms,
            limit=1,
        )
        if not rows:
            return None
        return self._provider_session_from_row(repo_path, rows[0])

    def initial_monitor_offset(self, session: ProviderSession) -> int:
        return self._latest_part_rowid(session.session_id)

    def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
        if not session_id:
            raise ProviderError("OpenCode monitoring requires an explicit session_id.")

        rows = self._part_rows(transcript_path, session_id=session_id, offset=offset)
        if not rows:
            return ParseBatch(events=(), new_offset=offset, last_event_ts=None, last_message_hash=None)

        events: list[ProviderTranscriptEvent] = []
        current_message_id: str | None = None
        current_texts: list[str] = []
        current_timestamp: str | None = None
        new_offset = offset

        def flush_current() -> None:
            nonlocal current_message_id, current_texts, current_timestamp
            if current_message_id and current_texts:
                events.append(
                    ProviderTranscriptEvent(
                        role="assistant",
                        content_type="text",
                        text="\n".join(current_texts).strip(),
                        timestamp=current_timestamp,
                        is_final=True,
                    )
                )
            current_message_id = None
            current_texts = []
            current_timestamp = None

        for row in rows:
            new_offset = max(new_offset, row["row_id"])
            if row["message_id"] != current_message_id:
                flush_current()
                current_message_id = row["message_id"]

            if row["role"] != "assistant" or row["part_type"] != "text" or not row["text"]:
                continue

            text = row["text"].strip()
            if not text:
                continue

            current_texts.append(text)
            if current_timestamp is None:
                current_timestamp = row["timestamp"]

        flush_current()

        return ParseBatch(
            events=tuple(events),
            new_offset=new_offset,
            last_event_ts=events[-1].timestamp if events else None,
            last_message_hash=compute_message_hash(events),
        )

    def history(
        self,
        transcript_path: Path,
        *,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[ProviderTranscriptEvent]:
        if not session_id:
            raise ProviderError("OpenCode history requires an explicit session_id.")

        message_rows = self._message_rows(transcript_path, session_id=session_id, limit=max(limit * 4, 20))
        if not message_rows:
            return []

        message_parts = self._message_parts(transcript_path, [row["id"] for row in message_rows])
        events: list[ProviderTranscriptEvent] = []

        for row in message_rows:
            role = row["role"]
            if role not in {"assistant", "user"}:
                continue
            texts = [part["text"].strip() for part in message_parts[row["id"]] if part["part_type"] == "text" and part["text"] and part["text"].strip()]
            if not texts:
                continue
            events.append(
                ProviderTranscriptEvent(
                    role=role,
                    content_type="text",
                    text="\n".join(texts),
                    timestamp=row["timestamp"],
                    is_final=True,
                )
            )

        return events[-limit:]

    def _required_command(self) -> tuple[str, ...]:
        if not self.config.opencode_command:
            raise ProviderError("OpenCode is not configured. Set opencode_command in ~/.turnmux/config.toml.")
        return self.config.opencode_command

    def _session_rows(
        self,
        *,
        repo_path: Path,
        requested_session_id: str | None = None,
        started_after_ms: int | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized_repo = repo_path.resolve(strict=False)
        query = [
            "SELECT id, directory, title, time_created, time_updated",
            "FROM session",
            "WHERE time_archived IS NULL",
        ]
        params: list[Any] = []
        if requested_session_id:
            query.append("AND id = ?")
            params.append(requested_session_id)
        if started_after_ms is not None:
            query.append("AND time_updated >= ?")
            params.append(started_after_ms)
        query.append("ORDER BY time_updated DESC")
        query.append("LIMIT ?")
        params.append(limit)

        rows: list[dict[str, Any]] = []
        with closing(self._connect(self.db_path)) as connection:
            for raw_row in connection.execute("\n".join(query), params):
                row = dict(raw_row)
                directory = row.get("directory")
                if not isinstance(directory, str):
                    continue
                if Path(directory).resolve(strict=False) != normalized_repo:
                    continue
                rows.append(row)
        return rows

    def _provider_session_from_row(self, repo_path: Path, row: dict[str, Any]) -> ProviderSession:
        session_id = str(row["id"])
        title = row.get("title") if isinstance(row.get("title"), str) else ""
        display_name = title.strip()
        if not display_name or display_name.startswith("New session - "):
            derived = self._first_user_prompt(self.db_path, session_id=session_id)
            display_name = derived or session_id

        return ProviderSession(
            session_id=session_id,
            display_name=shorten_text(display_name),
            updated_at=_epoch_ms_to_iso(row.get("time_updated")),
            repo_path=repo_path,
            transcript_path=self.db_path,
        )

    def _latest_part_rowid(self, session_id: str) -> int:
        with closing(self._connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) AS max_rowid FROM part WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["max_rowid"]) if row else 0

    def _part_rows(self, db_path: Path, *, session_id: str, offset: int) -> list[dict[str, Any]]:
        query = """
            SELECT
                p.rowid AS row_id,
                p.message_id AS message_id,
                p.data AS part_data,
                m.data AS message_data
            FROM part AS p
            JOIN message AS m ON m.id = p.message_id
            WHERE p.session_id = ? AND p.rowid > ?
            ORDER BY p.rowid ASC
        """
        rows: list[dict[str, Any]] = []
        with closing(self._connect(db_path)) as connection:
            for raw_row in connection.execute(query, (session_id, offset)):
                message_payload = _safe_json_loads(raw_row["message_data"])
                part_payload = _safe_json_loads(raw_row["part_data"])
                role = message_payload.get("role")
                if not isinstance(role, str):
                    continue
                rows.append(
                    {
                        "row_id": int(raw_row["row_id"]),
                        "message_id": raw_row["message_id"],
                        "role": role,
                        "part_type": part_payload.get("type"),
                        "text": part_payload.get("text"),
                        "timestamp": _opencode_part_timestamp(part_payload, message_payload),
                    }
                )
        return rows

    def _message_rows(self, db_path: Path, *, session_id: str, limit: int) -> list[dict[str, Any]]:
        query = """
            SELECT id, data, time_created
            FROM message
            WHERE session_id = ?
            ORDER BY time_created DESC
            LIMIT ?
        """
        rows: list[dict[str, Any]] = []
        with closing(self._connect(db_path)) as connection:
            raw_rows = connection.execute(query, (session_id, limit)).fetchall()

        for raw_row in reversed(raw_rows):
            payload = _safe_json_loads(raw_row["data"])
            role = payload.get("role")
            if not isinstance(role, str):
                continue
            rows.append(
                {
                    "id": raw_row["id"],
                    "role": role,
                    "timestamp": _epoch_ms_to_iso(payload.get("time", {}).get("created") if isinstance(payload.get("time"), dict) else raw_row["time_created"]),
                }
            )
        return rows

    def _message_parts(self, db_path: Path, message_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not message_ids:
            return {}

        placeholders = ", ".join("?" for _ in message_ids)
        query = f"""
            SELECT message_id, data
            FROM part
            WHERE message_id IN ({placeholders})
            ORDER BY rowid ASC
        """

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with closing(self._connect(db_path)) as connection:
            for raw_row in connection.execute(query, message_ids):
                payload = _safe_json_loads(raw_row["data"])
                grouped[raw_row["message_id"]].append(
                    {
                        "part_type": payload.get("type"),
                        "text": payload.get("text"),
                    }
                )
        return grouped

    def _first_user_prompt(self, db_path: Path, *, session_id: str) -> str | None:
        query = """
            SELECT p.data, m.data AS message_data
            FROM part AS p
            JOIN message AS m ON m.id = p.message_id
            WHERE p.session_id = ?
            ORDER BY p.rowid ASC
        """
        with closing(self._connect(db_path)) as connection:
            for raw_row in connection.execute(query, (session_id,)):
                message_payload = _safe_json_loads(raw_row["message_data"])
                if message_payload.get("role") != "user":
                    continue
                payload = _safe_json_loads(raw_row["data"])
                if payload.get("type") != "text":
                    continue
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    return shorten_text(text.strip())
        return None

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        if not db_path.exists():
            raise ProviderError(f"OpenCode database is missing: {db_path}")
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _epoch_ms_to_iso(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _datetime_to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _opencode_part_timestamp(part_payload: dict[str, Any], message_payload: dict[str, Any]) -> str | None:
    part_time = part_payload.get("time")
    if isinstance(part_time, dict):
        for key in ("end", "start", "created"):
            raw = part_time.get(key)
            if isinstance(raw, (int, float)):
                return _epoch_ms_to_iso(raw)

    message_time = message_payload.get("time")
    if isinstance(message_time, dict):
        for key in ("completed", "created"):
            raw = message_time.get(key)
            if isinstance(raw, (int, float)):
                return _epoch_ms_to_iso(raw)

    return None
