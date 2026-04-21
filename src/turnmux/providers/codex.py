from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any

from .base import ParseBatch, ProviderAdapter, ProviderSession, ProviderTranscriptEvent, compute_message_hash, parse_timestamp, read_jsonl_tail, shorten_text
from ..state.models import ProviderName


class CodexAdapter(ProviderAdapter):
    name = ProviderName.CODEX

    def __init__(self, config, codex_home: Path | None = None) -> None:
        super().__init__(config)
        self.codex_home = codex_home or (Path.home() / ".codex")
        self.session_index_path = self.codex_home / "session_index.jsonl"
        self.sessions_root = self.codex_home / "sessions"

    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        return build_codex_compatible_command(self.config.codex_command, initial_prompt=initial_prompt)

    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        return build_codex_compatible_command(self.config.codex_command, resume_session_id=session_id, repo_path=repo_path)

    def list_resumable_sessions(self, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        index = self._load_session_index()
        sessions: list[ProviderSession] = []

        for transcript_path in sorted(self.sessions_root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
            session = self._session_from_rollout(transcript_path, repo_path, index)
            if session:
                sessions.append(session)
            if len(sessions) >= limit:
                break

        return sessions

    def discover_session(
        self,
        repo_path: Path,
        *,
        started_after: str,
        requested_session_id: str | None = None,
    ) -> ProviderSession | None:
        index = self._load_session_index()
        started_after_dt = parse_timestamp(started_after)
        for transcript_path in sorted(self.sessions_root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
            session = self._session_from_rollout(transcript_path, repo_path, index)
            if not session:
                continue
            if requested_session_id and session.session_id == requested_session_id:
                return session
            updated_at_dt = parse_timestamp(session.updated_at)
            if not requested_session_id and started_after_dt and updated_at_dt and updated_at_dt >= started_after_dt:
                return session
        return None

    def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
        records, new_offset = read_jsonl_tail(transcript_path, offset)
        events: list[ProviderTranscriptEvent] = []

        for record in records:
            record_type = record.get("type")
            payload = record.get("payload")
            timestamp = record.get("timestamp")
            if not isinstance(payload, dict):
                continue

            if record_type == "response_item" and payload.get("type") == "message" and payload.get("role") == "assistant":
                for item in payload.get("content", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "output_text" and isinstance(item.get("text"), str) and item["text"].strip():
                        events.append(
                            ProviderTranscriptEvent(
                                role="assistant",
                                content_type="text",
                                text=item["text"].strip(),
                                timestamp=timestamp,
                                is_final=payload.get("phase") != "commentary",
                            )
                        )
            elif record_type == "event_msg" and payload.get("type") == "exec_command_end" and payload.get("exit_code") not in {None, 0}:
                command = payload.get("command") or []
                command_text = shlex.join(command) if isinstance(command, list) else str(command)
                aggregated_output = payload.get("aggregated_output") or ""
                summary = f"Command failed ({payload.get('exit_code')}): {command_text}".strip()
                if isinstance(aggregated_output, str) and aggregated_output.strip():
                    summary += "\n" + shorten_text(aggregated_output, limit=500)
                events.append(
                    ProviderTranscriptEvent(
                        role="tool",
                        content_type="shell",
                        text=summary,
                        timestamp=timestamp,
                        is_final=True,
                    )
                )

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
        events: list[ProviderTranscriptEvent] = []
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "response_item" and payload.get("type") == "message":
                role = payload.get("role")
                text = _extract_codex_message_text(payload)
                if role in {"assistant", "user"} and text:
                    events.append(
                        ProviderTranscriptEvent(
                            role=role,
                            content_type="text",
                            text=text,
                            timestamp=record.get("timestamp"),
                            is_final=payload.get("phase") != "commentary",
                        )
                    )
        return events[-limit:]

    def _load_session_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        if not self.session_index_path.exists():
            return index

        for line in self.session_index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                index[payload["id"]] = payload
        return index

    def _session_from_rollout(
        self,
        transcript_path: Path,
        repo_path: Path,
        index: dict[str, dict[str, Any]],
    ) -> ProviderSession | None:
        meta = _read_codex_session_meta(transcript_path)
        if not meta:
            return None
        cwd = meta.get("cwd")
        if not isinstance(cwd, str) or Path(cwd).resolve(strict=False) != repo_path.resolve(strict=False):
            return None

        session_id = meta.get("id")
        if not isinstance(session_id, str):
            return None

        index_entry = index.get(session_id, {})
        display_name = _codex_display_name(transcript_path, index_entry, session_id)
        updated_at = index_entry.get("updated_at") if isinstance(index_entry.get("updated_at"), str) else meta.get("timestamp")
        if not isinstance(updated_at, str):
            updated_at = _timestamp_from_stat(transcript_path)

        return ProviderSession(
            session_id=session_id,
            display_name=shorten_text(display_name),
            updated_at=updated_at,
            repo_path=repo_path,
            transcript_path=transcript_path,
        )


def _ensure_no_alt_screen(command: tuple[str, ...] | list[str]) -> list[str]:
    normalized = list(command)
    if "--no-alt-screen" not in normalized:
        normalized.append("--no-alt-screen")
    return normalized


def build_codex_compatible_command(
    command: tuple[str, ...] | list[str],
    *,
    initial_prompt: str | None = None,
    resume_session_id: str | None = None,
    repo_path: Path | None = None,
) -> list[str]:
    normalized = _ensure_no_alt_screen(command)
    if resume_session_id:
        if repo_path is None:
            raise ValueError("repo_path is required when resuming a Codex-compatible session.")
        normalized.extend(["resume", resume_session_id, "--cd", str(repo_path)])
    elif initial_prompt:
        normalized.append(initial_prompt)
    return normalized


def _extract_codex_message_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("content", []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"output_text", "input_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"].strip())
    return "\n".join(part for part in parts if part).strip()


def _read_codex_session_meta(transcript_path: Path) -> dict[str, Any] | None:
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "session_meta" and isinstance(record.get("payload"), dict):
                return record["payload"]
    return None


def _codex_display_name(transcript_path: Path, index_entry: dict[str, Any], session_id: str) -> str:
    thread_name = index_entry.get("thread_name")
    if isinstance(thread_name, str) and thread_name.strip():
        return shorten_text(thread_name.strip())

    derived = _read_codex_first_user_prompt(transcript_path)
    if derived:
        return shorten_text(derived)

    return session_id


def _read_codex_first_user_prompt(transcript_path: Path) -> str | None:
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload")
            if record.get("type") != "response_item" or not isinstance(payload, dict):
                continue
            if payload.get("type") != "message" or payload.get("role") != "user":
                continue
            text = _extract_codex_message_text(payload)
            if not text:
                continue
            normalized = text.strip()
            if normalized.startswith("<environment_context>") or normalized.startswith("<cwd>"):
                continue
            return normalized
    return None


def _timestamp_from_stat(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
