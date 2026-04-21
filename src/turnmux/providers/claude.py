from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from .base import ParseBatch, ProviderAdapter, ProviderSession, ProviderTranscriptEvent, compute_message_hash, parse_timestamp, read_jsonl_tail, shorten_text
from ..state.models import ProviderName


class ClaudeAdapter(ProviderAdapter):
    name = ProviderName.CLAUDE

    def __init__(self, config, claude_home: Path | None = None) -> None:
        super().__init__(config)
        base_dir = claude_home or Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        self.projects_root = base_dir / "projects"

    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        command = list(self.config.claude_command)
        if initial_prompt:
            command.append(initial_prompt)
        return command

    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        return [*self.config.claude_command, "--resume", session_id]

    def list_resumable_sessions(self, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        sessions: list[ProviderSession] = []
        for path in self._candidate_project_files(repo_path):
            session = self._session_from_transcript(path, repo_path)
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
        started_after_dt = parse_timestamp(started_after)
        for path in self._candidate_project_files(repo_path, limit=30):
            session = self._session_from_transcript(path, repo_path)
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
        relay_thinking = self.config.relay_claude_thinking

        for record in records:
            if record.get("type") != "assistant":
                continue
            message = record.get("message")
            timestamp = record.get("timestamp")
            if not isinstance(message, dict):
                continue
            for item in message.get("content", []):
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str) and item["text"].strip():
                    events.append(
                        ProviderTranscriptEvent(
                            role="assistant",
                            content_type="text",
                            text=item["text"].strip(),
                            timestamp=timestamp,
                            is_final=True,
                        )
                    )
                elif relay_thinking and item_type == "thinking" and isinstance(item.get("thinking"), str) and item["thinking"].strip():
                    events.append(
                        ProviderTranscriptEvent(
                            role="assistant",
                            content_type="thinking",
                            text=item["thinking"].strip(),
                            timestamp=timestamp,
                            is_final=False,
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
        for record in self._read_all_records(transcript_path):
            event_type = record.get("type")
            if event_type not in {"user", "assistant"}:
                continue
            message = record.get("message")
            timestamp = record.get("timestamp")
            if not isinstance(message, dict):
                continue
            role = "assistant" if event_type == "assistant" else "user"
            text = _extract_claude_message_text(message.get("content"), include_thinking=self.config.relay_claude_thinking)
            if text:
                events.append(
                    ProviderTranscriptEvent(
                        role=role,
                        content_type="text",
                        text=text,
                        timestamp=timestamp,
                        is_final=True,
                    )
                )
        return events[-limit:]

    def _candidate_project_files(self, repo_path: Path, *, limit: int = 15) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()
        project_dir = self.projects_root / self._project_dir_name(repo_path)
        if project_dir.is_dir():
            for path in sorted(project_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
                candidates.append(path)
                seen.add(path)
                if len(candidates) >= limit:
                    return candidates

        for path in sorted(self.projects_root.rglob("*.jsonl"), key=lambda candidate: candidate.stat().st_mtime, reverse=True):
            if path in seen:
                continue
            candidates.append(path)
            if len(candidates) >= limit:
                break
        return candidates

    def _session_from_transcript(self, transcript_path: Path, repo_path: Path) -> ProviderSession | None:
        records = self._read_all_records(transcript_path)
        session_id: str | None = None
        updated_at: str | None = None
        display_name = transcript_path.stem

        for record in records:
            if record.get("type") in {"user", "assistant"}:
                session_id = session_id or record.get("sessionId")
                updated_at = record.get("timestamp") or updated_at
                cwd = record.get("cwd")
                if isinstance(cwd, str) and Path(cwd).resolve(strict=False) != repo_path.resolve(strict=False):
                    return None
                if record.get("type") == "user" and display_name == transcript_path.stem:
                    text = _extract_claude_message_text(record.get("message", {}).get("content"))
                    if text:
                        display_name = shorten_text(text)
            elif record.get("type") == "last-prompt" and isinstance(record.get("lastPrompt"), str):
                display_name = shorten_text(record["lastPrompt"])

        if not session_id:
            session_id = transcript_path.stem
        if not updated_at:
            updated_at = _timestamp_from_stat(transcript_path)

        return ProviderSession(
            session_id=session_id,
            display_name=display_name,
            updated_at=updated_at,
            repo_path=repo_path,
            transcript_path=transcript_path,
        )

    @staticmethod
    def _project_dir_name(repo_path: Path) -> str:
        raw = str(repo_path.resolve(strict=False))
        normalized = re.sub(r"[^A-Za-z0-9]", "-", raw)
        return normalized or "-"

    @staticmethod
    def _read_all_records(transcript_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records


def _extract_claude_message_text(content: Any, *, include_thinking: bool = False) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"].strip())
        elif include_thinking and item_type == "thinking" and isinstance(item.get("thinking"), str) and item["thinking"].strip():
            parts.append(item["thinking"].strip())

    return "\n".join(part for part in parts if part).strip()


def _timestamp_from_stat(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
