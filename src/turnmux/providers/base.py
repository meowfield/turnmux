from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

from ..state.models import ProviderName


class ProviderError(RuntimeError):
    """Raised when provider session discovery or parsing fails."""


@dataclass(frozen=True, slots=True)
class ProviderSession:
    session_id: str
    display_name: str
    updated_at: str
    repo_path: Path
    transcript_path: Path


@dataclass(frozen=True, slots=True)
class ProviderTranscriptEvent:
    role: str
    content_type: str
    text: str
    timestamp: str | None
    is_final: bool


@dataclass(frozen=True, slots=True)
class ParseBatch:
    events: tuple[ProviderTranscriptEvent, ...]
    new_offset: int
    last_event_ts: str | None
    last_message_hash: str | None


class ProviderAdapter(ABC):
    name: ProviderName

    def __init__(self, config) -> None:
        self.config = config

    def runtime_env(self) -> dict[str, str]:
        return {}

    def initial_monitor_offset(self, session: ProviderSession) -> int:
        return session.transcript_path.stat().st_size if session.transcript_path.exists() else 0

    @abstractmethod
    def build_start_command(self, repo_path: Path, *, initial_prompt: str | None = None) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def build_resume_command(self, repo_path: Path, session_id: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_resumable_sessions(self, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        raise NotImplementedError

    @abstractmethod
    def discover_session(
        self,
        repo_path: Path,
        *,
        started_after: str,
        requested_session_id: str | None = None,
    ) -> ProviderSession | None:
        raise NotImplementedError

    @abstractmethod
    def parse_new_events(self, transcript_path: Path, offset: int, *, session_id: str | None = None) -> ParseBatch:
        raise NotImplementedError

    @abstractmethod
    def history(
        self,
        transcript_path: Path,
        *,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[ProviderTranscriptEvent]:
        raise NotImplementedError


def read_jsonl_tail(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    with path.open("rb") as handle:
        file_size = handle.seek(0, 2)
        if offset > file_size:
            offset = 0
        handle.seek(offset)
        chunk = handle.read()

    if not chunk:
        return [], offset

    lines = chunk.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines = lines[:-1]

    new_offset = offset + sum(len(line) for line in lines)
    records: list[dict[str, Any]] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)

    return records, new_offset


def compute_message_hash(events: Sequence[ProviderTranscriptEvent]) -> str | None:
    if not events:
        return None

    last_event = events[-1]
    digest = sha256()
    digest.update(last_event.role.encode("utf-8"))
    digest.update(b"\0")
    digest.update(last_event.content_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(last_event.text.encode("utf-8"))
    return digest.hexdigest()


def shorten_text(text: str, *, limit: int = 90) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
