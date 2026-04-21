from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ProviderName(StrEnum):
    CLAUDE = "claude"
    CODEX = "codex"
    OPENCODE = "opencode"


class BindingStatus(StrEnum):
    PENDING_START = "pending_start"
    ACTIVE = "active"
    STOPPED = "stopped"
    MISSING = "missing"


class OnboardingStep(StrEnum):
    CHOOSE_PROVIDER = "choose_provider"
    CHOOSE_REPO = "choose_repo"
    CHOOSE_MODE = "choose_mode"
    CHOOSE_RESUME = "choose_resume"


@dataclass(frozen=True, slots=True)
class Binding:
    id: int
    chat_id: int
    thread_id: int
    provider: ProviderName
    repo_path: Path
    tmux_session_name: str
    tmux_window_id: str | None
    tmux_window_name: str | None
    provider_session_id: str | None
    transcript_path: Path | None
    status: BindingStatus
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class MonitorOffset:
    binding_id: int
    byte_offset: int
    last_event_ts: str | None
    last_message_hash: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class PendingLaunch:
    id: int
    binding_id: int
    provider: ProviderName
    repo_path: Path
    started_at: str
    discovery_deadline_at: str
    requested_session_id: str | None


@dataclass(frozen=True, slots=True)
class PendingApproval:
    id: int
    binding_id: int
    provider: ProviderName
    fingerprint: str
    prompt_text: str
    approve_keys: tuple[str, ...]
    deny_keys: tuple[str, ...] | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class OnboardingState:
    chat_id: int
    thread_id: int
    step: OnboardingStep
    provider: ProviderName | None
    repo_path: Path | None
    mode: str | None
    pending_user_text: str | None
    resume_candidates_json: str | None
    updated_at: str
