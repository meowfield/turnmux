from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from typing import Sequence

from .db import connect
from .models import Binding, BindingStatus, MonitorOffset, OnboardingState, OnboardingStep, PendingApproval, PendingLaunch, ProviderName


class StateRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve(strict=False)

    def get_binding(self, chat_id: int, thread_id: int) -> Binding | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM bindings WHERE chat_id = ? AND thread_id = ?;",
                (chat_id, thread_id),
            ).fetchone()
        return _row_to_binding(row) if row else None

    def get_binding_by_id(self, binding_id: int) -> Binding | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute("SELECT * FROM bindings WHERE id = ?;", (binding_id,)).fetchone()
        return _row_to_binding(row) if row else None

    def list_bindings(self, statuses: Sequence[BindingStatus] | None = None) -> list[Binding]:
        query = "SELECT * FROM bindings"
        params: list[str] = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(status.value for status in statuses)
        query += " ORDER BY updated_at DESC;"

        with closing(connect(self.db_path)) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [_row_to_binding(row) for row in rows]

    def save_binding(
        self,
        *,
        chat_id: int,
        thread_id: int,
        provider: ProviderName,
        repo_path: Path,
        tmux_session_name: str,
        tmux_window_id: str | None,
        tmux_window_name: str | None,
        status: BindingStatus,
        provider_session_id: str | None = None,
        transcript_path: Path | None = None,
    ) -> Binding:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO bindings (
                    chat_id, thread_id, provider, repo_path, tmux_session_name,
                    tmux_window_id, tmux_window_name, provider_session_id, transcript_path, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    provider = excluded.provider,
                    repo_path = excluded.repo_path,
                    tmux_session_name = excluded.tmux_session_name,
                    tmux_window_id = excluded.tmux_window_id,
                    tmux_window_name = excluded.tmux_window_name,
                    provider_session_id = excluded.provider_session_id,
                    transcript_path = excluded.transcript_path,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    chat_id,
                    thread_id,
                    provider.value,
                    str(repo_path),
                    tmux_session_name,
                    tmux_window_id,
                    tmux_window_name,
                    provider_session_id,
                    str(transcript_path) if transcript_path else None,
                    status.value,
                ),
            )
            connection.commit()

        binding = self.get_binding(chat_id, thread_id)
        if binding is None:
            raise RuntimeError("Failed to persist binding.")
        return binding

    def update_binding_session(
        self,
        binding_id: int,
        *,
        provider_session_id: str | None,
        transcript_path: Path | None,
        status: BindingStatus,
    ) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                UPDATE bindings
                SET provider_session_id = ?, transcript_path = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (
                    provider_session_id,
                    str(transcript_path) if transcript_path else None,
                    status.value,
                    binding_id,
                ),
            )
            connection.commit()

    def update_binding_status(self, binding_id: int, status: BindingStatus) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE bindings SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;",
                (status.value, binding_id),
            )
            connection.commit()

    def delete_binding(self, binding_id: int) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute("DELETE FROM bindings WHERE id = ?;", (binding_id,))
            connection.commit()

    def upsert_monitor_offset(
        self,
        binding_id: int,
        *,
        byte_offset: int,
        last_event_ts: str | None = None,
        last_message_hash: str | None = None,
    ) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO monitor_offsets (binding_id, byte_offset, last_event_ts, last_message_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(binding_id) DO UPDATE SET
                    byte_offset = excluded.byte_offset,
                    last_event_ts = excluded.last_event_ts,
                    last_message_hash = excluded.last_message_hash,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (binding_id, byte_offset, last_event_ts, last_message_hash),
            )
            connection.commit()

    def get_monitor_offset(self, binding_id: int) -> MonitorOffset | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM monitor_offsets WHERE binding_id = ?;",
                (binding_id,),
            ).fetchone()
        return _row_to_monitor_offset(row) if row else None

    def save_pending_launch(
        self,
        *,
        binding_id: int,
        provider: ProviderName,
        repo_path: Path,
        started_at: str | None = None,
        discovery_deadline_at: str,
        requested_session_id: str | None = None,
    ) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO pending_launches (
                    binding_id, provider, repo_path, started_at, discovery_deadline_at, requested_session_id
                )
                VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?)
                ON CONFLICT(binding_id) DO UPDATE SET
                    provider = excluded.provider,
                    repo_path = excluded.repo_path,
                    discovery_deadline_at = excluded.discovery_deadline_at,
                    requested_session_id = excluded.requested_session_id,
                    started_at = COALESCE(excluded.started_at, CURRENT_TIMESTAMP);
                """,
                (
                    binding_id,
                    provider.value,
                    str(repo_path),
                    started_at,
                    discovery_deadline_at,
                    requested_session_id,
                ),
            )
            connection.commit()

    def list_pending_launches(self) -> list[PendingLaunch]:
        with closing(connect(self.db_path)) as connection:
            rows = connection.execute(
                "SELECT * FROM pending_launches ORDER BY started_at ASC;"
            ).fetchall()
        return [_row_to_pending_launch(row) for row in rows]

    def get_pending_launch(self, binding_id: int) -> PendingLaunch | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM pending_launches WHERE binding_id = ?;",
                (binding_id,),
            ).fetchone()
        return _row_to_pending_launch(row) if row else None

    def delete_pending_launch(self, binding_id: int) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute("DELETE FROM pending_launches WHERE binding_id = ?;", (binding_id,))
            connection.commit()

    def save_pending_approval(
        self,
        *,
        binding_id: int,
        provider: ProviderName,
        fingerprint: str,
        prompt_text: str,
        approve_keys: Sequence[str],
        deny_keys: Sequence[str] | None = None,
    ) -> PendingApproval:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO pending_approvals (
                    binding_id, provider, fingerprint, prompt_text, approve_keys_json, deny_keys_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(binding_id) DO UPDATE SET
                    provider = excluded.provider,
                    fingerprint = excluded.fingerprint,
                    prompt_text = excluded.prompt_text,
                    approve_keys_json = excluded.approve_keys_json,
                    deny_keys_json = excluded.deny_keys_json,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    binding_id,
                    provider.value,
                    fingerprint,
                    prompt_text,
                    json.dumps(list(approve_keys), ensure_ascii=False),
                    json.dumps(list(deny_keys), ensure_ascii=False) if deny_keys is not None else None,
                ),
            )
            connection.commit()

        pending = self.get_pending_approval(binding_id)
        if pending is None:
            raise RuntimeError("Failed to persist pending approval.")
        return pending

    def get_pending_approval(self, binding_id: int) -> PendingApproval | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM pending_approvals WHERE binding_id = ?;",
                (binding_id,),
            ).fetchone()
        return _row_to_pending_approval(row) if row else None

    def delete_pending_approval(self, binding_id: int) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute("DELETE FROM pending_approvals WHERE binding_id = ?;", (binding_id,))
            connection.commit()

    def save_onboarding_state(
        self,
        *,
        chat_id: int,
        thread_id: int,
        step: OnboardingStep,
        provider: ProviderName | None = None,
        repo_path: Path | None = None,
        mode: str | None = None,
        pending_user_text: str | None = None,
        resume_candidates_json: str | None = None,
    ) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO onboarding_states (
                    chat_id, thread_id, step, provider, repo_path, mode, pending_user_text, resume_candidates_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    step = excluded.step,
                    provider = excluded.provider,
                    repo_path = excluded.repo_path,
                    mode = excluded.mode,
                    pending_user_text = excluded.pending_user_text,
                    resume_candidates_json = excluded.resume_candidates_json,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    chat_id,
                    thread_id,
                    step.value,
                    provider.value if provider else None,
                    str(repo_path) if repo_path else None,
                    mode,
                    pending_user_text,
                    resume_candidates_json,
                ),
            )
            connection.commit()

    def get_onboarding_state(self, chat_id: int, thread_id: int) -> OnboardingState | None:
        with closing(connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT * FROM onboarding_states WHERE chat_id = ? AND thread_id = ?;",
                (chat_id, thread_id),
            ).fetchone()
        return _row_to_onboarding_state(row) if row else None

    def clear_onboarding_state(self, chat_id: int, thread_id: int) -> None:
        with closing(connect(self.db_path)) as connection:
            connection.execute(
                "DELETE FROM onboarding_states WHERE chat_id = ? AND thread_id = ?;",
                (chat_id, thread_id),
            )
            connection.commit()


def _row_to_binding(row) -> Binding:
    return Binding(
        id=row["id"],
        chat_id=row["chat_id"],
        thread_id=row["thread_id"],
        provider=ProviderName(row["provider"]),
        repo_path=Path(row["repo_path"]),
        tmux_session_name=row["tmux_session_name"],
        tmux_window_id=row["tmux_window_id"],
        tmux_window_name=row["tmux_window_name"],
        provider_session_id=row["provider_session_id"],
        transcript_path=Path(row["transcript_path"]) if row["transcript_path"] else None,
        status=BindingStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_monitor_offset(row) -> MonitorOffset:
    return MonitorOffset(
        binding_id=row["binding_id"],
        byte_offset=row["byte_offset"],
        last_event_ts=row["last_event_ts"],
        last_message_hash=row["last_message_hash"],
        updated_at=row["updated_at"],
    )


def _row_to_pending_launch(row) -> PendingLaunch:
    return PendingLaunch(
        id=row["id"],
        binding_id=row["binding_id"],
        provider=ProviderName(row["provider"]),
        repo_path=Path(row["repo_path"]),
        started_at=row["started_at"],
        discovery_deadline_at=row["discovery_deadline_at"],
        requested_session_id=row["requested_session_id"],
    )


def _row_to_pending_approval(row) -> PendingApproval:
    return PendingApproval(
        id=row["id"],
        binding_id=row["binding_id"],
        provider=ProviderName(row["provider"]),
        fingerprint=row["fingerprint"],
        prompt_text=row["prompt_text"],
        approve_keys=tuple(_decode_keys(row["approve_keys_json"])),
        deny_keys=tuple(_decode_keys(row["deny_keys_json"])) if row["deny_keys_json"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_onboarding_state(row) -> OnboardingState:
    return OnboardingState(
        chat_id=row["chat_id"],
        thread_id=row["thread_id"],
        step=OnboardingStep(row["step"]),
        provider=ProviderName(row["provider"]) if row["provider"] else None,
        repo_path=Path(row["repo_path"]) if row["repo_path"] else None,
        mode=row["mode"],
        pending_user_text=row["pending_user_text"],
        resume_candidates_json=row["resume_candidates_json"],
        updated_at=row["updated_at"],
    )


def _decode_keys(value: str) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, str) and item]
