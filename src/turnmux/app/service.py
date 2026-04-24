from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable

from ..attachments import AttachmentStore, inline_excerpt
from ..config import TurnmuxConfig, validate_repo_path
from ..input_types import AttachmentRef, AttachmentMediaClass, UserTurn
from ..providers import ProviderRegistry
from ..providers.base import ProviderSession, ProviderTranscriptEvent, parse_timestamp
from ..providers.trust import ensure_provider_trust
from ..runtime.approvals import detect_approval_request, detect_non_approval_prompt_response
from ..runtime import tmux
from ..state.models import Binding, BindingStatus, ProviderName
from ..state.repository import StateRepository


DISCOVERY_TIMEOUT_SECONDS = 90


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    chat_id: int
    thread_id: int
    text: str
    markup_kind: str | None = None
    markup_has_deny: bool = False
    binding_id: int | None = None
    next_byte_offset: int | None = None
    last_event_ts: str | None = None
    last_message_hash: str | None = None
    finalize_monitor_offset: bool = False


class AppService:
    def __init__(
        self,
        *,
        config: TurnmuxConfig,
        repository: StateRepository,
        providers: ProviderRegistry,
        runtime_home: Path | None = None,
        attachment_store: AttachmentStore | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.providers = providers
        self.runtime_home = runtime_home.expanduser().resolve(strict=False) if runtime_home is not None else None
        self.attachment_store = attachment_store
        self._auto_answered_prompt_fingerprints: set[tuple[int, str]] = set()

    def validate_repo(self, repo_path_text: str) -> Path:
        return validate_repo_path(Path(repo_path_text), self.config.allowed_roots)

    def suggest_repos(self, *, limit: int = 6) -> list[Path]:
        ranked: dict[Path, float] = {}

        def add_candidate(repo_path: Path, *, bonus: float = 0.0) -> None:
            normalized = repo_path.expanduser().resolve(strict=False)
            if not normalized.exists() or not normalized.is_dir():
                return
            if not (normalized / ".git").exists():
                return
            try:
                validated = validate_repo_path(normalized, self.config.allowed_roots)
            except Exception:
                return
            score = _repo_mtime(validated) + bonus
            previous = ranked.get(validated)
            if previous is None or score > previous:
                ranked[validated] = score

        add_candidate(Path.cwd(), bonus=10_000_000.0)
        for binding in self.repository.list_bindings():
            add_candidate(binding.repo_path, bonus=5_000_000.0)
        for root in self.config.allowed_roots:
            for repo in _discover_git_repos(root, max_depth=2):
                add_candidate(repo)

        return [path for path, _ in sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:limit]]

    async def launch_binding(
        self,
        *,
        chat_id: int,
        thread_id: int,
        provider: ProviderName,
        repo_path: Path,
        mode: str,
        requested_session_id: str | None = None,
    ) -> Binding:
        existing = self.repository.get_binding(chat_id, thread_id)
        if existing and existing.status == BindingStatus.ACTIVE:
            raise RuntimeError("This topic is already bound to an active session. Use /kill first.")

        adapter = self.providers.get(provider)
        window_env = adapter.runtime_env()
        ensure_provider_trust(provider, repo_path, runtime_home=self.runtime_home)
        tmux.ensure_session(self.config.tmux_session_name)
        window = tmux.create_window(
            self.config.tmux_session_name,
            repo_path,
            window_name=_window_name(provider, repo_path),
            env=window_env,
        )

        binding = self.repository.save_binding(
            chat_id=chat_id,
            thread_id=thread_id,
            provider=provider,
            repo_path=repo_path,
            tmux_session_name=self.config.tmux_session_name,
            tmux_window_id=window.window_id,
            tmux_window_name=window.name,
            status=BindingStatus.PENDING_START,
            provider_session_id=requested_session_id if mode == "resume" else None,
            transcript_path=None,
        )
        self.repository.upsert_monitor_offset(binding.id, byte_offset=0)

        if mode == "fresh":
            rebound = self.repository.get_binding(chat_id, thread_id)
            if rebound is None:
                raise RuntimeError("Binding disappeared after launch.")
            return rebound

        started_after = utc_now_iso()
        command = (
            adapter.build_resume_command(repo_path, requested_session_id)
            if mode == "resume" and requested_session_id
            else adapter.build_start_command(repo_path)
        )
        tmux.launch_command(window.window_id, command)

        discovered = await self._wait_for_discovery(
            provider=provider,
            repo_path=repo_path,
            started_after=started_after,
            requested_session_id=requested_session_id,
            tmux_session_name=binding.tmux_session_name,
            tmux_window_id=binding.tmux_window_id,
        )

        if discovered:
            self._activate_binding(binding.id, discovered)
        else:
            self.repository.save_pending_launch(
                binding_id=binding.id,
                provider=provider,
                repo_path=repo_path,
                started_at=started_after,
                discovery_deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=DISCOVERY_TIMEOUT_SECONDS)).isoformat(),
                requested_session_id=requested_session_id,
            )

        rebound = self.repository.get_binding(chat_id, thread_id)
        if rebound is None:
            raise RuntimeError("Binding disappeared after launch.")
        return rebound

    async def _wait_for_discovery(
        self,
        *,
        provider: ProviderName,
        repo_path: Path,
        started_after: str,
        requested_session_id: str | None,
        tmux_session_name: str | None,
        tmux_window_id: str | None,
    ) -> ProviderSession | None:
        adapter = self.providers.get(provider)
        for _ in range(10):
            discovered = adapter.discover_session(
                repo_path,
                started_after=started_after,
                requested_session_id=requested_session_id,
                tmux_session_name=tmux_session_name,
                tmux_window_id=tmux_window_id,
            )
            if discovered:
                return discovered
            await asyncio.sleep(0.5)
        return None

    def _activate_binding(self, binding_id: int, session: ProviderSession, *, initial_offset: int | None = None) -> None:
        self.repository.update_binding_session(
            binding_id,
            provider_session_id=session.session_id,
            transcript_path=session.transcript_path,
            status=BindingStatus.ACTIVE,
        )
        binding = self.repository.get_binding_by_id(binding_id)
        if binding is None:
            raise RuntimeError("Binding disappeared during activation.")
        adapter = self.providers.get(binding.provider)
        byte_offset = adapter.initial_monitor_offset(session) if initial_offset is None else initial_offset
        self.repository.upsert_monitor_offset(binding_id, byte_offset=byte_offset)
        self.repository.delete_pending_launch(binding_id)

    def send_user_text(self, binding: Binding, text: str) -> None:
        self.send_user_turn(binding, UserTurn(text=text))

    def send_user_turn(self, binding: Binding, turn: UserTurn) -> None:
        if not turn.has_content():
            return
        if binding.status not in {BindingStatus.ACTIVE, BindingStatus.PENDING_START} or not binding.tmux_window_id:
            raise RuntimeError("This topic does not have an active runtime session yet.")
        if self.repository.get_pending_approval(binding.id) is not None:
            raise RuntimeError("This topic is waiting for an approval decision. Use the Telegram buttons first.")
        rendered_text = self._render_turn_for_binding(binding, turn)
        if binding.status == BindingStatus.PENDING_START and not binding.provider_session_id and not binding.transcript_path:
            if self.repository.get_pending_launch(binding.id) is not None:
                raise RuntimeError("The provider is still starting. Wait for the activation message before sending more text.")
            self._launch_pending_fresh_binding(binding, rendered_text)
            return
        tmux.paste_text(binding.tmux_window_id, rendered_text, enter=True)

    def interrupt_binding(self, binding: Binding) -> None:
        if not binding.tmux_window_id:
            raise RuntimeError("Binding has no tmux window.")
        tmux.send_interrupt(binding.tmux_window_id)

    def kill_binding(self, binding: Binding) -> None:
        if binding.tmux_window_id:
            tmux.kill_window(binding.tmux_session_name, binding.tmux_window_id)
        if self.attachment_store is not None:
            self.attachment_store.clear_topic(binding.chat_id, binding.thread_id, repo_path=binding.repo_path)
        self.repository.delete_binding(binding.id)

    def _launch_pending_fresh_binding(self, binding: Binding, text: str) -> None:
        if not binding.tmux_window_id:
            raise RuntimeError("Binding has no tmux window.")

        adapter = self.providers.get(binding.provider)
        ensure_provider_trust(binding.provider, binding.repo_path, runtime_home=self.runtime_home)
        self.repository.upsert_monitor_offset(binding.id, byte_offset=0)
        started_after = utc_now_iso()

        tmux.launch_command(
            binding.tmux_window_id,
            adapter.build_start_command(binding.repo_path, initial_prompt=text),
        )

        self.repository.save_pending_launch(
            binding_id=binding.id,
            provider=binding.provider,
            repo_path=binding.repo_path,
            started_at=started_after,
            discovery_deadline_at=(datetime.now(timezone.utc) + timedelta(seconds=DISCOVERY_TIMEOUT_SECONDS)).isoformat(),
        )

    def _render_turn_for_binding(self, binding: Binding, turn: UserTurn) -> str:
        normalized_text = turn.normalized_text()
        projected_attachments = tuple(
            self._project_attachment_for_binding(binding, attachment)
            for attachment in turn.attachments
        )
        if not projected_attachments:
            return normalized_text or ""

        sections: list[str] = []
        if normalized_text:
            sections.append(normalized_text)

        attachment_lines = [f"TurnMux attached {len(projected_attachments)} file(s) for this message:"]
        for index, attachment in enumerate(projected_attachments, start=1):
            attachment_lines.extend(_render_attachment_block(binding.repo_path, index=index, attachment=attachment))
        sections.append("\n".join(attachment_lines))
        return "\n\n".join(section for section in sections if section.strip())

    def _project_attachment_for_binding(self, binding: Binding, attachment: AttachmentRef) -> AttachmentRef:
        if self.attachment_store is None:
            return attachment
        return self.attachment_store.project_attachment(
            binding.repo_path,
            chat_id=binding.chat_id,
            thread_id=binding.thread_id,
            attachment=attachment,
        )

    def status_text(self, binding: Binding) -> str:
        pending_approval = self.repository.get_pending_approval(binding.id)
        lines = [
            f"provider: {binding.provider.value}",
            f"repo: {binding.repo_path}",
            f"status: {binding.status.value}",
            f"tmux window: {binding.tmux_window_name or '-'} ({binding.tmux_window_id or '-'})",
            f"provider session: {binding.provider_session_id or '-'}",
            f"transcript: {binding.transcript_path or '-'}",
            f"pending approval: {'yes' if pending_approval is not None else 'no'}",
        ]
        return "\n".join(lines)

    def list_resumable_sessions(self, provider: ProviderName, repo_path: Path, *, limit: int = 5) -> list[ProviderSession]:
        return self.providers.get(provider).list_resumable_sessions(repo_path, limit=limit)

    def history_text(self, binding: Binding, *, limit: int = 10) -> str:
        if not binding.transcript_path:
            return "No transcript has been discovered for this binding yet."
        events = self.providers.get(binding.provider).history(
            binding.transcript_path,
            session_id=binding.provider_session_id,
            limit=limit,
        )
        if not events:
            return "No history available yet."
        return "\n\n".join(_format_history_event(event) for event in events)

    def resolve_pending_approval(self, binding: Binding, *, approve: bool) -> str:
        pending = self.repository.get_pending_approval(binding.id)
        if pending is None:
            raise RuntimeError("There is no pending approval in this topic.")
        if not binding.tmux_window_id:
            raise RuntimeError("Binding has no tmux window.")

        keys = pending.approve_keys if approve else pending.deny_keys
        if not keys:
            raise RuntimeError("This approval prompt does not expose that action.")

        tmux.send_keys(binding.tmux_window_id, *keys)
        self.repository.delete_pending_approval(binding.id)
        return "Approval sent." if approve else "Denial sent."

    def mark_outbound_delivered(self, message: OutboundMessage) -> None:
        if (
            message.binding_id is None
            or not message.finalize_monitor_offset
            or message.next_byte_offset is None
        ):
            return
        self.repository.upsert_monitor_offset(
            message.binding_id,
            byte_offset=message.next_byte_offset,
            last_event_ts=message.last_event_ts,
            last_message_hash=message.last_message_hash,
        )

    def refresh_pending_and_active_bindings(self) -> list[OutboundMessage]:
        outbound: list[OutboundMessage] = []
        now = datetime.now(timezone.utc)

        # Pending launches are promoted first so a newly discovered session can
        # become ACTIVE before approval detection or transcript polling runs.
        for pending in self.repository.list_pending_launches():
            binding = self.repository.get_binding_by_id(pending.binding_id)
            if binding is None:
                self.repository.delete_pending_launch(pending.binding_id)
                continue
            if binding.status == BindingStatus.ACTIVE:
                self.repository.delete_pending_launch(pending.binding_id)
                continue
            session = self.providers.get(pending.provider).discover_session(
                pending.repo_path,
                started_after=pending.started_at,
                requested_session_id=pending.requested_session_id,
                tmux_session_name=binding.tmux_session_name,
                tmux_window_id=binding.tmux_window_id,
            )
            if session:
                # Fresh launches can emit output before discovery notices the
                # transcript file. Start monitoring from byte 0 so the first
                # progress/final messages are not dropped on activation.
                initial_offset = None
                if (
                    pending.requested_session_id is None
                    and binding.provider_session_id is None
                    and binding.transcript_path is None
                ):
                    initial_offset = 0
                self._activate_binding(binding.id, session, initial_offset=initial_offset)
                continue

            deadline = parse_timestamp(pending.discovery_deadline_at)
            if deadline and deadline <= now:
                self.repository.delete_pending_launch(pending.binding_id)
                self.repository.delete_pending_approval(pending.binding_id)
                self.repository.update_binding_status(binding.id, BindingStatus.MISSING)
                outbound.append(
                    OutboundMessage(
                        chat_id=binding.chat_id,
                        thread_id=binding.thread_id,
                        text="Session did not expose a transcript before the discovery timeout. Binding marked as missing.",
                    )
                )

        # Approval prompts live in tmux scrollback rather than provider
        # transcripts, so we poll the pane for both ACTIVE and PENDING_START.
        for binding in self.repository.list_bindings(statuses=[BindingStatus.ACTIVE, BindingStatus.PENDING_START]):
            if not binding.tmux_window_id:
                continue
            if not tmux.window_exists(binding.tmux_session_name, binding.tmux_window_id):
                self.repository.delete_pending_approval(binding.id)
                if binding.status != BindingStatus.ACTIVE:
                    continue
                self.repository.update_binding_status(binding.id, BindingStatus.MISSING)
                outbound.append(
                    OutboundMessage(
                        chat_id=binding.chat_id,
                        thread_id=binding.thread_id,
                        text="tmux window is missing. Binding marked as missing.",
                    )
                )
                continue

            approval_outbound = self._sync_pending_approval(binding)
            if approval_outbound is not None:
                outbound.append(approval_outbound)

        # Transcript parsing is last because it depends on the binding already
        # being ACTIVE and having a discovered transcript path.
        for binding in self.repository.list_bindings(statuses=[BindingStatus.ACTIVE]):
            if not binding.tmux_window_id or not tmux.window_exists(binding.tmux_session_name, binding.tmux_window_id):
                continue

            if not binding.transcript_path or not binding.transcript_path.exists():
                self.repository.delete_pending_approval(binding.id)
                self.repository.update_binding_status(binding.id, BindingStatus.MISSING)
                outbound.append(
                    OutboundMessage(
                        chat_id=binding.chat_id,
                        thread_id=binding.thread_id,
                        text="Transcript file is missing. Binding marked as missing.",
                    )
                )
                continue

            adapter = self.providers.get(binding.provider)
            current_offset = self.repository.get_monitor_offset(binding.id)
            offset = current_offset.byte_offset if current_offset else 0
            batch = adapter.parse_new_events(
                binding.transcript_path,
                offset,
                session_id=binding.provider_session_id,
            )
            if batch.events:
                for index, event in enumerate(batch.events):
                    outbound.append(
                        OutboundMessage(
                            chat_id=binding.chat_id,
                            thread_id=binding.thread_id,
                            text=_format_transcript_event_message(event),
                            binding_id=binding.id,
                            next_byte_offset=batch.new_offset,
                            last_event_ts=batch.last_event_ts,
                            last_message_hash=batch.last_message_hash,
                            finalize_monitor_offset=index == len(batch.events) - 1,
                        )
                    )
            elif batch.new_offset != offset:
                self.repository.upsert_monitor_offset(
                    binding.id,
                    byte_offset=batch.new_offset,
                    last_event_ts=batch.last_event_ts,
                    last_message_hash=batch.last_message_hash,
                )

        return outbound

    def _sync_pending_approval(self, binding: Binding) -> OutboundMessage | None:
        assert binding.tmux_window_id is not None
        pane = tmux.capture_pane(binding.tmux_window_id, history_lines=120)
        current = self.repository.get_pending_approval(binding.id)
        prompt_response = detect_non_approval_prompt_response(binding.provider, pane)
        if prompt_response is not None:
            prompt_key = (binding.id, prompt_response.fingerprint)
            if prompt_key not in self._auto_answered_prompt_fingerprints:
                tmux.send_keys(binding.tmux_window_id, *prompt_response.keys)
                self._auto_answered_prompt_fingerprints.add(prompt_key)
            if current is not None:
                self.repository.delete_pending_approval(binding.id)
            return None

        request = detect_approval_request(binding.provider, pane)
        if request is None:
            if current is not None:
                self.repository.delete_pending_approval(binding.id)
            return None

        saved = self.repository.save_pending_approval(
            binding_id=binding.id,
            provider=binding.provider,
            fingerprint=request.fingerprint,
            prompt_text=request.prompt_text,
            approve_keys=request.approve_keys,
            deny_keys=request.deny_keys,
        )
        if current is not None and current.fingerprint == saved.fingerprint:
            return None

        return OutboundMessage(
            chat_id=binding.chat_id,
            thread_id=binding.thread_id,
            text=_format_pending_approval_message(binding.provider, saved.prompt_text),
            markup_kind="approval",
            markup_has_deny=saved.deny_keys is not None,
        )


def encode_resume_candidates(candidates: Iterable[ProviderSession]) -> str:
    return json.dumps(
        [
            {
                "session_id": candidate.session_id,
                "display_name": candidate.display_name,
                "updated_at": candidate.updated_at,
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
    )


def encode_repo_candidates(paths: Iterable[Path]) -> str:
    return json.dumps([str(path) for path in paths], ensure_ascii=False)


def _format_history_event(event: ProviderTranscriptEvent) -> str:
    if not event.is_final:
        return f"[progress] {event.text}"
    return f"[{event.role}] {event.text}"


def _format_transcript_event_message(event: ProviderTranscriptEvent) -> str:
    return event.text


def decode_repo_candidates(value: str | None) -> list[Path]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    candidates: list[Path] = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            candidates.append(Path(item))
    return candidates


def decode_resume_candidates(value: str | None) -> list[dict[str, str]]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def format_resume_candidates(candidates: list[ProviderSession]) -> str:
    lines = ["Reply with the number of the session to resume:"]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(f"{index}. {candidate.display_name} [{candidate.updated_at}] ({candidate.session_id})")
    return "\n".join(lines)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _window_name(provider: ProviderName, repo_path: Path) -> str:
    return f"{provider.value}:{repo_path.name}"


def _format_pending_approval_message(provider: ProviderName, prompt_text: str) -> str:
    excerpt = prompt_text.strip()
    if len(excerpt) > 1200:
        excerpt = excerpt[:1197] + "..."
    return f"Approval required by {provider.value}.\n\n{excerpt}\n\nUse the buttons below."


def _discover_git_repos(root: Path, *, max_depth: int) -> list[Path]:
    discovered: list[Path] = []
    stack: list[tuple[Path, int]] = [(root.expanduser().resolve(strict=False), 0)]
    skip_names = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}

    while stack:
        current, depth = stack.pop()
        if not current.exists() or not current.is_dir():
            continue
        if (current / ".git").exists():
            discovered.append(current)
            continue
        if depth >= max_depth:
            continue

        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir() and child.name not in skip_names),
                key=lambda child: child.name,
                reverse=True,
            )
        except OSError:
            continue

        for child in children:
            stack.append((child, depth + 1))

    return discovered


def _render_attachment_block(repo_path: Path, *, index: int, attachment: AttachmentRef) -> list[str]:
    lines = [f"[Attachment {index}]"]
    lines.append(f"type: {attachment.media_class.value}")
    if attachment.original_name:
        lines.append(f"original_name: {attachment.original_name}")
    if attachment.mime_type:
        lines.append(f"mime: {attachment.mime_type}")
    if attachment.file_size is not None:
        lines.append(f"bytes: {attachment.file_size}")
    lines.append(f"path: {_display_attachment_path(repo_path, attachment.local_path)}")

    derived_excerpt = inline_excerpt(attachment.derived_text_path)
    if attachment.derived_text_path is not None:
        lines.append(f"derived_text_path: {_display_attachment_path(repo_path, attachment.derived_text_path)}")
        if derived_excerpt:
            lines.append("derived_text_excerpt:")
            lines.append(derived_excerpt)
    elif attachment.media_class == AttachmentMediaClass.IMAGE:
        lines.append("note: inspect the image at the path above.")

    metadata = attachment.metadata()
    width = metadata.get("width")
    height = metadata.get("height")
    if isinstance(width, int) and isinstance(height, int):
        lines.append(f"dimensions: {width}x{height}")
    return lines


def _display_attachment_path(repo_path: Path, path: Path) -> str:
    normalized_path = path.expanduser().resolve(strict=False)
    try:
        return str(normalized_path.relative_to(repo_path))
    except ValueError:
        return str(normalized_path)


def _repo_mtime(repo_path: Path) -> float:
    git_dir = repo_path / ".git"
    target = git_dir if git_dir.exists() else repo_path
    try:
        return target.stat().st_mtime
    except OSError:
        return 0.0
