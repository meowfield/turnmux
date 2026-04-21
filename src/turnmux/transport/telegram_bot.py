from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from ..audio_transcription import (
    AudioTranscriptionError,
    AudioTranscriptionNotConfiguredError,
    close_transcription_client,
    transcribe_audio,
)
from ..app.service import (
    AppService,
    decode_resume_candidates,
    encode_resume_candidates,
)
from ..providers import ProviderRegistry
from ..providers.base import ProviderSession
from ..runtime.home import RuntimePaths
from ..runtime.lifecycle import HeartbeatWriter, install_asyncio_exception_logging, install_unix_signal_handlers
from ..state.models import BindingStatus, OnboardingStep, ProviderName
from ..state.repository import StateRepository


logger = logging.getLogger(__name__)
CONTROL_COMMANDS = {
    "/start",
    "/new",
    "/resume",
    "/cancel",
    "/status",
    "/history",
    "/interrupt",
    "/kill",
}
ONBOARDING_CALLBACK_PREFIX = "ob"
APPROVAL_CALLBACK_PREFIX = "ap"
REPO_BROWSER_PAGE_SIZE = 10


@dataclass(frozen=True, slots=True)
class RepoBrowserEntry:
    path: Path
    kind: str
    label: str


@dataclass(frozen=True, slots=True)
class RepoBrowserState:
    recent_repos: tuple[Path, ...]
    browse_dir: Path | None = None
    browse_entries: tuple[RepoBrowserEntry, ...] = ()
    browse_page: int = 0


class TurnmuxTelegramBot:
    def __init__(
        self,
        *,
        config,
        repository: StateRepository,
        providers: ProviderRegistry,
        runtime_paths: RuntimePaths | None = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self.available_providers = providers.available_providers()
        self.repository = repository
        self.service = AppService(config=config, repository=repository, providers=providers)
        self.runtime_paths = runtime_paths
        self._monitor_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_writer: HeartbeatWriter | None = None
        self._stop_event: asyncio.Event | None = None
        self._runtime_status = "starting"
        self._runtime_note: str | None = None
        self._last_monitor_error_signature: str | None = None
        self._last_monitor_error_at = 0.0

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        install_asyncio_exception_logging(loop)
        self._stop_event = asyncio.Event()
        install_unix_signal_handlers(loop, on_shutdown=self._request_shutdown)
        if self.runtime_paths is not None:
            self._heartbeat_writer = HeartbeatWriter(self.runtime_paths.heartbeat_path)
            self._heartbeat_writer.write(status="starting")

        application = Application.builder().token(self.config.telegram_bot_token).build()
        self._register_handlers(application)
        await application.initialize()
        await self._configure_bot_commands(application)
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        self._monitor_task = asyncio.create_task(self._monitor_loop(application), name="turnmux-monitor")
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="turnmux-heartbeat")
        self._set_runtime_health("running")
        logger.info(
            "TurnMux bot started pid=%s ppid=%s runtime_home=%s providers=%s",
            os.getpid(),
            os.getppid(),
            self.runtime_paths.home if self.runtime_paths else "-",
            ", ".join(provider.value for provider in self.available_providers),
        )
        try:
            await self._stop_event.wait()
        finally:
            logger.info("TurnMux bot stopping")
            self._set_runtime_health("stopping")
            if application.updater:
                await application.updater.stop()
            await application.stop()
            await application.shutdown()
            if self._monitor_task:
                self._monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._monitor_task
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._heartbeat_task
            if self._heartbeat_writer:
                self._heartbeat_writer.write(status="stopped")
            await close_transcription_client()

    def _register_handlers(self, application: Application) -> None:
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("new", self._handle_new))
        application.add_handler(CommandHandler("resume", self._handle_resume))
        application.add_handler(CommandHandler("cancel", self._handle_cancel))
        application.add_handler(CommandHandler("status", self._handle_status))
        application.add_handler(CommandHandler("history", self._handle_history))
        application.add_handler(CommandHandler("interrupt", self._handle_interrupt))
        application.add_handler(CommandHandler("kill", self._handle_kill))
        application.add_handler(CallbackQueryHandler(self._handle_approval_callback, pattern=rf"^{APPROVAL_CALLBACK_PREFIX}:"))
        application.add_handler(CallbackQueryHandler(self._handle_onboarding_callback, pattern=rf"^{ONBOARDING_CALLBACK_PREFIX}:"))
        application.add_handler(MessageHandler(~filters.COMMAND, self._handle_message))
        application.add_handler(MessageHandler(filters.COMMAND, self._handle_forwarded_command), group=1)

    async def _configure_bot_commands(self, application: Application) -> None:
        try:
            await application.bot.set_my_commands(_bot_commands(self.available_providers))
        except TelegramError:
            logger.exception("Failed to configure Telegram bot commands")

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        binding = self._binding_for_update(update)
        topic_label = "This topic is ready for a new binding."
        if binding:
            topic_label = (
                f"This topic is bound to `{binding.provider.value}` at `{binding.repo_path}`.\n"
                f"Current status: `{binding.status.value}`."
            )
        mode_note = "Tap `New Binding` or `Resume Existing` to continue."
        if _is_private_chat(update):
            mode_note = (
                "This is a private chat, so it can hold one live binding at a time.\n"
                "For parallel sessions, use TurnMux inside a forum supergroup with topics."
            )
        elif _is_forum_lobby(update):
            mode_note = (
                "This is the forum lobby.\n"
                "Send any message here or tap `New Binding` and TurnMux will open a dedicated topic for the new session."
            )
        await self._reply(
            update,
            "\n".join(
                [
                    "TurnMux is ready.",
                    topic_label,
                    "",
                    mode_note,
                    "You can also use /new, /resume, /status, /history, /interrupt, /kill, /cancel.",
                ]
            ),
            reply_markup=_build_start_keyboard(),
        )

    async def _handle_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        if await self._maybe_redirect_new_session_to_topic(update, mode="fresh"):
            return
        if _is_private_chat(update) and self._binding_for_update(update):
            await self._reply(
                update,
                "This private chat already has a live binding.\n"
                "For multiple parallel sessions, use TurnMux in a forum supergroup with one topic per session.",
            )
            return
        chat_id, thread_id = topic_key(update)
        self.repository.save_onboarding_state(
            chat_id=chat_id,
            thread_id=thread_id,
            step=OnboardingStep.CHOOSE_PROVIDER,
            mode="fresh",
        )
        await self._reply(
            update,
            "Choose a provider for this topic.",
            reply_markup=_build_provider_keyboard(self.available_providers),
        )

    async def _handle_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        if await self._maybe_redirect_new_session_to_topic(update, mode="resume"):
            return
        if _is_private_chat(update) and self._binding_for_update(update):
            await self._reply(
                update,
                "This private chat already has a live binding.\n"
                "For multiple parallel sessions, use TurnMux in a forum supergroup with one topic per session.",
            )
            return
        chat_id, thread_id = topic_key(update)
        self.repository.save_onboarding_state(
            chat_id=chat_id,
            thread_id=thread_id,
            step=OnboardingStep.CHOOSE_PROVIDER,
            mode="resume",
        )
        await self._reply(
            update,
            "Choose which provider you want to resume in this topic.",
            reply_markup=_build_provider_keyboard(self.available_providers),
        )

    async def _handle_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        chat_id, thread_id = topic_key(update)
        self.repository.clear_onboarding_state(chat_id, thread_id)
        await self._reply(update, "Setup cancelled.")

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        binding = self._binding_for_update(update)
        if not binding:
            await self._reply(update, "This topic is not bound.")
            return
        try:
            await self._reply(update, self.service.status_text(binding))
        except Exception as exc:
            logger.exception("Failed to build status text")
            await self._reply(update, f"Failed to read status: {exc}")

    async def _handle_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        binding = self._binding_for_update(update)
        if not binding:
            await self._reply(update, "This topic is not bound.")
            return
        try:
            await self._reply(update, self.service.history_text(binding))
        except Exception as exc:
            logger.exception("Failed to read history")
            await self._reply(update, f"Failed to read history: {exc}")

    async def _handle_interrupt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        binding = self._binding_for_update(update)
        if not binding:
            await self._reply(update, "This topic is not bound.")
            return
        try:
            self.service.interrupt_binding(binding)
        except Exception as exc:
            logger.exception("Failed to send interrupt")
            await self._reply(update, f"Failed to send interrupt: {exc}")
            return
        await self._reply(update, "Interrupt sent.")

    async def _handle_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        binding = self._binding_for_update(update)
        if not binding:
            if await self._maybe_delete_topic(update):
                return
            await self._reply(update, "This topic is not bound.")
            return
        try:
            self.service.kill_binding(binding)
        except Exception as exc:
            logger.exception("Failed to kill binding")
            await self._reply(update, f"Failed to kill binding: {exc}")
            return
        if await self._maybe_delete_topic(update):
            return
        await self._reply(update, "Session killed and binding cleared.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return

        message = update.effective_message
        if message is None:
            return

        chat_id, thread_id = topic_key(update)
        message_kind = _attachment_kind(message)
        logger.info("Received message type=%s chat_id=%s thread_id=%s", message_kind, chat_id, thread_id)

        message_text = getattr(message, "text", None)
        if message_text is not None:
            await self._route_incoming_text(update, message_text)
            return

        if any(getattr(message, field, None) is not None for field in ("voice", "audio", "video_note")):
            await self._handle_audio_message(update, message)
            return

        await self._handle_attachment(update, message)

    async def _handle_audio_message(self, update: Update, message) -> None:
        try:
            filename, content_type, payload = await _download_audio_payload(message)
            transcribed = await transcribe_audio(
                self.config,
                filename=filename,
                content_type=content_type,
                payload=payload,
            )
        except AudioTranscriptionNotConfiguredError as exc:
            await self._reply(update, str(exc))
            return
        except AudioTranscriptionError as exc:
            logger.exception("Failed to transcribe Telegram audio")
            await self._reply(update, f"Failed to transcribe audio: {exc}")
            return
        except Exception as exc:
            logger.exception("Failed to download Telegram audio")
            await self._reply(update, f"Failed to read Telegram audio: {exc}")
            return

        if not transcribed.strip():
            await self._reply(update, "The transcription was empty. Try a clearer or shorter recording.")
            return
        await self._route_incoming_text(update, transcribed)

    async def _handle_attachment(self, update: Update, message) -> None:
        attachment_kind = _attachment_kind(message)
        chat_id, thread_id = topic_key(update)
        logger.info(
            "Ignoring unsupported attachment type=%s chat_id=%s thread_id=%s",
            attachment_kind,
            chat_id,
            thread_id,
        )
        await self._reply(
            update,
            f"Unsupported attachment type: {attachment_kind}. Send text, voice, or audio.",
        )

    async def _route_incoming_text(self, update: Update, raw_text: str) -> None:
        text = raw_text.strip()
        if not text:
            return

        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        if onboarding:
            await self._advance_onboarding(update, text, onboarding.mode)
            return

        binding = self.repository.get_binding(chat_id, thread_id)
        if not binding:
            seed_text = text
            if _is_forum_lobby(update) and seed_text:
                await self._maybe_redirect_new_session_to_topic(update, mode="fresh", seed_text=seed_text)
                return
            if _is_named_topic(update) and seed_text:
                self.repository.save_onboarding_state(
                    chat_id=chat_id,
                    thread_id=thread_id,
                    step=OnboardingStep.CHOOSE_PROVIDER,
                    mode="fresh",
                    pending_user_text=_encode_pending_state(seed_text=seed_text),
                )
                await self._reply(
                    update,
                    "Choose a provider for this topic.\nYour first message is saved and will be sent after setup.",
                    reply_markup=_build_provider_keyboard(self.available_providers),
                )
                return
            await self._reply(update, "This topic is not bound. Use /new or /resume.")
            return
        if binding.status not in {BindingStatus.ACTIVE, BindingStatus.PENDING_START}:
            await self._reply(update, f"Binding is `{binding.status.value}`. Wait for activation or use /status.")
            return

        try:
            self.service.send_user_text(binding, text)
        except Exception as exc:
            logger.exception("Failed to forward user text")
            await self._reply(update, f"Failed to send text to the live session: {exc}")

    async def _handle_onboarding_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()

        # Inline keyboard callbacks use a compact protocol:
        # `ob:<action>:...` for onboarding and `ap:<action>:...` for approvals.
        payload = query.data.split(":")
        if len(payload) < 2:
            return

        action = payload[1]
        if action == "new":
            await self._handle_new(update, context)
            return
        if action == "resume":
            await self._handle_resume(update, context)
            return
        if action == "cancel":
            await self._handle_cancel(update, context)
            return

        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        if onboarding is None:
            await self._reply(update, "Setup expired. Start again with /new.")
            return

        if action == "provider" and len(payload) >= 3:
            provider = parse_provider(payload[2])
            if provider is None:
                await self._reply(update, "Unknown provider selection.")
                return
            if provider not in self.available_providers:
                await self._reply(update, f"{provider.value} is not configured for this TurnMux runtime.")
                return
            await self._present_repo_picker(update, provider=provider, mode=onboarding.mode)
            return

        if action == "repo" and len(payload) >= 3:
            browser_state = _decode_repo_browser_state(onboarding.pending_user_text)
            selected = _indexed_choice(list(browser_state.recent_repos), payload[2])
            if not isinstance(selected, Path):
                await self._reply(update, "That repo shortcut expired. Send the repo path again.")
                return
            await self._present_launch_choices(update, provider=onboarding.provider, repo_path=selected, preferred_mode=onboarding.mode)
            return

        if action == "browse" and len(payload) >= 3:
            browser_state = _decode_repo_browser_state(onboarding.pending_user_text)
            recent_repos = list(browser_state.recent_repos)
            subaction = payload[2]

            if subaction == "recent":
                await self._present_repo_picker(
                    update,
                    provider=onboarding.provider,
                    mode=onboarding.mode,
                    recent_repos=recent_repos,
                )
                return

            if subaction == "root" and len(payload) >= 4:
                selected = _indexed_choice(list(self.config.allowed_roots), payload[3])
                if not isinstance(selected, Path):
                    await self._reply(update, "That folder shortcut expired. Start again with /new.")
                    return
                await self._present_repo_picker(
                    update,
                    provider=onboarding.provider,
                    mode=onboarding.mode,
                    recent_repos=recent_repos,
                    browse_dir=selected,
                )
                return

            if subaction == "dir" and len(payload) >= 4:
                selected = _indexed_choice(list(browser_state.browse_entries), payload[3])
                if not isinstance(selected, RepoBrowserEntry) or selected.kind != "dir":
                    await self._reply(update, "That folder entry expired. Open the folder again.")
                    return
                await self._present_repo_picker(
                    update,
                    provider=onboarding.provider,
                    mode=onboarding.mode,
                    recent_repos=recent_repos,
                    browse_dir=selected.path,
                )
                return

            if subaction == "repo" and len(payload) >= 4:
                selected = _indexed_choice(list(browser_state.browse_entries), payload[3])
                if not isinstance(selected, RepoBrowserEntry) or selected.kind != "repo":
                    await self._reply(update, "That repo entry expired. Open the folder again.")
                    return
                await self._present_launch_choices(update, provider=onboarding.provider, repo_path=selected.path, preferred_mode=onboarding.mode)
                return

            if subaction == "use-current":
                if browser_state.browse_dir is None or not _is_git_repo(browser_state.browse_dir):
                    await self._reply(update, "This folder is not a repo. Pick a repo or open another folder.")
                    return
                await self._present_launch_choices(
                    update,
                    provider=onboarding.provider,
                    repo_path=browser_state.browse_dir,
                    preferred_mode=onboarding.mode,
                )
                return

            if subaction == "up":
                if browser_state.browse_dir is None:
                    await self._present_repo_picker(
                        update,
                        provider=onboarding.provider,
                        mode=onboarding.mode,
                        recent_repos=recent_repos,
                    )
                    return
                parent = _browser_parent(browser_state.browse_dir, self.config.allowed_roots)
                await self._present_repo_picker(
                    update,
                    provider=onboarding.provider,
                    mode=onboarding.mode,
                    recent_repos=recent_repos,
                    browse_dir=parent,
                )
                return

            if subaction == "page" and len(payload) >= 4:
                if browser_state.browse_dir is None:
                    await self._present_repo_picker(
                        update,
                        provider=onboarding.provider,
                        mode=onboarding.mode,
                        recent_repos=recent_repos,
                    )
                    return
                page_delta = 1 if payload[3] == "next" else -1
                await self._present_repo_picker(
                    update,
                    provider=onboarding.provider,
                    mode=onboarding.mode,
                    recent_repos=recent_repos,
                    browse_dir=browser_state.browse_dir,
                    browse_page=browser_state.browse_page + page_delta,
                )
                return

        if action == "mode" and len(payload) >= 3 and payload[2] == "fresh":
            await self._launch_from_onboarding(update, provider=onboarding.provider, repo_path=onboarding.repo_path, mode="fresh")
            return

        if action == "session" and len(payload) >= 3:
            selected = _indexed_choice(decode_resume_candidates(onboarding.resume_candidates_json), payload[2])
            if not isinstance(selected, dict) or not isinstance(selected.get("session_id"), str):
                await self._reply(update, "That resume option expired. Use /resume again.")
                return
            await self._launch_from_onboarding(
                update,
                provider=onboarding.provider,
                repo_path=onboarding.repo_path,
                mode="resume",
                requested_session_id=selected["session_id"],
            )
            return

    async def _handle_approval_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()

        payload = query.data.split(":")
        if len(payload) < 2 or payload[1] not in {"approve", "deny"}:
            return

        binding = self._binding_for_update(update)
        if not binding:
            await self._reply(update, "This topic is not bound.")
            return

        try:
            response_text = self.service.resolve_pending_approval(binding, approve=payload[1] == "approve")
        except Exception as exc:
            logger.exception("Failed to resolve pending approval")
            await self._reply(update, f"Failed to answer the approval prompt: {exc}")
            return

        if query.message is not None:
            with suppress(TelegramError):
                await query.message.edit_reply_markup(reply_markup=None)
        await self._reply(update, response_text)

    async def _handle_forwarded_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_allowed(update):
            return
        message = update.effective_message
        if message is None or message.text is None:
            return
        command_name = message.text.split()[0].lower()
        if command_name in CONTROL_COMMANDS:
            return
        binding = self._binding_for_update(update)
        if not binding or binding.status not in {BindingStatus.ACTIVE, BindingStatus.PENDING_START}:
            return
        try:
            self.service.send_user_text(binding, message.text)
        except Exception:
            logger.exception("Failed to forward slash command to provider")

    async def _advance_onboarding(self, update: Update, text: str, mode_override: str | None) -> None:
        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        if onboarding is None:
            return

        if onboarding.step == OnboardingStep.CHOOSE_PROVIDER:
            provider = parse_provider(text)
            if provider is None:
                await self._reply(
                    update,
                    f"Reply with {_provider_reply_hint(self.available_providers)}, or tap one of the buttons.",
                    reply_markup=_build_provider_keyboard(self.available_providers),
                )
                return
            if provider not in self.available_providers:
                await self._reply(update, f"{provider.value} is not configured for this TurnMux runtime.")
                return
            await self._present_repo_picker(update, provider=provider, mode=mode_override)
            return

        if onboarding.step == OnboardingStep.CHOOSE_REPO:
            try:
                repo_path = self.service.validate_repo(text)
            except Exception as exc:
                await self._reply(update, f"Invalid repo path: {exc}")
                return

            await self._present_launch_choices(update, provider=onboarding.provider, repo_path=repo_path, preferred_mode=onboarding.mode)
            return

        if onboarding.step == OnboardingStep.CHOOSE_MODE:
            mode = text.lower()
            if mode not in {"fresh", "resume"}:
                await self._reply(update, "Reply with `fresh` or `resume`.")
                return

            if mode == "fresh":
                await self._launch_from_onboarding(update, provider=onboarding.provider, repo_path=onboarding.repo_path, mode="fresh")
                return

            self.repository.save_onboarding_state(
                chat_id=chat_id,
                thread_id=thread_id,
                step=OnboardingStep.CHOOSE_RESUME,
                provider=onboarding.provider,
                repo_path=onboarding.repo_path,
                mode="resume",
                pending_user_text=_encode_pending_state(seed_text=_extract_seed_text(onboarding.pending_user_text)),
            )
            await self._present_resume_candidates(update, onboarding.provider, onboarding.repo_path)
            return

        if onboarding.step == OnboardingStep.CHOOSE_RESUME:
            if text.strip().lower() == "fresh":
                await self._launch_from_onboarding(update, provider=onboarding.provider, repo_path=onboarding.repo_path, mode="fresh")
                return

            candidates = decode_resume_candidates(onboarding.resume_candidates_json)
            if not candidates:
                await self._launch_from_onboarding(update, provider=onboarding.provider, repo_path=onboarding.repo_path, mode="fresh")
                return

            selected = None
            if text.isdigit():
                index = int(text) - 1
                if 0 <= index < len(candidates):
                    selected = candidates[index]
            else:
                selected = next((item for item in candidates if item.get("session_id") == text), None)

            if selected is None:
                await self._reply(update, "Reply with `fresh` or an exact session ID.")
                return

            await self._launch_from_onboarding(
                update,
                provider=onboarding.provider,
                repo_path=onboarding.repo_path,
                mode="resume",
                requested_session_id=selected["session_id"],
            )

    async def _present_repo_picker(
        self,
        update: Update,
        *,
        provider: ProviderName | None,
        mode: str | None,
        recent_repos: list[Path] | None = None,
        browse_dir: Path | None = None,
        browse_page: int = 0,
        seed_text: str | None = None,
    ) -> None:
        if provider is None:
            await self._reply(update, "Provider is missing. Restart with /new.")
            return

        await self._maybe_name_topic_for_provider(update, provider=provider)

        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        if recent_repos is None:
            existing_state = _decode_repo_browser_state(onboarding.pending_user_text if onboarding else None)
            recent_repos = list(existing_state.recent_repos) or self.service.suggest_repos(limit=6)
        if seed_text is None:
            seed_text = _extract_seed_text(onboarding.pending_user_text if onboarding else None)
        browser_state = _make_repo_browser_state(
            recent_repos=recent_repos,
            allowed_roots=self.config.allowed_roots,
            browse_dir=browse_dir,
            browse_page=browse_page,
        )
        self.repository.save_onboarding_state(
            chat_id=chat_id,
            thread_id=thread_id,
            step=OnboardingStep.CHOOSE_REPO,
            provider=provider,
            mode=mode,
            pending_user_text=_encode_pending_state(seed_text=seed_text, repo_browser_state=browser_state),
        )
        await self._reply(
            update,
            _format_repo_prompt(provider, mode, browser_state),
            reply_markup=_build_repo_keyboard(browser_state, self.config.allowed_roots),
        )

    async def _present_resume_candidates(self, update: Update, provider: ProviderName | None, repo_path: Path | None) -> None:
        if provider is None or repo_path is None:
            await self._reply(update, "Provider or repo path is missing. Restart with /new.")
            return

        try:
            candidates = self.service.list_resumable_sessions(provider, repo_path)
        except Exception as exc:
            logger.exception("Failed to list resumable sessions")
            await self._reply(update, f"Failed to inspect resumable sessions: {exc}")
            return
        if not candidates:
            chat_id, thread_id = topic_key(update)
            onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
            self.repository.save_onboarding_state(
                chat_id=chat_id,
                thread_id=thread_id,
                step=OnboardingStep.CHOOSE_MODE,
                provider=provider,
                repo_path=repo_path,
                mode="fresh",
                pending_user_text=_encode_pending_state(seed_text=_extract_seed_text(onboarding.pending_user_text if onboarding else None)),
            )
            await self._reply(
                update,
                _format_launch_prompt(provider, repo_path, [], preferred_mode="fresh"),
                reply_markup=_build_launch_keyboard([], preferred_mode="fresh"),
            )
            return

        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        self.repository.save_onboarding_state(
            chat_id=chat_id,
            thread_id=thread_id,
            step=OnboardingStep.CHOOSE_RESUME,
            provider=provider,
            repo_path=repo_path,
            mode="resume",
            pending_user_text=_encode_pending_state(seed_text=_extract_seed_text(onboarding.pending_user_text if onboarding else None)),
            resume_candidates_json=encode_resume_candidates(candidates),
        )
        await self._reply(
            update,
            _format_launch_prompt(provider, repo_path, candidates, preferred_mode="resume"),
            reply_markup=_build_launch_keyboard(candidates, preferred_mode="resume"),
        )

    async def _present_launch_choices(
        self,
        update: Update,
        *,
        provider: ProviderName | None,
        repo_path: Path | None,
        preferred_mode: str | None,
    ) -> None:
        if provider is None or repo_path is None:
            await self._reply(update, "Provider or repo path is missing. Restart with /new.")
            return

        try:
            candidates = self.service.list_resumable_sessions(provider, repo_path)
        except Exception as exc:
            logger.exception("Failed to list resumable sessions")
            await self._reply(update, f"Failed to inspect resumable sessions: {exc}")
            return

        if preferred_mode == "fresh" and not candidates:
            await self._launch_from_onboarding(update, provider=provider, repo_path=repo_path, mode="fresh")
            return

        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        self.repository.save_onboarding_state(
            chat_id=chat_id,
            thread_id=thread_id,
            step=OnboardingStep.CHOOSE_RESUME,
            provider=provider,
            repo_path=repo_path,
            mode=preferred_mode,
            pending_user_text=_encode_pending_state(seed_text=_extract_seed_text(onboarding.pending_user_text if onboarding else None)),
            resume_candidates_json=encode_resume_candidates(candidates),
        )
        await self._reply(
            update,
            _format_launch_prompt(provider, repo_path, candidates, preferred_mode),
            reply_markup=_build_launch_keyboard(candidates, preferred_mode=preferred_mode),
        )

    async def _launch_from_onboarding(
        self,
        update: Update,
        *,
        provider: ProviderName | None,
        repo_path: Path | None,
        mode: str,
        requested_session_id: str | None = None,
    ) -> None:
        if provider is None or repo_path is None:
            await self._reply(update, "Setup state is incomplete. Restart with /new.")
            return
        chat_id, thread_id = topic_key(update)
        onboarding = self.repository.get_onboarding_state(chat_id, thread_id)
        seed_text = _extract_seed_text(onboarding.pending_user_text if onboarding else None)
        try:
            binding = await self.service.launch_binding(
                chat_id=chat_id,
                thread_id=thread_id,
                provider=provider,
                repo_path=repo_path,
                mode=mode,
                requested_session_id=requested_session_id,
            )
        except Exception as exc:
            logger.exception("Failed to launch binding")
            await self._reply(update, f"Failed to launch {provider.value}: {exc}")
            return
        auto_sent = False
        if seed_text and mode == "fresh":
            try:
                self.service.send_user_text(binding, seed_text)
                auto_sent = True
            except Exception:
                logger.exception("Failed to auto-send saved first message")
        self.repository.clear_onboarding_state(chat_id, thread_id)
        status_note = f"Binding created for {provider.value} at {repo_path}.\nStatus: {binding.status.value}"
        if auto_sent:
            status_note += "\nYour first message was sent to start the session."
        elif binding.status == BindingStatus.PENDING_START:
            status_note += "\nThe tmux window is ready. Send the first message in this topic to let the provider create its transcript and activate the binding."
        await self._maybe_name_topic(update, provider=provider, repo_path=repo_path)
        await self._reply(
            update,
            status_note,
        )

    async def _monitor_loop(self, application: Application) -> None:
        while True:
            try:
                outbound = self.service.refresh_pending_and_active_bindings()
                for message in outbound:
                    await send_thread_message(
                        application,
                        chat_id=message.chat_id,
                        thread_id=message.thread_id,
                        text=message.text,
                        reply_markup=_build_approval_keyboard(has_deny=message.markup_has_deny) if message.markup_kind == "approval" else None,
                    )
                self._set_runtime_health("running")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_runtime_health("degraded", note=f"monitor:{exc}")
                signature = f"{type(exc).__name__}:{exc}"
                now = asyncio.get_running_loop().time()
                should_log = (
                    signature != self._last_monitor_error_signature
                    or (now - self._last_monitor_error_at) >= 60.0
                )
                self._last_monitor_error_signature = signature
                self._last_monitor_error_at = now
                if should_log:
                    logger.exception("TurnMux monitor loop failed")
            await asyncio.sleep(2.0)

    async def _heartbeat_loop(self) -> None:
        if self._heartbeat_writer is None:
            return
        while True:
            try:
                self._heartbeat_writer.write(status=self._runtime_status, note=self._runtime_note)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to write runtime heartbeat")
            await asyncio.sleep(30.0)

    def _request_shutdown(self, reason: str) -> None:
        logger.warning("Shutdown requested via %s", reason)
        self._set_runtime_health("stopping", note=f"signal:{reason}")
        if self._heartbeat_writer is not None:
            self._heartbeat_writer.write(status=self._runtime_status, note=self._runtime_note)
        if self._stop_event is not None and not self._stop_event.is_set():
            self._stop_event.set()

    def _set_runtime_health(self, status: str, *, note: str | None = None) -> None:
        self._runtime_status = status
        self._runtime_note = note

    async def _ensure_allowed(self, update: Update) -> bool:
        user = update.effective_user
        if user and user.id in self.config.allowed_user_ids:
            return True
        await self._reply(update, "You are not allowed to use this TurnMux bot.")
        return False

    async def _maybe_redirect_new_session_to_topic(self, update: Update, *, mode: str, seed_text: str | None = None) -> bool:
        if not _is_forum_lobby(update):
            return False

        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return False

        topic_name = "Session Setup" if mode == "fresh" else "Resume Setup"
        try:
            forum_topic = await message.get_bot().create_forum_topic(
                chat_id=chat.id,
                name=topic_name,
            )
        except TelegramError as exc:
            logger.exception("Failed to create forum topic")
            await self._reply(
                update,
                f"Failed to create a new topic automatically: {exc}\n"
                "Create a topic manually and run /new inside it.",
            )
            return True

        self.repository.save_onboarding_state(
            chat_id=chat.id,
            thread_id=forum_topic.message_thread_id,
            step=OnboardingStep.CHOOSE_PROVIDER,
            mode=mode,
            pending_user_text=_encode_pending_state(seed_text=seed_text),
        )
        await send_thread_message(
            message.get_bot(),
            chat_id=chat.id,
            thread_id=forum_topic.message_thread_id,
            text=(
                "Choose a provider for this topic.\nYour first message is saved and will be sent after setup."
                if seed_text
                else "Choose a provider for this topic."
            ),
            reply_markup=_build_provider_keyboard(self.available_providers),
        )
        followup = f"Created topic `{topic_name}`. Continue there."
        if seed_text:
            followup = f"Created topic `{topic_name}` from your message. Continue there."
        await self._reply(update, followup)
        return True

    async def _maybe_name_topic_for_provider(self, update: Update, *, provider: ProviderName) -> None:
        if not _is_named_topic(update) or update.effective_chat is None:
            return
        message = update.effective_message
        if message is None:
            return
        try:
            await message.get_bot().edit_forum_topic(
                chat_id=update.effective_chat.id,
                message_thread_id=topic_key(update)[1],
                name=_format_topic_setup_name(provider),
            )
        except TelegramError:
            logger.exception("Failed to rename forum topic for provider setup")

    async def _maybe_name_topic(self, update: Update, *, provider: ProviderName, repo_path: Path) -> None:
        if not _is_named_topic(update) or update.effective_chat is None:
            return
        message = update.effective_message
        if message is None:
            return
        try:
            await message.get_bot().edit_forum_topic(
                chat_id=update.effective_chat.id,
                message_thread_id=topic_key(update)[1],
                name=_format_topic_name(provider, repo_path),
            )
        except TelegramError:
            logger.exception("Failed to rename forum topic")

    async def _maybe_delete_topic(self, update: Update) -> bool:
        if not _is_named_topic(update) or update.effective_chat is None:
            return False
        message = update.effective_message
        if message is None:
            return False
        try:
            await message.get_bot().delete_forum_topic(
                chat_id=update.effective_chat.id,
                message_thread_id=topic_key(update)[1],
            )
            return True
        except TelegramError as exc:
            logger.exception("Failed to delete forum topic")
            await self._reply(
                update,
                f"Session killed, but the Telegram topic could not be deleted: {exc}",
            )
            return True

    def _binding_for_update(self, update: Update):
        chat_id, thread_id = topic_key(update)
        return self.repository.get_binding(chat_id, thread_id)

    async def _reply(self, update: Update, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
        message = update.effective_message
        if message is None:
            return
        chat_id, thread_id = topic_key(update)
        for index, chunk in enumerate(split_text(text)):
            await message.get_bot().send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=thread_id or None,
                reply_markup=reply_markup if index == 0 else None,
            )


def parse_provider(value: str) -> ProviderName | None:
    normalized = value.strip().lower()
    if normalized == "claude":
        return ProviderName.CLAUDE
    if normalized == "codex":
        return ProviderName.CODEX
    if normalized in {"opencode", "open code"}:
        return ProviderName.OPENCODE
    return None


def _provider_button_label(provider: ProviderName) -> str:
    labels = {
        ProviderName.CLAUDE: "Claude",
        ProviderName.CODEX: "Codex",
        ProviderName.OPENCODE: "OpenCode",
    }
    return labels[provider]


def _provider_hint_text(available_providers: tuple[ProviderName, ...]) -> str:
    if not available_providers:
        return "provider-backed"
    labels = [_provider_button_label(provider).lower() for provider in available_providers]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _provider_reply_hint(available_providers: tuple[ProviderName, ...]) -> str:
    if not available_providers:
        return "`provider-name`"
    if len(available_providers) == 1:
        return f"`{available_providers[0].value}`"
    head = ", ".join(f"`{provider.value}`" for provider in available_providers[:-1])
    return f"{head}, or `{available_providers[-1].value}`"


def topic_key(update: Update) -> tuple[int, int]:
    message = update.effective_message
    if update.effective_chat is None:
        raise RuntimeError("Telegram update does not have a chat.")
    thread_id = 0
    if message and message.message_thread_id and message.message_thread_id != 1:
        thread_id = message.message_thread_id
    return update.effective_chat.id, thread_id


async def send_thread_message(bot_or_application, *, chat_id: int, thread_id: int, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    bot = getattr(bot_or_application, "bot", bot_or_application)
    for chunk in split_text(text):
        await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            message_thread_id=thread_id or None,
            reply_markup=reply_markup,
        )
        reply_markup = None


def split_text(text: str, *, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _bot_commands(available_providers: tuple[ProviderName, ...] | None = None) -> list[BotCommand]:
    provider_hint = _provider_hint_text(available_providers or ())
    return [
        BotCommand("start", "Show the TurnMux entry screen"),
        BotCommand("new", f"Start a new {provider_hint} session"),
        BotCommand("resume", f"Resume an existing {provider_hint} session"),
        BotCommand("status", "Show binding and runtime status"),
        BotCommand("history", "Show recent transcript history"),
        BotCommand("interrupt", "Send Ctrl-C to the live session"),
        BotCommand("kill", "Kill the session and remove the topic binding"),
        BotCommand("cancel", "Cancel the current setup flow"),
    ]


async def _download_audio_payload(message) -> tuple[str, str, bytes]:
    if getattr(message, "voice", None) is not None:
        voice = message.voice
        telegram_file = await voice.get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        return "voice.ogg", "audio/ogg", payload

    if getattr(message, "audio", None) is not None:
        audio = message.audio
        telegram_file = await audio.get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        filename = audio.file_name or "audio.bin"
        content_type = audio.mime_type or "application/octet-stream"
        return filename, content_type, payload

    if getattr(message, "video_note", None) is not None:
        video_note = message.video_note
        telegram_file = await video_note.get_file()
        payload = bytes(await telegram_file.download_as_bytearray())
        return "video_note.mp4", "video/mp4", payload

    raise ValueError("Message does not contain a supported audio attachment.")


def _attachment_kind(message) -> str:
    for field_name in (
        "voice",
        "audio",
        "video_note",
        "photo",
        "video",
        "document",
        "sticker",
        "animation",
    ):
        value = getattr(message, field_name, None)
        if value:
            return field_name
    return "attachment"


def _build_start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("New Binding", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:new"),
                InlineKeyboardButton("Resume Existing", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:resume"),
            ]
        ]
    )


def _build_provider_keyboard(available_providers: tuple[ProviderName, ...]) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(_provider_button_label(provider), callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:provider:{provider.value}")
        for provider in available_providers
    ]
    rows = [buttons[index : index + 2] for index in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("Cancel", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:cancel")])
    return InlineKeyboardMarkup(rows)


def _build_repo_keyboard(state: RepoBrowserState, allowed_roots: tuple[Path, ...]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    if state.browse_dir is None:
        for index, path in enumerate(state.recent_repos):
            rows.append([InlineKeyboardButton(_repo_button_label(path), callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:repo:{index}")])
        if len(allowed_roots) == 1:
            rows.append(
                [
                    InlineKeyboardButton(
                        "Browse Folders",
                        callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:root:0",
                    )
                ]
            )
        else:
            for index, root in enumerate(allowed_roots):
                rows.append(
                    [
                        InlineKeyboardButton(
                            f"Browse {_root_button_label(root)}",
                            callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:root:{index}",
                        )
                    ]
                )
        rows.append([InlineKeyboardButton("Cancel", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:cancel")])
        return InlineKeyboardMarkup(rows)

    if _is_git_repo(state.browse_dir):
        rows.append(
            [
                InlineKeyboardButton(
                    f"Use {_truncate_button_text(state.browse_dir.name or str(state.browse_dir), 28)}",
                    callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:use-current",
                )
            ]
        )

    start = state.browse_page * REPO_BROWSER_PAGE_SIZE
    visible_entries = state.browse_entries[start : start + REPO_BROWSER_PAGE_SIZE]
    for offset, entry in enumerate(visible_entries, start=start):
        action = "repo" if entry.kind == "repo" else "dir"
        rows.append(
            [
                InlineKeyboardButton(
                    _browser_entry_button_label(entry),
                    callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:{action}:{offset}",
                )
            ]
        )

    navigation_row: list[InlineKeyboardButton] = []
    if state.browse_page > 0:
        navigation_row.append(
            InlineKeyboardButton("Prev", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:page:prev")
        )
    if _browser_parent(state.browse_dir, allowed_roots) is not None:
        navigation_row.append(InlineKeyboardButton("Up", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:up"))
    if (start + REPO_BROWSER_PAGE_SIZE) < len(state.browse_entries):
        navigation_row.append(
            InlineKeyboardButton("Next", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:page:next")
        )
    if navigation_row:
        rows.append(navigation_row)
    rows.append(
        [
            InlineKeyboardButton("Recent", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:browse:recent"),
            InlineKeyboardButton("Cancel", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:cancel"),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _build_launch_keyboard(candidates: list[object], *, preferred_mode: str | None) -> InlineKeyboardMarkup:
    fresh_label = "Start Fresh"
    if preferred_mode == "fresh":
        fresh_label = "Start Fresh (Recommended)"
    rows = [[InlineKeyboardButton(fresh_label, callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:mode:fresh")]]
    for index, candidate in enumerate(candidates):
        session_id, label = _resume_candidate_fields(candidate, index)
        if session_id is None:
            continue
        rows.append(
            [
                InlineKeyboardButton(
                    f"Resume {index + 1}: {_truncate_button_text(str(label), 36)}",
                    callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:session:{index}",
                )
            ]
        )
    rows.append([InlineKeyboardButton("Cancel", callback_data=f"{ONBOARDING_CALLBACK_PREFIX}:cancel")])
    return InlineKeyboardMarkup(rows)


def _build_approval_keyboard(*, has_deny: bool) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton("Approve", callback_data=f"{APPROVAL_CALLBACK_PREFIX}:approve")]
    if has_deny:
        row.append(InlineKeyboardButton("Deny", callback_data=f"{APPROVAL_CALLBACK_PREFIX}:deny"))
    return InlineKeyboardMarkup([row])


def _format_repo_prompt(provider: ProviderName, mode: str | None, state: RepoBrowserState) -> str:
    lines = [f"Provider: `{provider.value}`"]
    if state.browse_dir is None:
        lines.append("Choose a repo.")
        lines.append("Tap a recent repo, browse folders, or send an absolute path.")
    else:
        lines.append(f"Browsing `{state.browse_dir}`")
        lines.append("Tap a repo to select it, open a folder, or send an absolute path.")
        total_entries = len(state.browse_entries)
        if total_entries > REPO_BROWSER_PAGE_SIZE:
            start = state.browse_page * REPO_BROWSER_PAGE_SIZE + 1
            end = min(total_entries, (state.browse_page + 1) * REPO_BROWSER_PAGE_SIZE)
            lines.append(f"Showing {start}-{end} of {total_entries}.")
        elif total_entries == 0 and not _is_git_repo(state.browse_dir):
            lines.append("No repos or subfolders are visible here.")
    if mode == "resume":
        lines.append("TurnMux will show resumable sessions after repo selection.")
    return "\n".join(lines)


def _format_launch_prompt(
    provider: ProviderName,
    repo_path: Path,
    candidates: list[object],
    preferred_mode: str | None,
) -> str:
    lines = [
        f"Provider: `{provider.value}`",
        f"Repo: `{repo_path}`",
    ]
    if candidates:
        lines.append("Choose a resumable session below or start fresh.")
        lines.append("You can also send an exact session ID as text.")
    else:
        lines.append("No resumable sessions were found for that repo. Start fresh.")
    if preferred_mode == "resume":
        lines.append("If none of them fit, `Start Fresh` is still available.")
    return "\n".join(lines)


def _repo_button_label(path: Path) -> str:
    parent = path.parent.name
    name = path.name
    if parent and parent != path.anchor.strip("/"):
        return _truncate_button_text(f"{parent}/{name}", 36)
    return _truncate_button_text(name or str(path), 36)


def _truncate_button_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _indexed_choice(items: list[object], raw_index: str) -> object | None:
    try:
        index = int(raw_index)
    except ValueError:
        return None
    if 0 <= index < len(items):
        return items[index]
    return None


def _resume_candidate_fields(candidate: object, index: int) -> tuple[str | None, str]:
    if isinstance(candidate, ProviderSession):
        return candidate.session_id, candidate.display_name or candidate.session_id
    if isinstance(candidate, dict):
        session_id = candidate.get("session_id")
        if isinstance(session_id, str):
            label = candidate.get("display_name") if isinstance(candidate.get("display_name"), str) else session_id
            return session_id, label
    return None, f"Resume {index + 1}"


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _browser_entry_button_label(entry: RepoBrowserEntry) -> str:
    prefix = "Repo" if entry.kind == "repo" else "Open"
    return _truncate_button_text(f"{prefix}: {entry.label}", 36)


def _root_button_label(root: Path) -> str:
    parts = [part for part in root.parts if part not in {root.anchor, "/"}]
    if len(parts) >= 2:
        return _truncate_button_text("/".join(parts[-2:]), 28)
    return _truncate_button_text(root.name or str(root), 28)


def _make_repo_browser_state(
    *,
    recent_repos: Iterable[Path],
    allowed_roots: tuple[Path, ...],
    browse_dir: Path | None,
    browse_page: int,
) -> RepoBrowserState:
    normalized_recent: list[Path] = []
    for path in recent_repos:
        try:
            normalized = _normalize_browse_dir(path, allowed_roots)
        except ValueError:
            continue
        if _is_git_repo(normalized):
            normalized_recent.append(normalized)

    normalized_browse_dir: Path | None = None
    browse_entries: tuple[RepoBrowserEntry, ...] = ()
    if browse_dir is not None:
        normalized_browse_dir = _normalize_browse_dir(browse_dir, allowed_roots)
        browse_entries = _list_repo_browser_entries(normalized_browse_dir, allowed_roots)

    max_page = max((len(browse_entries) - 1) // REPO_BROWSER_PAGE_SIZE, 0)
    clamped_page = min(max(browse_page, 0), max_page)
    return RepoBrowserState(
        recent_repos=tuple(normalized_recent[:6]),
        browse_dir=normalized_browse_dir,
        browse_entries=browse_entries,
        browse_page=clamped_page,
    )


def _encode_pending_state(*, seed_text: str | None = None, repo_browser_state: RepoBrowserState | None = None) -> str | None:
    payload: dict[str, object] = {}
    # Onboarding keeps one persisted blob so button callbacks and free-text
    # replies can share the same seed message and repo-browser snapshot.
    if seed_text:
        payload["seed_text"] = seed_text
    if repo_browser_state is not None:
        payload["repo_browser"] = _repo_browser_to_payload(repo_browser_state)
    if not payload:
        return None
    return json.dumps(payload, ensure_ascii=False)


def _encode_repo_browser_state(state: RepoBrowserState) -> str:
    return json.dumps(_repo_browser_to_payload(state), ensure_ascii=False)


def _decode_repo_browser_state(value: str | None) -> RepoBrowserState:
    if not value:
        return RepoBrowserState(recent_repos=())
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return RepoBrowserState(recent_repos=())
    if not isinstance(payload, dict):
        return RepoBrowserState(recent_repos=())
    repo_browser_payload = payload.get("repo_browser") if "repo_browser" in payload else payload
    if not isinstance(repo_browser_payload, dict):
        return RepoBrowserState(recent_repos=())
    return _repo_browser_from_payload(repo_browser_payload)


def _extract_seed_text(value: str | None) -> str | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value.strip() or None
    if not isinstance(payload, dict):
        return None
    seed_text = payload.get("seed_text")
    if isinstance(seed_text, str) and seed_text.strip():
        return seed_text
    return None


def _repo_browser_to_payload(state: RepoBrowserState) -> dict[str, object]:
    return {
        "recent_repos": [str(path) for path in state.recent_repos],
        "browse_dir": str(state.browse_dir) if state.browse_dir else None,
        "browse_entries": [
            {"path": str(entry.path), "kind": entry.kind, "label": entry.label}
            for entry in state.browse_entries
        ],
        "browse_page": state.browse_page,
    }


def _repo_browser_from_payload(payload: dict[str, object]) -> RepoBrowserState:
    raw_recent = payload.get("recent_repos")
    recent_repos = tuple(Path(item) for item in raw_recent if isinstance(item, str) and item.strip()) if isinstance(raw_recent, list) else ()

    raw_browse_dir = payload.get("browse_dir")
    browse_dir = Path(raw_browse_dir) if isinstance(raw_browse_dir, str) and raw_browse_dir.strip() else None

    raw_entries = payload.get("browse_entries")
    browse_entries: list[RepoBrowserEntry] = []
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            kind = item.get("kind")
            label = item.get("label")
            if isinstance(path, str) and path.strip() and isinstance(kind, str) and isinstance(label, str):
                browse_entries.append(RepoBrowserEntry(path=Path(path), kind=kind, label=label))

    raw_page = payload.get("browse_page")
    browse_page = raw_page if isinstance(raw_page, int) and raw_page >= 0 else 0

    return RepoBrowserState(
        recent_repos=recent_repos,
        browse_dir=browse_dir,
        browse_entries=tuple(browse_entries),
        browse_page=browse_page,
    )


def _normalize_browse_dir(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    normalized = path.expanduser().resolve(strict=False)
    if not normalized.exists() or not normalized.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {path}")
    for root in allowed_roots:
        normalized_root = root.expanduser().resolve(strict=False)
        try:
            normalized.relative_to(normalized_root)
            return normalized
        except ValueError:
            continue
    raise ValueError(f"Path is outside allowed_roots: {path}")


def _list_repo_browser_entries(directory: Path, allowed_roots: tuple[Path, ...]) -> tuple[RepoBrowserEntry, ...]:
    try:
        children = list(directory.iterdir())
    except OSError:
        return ()

    entries: list[RepoBrowserEntry] = []
    for child in children:
        if child.name.startswith(".") or not child.is_dir():
            continue
        try:
            normalized = _normalize_browse_dir(child, allowed_roots)
        except ValueError:
            continue
        entries.append(
            RepoBrowserEntry(
                path=normalized,
                kind="repo" if _is_git_repo(normalized) else "dir",
                label=child.name,
            )
        )

    entries.sort(key=lambda entry: (0 if entry.kind == "repo" else 1, entry.label.lower()))
    return tuple(entries)


def _browser_parent(path: Path, allowed_roots: tuple[Path, ...]) -> Path | None:
    normalized = _normalize_browse_dir(path, allowed_roots)
    for root in allowed_roots:
        normalized_root = root.expanduser().resolve(strict=False)
        try:
            normalized.relative_to(normalized_root)
        except ValueError:
            continue
        if normalized == normalized_root:
            return None
        parent = normalized.parent.resolve(strict=False)
        try:
            parent.relative_to(normalized_root)
            return parent
        except ValueError:
            return None
    return None


def _is_named_topic(update: Update) -> bool:
    return topic_key(update)[1] > 0


def _is_private_chat(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type == "private")


def _is_forum_lobby(update: Update) -> bool:
    chat = update.effective_chat
    if chat is None or chat.type != "supergroup" or not getattr(chat, "is_forum", False):
        return False
    return topic_key(update)[1] == 0


def _provider_topic_badge(provider: ProviderName) -> str:
    badges = {
        ProviderName.CLAUDE: "🟠 claude",
        ProviderName.CODEX: "🔵 codex",
        ProviderName.OPENCODE: "🟢 opencode",
    }
    return badges[provider]


def _format_topic_setup_name(provider: ProviderName) -> str:
    return f"{_provider_topic_badge(provider)} · setup"


def _format_topic_name(provider: ProviderName, repo_path: Path) -> str:
    name = f"{_provider_topic_badge(provider)} · {repo_path.name or repo_path}"
    return name[:128]
