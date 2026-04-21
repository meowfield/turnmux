from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import textwrap

from .config import ConfigError, TurnmuxConfig, load_config, validate_repo_path
from .providers import ProviderRegistry
from .runtime.binaries import resolve_binary
from .providers.trust import (
    is_claude_skip_dangerous_prompt_enabled,
    is_provider_trusted,
)
from .runtime.home import RuntimePaths, set_private_file_permissions
from .service_manager import HEARTBEAT_STALE_AFTER_SECONDS, evaluate_launch_agent_health, heartbeat_age_seconds, read_launch_agent_status
from .state.models import ProviderName


TOKEN_PLACEHOLDER = "REPLACE_WITH_BOT_TOKEN"
USER_ID_PLACEHOLDER = 123456789


@dataclass(frozen=True, slots=True)
class DoctorReport:
    ok: bool
    text: str


def write_sample_config(
    runtime_paths: RuntimePaths,
    *,
    force: bool = False,
    working_dir: Path | None = None,
) -> Path:
    config_path = runtime_paths.config_path
    if config_path.exists() and not force:
        raise FileExistsError(f"Config already exists: {config_path}")
    config_path.write_text(render_sample_config(runtime_paths, working_dir=working_dir), encoding="utf-8")
    set_private_file_permissions(config_path)
    return config_path


def render_sample_config(runtime_paths: RuntimePaths, *, working_dir: Path | None = None) -> str:
    allowed_root = _recommended_allowed_root(working_dir)
    detected_claude_binary = _detect_binary("claude")
    detected_codex_binary = _detect_binary("codex")
    detected_opencode_binary = _detect_binary("opencode")
    claude_block = _render_command_block(
        "Claude",
        "claude_command",
        [detected_claude_binary or "claude", "--dangerously-skip-permissions"],
        enabled=detected_claude_binary is not None,
    )
    codex_block = _render_command_block(
        "Codex",
        "codex_command",
        [detected_codex_binary or "codex", "--ask-for-approval", "on-request", "--sandbox", "danger-full-access", "--no-alt-screen"],
        enabled=detected_codex_binary is not None,
    )
    opencode_block = _render_command_block(
        "OpenCode",
        "opencode_command",
        [detected_opencode_binary or str((Path.home() / ".opencode" / "bin" / "opencode").resolve(strict=False))],
        enabled=detected_opencode_binary is not None,
        optional=True,
        trailing_lines=['opencode_model = "provider/model"'],
    )
    return textwrap.dedent(
        f"""\
        # Replace telegram_bot_token and allowed_user_ids before running TurnMux.
        # Configure at least one provider command below.
        # TurnMux needs to receive plain text from forum topics, so either:
        # - disable bot privacy mode in @BotFather, or
        # - add the bot as a supergroup admin.

        telegram_bot_token = "{TOKEN_PLACEHOLDER}"
        allowed_user_ids = [{USER_ID_PLACEHOLDER}]
        # Start narrow. Add more roots only if you intentionally want broader access.
        allowed_roots = ["{allowed_root}"]
        tmux_session_name = "turnmux"

        {claude_block}

        {codex_block}

        # Optional: relay Claude thinking blocks back into Telegram and /history.
        # relay_claude_thinking = true

        {opencode_block}

        # Optional: voice/audio transcription via OpenAI.
        # openai_api_key = "sk-..."
        # openai_base_url = "https://api.openai.com/v1"
        # openai_transcription_model = "gpt-4o-transcribe"
        """
    )


def run_doctor(
    runtime_paths: RuntimePaths,
    *,
    config_path: Path | None = None,
    repo_path: Path | None = None,
) -> DoctorReport:
    lines = ["TurnMux doctor"]
    ok = True

    lines.append(f"[ok] runtime home: {runtime_paths.home}")
    lines.append(f"[ok] logs dir: {runtime_paths.logs_dir}")
    if runtime_paths.state_db_path.exists():
        lines.append(f"[ok] state db: {runtime_paths.state_db_path}")
    else:
        lines.append(f"[warn] state db is missing: {runtime_paths.state_db_path}")
        lines.append("[hint] Run `turnmux bootstrap` once before first use, or let `turnmux service install` create it.")

    tmux_binary = _detect_binary("tmux")
    if tmux_binary:
        tmux_version = _command_output([tmux_binary, "-V"]) or "tmux detected"
        lines.append(f"[ok] tmux: {tmux_binary} ({tmux_version})")
        tmux_probe_error = _probe_tmux(tmux_binary)
        if tmux_probe_error is None:
            lines.append("[ok] tmux control check: ready")
        else:
            ok = False
            lines.append(f"[error] tmux control check failed: {tmux_probe_error}")
            lines.append("[hint] Make sure the current user can create and access tmux sockets, then rerun `turnmux doctor`.")
    else:
        ok = False
        lines.append("[error] tmux is not installed or not on PATH.")
        lines.append("[hint] Install tmux and confirm `tmux -V` works in the same shell environment.")

    resolved_config_path = (config_path or runtime_paths.config_path).expanduser().resolve(strict=False)
    if not resolved_config_path.exists():
        ok = False
        lines.append(f"[error] config file is missing: {resolved_config_path}")
        lines.append("[hint] Run `turnmux init-config`, then replace the placeholder bot token and user id.")
        return DoctorReport(ok=ok, text="\n".join(lines))

    try:
        config = load_config(resolved_config_path)
    except ConfigError as exc:
        ok = False
        lines.append(f"[error] config failed to load: {exc}")
        if "Configure at least one provider command" in str(exc):
            lines.append("[hint] Uncomment one provider block and make sure its executable exists on PATH before retrying.")
        return DoctorReport(ok=ok, text="\n".join(lines))

    lines.extend(_config_checks(config))
    if repo_path is not None:
        lines.extend(_repo_checks(config, repo_path))
    lines.extend(_service_checks(runtime_paths))

    lines.append(
        "[note] TurnMux needs either disabled bot privacy mode or bot admin rights in the forum supergroup to receive plain text messages."
    )
    lines.append(
        "[note] Treat Telegram topics as a high-trust control surface. Avoid sending raw API keys or other secrets there."
    )
    return DoctorReport(ok=ok and not any(line.startswith("[error]") for line in lines), text="\n".join(lines))


def _config_checks(config: TurnmuxConfig) -> list[str]:
    lines = [f"[ok] config file: {config.config_path}"]
    lines.append(f"[ok] allowed roots: {', '.join(str(path) for path in config.allowed_roots)}")
    lines.append(f"[ok] tmux session name: {config.tmux_session_name}")

    if _looks_like_placeholder_token(config.telegram_bot_token):
        lines.append("[error] telegram_bot_token is still the sample placeholder.")
        lines.append("[hint] Replace it with the real bot token from @BotFather before starting TurnMux.")
    else:
        lines.append("[ok] telegram_bot_token looks non-placeholder.")

    if USER_ID_PLACEHOLDER in config.allowed_user_ids:
        lines.append(f"[error] allowed_user_ids still contains the placeholder value {USER_ID_PLACEHOLDER}.")
        lines.append("[hint] Replace it with the Telegram numeric user id that should control this runtime.")
    else:
        lines.append(f"[ok] allowed_user_ids: {', '.join(str(value) for value in config.allowed_user_ids)}")

    if config.claude_command:
        lines.extend(_command_checks("claude_command", config.claude_command))
        if not _has_claude_noninteractive_permissions(config.claude_command):
            lines.append("[warn] claude_command is missing a full-access permission flag; Claude will no longer start in the default bypass mode.")
    else:
        lines.append("[note] claude provider: not configured.")

    if config.codex_command:
        lines.extend(_command_checks("codex_command", config.codex_command))
        if not _command_has_pair(config.codex_command, "--ask-for-approval", "on-request"):
            lines.append("[warn] codex_command should include `--ask-for-approval on-request` so Telegram approval buttons can answer prompts.")
        if not _command_has_pair(config.codex_command, "--sandbox", "danger-full-access"):
            lines.append("[warn] codex_command should include `--sandbox danger-full-access` to match the default full-access TurnMux mode.")
        if "--no-alt-screen" not in config.codex_command:
            lines.append("[warn] codex_command should include `--no-alt-screen` for tmux scrollback and stable capture.")
    else:
        lines.append("[note] codex provider: not configured.")
    if config.opencode_command:
        lines.extend(_command_checks("opencode_command", config.opencode_command))
    else:
        lines.append("[note] opencode provider: not configured.")
    if config.opencode_model:
        lines.append(f"[ok] opencode_model: {config.opencode_model}")
    lines.append(f"[ok] relay_claude_thinking: {'enabled' if config.relay_claude_thinking else 'disabled'}")
    if config.openai_api_key:
        lines.append(f"[ok] voice transcription: configured ({config.openai_transcription_model} via {config.openai_base_url})")
    else:
        lines.append("[note] voice transcription: disabled (set openai_api_key only if you want Telegram audio/voice input)")

    return lines


def _repo_checks(config: TurnmuxConfig, repo_path: Path) -> list[str]:
    lines: list[str] = []
    try:
        normalized_repo = validate_repo_path(repo_path, config.allowed_roots)
    except Exception as exc:
        lines.append(f"[error] repo path is not launchable: {exc}")
        return lines

    lines.append(f"[ok] repo path: {normalized_repo}")
    registry = ProviderRegistry(config)
    for provider in registry.available_providers():
        trust_state = "ok" if is_provider_trusted(provider, normalized_repo) else "warn"
        trust_suffix = "trusted" if trust_state == "ok" else "not trusted yet"
        lines.append(f"[{trust_state}] {provider.value} workspace trust: {trust_suffix}")
        if provider == ProviderName.CLAUDE:
            prompt_state = "ok" if is_claude_skip_dangerous_prompt_enabled() else "warn"
            prompt_suffix = "enabled" if prompt_state == "ok" else "not enabled yet"
            lines.append(f"[{prompt_state}] claude dangerous-mode prompt skip: {prompt_suffix}")
        try:
            count = len(registry.get(provider).list_resumable_sessions(normalized_repo, limit=5))
        except Exception as exc:
            lines.append(f"[warn] {provider.value} resume discovery failed for repo: {exc}")
            continue
        lines.append(f"[ok] {provider.value} resumable sessions found for repo: {count}")
    return lines


def _service_checks(runtime_paths: RuntimePaths) -> list[str]:
    if sys.platform != "darwin":
        return []

    status = read_launch_agent_status(runtime_paths)
    health = evaluate_launch_agent_health(status, expected_runtime_home=runtime_paths.home)
    lines = [
        f"[{'ok' if status.installed else 'warn'}] launchd plist installed: {'yes' if status.installed else 'no'}",
        f"[{'ok' if status.loaded else 'warn'}] launchd service loaded: {'yes' if status.loaded else 'no'}",
        f"[{health.level}] launchd service health: {health.summary}",
    ]
    if status.runtime_home is not None:
        runtime_home_level = "ok" if status.runtime_home == runtime_paths.home else "warn"
        lines.append(f"[{runtime_home_level}] launchd runtime home: {status.runtime_home}")
    if status.config_path is not None:
        lines.append(f"[ok] launchd config path: {status.config_path}")
    if status.heartbeat:
        heartbeat_line = f"runtime heartbeat: {status.heartbeat.get('status', '-')}, last={status.heartbeat.get('last_heartbeat_at', '-')}"
        heartbeat_age = heartbeat_age_seconds(status.heartbeat)
        heartbeat_level = "ok"
        if heartbeat_age is not None:
            heartbeat_line = f"{heartbeat_line}, age={_format_age(heartbeat_age)}"
            if heartbeat_age > HEARTBEAT_STALE_AFTER_SECONDS:
                heartbeat_level = "error"
        else:
            heartbeat_level = "warn"
        heartbeat_status = str(status.heartbeat.get("status", "-"))
        if heartbeat_status in {"starting", "stopping"} and heartbeat_level == "ok":
            heartbeat_level = "warn"
        elif heartbeat_status != "running":
            heartbeat_level = "error"
        lines.append(f"[{heartbeat_level}] {heartbeat_line}")
    else:
        lines.append("[warn] runtime heartbeat: missing")
    for hint in health.hints:
        lines.append(f"[hint] {hint}")
    return lines


def _command_checks(field_name: str, command: tuple[str, ...]) -> list[str]:
    resolved = _resolve_command(command[0])
    if resolved:
        return [f"[ok] {field_name}: {resolved}"]
    return [f"[error] {field_name} executable not found: {command[0]}"]


def _looks_like_placeholder_token(value: str) -> bool:
    if value == TOKEN_PLACEHOLDER:
        return True
    return re.fullmatch(r"\d+:[A-Za-z0-9_-]{20,}", value) is None


def _recommended_allowed_root(working_dir: Path | None) -> Path:
    candidate = (working_dir or Path.cwd()).expanduser().resolve(strict=False)
    git_root = _discover_git_root(candidate)
    if git_root is not None:
        return git_root
    if candidate.is_dir():
        return candidate
    if candidate.parent.is_dir():
        return candidate.parent
    return Path.home()


def _discover_git_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    while True:
        if (current / ".git").exists():
            return current
        if current == current.parent:
            return None
        current = current.parent


def _detect_binary(name: str) -> str | None:
    fallback_paths: tuple[Path, ...] = ()
    env_var = None
    if name == "opencode":
        fallback_paths = (Path.home() / ".opencode" / "bin" / "opencode",)
    elif name == "tmux":
        env_var = "TURNMUX_TMUX_BINARY"
    return resolve_binary(name, env_var=env_var, fallback_paths=fallback_paths)


def _resolve_command(binary: str) -> str | None:
    path = Path(binary).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve(strict=False))
    return _detect_binary(binary)


def _command_output(command: list[str]) -> str | None:
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    output = process.stdout.strip() or process.stderr.strip()
    return output or None


def _probe_tmux(binary: str) -> str | None:
    process = subprocess.run(
        [binary, "display-message", "-p", "#{version}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode == 0:
        return None
    return process.stderr.strip() or process.stdout.strip() or "tmux control probe failed"


def _has_claude_noninteractive_permissions(command: tuple[str, ...]) -> bool:
    if "--dangerously-skip-permissions" in command:
        return True
    return _command_has_pair(command, "--permission-mode", "bypassPermissions")


def _command_has_pair(command: tuple[str, ...], flag: str, value: str) -> bool:
    for index, item in enumerate(command[:-1]):
        if item == flag and command[index + 1] == value:
            return True
    return False


def _format_age(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _render_command_block(
    provider_name: str,
    field_name: str,
    command: list[str],
    *,
    enabled: bool,
    optional: bool = False,
    trailing_lines: list[str] | None = None,
) -> str:
    prefix = "" if enabled else "# "
    header = f"# {provider_name} provider."
    if optional:
        header = f"# Optional: {provider_name} provider."
    if not enabled:
        header = f"{header} Uncomment if you want to expose {provider_name}."
    lines = [header, f"{prefix}{field_name} = ["]
    for token in command:
        lines.append(f'{prefix}  "{token}",')
    lines.append(f"{prefix}]")
    for trailing in trailing_lines or ():
        lines.append(f"{prefix}{trailing}")
    return "\n".join(lines)
