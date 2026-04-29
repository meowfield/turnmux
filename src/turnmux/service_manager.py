from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import time

from .runtime.binaries import build_runtime_path
from .runtime.home import RuntimePaths
from .runtime.lifecycle import read_heartbeat


DEFAULT_SERVICE_LABEL = "io.turnmux.bot"
HEARTBEAT_STALE_AFTER_SECONDS = 90.0
_SEVERITY_ORDER = {"ok": 0, "warn": 1, "error": 2}
SERVICE_ENV_PASSTHROUGH = ("TURNMUX_TMUX_BINARY",)


@dataclass(frozen=True, slots=True)
class LaunchAgentSpec:
    label: str
    plist_path: Path
    program_arguments: tuple[str, ...]
    working_directory: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True, slots=True)
class LaunchAgentStatus:
    label: str
    plist_path: Path
    installed: bool
    loaded: bool
    pid: int | None
    last_exit_status: int | None
    heartbeat: dict[str, object] | None
    runtime_home: Path | None = None
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class LaunchAgentHealth:
    level: str
    summary: str
    details: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()


def build_launch_agent_spec(
    runtime_paths: RuntimePaths,
    *,
    config_path: Path,
    label: str = DEFAULT_SERVICE_LABEL,
    python_executable: Path | None = None,
    working_directory: Path | None = None,
) -> LaunchAgentSpec:
    raw_executable = (python_executable or Path(sys.executable)).expanduser()
    executable = raw_executable if raw_executable.is_absolute() else (Path.cwd() / raw_executable)
    cwd = (working_directory or Path.cwd()).expanduser().resolve(strict=False)
    plist_path = launch_agent_path(label)
    return LaunchAgentSpec(
        label=label,
        plist_path=plist_path,
        program_arguments=(
            str(executable),
            "-m",
            "turnmux",
            "--runtime-home",
            str(runtime_paths.home),
            "--config",
            str(config_path.expanduser().resolve(strict=False)),
            "run",
        ),
        working_directory=cwd,
        stdout_path=runtime_paths.service_stdout_path,
        stderr_path=runtime_paths.service_stderr_path,
    )


def render_launch_agent_plist(spec: LaunchAgentSpec) -> bytes:
    existing_environment = _read_launch_agent_environment(spec.plist_path)
    environment = {
        "PATH": build_runtime_path(),
        "PYTHONUNBUFFERED": "1",
    }
    for name in SERVICE_ENV_PASSTHROUGH:
        value = os.environ.get(name) or existing_environment.get(name)
        if value:
            environment[name] = value

    payload = {
        "Label": spec.label,
        "ProgramArguments": list(spec.program_arguments),
        "WorkingDirectory": str(spec.working_directory),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(spec.stdout_path),
        "StandardErrorPath": str(spec.stderr_path),
        "EnvironmentVariables": environment,
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def _read_launch_agent_environment(plist_path: Path) -> dict[str, str]:
    try:
        payload = plistlib.loads(plist_path.read_bytes())
    except (FileNotFoundError, plistlib.InvalidFileException, OSError):
        return {}

    environment = payload.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return {}
    return {str(key): value for key, value in environment.items() if isinstance(value, str)}


def install_launch_agent(
    runtime_paths: RuntimePaths,
    *,
    config_path: Path,
    label: str = DEFAULT_SERVICE_LABEL,
    python_executable: Path | None = None,
    working_directory: Path | None = None,
) -> LaunchAgentSpec:
    spec = build_launch_agent_spec(
        runtime_paths,
        config_path=config_path,
        label=label,
        python_executable=python_executable,
        working_directory=working_directory,
    )
    spec.plist_path.parent.mkdir(parents=True, exist_ok=True)
    spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    spec.plist_path.write_bytes(render_launch_agent_plist(spec))
    return spec


def uninstall_launch_agent(label: str = DEFAULT_SERVICE_LABEL) -> Path:
    plist_path = launch_agent_path(label)
    if plist_path.exists():
        plist_path.unlink()
    return plist_path


def start_launch_agent(spec: LaunchAgentSpec) -> None:
    domain = _launchd_domain()
    service_target = f"{domain}/{spec.label}"
    if is_launch_agent_loaded(spec.label):
        _run_launchctl(["bootout", service_target], check=False)
        _wait_for_launch_agent_state(spec.label, loaded=False)
    _bootstrap_launch_agent(spec.plist_path, domain=domain)
    _run_launchctl(["kickstart", "-k", service_target])


def stop_launch_agent(label: str = DEFAULT_SERVICE_LABEL) -> None:
    domain = _launchd_domain()
    _run_launchctl(["bootout", f"{domain}/{label}"], check=False)
    _wait_for_launch_agent_state(label, loaded=False)


def restart_launch_agent(spec: LaunchAgentSpec) -> None:
    stop_launch_agent(spec.label)
    start_launch_agent(spec)


def is_launch_agent_loaded(label: str = DEFAULT_SERVICE_LABEL) -> bool:
    domain = _launchd_domain()
    process = _run_launchctl(["print", f"{domain}/{label}"], check=False)
    return process.returncode == 0


def read_launch_agent_status(runtime_paths: RuntimePaths, *, label: str = DEFAULT_SERVICE_LABEL) -> LaunchAgentStatus:
    plist_path = launch_agent_path(label)
    domain = _launchd_domain()
    installed = plist_path.exists()
    service_runtime_home, service_config_path = _read_launch_agent_runtime_context(plist_path)
    process = _run_launchctl(["print", f"{domain}/{label}"], check=False)
    loaded = process.returncode == 0
    pid = None
    last_exit_status = None
    if loaded:
        output = process.stdout or process.stderr
        pid_match = re.search(r"\bpid = (\d+)", output)
        if pid_match:
            pid = int(pid_match.group(1))
        exit_match = re.search(r"\blast exit code = (\d+)", output)
        if exit_match:
            last_exit_status = int(exit_match.group(1))

    return LaunchAgentStatus(
        label=label,
        plist_path=plist_path,
        installed=installed,
        loaded=loaded,
        pid=pid,
        last_exit_status=last_exit_status,
        heartbeat=read_heartbeat((service_runtime_home or runtime_paths.home) / "heartbeat.json"),
        runtime_home=service_runtime_home,
        config_path=service_config_path,
    )


def format_launch_agent_status(status: LaunchAgentStatus) -> str:
    health = evaluate_launch_agent_health(status)
    lines = [
        f"health: {health.level}",
        f"summary: {health.summary}",
        f"label: {status.label}",
        f"plist: {status.plist_path}",
        f"installed: {'yes' if status.installed else 'no'}",
        f"loaded: {'yes' if status.loaded else 'no'}",
        f"pid: {status.pid if status.pid is not None else '-'}",
        f"last exit: {status.last_exit_status if status.last_exit_status is not None else '-'}",
    ]
    if status.runtime_home is not None:
        lines.append(f"runtime home: {status.runtime_home}")
        lines.append(f"logs dir: {status.runtime_home / 'logs'}")
    if status.config_path is not None:
        lines.append(f"config: {status.config_path}")
    if status.heartbeat:
        lines.append(f"heartbeat status: {status.heartbeat.get('status', '-')}")
        lines.append(f"heartbeat at: {status.heartbeat.get('last_heartbeat_at', '-')}")
        lines.append(f"heartbeat pid: {status.heartbeat.get('pid', '-')}")
        heartbeat_age = heartbeat_age_seconds(status.heartbeat)
        if heartbeat_age is not None:
            lines.append(f"heartbeat age: {_format_age(heartbeat_age)}")
        note = status.heartbeat.get("note")
        if isinstance(note, str) and note.strip():
            lines.append(f"heartbeat note: {note.strip()}")
    else:
        lines.append("heartbeat status: missing")
    for detail in health.details:
        if detail == health.summary:
            continue
        lines.append(f"detail: {detail}")
    for hint in health.hints:
        lines.append(f"hint: {hint}")
    return "\n".join(lines)


def evaluate_launch_agent_health(
    status: LaunchAgentStatus,
    *,
    expected_runtime_home: Path | None = None,
    stale_after_seconds: float = HEARTBEAT_STALE_AFTER_SECONDS,
) -> LaunchAgentHealth:
    level = "ok"
    summary = "healthy"
    details: list[str] = []
    hints: list[str] = []

    def record(new_level: str, detail: str, hint: str | None = None) -> None:
        nonlocal level, summary
        if _SEVERITY_ORDER[new_level] > _SEVERITY_ORDER[level]:
            level = new_level
            summary = detail
        if detail not in details:
            details.append(detail)
        if hint and hint not in hints:
            hints.append(hint)

    if not status.installed and not status.loaded:
        record("warn", "launchd service is not installed.", "Run `turnmux service install` if you want TurnMux managed by launchd.")
        return LaunchAgentHealth(level=level, summary=summary, details=tuple(details), hints=tuple(hints))

    if status.loaded and not status.installed:
        record("error", "launchd reports the service as loaded but the plist file is missing.", "Run `turnmux service restart` to rewrite the plist cleanly.")

    if status.installed and not status.loaded:
        record("error", "launchd service is not loaded.", "Run `turnmux service start` or inspect the launchd stderr log.")

    if status.last_exit_status not in (None, 0):
        record("warn", f"last exit status: {status.last_exit_status}", "Inspect the launchd stderr log for the startup failure or crash.")

    expected_home = expected_runtime_home.expanduser().resolve(strict=False) if expected_runtime_home is not None else None
    if status.runtime_home is not None and expected_home is not None and status.runtime_home != expected_home:
        record(
            "warn",
            f"service runtime home differs from this invocation: {status.runtime_home}",
            "Reinstall or restart the service from the runtime home you want launchd to use.",
        )

    heartbeat = status.heartbeat
    if status.loaded:
        if heartbeat is None:
            record(
                "error",
                "launchd is loaded but heartbeat.json is missing.",
                "Inspect the launchd stderr log and run `turnmux service restart`.",
            )
        else:
            heartbeat_age = heartbeat_age_seconds(heartbeat)
            if heartbeat_age is None:
                record("warn", "heartbeat timestamp is missing or invalid.", "Check whether TurnMux can still update heartbeat.json.")
            elif heartbeat_age > stale_after_seconds:
                record(
                    "error",
                    f"heartbeat is stale ({_format_age(heartbeat_age)} old).",
                    "Inspect the launchd stderr log and run `turnmux service restart`.",
                )

            heartbeat_status = str(heartbeat.get("status", "-"))
            if heartbeat_status == "running":
                pass
            elif heartbeat_status in {"starting", "stopping"}:
                record("warn", f"runtime heartbeat status is `{heartbeat_status}`.", "Wait a few seconds and run `turnmux service status` again.")
            else:
                record(
                    "error",
                    f"runtime heartbeat status is `{heartbeat_status}`.",
                    "Inspect the launchd stderr log and run `turnmux service restart`.",
                )

            heartbeat_pid = heartbeat.get("pid")
            if status.pid is not None and isinstance(heartbeat_pid, int) and heartbeat_pid != status.pid:
                record(
                    "warn",
                    f"launchd pid {status.pid} does not match heartbeat pid {heartbeat_pid}.",
                    "Inspect the launchd stderr log; the process may have restarted without refreshing heartbeat yet.",
                )
    elif heartbeat is not None:
        record("warn", "heartbeat exists but launchd does not report the service as loaded.", "Inspect the launchd stderr log before restarting the service.")

    return LaunchAgentHealth(level=level, summary=summary, details=tuple(details), hints=tuple(hints))


def launch_agent_path(label: str = DEFAULT_SERVICE_LABEL) -> Path:
    return (Path.home() / "Library" / "LaunchAgents" / f"{label}.plist").expanduser().resolve(strict=False)


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _run_launchctl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "launchctl command failed"
        raise RuntimeError(message)
    return process


def heartbeat_age_seconds(
    heartbeat: dict[str, object],
    *,
    now: datetime | None = None,
) -> float | None:
    heartbeat_at = _parse_timestamp(heartbeat.get("last_heartbeat_at"))
    if heartbeat_at is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - heartbeat_at).total_seconds())


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_age(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _read_launch_agent_runtime_context(plist_path: Path) -> tuple[Path | None, Path | None]:
    if not plist_path.exists():
        return None, None
    try:
        payload = plistlib.loads(plist_path.read_bytes())
    except (OSError, plistlib.InvalidFileException):
        return None, None
    program_arguments = payload.get("ProgramArguments")
    if not isinstance(program_arguments, list):
        return None, None
    return _argument_path(program_arguments, "--runtime-home"), _argument_path(program_arguments, "--config")


def _argument_path(arguments: list[object], flag: str) -> Path | None:
    for index, item in enumerate(arguments[:-1]):
        if item == flag and isinstance(arguments[index + 1], str) and arguments[index + 1].strip():
            return Path(arguments[index + 1]).expanduser().resolve(strict=False)
    return None


def _bootstrap_launch_agent(plist_path: Path, *, domain: str, retries: int = 5) -> None:
    for attempt in range(retries):
        process = _run_launchctl(["bootstrap", domain, str(plist_path)], check=False)
        if process.returncode == 0:
            return

        message = process.stderr.strip() or process.stdout.strip() or "launchctl bootstrap failed"
        is_transient_io_error = "Input/output error" in message
        if not is_transient_io_error or attempt == retries - 1:
            raise RuntimeError(message)
        time.sleep(0.2 * (attempt + 1))


def _wait_for_launch_agent_state(
    label: str,
    *,
    loaded: bool,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_launch_agent_loaded(label) == loaded:
            return
        time.sleep(0.1)
