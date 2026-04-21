from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Mapping, Sequence
from uuid import uuid4

from .binaries import build_runtime_path, resolve_binary


class TmuxError(RuntimeError):
    """Raised when a tmux command fails or runtime assumptions are violated."""


@dataclass(frozen=True, slots=True)
class TmuxWindow:
    window_id: str
    name: str
    current_path: Path
    active: bool


INTERNAL_MAIN_WINDOW = "__turnmux__"
SENSITIVE_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USERS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


def ensure_session(session_name: str) -> None:
    probe = _run_tmux(["has-session", "-t", session_name], check=False)
    if probe.returncode == 0:
        _scrub_session_env(session_name)
        return

    _run_tmux(["new-session", "-d", "-s", session_name, "-n", INTERNAL_MAIN_WINDOW])
    _scrub_session_env(session_name)


def list_windows(session_name: str, *, include_internal: bool = False) -> list[TmuxWindow]:
    output = _run_tmux(
        [
            "list-windows",
            "-t",
            session_name,
            "-F",
            "#{window_id}\t#{window_name}\t#{pane_current_path}\t#{window_active}",
        ]
    ).stdout

    windows: list[TmuxWindow] = []
    for line in output.strip().splitlines():
        window_id, name, current_path, active_flag = line.split("\t", maxsplit=3)
        if not include_internal and name == INTERNAL_MAIN_WINDOW:
            continue
        windows.append(
            TmuxWindow(
                window_id=window_id,
                name=name,
                current_path=Path(current_path),
                active=active_flag == "1",
            )
        )
    return windows


def create_window(
    session_name: str,
    repo_path: Path,
    *,
    window_name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> TmuxWindow:
    normalized_repo_path = repo_path.expanduser().resolve(strict=True)
    if not normalized_repo_path.is_dir():
        raise TmuxError(f"Repo path is not a directory: {normalized_repo_path}")

    ensure_session(session_name)

    command = [
        "new-window",
        "-P",
        "-F",
        "#{window_id}\t#{window_name}\t#{pane_current_path}\t#{window_active}",
        "-t",
        session_name,
        "-c",
        str(normalized_repo_path),
    ]
    if window_name:
        command.extend(["-n", window_name])
    if env:
        for key, value in env.items():
            command.extend(["-e", f"{key}={value}"])

    output = _run_tmux(command).stdout.strip()
    window_id, name, current_path, active_flag = output.split("\t", maxsplit=3)
    return TmuxWindow(
        window_id=window_id,
        name=name,
        current_path=Path(current_path),
        active=active_flag == "1",
    )


def launch_command(window_target: str, command: Sequence[str]) -> None:
    paste_text(window_target, shlex.join(command), enter=True, enter_delay_seconds=0.0)


def paste_text(window_target: str, text: str, *, enter: bool = True, enter_delay_seconds: float = 0.35) -> None:
    pane_target = _active_pane_target(window_target)
    buffer_name = f"turnmux-{uuid4().hex}"
    _run_tmux(["load-buffer", "-b", buffer_name, "-"], input_text=text)
    _run_tmux(["paste-buffer", "-b", buffer_name, "-d", "-t", pane_target])
    if enter:
        if enter_delay_seconds > 0:
            time.sleep(enter_delay_seconds)
        _run_tmux(["send-keys", "-t", pane_target, "Enter"])


def send_interrupt(window_target: str) -> None:
    _run_tmux(["send-keys", "-t", _active_pane_target(window_target), "C-c"])


def send_keys(window_target: str, *keys: str) -> None:
    _run_tmux(["send-keys", "-t", _active_pane_target(window_target), *keys])


def capture_pane(window_target: str, *, history_lines: int = 200) -> str:
    return _run_tmux(
        [
            "capture-pane",
            "-p",
            "-S",
            f"-{history_lines}",
            "-t",
            _active_pane_target(window_target),
        ]
    ).stdout


def window_exists(session_name: str, target: str) -> bool:
    return _run_tmux(["list-windows", "-t", _window_target(session_name, target)], check=False).returncode == 0


def kill_window(session_name: str, target: str) -> None:
    _run_tmux(["kill-window", "-t", _window_target(session_name, target)])


def _window_target(session_name: str, target: str) -> str:
    return target if target.startswith("@") else f"{session_name}:{target}"


def _active_pane_target(window_target: str) -> str:
    return _run_tmux(["display-message", "-p", "-t", window_target, "#{pane_id}"]).stdout.strip()


def _run_tmux(
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    tmux_binary = _tmux_binary()
    try:
        process = subprocess.run(
            [tmux_binary, *args],
            capture_output=True,
            text=True,
            check=False,
            input=input_text,
        )
    except FileNotFoundError as exc:
        raise TmuxError(
            "tmux executable is not available. "
            f"Searched PATH={build_runtime_path()} "
            "and TURNMUX_TMUX_BINARY if set."
        ) from exc

    if check and process.returncode != 0:
        error_message = process.stderr.strip() or process.stdout.strip() or "tmux command failed"
        raise TmuxError(error_message)

    return process


def _scrub_session_env(session_name: str) -> None:
    for var_name in SENSITIVE_ENV_VARS:
        _run_tmux(["set-environment", "-t", session_name, "-u", var_name], check=False)


@lru_cache(maxsize=1)
def _tmux_binary() -> str:
    resolved = resolve_binary("tmux", env_var="TURNMUX_TMUX_BINARY")
    if not resolved:
        raise TmuxError(
            "tmux executable not found. "
            f"Searched PATH={build_runtime_path()} "
            "plus common Homebrew/system locations."
        )
    return resolved


# Arbitrary multiline user text goes through load-buffer/paste-buffer to preserve
# newlines and avoid shell-escaping bugs.
