from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUNTIME_HOME = Path.home() / ".turnmux"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    home: Path
    config_path: Path
    logs_dir: Path
    log_path: Path
    heartbeat_path: Path
    service_stdout_path: Path
    service_stderr_path: Path
    state_db_path: Path


def initialize_runtime_home(base_dir: Path | None = None) -> RuntimePaths:
    home = ensure_private_directory(base_dir or DEFAULT_RUNTIME_HOME)
    logs_dir = ensure_private_directory(home / "logs")

    return RuntimePaths(
        home=home,
        config_path=home / "config.toml",
        logs_dir=logs_dir,
        log_path=logs_dir / "turnmux.log",
        heartbeat_path=home / "heartbeat.json",
        service_stdout_path=logs_dir / "launchd.stdout.log",
        service_stderr_path=logs_dir / "launchd.stderr.log",
        state_db_path=home / "state.db",
    )


def ensure_private_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)
    _maybe_chmod(resolved, PRIVATE_DIR_MODE)
    return resolved


def ensure_private_file(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    ensure_private_directory(resolved.parent)
    if not resolved.exists():
        resolved.touch()
    _maybe_chmod(resolved, PRIVATE_FILE_MODE)
    return resolved


def set_private_file_permissions(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.exists():
        return
    _maybe_chmod(resolved, PRIVATE_FILE_MODE)


def _maybe_chmod(path: Path, mode: int) -> None:
    if os.name != "posix":
        return
    try:
        current_mode = path.stat().st_mode & 0o777
    except OSError:
        return
    if current_mode == mode:
        return
    try:
        os.chmod(path, mode)
    except OSError:
        return
