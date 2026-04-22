from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from ..runtime.home import DEFAULT_RUNTIME_HOME, ensure_private_directory, set_private_file_permissions

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]


CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
LEGACY_CCBOT_SESSION_MAP_PATH = Path.home() / ".ccbot" / "session_map.json"
HOOK_COMMAND_MARKER = "turnmux hook claude-session-start"
SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass(frozen=True, slots=True)
class ClaudeSessionMapEntry:
    session_id: str
    cwd: Path | None
    window_name: str | None
    source_path: Path


def default_claude_session_map_path(runtime_home: Path | None = None) -> Path:
    home = (runtime_home or DEFAULT_RUNTIME_HOME).expanduser().resolve(strict=False)
    return home / "claude_session_map.json"


def ensure_claude_session_start_hook(
    *,
    settings_path: Path | None = None,
    executable_path: str | None = None,
    runtime_home: Path | None = None,
) -> None:
    path = (settings_path or CLAUDE_SETTINGS_PATH).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_json_object_for_update(path)

    hooks = payload.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        payload["hooks"] = hooks

    session_start = hooks.setdefault("SessionStart", [])
    if not isinstance(session_start, list):
        session_start = []
        hooks["SessionStart"] = session_start

    expected_command = _build_session_start_hook_command(
        executable_path=executable_path,
        runtime_home=runtime_home,
    )
    for hook in _iter_session_start_hooks(session_start):
        command = _coerce_non_empty_str(hook.get("command"))
        if command is None or not _session_start_hook_matches_runtime(command, runtime_home):
            continue
        if command == expected_command:
            return
        hook["type"] = "command"
        hook["command"] = expected_command
        hook.setdefault("timeout", 5)
        _write_json_atomic(path, payload)
        return

    session_start.append({"hooks": [{"type": "command", "command": expected_command, "timeout": 5}]})
    _write_json_atomic(path, payload)


def is_claude_session_start_hook_installed(
    *,
    settings_path: Path | None = None,
    runtime_home: Path | None = None,
) -> bool:
    payload = _load_json_object((settings_path or CLAUDE_SETTINGS_PATH).expanduser().resolve(strict=False))
    if payload is None:
        return False
    return _is_session_start_hook_installed_payload(payload, runtime_home=runtime_home)


def process_claude_session_start_hook(
    *,
    runtime_home: Path | None = None,
    payload: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    tmux_display_message=None,
) -> int:
    data = payload if payload is not None else _load_hook_payload_from_stdin()
    if not isinstance(data, Mapping):
        return 0

    event = _coerce_non_empty_str(data.get("hook_event_name"))
    session_id = _coerce_non_empty_str(data.get("session_id"))
    cwd = _coerce_non_empty_str(data.get("cwd"))

    if event != "SessionStart" or session_id is None:
        return 0
    if SESSION_ID_RE.match(session_id) is None:
        return 0
    if cwd and not os.path.isabs(cwd):
        return 0

    environment = env or os.environ
    pane_id = environment.get("TMUX_PANE")
    if not pane_id:
        return 0

    tmux_query = tmux_display_message or _tmux_display_message
    raw_display = tmux_query(pane_id).strip()
    parts = raw_display.split(":", 2)
    if len(parts) != 3:
        return 0

    tmux_session_name, window_id, window_name = parts
    map_path = default_claude_session_map_path(runtime_home)
    ensure_private_directory(map_path.parent)

    entry = {
        "session_id": session_id,
        "cwd": cwd or "",
        "window_name": window_name,
    }
    _update_session_map(map_path, f"{tmux_session_name}:{window_id}", entry)
    return 0


def find_claude_session_map_entry(
    tmux_session_name: str,
    tmux_window_id: str,
    *,
    runtime_home: Path | None = None,
) -> ClaudeSessionMapEntry | None:
    key = f"{tmux_session_name}:{tmux_window_id}"
    for path in _candidate_session_map_paths(runtime_home):
        payload = _load_json_object(path)
        if payload is None:
            continue
        raw_entry = payload.get(key)
        if not isinstance(raw_entry, dict):
            continue
        session_id = _coerce_non_empty_str(raw_entry.get("session_id"))
        if session_id is None:
            continue
        cwd_text = _coerce_non_empty_str(raw_entry.get("cwd"))
        window_name = _coerce_non_empty_str(raw_entry.get("window_name"))
        return ClaudeSessionMapEntry(
            session_id=session_id,
            cwd=Path(cwd_text).expanduser().resolve(strict=False) if cwd_text else None,
            window_name=window_name,
            source_path=path,
        )
    return None


def _candidate_session_map_paths(runtime_home: Path | None) -> tuple[Path, ...]:
    turnmux_path = default_claude_session_map_path(runtime_home)
    if LEGACY_CCBOT_SESSION_MAP_PATH == turnmux_path:
        return (turnmux_path,)
    return (turnmux_path, LEGACY_CCBOT_SESSION_MAP_PATH)


def _load_hook_payload_from_stdin() -> Mapping[str, Any] | None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _tmux_display_message(pane_id: str) -> str:
    result = subprocess.run(
        [
            "tmux",
            "display-message",
            "-t",
            pane_id,
            "-p",
            "#{session_name}:#{window_id}:#{window_name}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _find_turnmux_executable() -> str:
    turnmux_path = shutil.which("turnmux")
    if turnmux_path:
        return turnmux_path
    python_dir = Path(sys.executable).parent
    turnmux_in_env = python_dir / "turnmux"
    if turnmux_in_env.exists():
        return str(turnmux_in_env)
    return "turnmux"


def _build_session_start_hook_command(
    *,
    executable_path: str | None = None,
    runtime_home: Path | None = None,
) -> str:
    command = [executable_path or _find_turnmux_executable(), "hook", "claude-session-start"]
    normalized_runtime_home = _normalize_runtime_home(runtime_home)
    if normalized_runtime_home != _normalize_runtime_home(DEFAULT_RUNTIME_HOME):
        command.extend(["--runtime-home", str(normalized_runtime_home)])
    return shlex.join(command)


def _is_session_start_hook_installed_payload(
    payload: Mapping[str, Any],
    *,
    runtime_home: Path | None = None,
) -> bool:
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False
    session_start = hooks.get("SessionStart")
    if not isinstance(session_start, list):
        return False

    for hook in _iter_session_start_hooks(session_start):
        command = _coerce_non_empty_str(hook.get("command"))
        if command and _session_start_hook_matches_runtime(command, runtime_home):
            return True
    return False


def _iter_session_start_hooks(session_start: list[Any]):
    for entry in session_start:
        if not isinstance(entry, dict):
            continue
        nested_hooks = entry.get("hooks")
        if not isinstance(nested_hooks, list):
            continue
        for hook in nested_hooks:
            if isinstance(hook, dict):
                yield hook


def _session_start_hook_matches_runtime(command: str, runtime_home: Path | None) -> bool:
    if not _looks_like_session_start_hook_command(command):
        return False
    if runtime_home is None:
        return True

    expected_runtime_home = _normalize_runtime_home(runtime_home)
    command_runtime_home = _extract_runtime_home(command)
    default_runtime_home = _normalize_runtime_home(DEFAULT_RUNTIME_HOME)
    if expected_runtime_home == default_runtime_home:
        return command_runtime_home is None or command_runtime_home == default_runtime_home
    return command_runtime_home == expected_runtime_home


def _looks_like_session_start_hook_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return HOOK_COMMAND_MARKER in command
    for index, token in enumerate(tokens[:-1]):
        if token == "hook" and tokens[index + 1] == "claude-session-start":
            return True
    return False


def _extract_runtime_home(command: str) -> Path | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for index, token in enumerate(tokens[:-1]):
        if token == "--runtime-home":
            return Path(tokens[index + 1]).expanduser().resolve(strict=False)
    return None


def _normalize_runtime_home(runtime_home: Path | None) -> Path:
    return (runtime_home or DEFAULT_RUNTIME_HOME).expanduser().resolve(strict=False)


def _update_session_map(path: Path, key: str, entry: dict[str, str]) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    ensure_private_directory(lock_path.parent)

    with lock_path.open("w", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        payload = _load_json_object(path) or {}
        payload[key] = entry
        _write_json_atomic(path, payload)
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_private_directory(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temp_path = Path(handle.name)
    set_private_file_permissions(temp_path)
    temp_path.replace(path)
    set_private_file_permissions(path)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object_for_update(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    return payload if payload is not None else {}


def _coerce_non_empty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
