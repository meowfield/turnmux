from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tomllib

from ..state.models import ProviderName


def ensure_provider_trust(provider: ProviderName, repo_path: Path) -> None:
    normalized_repo = repo_path.expanduser().resolve(strict=False)
    if provider == ProviderName.CLAUDE:
        ensure_claude_skip_dangerous_prompt()
        ensure_claude_project_trusted(normalized_repo)
        return
    if provider == ProviderName.CODEX:
        ensure_codex_project_trusted(normalized_repo)
        return
    if provider == ProviderName.OPENCODE:
        return
    raise ValueError(f"Unsupported provider: {provider}")


def is_provider_trusted(provider: ProviderName, repo_path: Path) -> bool:
    normalized_repo = repo_path.expanduser().resolve(strict=False)
    if provider == ProviderName.CLAUDE:
        return is_claude_project_trusted(normalized_repo)
    if provider == ProviderName.CODEX:
        return is_codex_project_trusted(normalized_repo)
    if provider == ProviderName.OPENCODE:
        return True
    return False


def ensure_claude_skip_dangerous_prompt(*, settings_path: Path | None = None) -> None:
    path = (settings_path or (Path.home() / ".claude" / "settings.json")).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_json_object_for_update(path)

    permissions = payload.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        payload["permissions"] = permissions

    if permissions.get("skipDangerousModePermissionPrompt") is True:
        return

    permissions["skipDangerousModePermissionPrompt"] = True
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_claude_skip_dangerous_prompt_enabled(*, settings_path: Path | None = None) -> bool:
    path = (settings_path or (Path.home() / ".claude" / "settings.json")).expanduser().resolve(strict=False)
    payload = _load_json_object(path)
    if payload is None:
        return False
    permissions = payload.get("permissions")
    return isinstance(permissions, dict) and permissions.get("skipDangerousModePermissionPrompt") is True


def ensure_claude_project_trusted(repo_path: Path, *, state_path: Path | None = None) -> None:
    path = (state_path or (Path.home() / ".claude.json")).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_json_object_for_update(path)

    projects = payload.setdefault("projects", {})
    if not isinstance(projects, dict):
        projects = {}
        payload["projects"] = projects

    entry = projects.get(str(repo_path))
    if not isinstance(entry, dict):
        entry = {}
        projects[str(repo_path)] = entry

    if entry.get("hasTrustDialogAccepted") is True:
        return

    entry["hasTrustDialogAccepted"] = True
    entry.setdefault("allowedTools", [])
    entry.setdefault("mcpContextUris", [])
    entry.setdefault("mcpServers", {})
    entry.setdefault("enabledMcpjsonServers", [])
    entry.setdefault("disabledMcpjsonServers", [])
    entry.setdefault("projectOnboardingSeenCount", 0)
    entry.setdefault("hasClaudeMdExternalIncludesApproved", False)
    entry.setdefault("hasClaudeMdExternalIncludesWarningShown", False)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_claude_project_trusted(repo_path: Path, *, state_path: Path | None = None) -> bool:
    path = (state_path or (Path.home() / ".claude.json")).expanduser().resolve(strict=False)
    payload = _load_json_object(path)
    if payload is None:
        return False
    projects = payload.get("projects")
    if not isinstance(projects, dict):
        return False
    entry = projects.get(str(repo_path))
    return isinstance(entry, dict) and entry.get("hasTrustDialogAccepted") is True


def ensure_codex_project_trusted(repo_path: Path, *, config_path: Path | None = None) -> None:
    path = (config_path or (Path.home() / ".codex" / "config.toml")).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_codex_project_block(repo_path), encoding="utf-8")
        return

    current = path.read_text(encoding="utf-8")
    if _load_toml_mapping(path) is None:
        _backup_invalid_file(path, current)
        path.write_text(_codex_project_block(repo_path), encoding="utf-8")
        return
    if is_codex_project_trusted(repo_path, config_path=path):
        return

    escaped_repo = _toml_basic_string(str(repo_path))
    section_header = f'[projects."{escaped_repo}"]'
    pattern = re.compile(
        rf"(?ms)^(?P<header>\[projects\.\"{re.escape(escaped_repo)}\"\]\n)(?P<body>.*?)(?=^\[|\Z)"
    )
    match = pattern.search(current)
    if not match:
        suffix = "" if current.endswith("\n") or not current else "\n"
        path.write_text(current + suffix + _codex_project_block(repo_path), encoding="utf-8")
        return

    body = match.group("body")
    if re.search(r'^trust_level\s*=', body, flags=re.MULTILINE):
        updated_body = re.sub(r'^trust_level\s*=.*$', 'trust_level = "trusted"', body, count=1, flags=re.MULTILINE)
    else:
        updated_body = 'trust_level = "trusted"\n' + body
    path.write_text(current[: match.start()] + match.group("header") + updated_body + current[match.end() :], encoding="utf-8")


def is_codex_project_trusted(repo_path: Path, *, config_path: Path | None = None) -> bool:
    path = (config_path or (Path.home() / ".codex" / "config.toml")).expanduser().resolve(strict=False)
    payload = _load_toml_mapping(path)
    if payload is None:
        return False
    projects = payload.get("projects")
    if not isinstance(projects, dict):
        return False
    entry = projects.get(str(repo_path))
    return isinstance(entry, dict) and entry.get("trust_level") == "trusted"


def _codex_project_block(repo_path: Path) -> str:
    return f'[projects."{_toml_basic_string(str(repo_path))}"]\ntrust_level = "trusted"\n'


def _toml_basic_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object_for_update(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        _backup_invalid_file(path, raw)
        return {}
    if not isinstance(payload, dict):
        _backup_invalid_file(path, raw)
        return {}
    return payload


def _load_toml_mapping(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _backup_invalid_file(path: Path, raw: str) -> Path:
    # Preserve the original user-owned file before we rebuild the minimal
    # structure TurnMux needs for a non-interactive launch path.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.turnmux-invalid-{timestamp}.bak")
    suffix = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.turnmux-invalid-{timestamp}-{suffix}.bak")
        suffix += 1
    backup_path.write_text(raw, encoding="utf-8")
    return backup_path
