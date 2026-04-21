from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Iterable


COMMON_BINARY_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)


def build_runtime_path(*, extra_entries: Iterable[str | Path] = ()) -> str:
    entries: list[str] = []
    seen: set[str] = set()

    def add(entry: str | Path | None) -> None:
        if entry is None:
            return
        value = str(entry).strip()
        if not value:
            return
        if value not in seen:
            entries.append(value)
            seen.add(value)

    for entry in extra_entries:
        add(entry)

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        add(entry)

    for entry in COMMON_BINARY_DIRS:
        add(entry)

    return os.pathsep.join(entries)


def resolve_binary(
    name: str,
    *,
    env_var: str | None = None,
    fallback_paths: Iterable[str | Path] = (),
) -> str | None:
    search_path = build_runtime_path()

    if env_var:
        override = os.environ.get(env_var)
        resolved_override = _resolve_candidate(override, search_path=search_path)
        if resolved_override:
            return resolved_override

    resolved_name = _resolve_candidate(name, search_path=search_path)
    if resolved_name:
        return resolved_name

    for fallback in fallback_paths:
        resolved_fallback = _resolve_candidate(fallback, search_path=search_path)
        if resolved_fallback:
            return resolved_fallback

    return None


def _resolve_candidate(candidate: str | Path | None, *, search_path: str) -> str | None:
    if candidate is None:
        return None

    value = str(candidate).strip()
    if not value:
        return None

    if os.sep in value:
        path = Path(value).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path.resolve(strict=False))
        return None

    resolved = shutil.which(value, path=search_path)
    if resolved:
        return str(Path(resolved).resolve(strict=False))
    return None
