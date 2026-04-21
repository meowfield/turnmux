from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import tomllib
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when the runtime config is missing or invalid."""


class RepoPathValidationError(ValueError):
    """Raised when a requested repo path is outside configured allowed roots."""


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"


@dataclass(frozen=True, slots=True)
class TurnmuxConfig:
    telegram_bot_token: str
    allowed_user_ids: tuple[int, ...]
    allowed_roots: tuple[Path, ...]
    tmux_session_name: str
    claude_command: tuple[str, ...] | None
    codex_command: tuple[str, ...] | None
    opencode_command: tuple[str, ...] | None
    opencode_model: str | None
    config_path: Path
    relay_claude_thinking: bool = False
    openai_api_key: str | None = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_transcription_model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL


def default_config_path() -> Path:
    return Path.home() / ".turnmux" / "config.toml"


def load_config(path: Path | None = None) -> TurnmuxConfig:
    config_path = (path or default_config_path()).expanduser().resolve(strict=False)

    try:
        with config_path.open("rb") as handle:
            raw_config = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in config file {config_path}: {exc}") from exc

    return parse_config(raw_config, source=config_path)


def parse_config(raw_config: Mapping[str, Any], *, source: Path) -> TurnmuxConfig:
    if not isinstance(raw_config, Mapping):
        raise ConfigError("Config root must be a TOML table.")

    config = TurnmuxConfig(
        telegram_bot_token=_require_non_empty_string(raw_config, "telegram_bot_token"),
        allowed_user_ids=_parse_allowed_user_ids(raw_config.get("allowed_user_ids")),
        allowed_roots=_parse_allowed_roots(raw_config.get("allowed_roots")),
        tmux_session_name=_require_non_empty_string(raw_config, "tmux_session_name"),
        claude_command=_parse_optional_command(raw_config.get("claude_command"), field_name="claude_command"),
        codex_command=_parse_optional_command(raw_config.get("codex_command"), field_name="codex_command"),
        opencode_command=_parse_optional_command(raw_config.get("opencode_command"), field_name="opencode_command"),
        opencode_model=_optional_non_empty_string(raw_config.get("opencode_model"), field_name="opencode_model"),
        config_path=source.resolve(strict=False),
        relay_claude_thinking=_optional_bool(
            raw_config.get("relay_claude_thinking"),
            field_name="relay_claude_thinking",
            default=False,
        ),
        openai_api_key=_optional_non_empty_string(
            raw_config.get("openai_api_key", _env_or_none("OPENAI_API_KEY")),
            field_name="openai_api_key",
        ),
        openai_base_url=_string_or_default(
            raw_config.get("openai_base_url", _env_or_none("OPENAI_BASE_URL")),
            field_name="openai_base_url",
            default=DEFAULT_OPENAI_BASE_URL,
        ),
        openai_transcription_model=_string_or_default(
            raw_config.get("openai_transcription_model", _env_or_none("OPENAI_TRANSCRIPTION_MODEL")),
            field_name="openai_transcription_model",
            default=DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        ),
    )
    if not any((config.claude_command, config.codex_command, config.opencode_command)):
        raise ConfigError("Configure at least one provider command: claude_command, codex_command, or opencode_command.")
    return config


def validate_repo_path(repo_path: Path, allowed_roots: Sequence[Path]) -> Path:
    resolved_repo_path = repo_path.expanduser().resolve(strict=True)
    if not resolved_repo_path.is_dir():
        raise RepoPathValidationError(f"Repo path is not a directory: {resolved_repo_path}")

    for allowed_root in allowed_roots:
        normalized_root = allowed_root.expanduser().resolve(strict=True)
        if resolved_repo_path == normalized_root or resolved_repo_path.is_relative_to(normalized_root):
            return resolved_repo_path

    raise RepoPathValidationError(
        f"Repo path '{resolved_repo_path}' is outside configured allowed_roots."
    )


def _require_non_empty_string(raw_config: Mapping[str, Any], field_name: str) -> str:
    value = raw_config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def _optional_non_empty_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string when provided.")
    return value.strip()


def _string_or_default(value: Any, *, field_name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field_name}' must be a non-empty string when provided.")
    return value.strip()


def _optional_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"'{field_name}' must be a boolean when provided.")
    return value


def _parse_allowed_user_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'allowed_user_ids' must be a non-empty array of integers.")

    normalized_ids: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, int):
            raise ConfigError("'allowed_user_ids' must contain only integers.")
        if item in seen:
            continue
        normalized_ids.append(item)
        seen.add(item)
    return tuple(normalized_ids)


def _parse_allowed_roots(value: Any) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("'allowed_roots' must be a non-empty array of absolute directory paths.")

    normalized_roots: list[Path] = []
    seen: set[Path] = set()

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError("'allowed_roots' must contain only non-empty strings.")

        raw_path = Path(item).expanduser()
        if not raw_path.is_absolute():
            raise ConfigError("'allowed_roots' entries must be absolute paths.")

        resolved_path = raw_path.resolve(strict=True)
        if not resolved_path.is_dir():
            raise ConfigError(f"'allowed_roots' entry is not a directory: {resolved_path}")

        if resolved_path in seen:
            continue
        normalized_roots.append(resolved_path)
        seen.add(resolved_path)

    return tuple(normalized_roots)


def _parse_command(value: Any, *, field_name: str) -> tuple[str, ...]:
    tokens: list[str]

    if isinstance(value, str):
        tokens = shlex.split(value)
    elif isinstance(value, list):
        tokens = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(f"'{field_name}' array entries must be non-empty strings.")
            tokens.append(item.strip())
    else:
        raise ConfigError(f"'{field_name}' must be either a string or an array of strings.")

    if not tokens:
        raise ConfigError(f"'{field_name}' must not be empty.")

    return tuple(tokens)


def _parse_optional_command(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _parse_command(value, field_name=field_name)


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()
