from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable
import re

from ..state.models import ProviderName


OPTION_LINE_RE = re.compile(r"^\s*(?P<key>[0-9A-Za-z])(?:\s*[\).\]:-])\s+(?P<label>.+?)\s*$")
BRACKET_OPTION_RE = re.compile(r"^\s*\[(?P<key>[0-9A-Za-z])\]\s*(?P<label>.+?)\s*$")
YES_NO_RE = re.compile(r"\[(?P<yes>[Yy])(?:es)?\s*/\s*(?P<no>[Nn])(?:o)?\]")
CLAUDE_ENTER_ESCAPE_TOP_PATTERNS = (
    re.compile(r"^\s*Do you want to proceed\?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Do you want to make this edit", re.IGNORECASE),
    re.compile(r"^\s*Do you want to create \S", re.IGNORECASE),
    re.compile(r"^\s*Do you want to delete \S", re.IGNORECASE),
    re.compile(r"^\s*Bash command\s*$", re.IGNORECASE),
    re.compile(r"^\s*This command requires approval", re.IGNORECASE),
)
APPROVAL_CONTEXT_PATTERNS = (
    "approval",
    "approve",
    "permission",
    "permissions",
    "allow",
    "deny",
    "dangerous",
    "bypass permissions",
    "run this command",
    "execute this command",
    "continue with",
)
APPROVE_LABEL_PATTERNS = (
    "approve",
    "allow",
    "yes",
    "accept",
    "continue",
    "proceed",
    "run",
)
DENY_LABEL_PATTERNS = (
    "deny",
    "reject",
    "cancel",
    "exit",
    "stop",
    "no",
)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    fingerprint: str
    prompt_text: str
    approve_keys: tuple[str, ...]
    deny_keys: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _OptionChoice:
    key: str
    label: str
    line_index: int


def detect_approval_request(provider: ProviderName, pane_text: str) -> ApprovalRequest | None:
    lines = _clean_lines(pane_text)
    if not lines:
        return None

    detectors = (
        _detect_claude_bypass_warning,
        _detect_claude_enter_escape_prompt,
        _detect_numbered_or_lettered_choices,
        _detect_yes_no_prompt,
        _detect_enter_escape_prompt,
    )
    for detector in detectors:
        request = detector(provider, lines)
        if request is not None:
            return request
    return None


def _detect_claude_bypass_warning(provider: ProviderName, lines: list[str]) -> ApprovalRequest | None:
    if provider != ProviderName.CLAUDE:
        return None

    warning_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "warning: claude code running in bypass permissions mode" in line.lower()
        ),
        None,
    )
    if warning_index is None:
        return None

    choices = _parse_option_choices(lines[warning_index : warning_index + 8], base_index=warning_index)
    approve = next((choice for choice in choices if "yes, i accept" in choice.label.lower()), None)
    deny = next((choice for choice in choices if any(token in choice.label.lower() for token in ("no", "exit", "cancel"))), None)
    prompt_text = _excerpt(lines, warning_index, warning_index + 6)
    if approve is None:
        return _approval_request(prompt_text, approve_keys=("2", "Enter"), deny_keys=("1", "Enter"))
    return _approval_request(
        prompt_text,
        approve_keys=(approve.key, "Enter"),
        deny_keys=(deny.key, "Enter") if deny is not None else ("1", "Enter"),
    )


def _detect_numbered_or_lettered_choices(provider: ProviderName, lines: list[str]) -> ApprovalRequest | None:
    choices = _parse_option_choices(lines)
    if not choices:
        return None

    if not _has_approval_context(lines):
        return None

    approve = next((choice for choice in choices if _matches_any(choice.label.lower(), APPROVE_LABEL_PATTERNS)), None)
    deny = next((choice for choice in choices if _matches_any(choice.label.lower(), DENY_LABEL_PATTERNS)), None)
    if approve is None or deny is None:
        return None

    first_index = min(approve.line_index, deny.line_index)
    last_index = max(approve.line_index, deny.line_index)
    prompt_text = _excerpt(lines, max(0, first_index - 3), min(len(lines) - 1, last_index + 2))
    return _approval_request(
        prompt_text,
        approve_keys=(approve.key, "Enter"),
        deny_keys=(deny.key, "Enter"),
    )


def _detect_claude_enter_escape_prompt(provider: ProviderName, lines: list[str]) -> ApprovalRequest | None:
    if provider != ProviderName.CLAUDE:
        return None

    top_index = next(
        (
            index
            for index, line in enumerate(lines)
            if any(pattern.search(line) for pattern in CLAUDE_ENTER_ESCAPE_TOP_PATTERNS)
        ),
        None,
    )
    if top_index is None:
        return None

    bottom_index = next(
        (
            index
            for index in range(top_index + 1, len(lines))
            if "esc to cancel" in lines[index].lower()
        ),
        None,
    )
    if bottom_index is None:
        return None

    prompt_text = _excerpt(lines, top_index, bottom_index)
    return _approval_request(prompt_text, approve_keys=("Enter",), deny_keys=("Escape",))


def _detect_yes_no_prompt(provider: ProviderName, lines: list[str]) -> ApprovalRequest | None:
    if not _has_approval_context(lines):
        return None

    for index, line in enumerate(lines):
        match = YES_NO_RE.search(line)
        if match is None:
            continue
        prompt_text = _excerpt(lines, max(0, index - 2), min(len(lines) - 1, index + 2))
        return _approval_request(
            prompt_text,
            approve_keys=(match.group("yes").lower(), "Enter"),
            deny_keys=(match.group("no").lower(), "Enter"),
        )
    return None


def _detect_enter_escape_prompt(provider: ProviderName, lines: list[str]) -> ApprovalRequest | None:
    if not _has_approval_context(lines):
        return None

    for index, line in enumerate(lines):
        lower = line.lower()
        if "enter" not in lower:
            continue
        if not any(token in lower for token in ("approve", "allow", "accept", "continue", "run")):
            continue

        prompt_text = _excerpt(lines, max(0, index - 2), min(len(lines) - 1, index + 2))
        deny_keys: tuple[str, ...] | None = None
        if "esc" in lower and any(token in lower for token in ("deny", "cancel", "reject", "stop")):
            deny_keys = ("Escape",)
        elif "esc" in lower:
            deny_keys = ("Escape",)
        return _approval_request(prompt_text, approve_keys=("Enter",), deny_keys=deny_keys)
    return None


def _clean_lines(pane_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in pane_text.splitlines():
        normalized = " ".join(raw_line.strip().split())
        if normalized:
            lines.append(normalized)
    return lines[-80:]


def _parse_option_choices(lines: Iterable[str], *, base_index: int = 0) -> list[_OptionChoice]:
    choices: list[_OptionChoice] = []
    for offset, line in enumerate(lines):
        match = OPTION_LINE_RE.match(line) or BRACKET_OPTION_RE.match(line)
        if match is None:
            continue
        choices.append(
            _OptionChoice(
                key=match.group("key"),
                label=match.group("label").strip(),
                line_index=base_index + offset,
            )
        )
    return choices


def _has_approval_context(lines: list[str]) -> bool:
    joined = "\n".join(lines).lower()
    return any(pattern in joined for pattern in APPROVAL_CONTEXT_PATTERNS)


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in value for pattern in patterns)


def _excerpt(lines: list[str], start: int, end: int) -> str:
    start = max(start, 0)
    end = min(end, len(lines) - 1)
    excerpt = lines[start : end + 1]
    if len(excerpt) > 8:
        excerpt = excerpt[-8:]
    return "\n".join(excerpt).strip()


def _approval_request(prompt_text: str, *, approve_keys: tuple[str, ...], deny_keys: tuple[str, ...] | None) -> ApprovalRequest:
    digest = sha256()
    digest.update(prompt_text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(",".join(approve_keys).encode("utf-8"))
    digest.update(b"\0")
    digest.update(",".join(deny_keys or ()).encode("utf-8"))
    return ApprovalRequest(
        fingerprint=digest.hexdigest(),
        prompt_text=prompt_text,
        approve_keys=approve_keys,
        deny_keys=deny_keys,
    )
