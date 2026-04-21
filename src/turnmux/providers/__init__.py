from .base import ProviderAdapter, ProviderSession, ProviderTranscriptEvent
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .opencode import OpenCodeAdapter
from ..state.models import ProviderName


class ProviderRegistry:
    def __init__(self, config) -> None:
        self._providers = {}
        if config.claude_command:
            self._providers[ProviderName.CLAUDE] = ClaudeAdapter(config)
        if config.codex_command:
            self._providers[ProviderName.CODEX] = CodexAdapter(config)
        if config.opencode_command:
            self._providers[ProviderName.OPENCODE] = OpenCodeAdapter(config)

    def get(self, provider: ProviderName) -> ProviderAdapter:
        return self._providers[provider]

    def available_providers(self) -> tuple[ProviderName, ...]:
        return tuple(self._providers)


__all__ = [
    "ClaudeAdapter",
    "CodexAdapter",
    "OpenCodeAdapter",
    "ProviderAdapter",
    "ProviderRegistry",
    "ProviderSession",
    "ProviderTranscriptEvent",
]
