from __future__ import annotations

import unittest

from turnmux.runtime.approvals import detect_approval_request
from turnmux.state.models import ProviderName


class ApprovalDetectionTests(unittest.TestCase):
    def test_detects_claude_bypass_warning(self) -> None:
        request = detect_approval_request(
            ProviderName.CLAUDE,
            (
                "WARNING: Claude Code running in Bypass Permissions mode\n"
                "1. No, exit\n"
                "2. Yes, I accept\n"
            ),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("2", "Enter"))
        self.assertEqual(request.deny_keys, ("1", "Enter"))
        self.assertIn("Bypass Permissions mode", request.prompt_text)

    def test_detects_codenumbered_approval_prompt(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            (
                "Approval required before running this command.\n"
                "1. Approve and continue\n"
                "2. Deny\n"
            ),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("1", "Enter"))
        self.assertEqual(request.deny_keys, ("2", "Enter"))

    def test_detects_yes_no_prompt(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            "Allow this command to run with full access? [Y/n]",
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("y", "Enter"))
        self.assertEqual(request.deny_keys, ("n", "Enter"))

    def test_detects_claude_enter_escape_permission_prompt(self) -> None:
        request = detect_approval_request(
            ProviderName.CLAUDE,
            (
                "Do you want to make this edit to settings.json?\n"
                "This will update the local project configuration.\n"
                "Esc to cancel\n"
            ),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("Enter",))
        self.assertEqual(request.deny_keys, ("Escape",))

    def test_detects_claude_bash_approval_prompt(self) -> None:
        request = detect_approval_request(
            ProviderName.CLAUDE,
            (
                "Bash command\n"
                "git push origin main\n"
                "Esc to cancel\n"
            ),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("Enter",))
        self.assertEqual(request.deny_keys, ("Escape",))

    def test_ignores_regular_transcript_text(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            "I inspected the repo and found no approval prompt in the output.",
        )
        self.assertIsNone(request)
