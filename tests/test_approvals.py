from __future__ import annotations

import unittest

from turnmux.runtime.approvals import detect_approval_request, detect_non_approval_prompt_response
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

    def test_detects_enter_continue_when_approval_context_is_local(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            (
                "Approval required before running this command.\n"
                "Press enter to continue\n"
            ),
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.approve_keys, ("Enter",))
        self.assertIsNone(request.deny_keys)

    def test_ignores_codex_update_prompt_even_with_approval_policy_in_scrollback(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            (
                "model: gpt-5.5\n"
                "approval policy: on-request\n"
                "sandbox: danger-full-access\n"
                "Update now (runs `npm install -g @openai/codex@latest`)\n"
                "2. Skip\n"
                "3. Skip until next version\n"
                "Press enter to continue\n"
            ),
        )

        self.assertIsNone(request)

    def test_ignores_assistant_prose_that_mentions_approve_and_enter(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            (
                "отделить мои изменения от уже существовавших.\n"
                "Исправил. Это был реальный logic bug: Codex update prompt Update available /\n"
                "Skip / Press enter to continue попадал в approval detector из-за строки --ask-\n"
                "for-approval/approval policy в scrollback. В худшем варианте кнопка Approve\n"
                "могла нажать Enter на выбранном Update now.\n"
            ),
        )

        self.assertIsNone(request)

    def test_codex_update_prompt_auto_response_skips_update(self) -> None:
        response = detect_non_approval_prompt_response(
            ProviderName.CODEX,
            (
                "model: gpt-5.5\n"
                "approval policy: on-request\n"
                "sandbox: danger-full-access\n"
                "Update available! 0.122.0 -> 0.124.0\n"
                "Release notes: https://github.com/openai/codex/releases/latest\n"
                "› 1. Update now (runs `npm install -g @openai/codex`)\n"
                "2. Skip\n"
                "3. Skip until next version\n"
                "Press enter to continue\n"
            ),
        )

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.keys, ("Down", "Enter"))
        self.assertIn("Update available", response.prompt_text)

    def test_ignores_regular_transcript_text(self) -> None:
        request = detect_approval_request(
            ProviderName.CODEX,
            "I inspected the repo and found no approval prompt in the output.",
        )
        self.assertIsNone(request)
