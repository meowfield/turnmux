from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from turnmux.state.db import bootstrap_database
from turnmux.state.models import BindingStatus, OnboardingStep, ProviderName
from turnmux.state.repository import StateRepository


class StateRepositoryTests(unittest.TestCase):
    def test_save_binding_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CLAUDE,
                repo_path=Path(tmp_dir) / "repo",
                tmux_session_name="turnmux",
                tmux_window_id="@1",
                tmux_window_name="claude:repo",
                status=BindingStatus.PENDING_START,
            )

            loaded = repository.get_binding(1, 10)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.id, binding.id)
            self.assertEqual(loaded.provider, ProviderName.CLAUDE)
            self.assertEqual(loaded.tmux_window_id, "@1")

    def test_pending_launch_and_onboarding_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CODEX,
                repo_path=Path(tmp_dir) / "repo",
                tmux_session_name="turnmux",
                tmux_window_id="@2",
                tmux_window_name="codex:repo",
                status=BindingStatus.PENDING_START,
            )

            repository.save_pending_launch(
                binding_id=binding.id,
                provider=ProviderName.CODEX,
                repo_path=Path(tmp_dir) / "repo",
                started_at="2026-04-20T10:00:00+00:00",
                discovery_deadline_at="2026-04-20T10:10:00+00:00",
                requested_session_id="session-123",
            )
            pending = repository.list_pending_launches()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].requested_session_id, "session-123")
            self.assertEqual(pending[0].started_at, "2026-04-20T10:00:00+00:00")

            repository.save_onboarding_state(
                chat_id=1,
                thread_id=10,
                step=OnboardingStep.CHOOSE_RESUME,
                provider=ProviderName.CODEX,
                repo_path=Path(tmp_dir) / "repo",
                mode="resume",
                resume_candidates_json='[{"session_id":"abc"}]',
            )
            onboarding = repository.get_onboarding_state(1, 10)
            self.assertIsNotNone(onboarding)
            assert onboarding is not None
            self.assertEqual(onboarding.step, OnboardingStep.CHOOSE_RESUME)
            self.assertEqual(onboarding.provider, ProviderName.CODEX)
            self.assertEqual(onboarding.mode, "resume")

            repository.clear_onboarding_state(1, 10)
            self.assertIsNone(repository.get_onboarding_state(1, 10))

    def test_pending_approval_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            bootstrap_database(db_path)
            repository = StateRepository(db_path)

            binding = repository.save_binding(
                chat_id=1,
                thread_id=10,
                provider=ProviderName.CLAUDE,
                repo_path=Path(tmp_dir) / "repo",
                tmux_session_name="turnmux",
                tmux_window_id="@3",
                tmux_window_name="claude:repo",
                status=BindingStatus.ACTIVE,
            )

            repository.save_pending_approval(
                binding_id=binding.id,
                provider=ProviderName.CLAUDE,
                fingerprint="prompt-1",
                prompt_text="Approve dangerous mode?",
                approve_keys=("2", "Enter"),
                deny_keys=("1", "Enter"),
            )

            pending = repository.get_pending_approval(binding.id)
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertEqual(pending.fingerprint, "prompt-1")
            self.assertEqual(pending.approve_keys, ("2", "Enter"))
            self.assertEqual(pending.deny_keys, ("1", "Enter"))

            repository.delete_pending_approval(binding.id)
            self.assertIsNone(repository.get_pending_approval(binding.id))
