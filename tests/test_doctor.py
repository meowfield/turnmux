from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from turnmux.doctor import TOKEN_PLACEHOLDER, USER_ID_PLACEHOLDER, run_doctor
from turnmux.runtime.home import initialize_runtime_home
from turnmux.service_manager import LaunchAgentStatus


def _write_config(path: Path, *, token: str, user_id: int) -> None:
    path.write_text(
        "\n".join(
            [
                f'telegram_bot_token = "{token}"',
                f"allowed_user_ids = [{user_id}]",
                f'allowed_roots = ["{path.parent.resolve(strict=False)}"]',
                'tmux_session_name = "turnmux"',
                f'codex_command = ["{Path(sys.executable).resolve(strict=False)}"]',
                "",
            ]
        ),
        encoding="utf-8",
    )


class DoctorTests(unittest.TestCase):
    def test_doctor_fails_on_placeholder_config_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            _write_config(runtime_paths.config_path, token=TOKEN_PLACEHOLDER, user_id=USER_ID_PLACEHOLDER)

            with patch("turnmux.doctor._detect_binary", return_value="/usr/bin/tmux"), patch(
                "turnmux.doctor._command_output", return_value="tmux 3.6a"
            ), patch("turnmux.doctor._probe_tmux", return_value=None), patch("turnmux.doctor.sys.platform", "linux"):
                report = run_doctor(runtime_paths)

        self.assertFalse(report.ok)
        self.assertIn("[error] telegram_bot_token is still the sample placeholder.", report.text)
        self.assertIn("[error] allowed_user_ids still contains the placeholder value 123456789.", report.text)
        self.assertIn("[note] voice transcription: disabled", report.text)

    def test_doctor_surfaces_service_runtime_home_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            _write_config(runtime_paths.config_path, token="123456:valid-looking-token-value", user_id=42)
            other_runtime = Path(tmp_dir) / "other-runtime"

            with patch("turnmux.doctor._detect_binary", return_value="/usr/bin/tmux"), patch(
                "turnmux.doctor._command_output", return_value="tmux 3.6a"
            ), patch("turnmux.doctor._probe_tmux", return_value=None), patch(
                "turnmux.doctor.read_launch_agent_status",
                return_value=LaunchAgentStatus(
                    label="io.turnmux.bot",
                    plist_path=Path("/tmp/io.turnmux.bot.plist"),
                    installed=True,
                    loaded=True,
                    pid=111,
                    last_exit_status=0,
                    heartbeat={
                        "status": "running",
                        "last_heartbeat_at": "2030-01-01T00:00:00+00:00",
                        "pid": 111,
                    },
                    runtime_home=other_runtime,
                    config_path=other_runtime / "config.toml",
                ),
            ), patch("turnmux.doctor.sys.platform", "darwin"):
                report = run_doctor(runtime_paths)

        self.assertTrue(report.ok)
        self.assertIn(f"[warn] launchd runtime home: {other_runtime}", report.text)
        self.assertIn("[warn] launchd service health: service runtime home differs from this invocation", report.text)
