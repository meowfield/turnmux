from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import call, patch

from turnmux.runtime.home import initialize_runtime_home
from turnmux.service_manager import (
    LaunchAgentStatus,
    build_launch_agent_spec,
    evaluate_launch_agent_health,
    format_launch_agent_status,
    read_launch_agent_status,
    render_launch_agent_plist,
    start_launch_agent,
    stop_launch_agent,
)


class ServiceManagerTests(unittest.TestCase):
    def test_render_launch_agent_plist_contains_keepalive_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            config_path = runtime_paths.config_path
            spec = build_launch_agent_spec(
                runtime_paths,
                config_path=config_path,
                label="io.turnmux.test",
                python_executable=Path("/usr/bin/python3"),
                working_directory=Path(tmp_dir),
            )

            payload = plistlib.loads(render_launch_agent_plist(spec))

            self.assertEqual(payload["Label"], "io.turnmux.test")
            self.assertEqual(
                payload["ProgramArguments"],
                [
                    "/usr/bin/python3",
                    "-m",
                    "turnmux",
                    "--runtime-home",
                    str(runtime_paths.home),
                    "--config",
                    str(config_path),
                    "run",
                ],
            )
            self.assertTrue(payload["RunAtLoad"])
            self.assertTrue(payload["KeepAlive"])
            self.assertEqual(payload["StandardOutPath"], str(runtime_paths.service_stdout_path))
            self.assertEqual(payload["StandardErrorPath"], str(runtime_paths.service_stderr_path))
            self.assertIn("PATH", payload["EnvironmentVariables"])
            self.assertIn("/opt/homebrew/bin", payload["EnvironmentVariables"]["PATH"])

    def test_render_launch_agent_plist_passes_through_tmux_binary_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            spec = build_launch_agent_spec(
                runtime_paths,
                config_path=runtime_paths.config_path,
                label="io.turnmux.test",
                python_executable=Path("/usr/bin/python3"),
                working_directory=Path(tmp_dir),
            )

            with patch.dict("os.environ", {"TURNMUX_TMUX_BINARY": "/tmp/tmux-wrapper"}):
                payload = plistlib.loads(render_launch_agent_plist(spec))

            self.assertEqual(payload["EnvironmentVariables"]["TURNMUX_TMUX_BINARY"], "/tmp/tmux-wrapper")

    def test_render_launch_agent_plist_preserves_existing_tmux_binary_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            spec = build_launch_agent_spec(
                runtime_paths,
                config_path=runtime_paths.config_path,
                label="io.turnmux.test",
                python_executable=Path("/usr/bin/python3"),
                working_directory=Path(tmp_dir),
            )
            spec.plist_path.parent.mkdir(parents=True, exist_ok=True)
            spec.plist_path.write_bytes(
                plistlib.dumps(
                    {
                        "EnvironmentVariables": {
                            "TURNMUX_TMUX_BINARY": "/tmp/existing-tmux-wrapper",
                        }
                    }
                )
            )

            with patch.dict("os.environ", {}, clear=True):
                payload = plistlib.loads(render_launch_agent_plist(spec))

            self.assertEqual(payload["EnvironmentVariables"]["TURNMUX_TMUX_BINARY"], "/tmp/existing-tmux-wrapper")

    def test_format_status_includes_heartbeat(self) -> None:
        status = LaunchAgentStatus(
            label="io.turnmux.bot",
            plist_path=Path("/tmp/example-home/Library/LaunchAgents/io.turnmux.bot.plist"),
            installed=True,
            loaded=True,
            pid=123,
            last_exit_status=0,
            heartbeat={
                "status": "running",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "pid": 123,
            },
        )

        rendered = format_launch_agent_status(status)
        self.assertIn("health: ok", rendered)
        self.assertIn("installed: yes", rendered)
        self.assertIn("loaded: yes", rendered)
        self.assertIn("heartbeat status: running", rendered)
        self.assertIn("heartbeat pid: 123", rendered)

    def test_read_launch_agent_status_prefers_runtime_home_from_installed_plist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            fallback_runtime = initialize_runtime_home(base_dir / "fallback-runtime")
            installed_runtime = initialize_runtime_home(base_dir / "installed-runtime")
            heartbeat_payload = {
                "status": "running",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "pid": 777,
            }
            installed_runtime.heartbeat_path.write_text(json.dumps(heartbeat_payload), encoding="utf-8")
            plist_path = base_dir / "io.turnmux.test.plist"

            with patch("turnmux.service_manager.launch_agent_path", return_value=plist_path):
                spec = build_launch_agent_spec(
                    installed_runtime,
                    config_path=installed_runtime.config_path,
                    label="io.turnmux.test",
                    python_executable=Path("/usr/bin/python3"),
                    working_directory=base_dir,
                )
                plist_path.write_bytes(render_launch_agent_plist(spec))

                with patch(
                    "turnmux.service_manager._run_launchctl",
                    return_value=subprocess.CompletedProcess(
                        args=["launchctl", "print", "gui/502/io.turnmux.test"],
                        returncode=0,
                        stdout="pid = 777\nlast exit code = 0\n",
                        stderr="",
                    ),
                ), patch("turnmux.service_manager._launchd_domain", return_value="gui/502"):
                    status = read_launch_agent_status(fallback_runtime, label="io.turnmux.test")

        self.assertEqual(status.runtime_home, installed_runtime.home)
        self.assertEqual(status.config_path, installed_runtime.config_path)
        self.assertEqual(status.heartbeat, heartbeat_payload)
        self.assertEqual(status.pid, 777)

    def test_evaluate_health_reports_stale_heartbeat_and_runtime_home_mismatch(self) -> None:
        status = LaunchAgentStatus(
            label="io.turnmux.bot",
            plist_path=Path("/tmp/example-home/Library/LaunchAgents/io.turnmux.bot.plist"),
            installed=True,
            loaded=True,
            pid=123,
            last_exit_status=0,
            heartbeat={
                "status": "running",
                "last_heartbeat_at": "2026-04-20T19:09:18+00:00",
                "pid": 123,
            },
            runtime_home=Path("/tmp/installed-runtime"),
        )

        health = evaluate_launch_agent_health(status, expected_runtime_home=Path("/tmp/requested-runtime"))

        self.assertEqual(health.level, "error")
        self.assertIn("heartbeat is stale", health.summary)
        self.assertTrue(any("runtime home differs" in detail for detail in health.details))

    def test_start_launch_agent_uses_service_target_for_bootout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_paths = initialize_runtime_home(Path(tmp_dir) / ".turnmux")
            spec = build_launch_agent_spec(
                runtime_paths,
                config_path=runtime_paths.config_path,
                label="io.turnmux.test",
                python_executable=Path("/usr/bin/python3"),
                working_directory=Path(tmp_dir),
            )
            with patch("turnmux.service_manager.is_launch_agent_loaded", return_value=True), patch(
                "turnmux.service_manager._run_launchctl"
            ) as run_launchctl, patch("turnmux.service_manager._bootstrap_launch_agent") as bootstrap, patch(
                "turnmux.service_manager._wait_for_launch_agent_state"
            ) as wait_for_state, patch("turnmux.service_manager._launchd_domain", return_value="gui/502"):
                start_launch_agent(spec)

        self.assertEqual(run_launchctl.call_args_list, [call(["bootout", "gui/502/io.turnmux.test"], check=False), call(["kickstart", "-k", "gui/502/io.turnmux.test"])])
        wait_for_state.assert_called_once_with("io.turnmux.test", loaded=False)
        bootstrap.assert_called_once_with(spec.plist_path, domain="gui/502")

    def test_stop_launch_agent_uses_service_target_for_bootout(self) -> None:
        with patch("turnmux.service_manager._run_launchctl") as run_launchctl, patch(
            "turnmux.service_manager._wait_for_launch_agent_state"
        ) as wait_for_state, patch(
            "turnmux.service_manager._launchd_domain",
            return_value="gui/502",
        ):
            stop_launch_agent("io.turnmux.test")

        run_launchctl.assert_called_once_with(["bootout", "gui/502/io.turnmux.test"], check=False)
        wait_for_state.assert_called_once_with("io.turnmux.test", loaded=False)
