from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turnmux.main import main


class MainCliTests(unittest.TestCase):
    def test_init_config_honors_runtime_home_before_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / ".turnmux"

            exit_code = main(["--runtime-home", str(runtime_home), "init-config"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((runtime_home / "config.toml").exists())

    def test_init_config_honors_runtime_home_after_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / ".turnmux"

            exit_code = main(["init-config", "--runtime-home", str(runtime_home)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((runtime_home / "config.toml").exists())

    def test_service_status_honors_runtime_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_home = Path(tmp_dir) / ".turnmux"
            with patch("turnmux.main.configure_logging"), patch("turnmux.main.read_launch_agent_status") as read_status:
                read_status.return_value = type(
                    "Status",
                    (),
                    {
                        "label": "io.turnmux.bot",
                        "plist_path": runtime_home / "io.turnmux.bot.plist",
                        "installed": False,
                        "loaded": False,
                        "pid": None,
                        "last_exit_status": None,
                        "heartbeat": None,
                    },
                )()
                with patch("turnmux.main.format_launch_agent_status", return_value="ok"):
                    exit_code = main(["service", "--runtime-home", str(runtime_home), "status"])

            self.assertEqual(exit_code, 0)
            read_status.assert_called_once()
