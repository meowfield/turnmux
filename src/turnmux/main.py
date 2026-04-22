from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Sequence

from .config import ConfigError, load_config
from .doctor import run_doctor, write_sample_config
from .log_setup import configure_logging
from .providers import ProviderRegistry
from .providers.claude_session_hook import process_claude_session_start_hook
from .runtime.home import RuntimePaths, initialize_runtime_home
from .runtime.lifecycle import install_global_exception_logging
from .service_manager import (
    format_launch_agent_status,
    install_launch_agent,
    read_launch_agent_status,
    restart_launch_agent,
    start_launch_agent,
    stop_launch_agent,
    uninstall_launch_agent,
)
from .state.db import bootstrap_database
from .state.repository import StateRepository


def add_common_args(parser: argparse.ArgumentParser, *, dest_prefix: str = "") -> None:
    parser.add_argument(
        "--runtime-home",
        dest=f"{dest_prefix}runtime_home",
        type=Path,
        default=None,
        help="Override the default runtime home (~/.turnmux).",
    )
    parser.add_argument(
        "--config",
        dest=f"{dest_prefix}config",
        type=Path,
        default=None,
        help="Override the default config path (~/.turnmux/config.toml).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurnMux runtime and Telegram transport.")
    add_common_args(parser, dest_prefix="root_")
    subparsers = parser.add_subparsers(dest="command")
    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create runtime home, load config, and bootstrap state.db.")
    add_common_args(bootstrap_parser)
    init_config_parser = subparsers.add_parser("init-config", help="Write a sample config.toml with detected local defaults.")
    add_common_args(init_config_parser)
    init_config_parser.add_argument("--force", action="store_true", help="Overwrite an existing config file.")
    doctor_parser = subparsers.add_parser("doctor", help="Validate runtime prerequisites and config.")
    add_common_args(doctor_parser)
    doctor_parser.add_argument("--repo", type=Path, default=None, help="Optionally validate a repo path and show provider resume counts.")
    run_parser = subparsers.add_parser("run", help="Run the Telegram bot and monitor loop.")
    add_common_args(run_parser)
    hook_parser = subparsers.add_parser("hook", help="Run internal provider hooks.")
    add_common_args(hook_parser)
    hook_subparsers = hook_parser.add_subparsers(dest="hook_command")
    claude_hook_parser = hook_subparsers.add_parser(
        "claude-session-start",
        help="Internal Claude SessionStart hook handler.",
    )
    add_common_args(claude_hook_parser)
    service_parser = subparsers.add_parser("service", help="Manage the persistent macOS launchd service.")
    add_common_args(service_parser)
    service_subparsers = service_parser.add_subparsers(dest="service_command")

    service_install_parser = service_subparsers.add_parser("install", help="Install the launchd agent and start it.")
    add_common_args(service_install_parser)
    service_install_parser.add_argument("--no-start", action="store_true", help="Write the plist but do not start the service.")

    for name, help_text in (
        ("start", "Start or restart the installed launchd service."),
        ("stop", "Stop the launchd service."),
        ("restart", "Restart the launchd service."),
        ("status", "Show launchd and heartbeat status."),
        ("uninstall", "Stop the launchd service and remove the plist."),
    ):
        subparser = service_subparsers.add_parser(name, help=help_text)
        add_common_args(subparser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_home = _arg_value(args, "runtime_home")
    config_override = _arg_value(args, "config")

    runtime_paths = initialize_runtime_home(runtime_home)
    configure_logging(runtime_paths.log_path)
    logger = logging.getLogger(__name__)

    config_path = config_override.expanduser().resolve(strict=False) if config_override else runtime_paths.config_path
    command = args.command or "bootstrap"

    if command == "init-config":
        try:
            written_path = write_sample_config(runtime_paths, force=args.force, working_dir=Path.cwd())
        except FileExistsError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Wrote sample TurnMux config to %s", written_path)
        return 0

    if command == "doctor":
        report = run_doctor(runtime_paths, config_path=config_path, repo_path=args.repo)
        print(report.text)
        return 0 if report.ok else 1

    if command == "hook":
        if args.hook_command == "claude-session-start":
            return process_claude_session_start_hook(runtime_home=runtime_paths.home)
        logger.error("Unknown hook command.")
        return 1

    if command == "service":
        return handle_service_command(args, runtime_paths, config_path)

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        logger.error("Failed to load TurnMux config: %s", exc)
        return 1

    db_path = bootstrap_database(runtime_paths.state_db_path)
    if command == "bootstrap":
        logger.info(
            "TurnMux bootstrap complete for tmux session '%s' with %d allowed root(s); state db=%s",
            config.tmux_session_name,
            len(config.allowed_roots),
            db_path,
        )
        return 0

    if command == "run":
        install_global_exception_logging()
        try:
            return asyncio.run(run_bot(runtime_paths, config))
        except KeyboardInterrupt:
            logger.info("TurnMux interrupted by user")
            return 130
        except Exception:
            logger.exception("TurnMux runtime crashed")
            return 1

    return 0


async def run_bot(runtime_paths: RuntimePaths, config) -> int:
    from .transport.telegram_bot import TurnmuxTelegramBot

    repository = StateRepository(runtime_paths.state_db_path)
    providers = ProviderRegistry(config, runtime_home=runtime_paths.home)
    bot = TurnmuxTelegramBot(config=config, repository=repository, providers=providers, runtime_paths=runtime_paths)
    await bot.run()
    return 0


def handle_service_command(args, runtime_paths: RuntimePaths, config_path: Path) -> int:
    logger = logging.getLogger(__name__)
    if sys.platform != "darwin":
        logger.error("The built-in TurnMux service manager currently supports only macOS launchd.")
        return 1

    action = args.service_command or "status"

    if action in {"install", "start", "restart"}:
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            logger.error("Failed to load TurnMux config: %s", exc)
            return 1
        bootstrap_database(runtime_paths.state_db_path)
        spec = install_launch_agent(runtime_paths, config_path=config.config_path)
        if action == "install":
            if not args.no_start:
                start_launch_agent(spec)
                logger.info("Installed and started launchd service at %s", spec.plist_path)
            else:
                logger.info("Installed launchd service at %s", spec.plist_path)
            return 0
        if action == "start":
            start_launch_agent(spec)
            logger.info("Started launchd service %s", spec.label)
            return 0
        restart_launch_agent(spec)
        logger.info("Restarted launchd service %s", spec.label)
        return 0

    if action == "stop":
        stop_launch_agent()
        logger.info("Stopped launchd service")
        return 0

    if action == "uninstall":
        stop_launch_agent()
        removed = uninstall_launch_agent()
        logger.info("Uninstalled launchd service plist %s", removed)
        return 0

    status = read_launch_agent_status(runtime_paths)
    print(format_launch_agent_status(status))
    return 0


def _arg_value(args, name: str):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return getattr(args, f"root_{name}", None)
