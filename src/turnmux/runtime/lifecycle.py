from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Callable

from .home import ensure_private_directory, set_private_file_permissions


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HeartbeatSnapshot:
    pid: int
    ppid: int
    started_at: str
    last_heartbeat_at: str
    status: str
    note: str | None = None


class HeartbeatWriter:
    def __init__(self, path: Path, *, started_at: str | None = None) -> None:
        self.path = path.expanduser().resolve(strict=False)
        ensure_private_directory(self.path.parent)
        self.started_at = started_at or utc_now_iso()
        self.pid = os.getpid()
        self.ppid = os.getppid()

    def write(self, *, status: str, note: str | None = None) -> None:
        snapshot = HeartbeatSnapshot(
            pid=self.pid,
            ppid=self.ppid,
            started_at=self.started_at,
            last_heartbeat_at=utc_now_iso(),
            status=status,
            note=note,
        )
        _atomic_write_json(self.path, asdict(snapshot))


def read_heartbeat(path: Path) -> dict[str, object] | None:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.exists():
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def install_global_exception_logging() -> None:
    def _sys_excepthook(exc_type, exc_value, exc_traceback) -> None:
        logger.critical("Unhandled exception reached sys.excepthook", exc_info=(exc_type, exc_value, exc_traceback))

    def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Unhandled exception reached threading.excepthook in thread %s",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook


def install_asyncio_exception_logging(loop: asyncio.AbstractEventLoop) -> None:
    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BaseException):
            logger.exception("Unhandled asyncio exception", exc_info=exc)
            return
        logger.error("Unhandled asyncio error: %s", context.get("message", "unknown error"))

    loop.set_exception_handler(_handler)


def install_unix_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    *,
    on_shutdown: Callable[[str], None],
) -> None:
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _ignore_sighup)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None), getattr(signal, "SIGQUIT", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, lambda sig=sig: on_shutdown(signal.Signals(sig).name))
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda signum, frame, sig=sig: on_shutdown(signal.Signals(sig).name))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ignore_sighup(signum, frame) -> None:
    logger.warning(
        "Received SIGHUP. Ignoring it so TurnMux keeps running. "
        "This usually means the parent terminal or PTY disappeared."
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    set_private_file_permissions(temp_path)
    temp_path.replace(path)
    set_private_file_permissions(path)
