from __future__ import annotations

import logging
from pathlib import Path

from .runtime.home import ensure_private_directory, set_private_file_permissions


_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_path: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Configure deterministic stderr + file logging for the TurnMux process."""
    resolved_log_path = log_path.expanduser().resolve(strict=False)
    ensure_private_directory(resolved_log_path.parent)
    logging.raiseExceptions = False

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    set_private_file_permissions(resolved_log_path)

    return root_logger
