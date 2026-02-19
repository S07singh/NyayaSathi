"""
NyayaSathi AI — Logging Utility
================================
Provides a consistent, reusable logger factory.  Every module calls
``get_logger(__name__)`` at the top so that log output is uniform and
filterable.
"""

import logging
import sys
from typing import Optional

# Avoid circular import — read level lazily on first call.
_DEFAULT_LEVEL: str = "INFO"


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : str | None
        Override log level.  Falls back to ``config.LOG_LEVEL``.
    log_file : str | None
        If supplied, a ``FileHandler`` is also attached.
    """
    # Late import to avoid circular dependency at module load time
    try:
        from config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
    except ImportError:
        LOG_LEVEL = _DEFAULT_LEVEL
        LOG_FORMAT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
        LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    effective_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(effective_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(effective_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(effective_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
