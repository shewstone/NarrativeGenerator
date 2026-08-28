"""Structured logging utilities for Narrative Engine."""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Optional

import structlog

# Context variables for request tracking
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
episode_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("episode_id", default=None)


def configure_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Minimum log level to capture
        json_format: Whether to output JSON formatted logs
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, defaults to caller module

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def set_context(request_id: Optional[str] = None, episode_id: Optional[str] = None) -> None:
    """Set context variables for the current execution context.

    Args:
        request_id: Unique request identifier
        episode_id: Episode identifier for the current operation
    """
    if request_id:
        request_id_var.set(request_id)
    if episode_id:
        episode_id_var.set(episode_id)


@contextmanager
def LogTimer(operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None):  # noqa: N802
    """Context manager to time and log operation duration.

    Args:
        operation: Name of the operation being timed
        logger: Optional logger instance

    Yields:
        None

    Example:
        with LogTimer("database_query"):
            results = await db.query(...)
    """
    log = logger or get_logger()
    start_time = time.monotonic()
    try:
        yield
        elapsed_ms = (time.monotonic() - start_time) * 1000
        log.debug(f"{operation}_completed", duration_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        log.error(
            f"{operation}_failed",
            duration_ms=elapsed_ms,
            error=str(e),
            exc_info=True,
        )
        raise


class EpisodeLogger:
    """Logger wrapper that adds episode context to all log entries."""

    def __init__(self, episode_id: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        """Initialize episode logger.

        Args:
            episode_id: The episode ID to include in logs
            logger: Optional underlying logger
        """
        self.episode_id = episode_id
        self._logger = logger or get_logger()

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal log method that adds episode context."""
        kwargs["episode_id"] = self.episode_id
        getattr(self._logger, level)(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with episode context."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with episode context."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with episode context."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with episode context."""
        self._log("error", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with episode context."""
        kwargs["episode_id"] = self.episode_id
        self._logger.exception(message, **kwargs)
