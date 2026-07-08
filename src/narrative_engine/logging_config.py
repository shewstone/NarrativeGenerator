"""Structured logging configuration for Narrative Engine.

Supports both development (human-readable) and production (JSON) formats.
Includes context propagation for tracing requests across async boundaries.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog

# Context variables for request tracing
request_id: ContextVar[str] = ContextVar("request_id", default="")
episode_id: ContextVar[str] = ContextVar("episode_id", default="")
operation: ContextVar[str] = ContextVar("operation", default="")


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for production, human-readable for dev
        log_file: Optional file path for logging
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_format:
        # Production: JSON format for log aggregation
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty printed
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Configure standard library logging
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with context."""
    logger = structlog.get_logger(name)
    return logger.bind(
        request_id=request_id.get(),
        episode_id=episode_id.get(),
        operation=operation.get(),
    )


def set_context(
    req_id: Optional[str] = None,
    ep_id: Optional[str] = None,
    op: Optional[str] = None,
) -> None:
    """Set logging context variables."""
    if req_id:
        request_id.set(req_id)
    if ep_id:
        episode_id.set(ep_id)
    if op:
        operation.set(op)


def clear_context() -> None:
    """Clear all context variables."""
    request_id.set("")
    episode_id.set("")
    operation.set("")


class LogTimer:
    """Context manager for timing operations and logging results.

    Example:
        with LogTimer(logger, "database_query", episode_id=str(episode.id)):
            result = await repository.get_by_id(episode.id)
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation_name: str,
        episode_id: Optional[str] = None,
        **extra_context: Any,
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.episode_id = episode_id
        self.extra_context = extra_context
        self.start_time: Optional[float] = None

    def __enter__(self) -> "LogTimer":
        import time

        self.start_time = time.time()
        self.logger.debug(
            f"{self.operation_name}_started",
            operation=self.operation_name,
            episode_id=self.episode_id,
            **self.extra_context,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        if exc_type:
            self.logger.error(
                f"{self.operation_name}_failed",
                operation=self.operation_name,
                episode_id=self.episode_id,
                duration_ms=duration_ms,
                error_type=exc_type.__name__,
                error=str(exc_val),
                **self.extra_context,
            )
        else:
            self.logger.debug(
                f"{self.operation_name}_completed",
                operation=self.operation_name,
                episode_id=self.episode_id,
                duration_ms=duration_ms,
                **self.extra_context,
            )


class EpisodeLogger:
    """Logger bound to an episode for consistent context."""

    def __init__(
        self,
        base_logger: structlog.stdlib.BoundLogger,
        episode_id: str,
        episode_title: Optional[str] = None,
    ):
        self.logger = base_logger.bind(
            episode_id=episode_id,
            episode_title=episode_title,
        )
        self.episode_id = episode_id

    def info(self, event: str, **kwargs: Any) -> None:
        self.logger.info(event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self.logger.debug(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self.logger.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self.logger.error(event, **kwargs)

    def timer(self, operation_name: str, **extra_context: Any) -> LogTimer:
        return LogTimer(
            self.logger,
            operation_name,
            episode_id=self.episode_id,
            **extra_context,
        )
