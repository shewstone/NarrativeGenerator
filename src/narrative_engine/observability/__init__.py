"""Observability package for Narrative Engine.

Provides structured logging, metrics, tracing, and error handling
for production-grade observability.
"""

from narrative_engine.observability.errors import (
    EpisodeNotFoundError,
    NarrativeEngineError,
    RepositoryError,
    ValidationError,
)
from narrative_engine.observability.logging import (
    EpisodeLogger,
    LogTimer,
    configure_logging,
    get_logger,
    set_context,
)
from narrative_engine.observability.metrics import (
    MetricsCollector,
    TimingContext,
)

__all__ = [
    # Errors
    "NarrativeEngineError",
    "EpisodeNotFoundError",
    "RepositoryError",
    "ValidationError",
    # Logging
    "configure_logging",
    "get_logger",
    "set_context",
    "LogTimer",
    "EpisodeLogger",
    # Metrics
    "MetricsCollector",
    "TimingContext",
]
