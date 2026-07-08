"""Custom exceptions with structured error context.

Provides application-specific errors that carry enough context
for debugging and user-friendly error messages.
"""

from typing import Any, Dict, Optional


class NarrativeEngineError(Exception):
    """Base exception for all Narrative Engine errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class EpisodeNotFoundError(NarrativeEngineError):
    """Raised when an episode is not found."""

    def __init__(self, episode_id: str, **context: Any):
        super().__init__(
            message=f"Episode not found: {episode_id}",
            error_code="EPISODE_NOT_FOUND",
            context={"episode_id": episode_id, **context},
        )


class RepositoryError(NarrativeEngineError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, details: str, **context: Any):
        super().__init__(
            message=f"Repository operation failed: {operation} - {details}",
            error_code="REPOSITORY_ERROR",
            context={"operation": operation, "details": details, **context},
        )


class ValidationError(NarrativeEngineError):
    """Raised when data validation fails."""

    def __init__(self, field: str, reason: str, **context: Any):
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            error_code="VALIDATION_ERROR",
            context={"field": field, "reason": reason, **context},
        )


class ExtractionError(NarrativeEngineError):
    """Raised when LLM extraction fails."""

    def __init__(
        self,
        stage: str,
        source_id: str,
        details: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(
            message=f"Extraction failed at stage '{stage}' for source '{source_id}': {details or 'Unknown error'}",
            error_code="EXTRACTION_ERROR",
            context={
                "stage": stage,
                "source_id": source_id,
                "details": details,
                **context,
            },
        )


class RetrievalError(NarrativeEngineError):
    """Raised when analog retrieval fails."""

    def __init__(self, query_id: str, reason: str, **context: Any):
        super().__init__(
            message=f"Retrieval failed for query '{query_id}': {reason}",
            error_code="RETRIEVAL_ERROR",
            context={"query_id": query_id, "reason": reason, **context},
        )


class TaxonomyError(NarrativeEngineError):
    """Raised when taxonomy operations fail."""

    def __init__(self, taxonomy_id: str, operation: str, **context: Any):
        super().__init__(
            message=f"Taxonomy operation failed: {operation} for {taxonomy_id}",
            error_code="TAXONOMY_ERROR",
            context={"taxonomy_id": taxonomy_id, "operation": operation, **context},
        )
