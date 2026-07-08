"""Metrics collection for performance monitoring.

Tracks operation timing, throughput, and success rates
for system health monitoring.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimingMetrics:
    """Timing statistics for an operation."""

    operation: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    errors: int = 0

    def record(self, duration_ms: float, success: bool = True) -> None:
        """Record a timing measurement."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        if not success:
            self.errors += 1

    @property
    def avg_ms(self) -> float:
        """Average duration in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        return (self.errors / self.count * 100) if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "operation": self.operation,
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else None,
            "max_ms": round(self.max_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "errors": self.errors,
            "error_rate": round(self.error_rate, 2),
        }


@dataclass
class CounterMetrics:
    """Simple counter for events."""

    name: str
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def increment(self, amount: int = 1) -> None:
        """Increment counter."""
        self.count += amount


class MetricsCollector:
    """Collects and reports metrics for the Narrative Engine.

    Example:
        metrics = MetricsCollector()
        metrics.timing("database_query", duration_ms=45.2)
        metrics.counter("episodes_created").increment()
    """

    def __init__(self) -> None:
        self._timings: Dict[str, TimingMetrics] = {}
        self._counters: Dict[str, CounterMetrics] = {}

    def timing(self, operation: str, duration_ms: float, success: bool = True) -> None:
        """Record a timing metric."""
        if operation not in self._timings:
            self._timings[operation] = TimingMetrics(operation=operation)
        self._timings[operation].record(duration_ms, success)

    def counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> CounterMetrics:
        """Get or create a counter."""
        key = f"{name}:{sorted(labels.items())}" if labels else name
        if key not in self._counters:
            self._counters[key] = CounterMetrics(name=name, labels=labels or {})
        return self._counters[key]

    def get_report(self) -> Dict[str, Any]:
        """Generate a metrics report."""
        return {
            "timings": {k: v.to_dict() for k, v in self._timings.items()},
            "counters": {
                k: {"name": v.name, "count": v.count, "labels": v.labels}
                for k, v in self._counters.items()
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._timings.clear()
        self._counters.clear()


# Global metrics collector
_global_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_metrics


@contextmanager
def TimingContext(operation: str, metrics: Optional[MetricsCollector] = None):
    """Context manager for timing operations.

    Example:
        with TimingContext("database_query"):
            result = await repository.get_by_id(id)
    """
    import time

    start = time.time()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration_ms = (time.time() - start) * 1000
        collector = metrics or _global_metrics
        collector.timing(operation, duration_ms, success)
