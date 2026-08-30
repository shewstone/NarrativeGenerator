"""Compatibility re-export for tests that use the runtime gold fixture."""

from narrative_engine.evaluation.extraction_fixture import EXTRACTION_CASES, ExtractionCase

__all__ = ["EXTRACTION_CASES", "ExtractionCase"]
