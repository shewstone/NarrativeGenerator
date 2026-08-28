"""Tests for Arc Instance composition."""

import uuid
from datetime import datetime

from narrative_engine.composition.arc_instance import (
    ArcInstance,
    CompositionStatus,
    PhaseCoverage,
)
from narrative_engine.models import ArcPhase, ArcType


class TestArcInstance:
    """Test Arc Instance creation and manipulation."""

    def test_create_arc_instance(self):
        """Test basic Arc Instance creation."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="CREDIT_BOOM_AND_BUST, 1920s America, 1922–1932",
        )

        assert instance.arc_type == ArcType.CREDIT_BOOM_AND_BUST
        assert instance.status == CompositionStatus.PENDING
        assert instance.overall_coverage == 0.0

    def test_add_episode_to_phase(self):
        """Test adding episodes with provenance tracking."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        episode_id = uuid.uuid4()
        instance.add_episode_to_phase(
            phase=ArcPhase.PANIC,
            episode_id=episode_id,
            source_id="book-a.txt",
            confidence=0.9,
            date=datetime(1929, 10, 24),
        )

        assert ArcPhase.PANIC in instance.phases
        coverage = instance.phases[ArcPhase.PANIC]
        assert episode_id in coverage.episode_ids
        assert "book-a.txt" in coverage.source_ids
        assert coverage.confidence == 0.9

    def test_multiple_sources_per_phase(self):
        """Test that multiple sources can cover the same phase."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        # Book 1 covers phases 1-2
        instance.add_episode_to_phase(
            phase=ArcPhase.BOOM,
            episode_id=uuid.uuid4(),
            source_id="book-1.txt",
            confidence=0.8,
        )

        # Book 2 covers phases 3-5
        instance.add_episode_to_phase(
            phase=ArcPhase.BOOM,
            episode_id=uuid.uuid4(),
            source_id="book-2.txt",
            confidence=0.85,
        )

        coverage = instance.phases[ArcPhase.BOOM]
        assert len(coverage.source_ids) == 2
        assert "book-1.txt" in coverage.source_ids
        assert "book-2.txt" in coverage.source_ids
        # Confidence should be averaged
        assert 0.8 < coverage.confidence < 0.85

    def test_identify_gaps(self):
        """Test gap identification."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        # Only cover PANIC phase
        instance.add_episode_to_phase(
            phase=ArcPhase.PANIC,
            episode_id=uuid.uuid4(),
            source_id="book.txt",
            confidence=0.9,
        )

        expected_phases = [
            ArcPhase.BOOM,
            ArcPhase.EUPHORIA,
            ArcPhase.DISTRESS,
            ArcPhase.PANIC,
            ArcPhase.REVULSION,
        ]

        gaps = instance.identify_gaps(expected_phases)

        assert len(gaps) == 4  # Missing 4 phases
        assert "Missing phase: boom" in gaps
        assert "Missing phase: euphoria" in gaps
        assert "Missing phase: distress" in gaps
        assert "Missing phase: revulsion" in gaps

    def test_update_status_complete(self):
        """Test status update for complete coverage."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        # Cover all phases of credit boom and bust
        phases = [
            ArcPhase.BOOM,
            ArcPhase.EUPHORIA,
            ArcPhase.DISTRESS,
            ArcPhase.PANIC,
            ArcPhase.REVULSION,
        ]

        for phase in phases:
            instance.add_episode_to_phase(
                phase=phase,
                episode_id=uuid.uuid4(),
                source_id="comprehensive-source.txt",
                confidence=0.9,
            )

        instance.identify_gaps(phases)
        status = instance.update_status()

        assert status == CompositionStatus.COMPLETE
        assert instance.status == CompositionStatus.COMPLETE

    def test_update_status_with_gaps(self):
        """Test status update when gaps exist."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        # Only cover 3 of 5 phases
        instance.add_episode_to_phase(
            phase=ArcPhase.BOOM,
            episode_id=uuid.uuid4(),
            source_id="book.txt",
            confidence=0.9,
        )
        instance.add_episode_to_phase(
            phase=ArcPhase.EUPHORIA,
            episode_id=uuid.uuid4(),
            source_id="book.txt",
            confidence=0.9,
        )
        instance.add_episode_to_phase(
            phase=ArcPhase.PANIC,
            episode_id=uuid.uuid4(),
            source_id="book.txt",
            confidence=0.9,
        )

        expected = [
            ArcPhase.BOOM,
            ArcPhase.EUPHORIA,
            ArcPhase.DISTRESS,
            ArcPhase.PANIC,
            ArcPhase.REVULSION,
        ]

        instance.identify_gaps(expected)
        status = instance.update_status()

        assert status == CompositionStatus.GAPS
        assert len(instance.coverage_gaps) == 2

    def test_coverage_score_calculation(self):
        """Test phase coverage score calculation."""
        coverage = PhaseCoverage(phase=ArcPhase.PANIC)

        # No sources = 0 coverage
        assert coverage.coverage_score == 0.0

        # Add sources
        coverage.source_ids = ["source-a", "source-b"]
        coverage.confidence = 0.8

        # Score should reflect sources and confidence
        score = coverage.coverage_score
        assert score > 0.0
        assert score <= 1.0

    def test_source_coverage_tracking(self):
        """Test that source coverage is tracked across the instance."""
        instance = ArcInstance(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            canonical_name="Test Instance",
        )

        # Book 1 covers multiple phases
        e1 = uuid.uuid4()
        for phase in [ArcPhase.BOOM, ArcPhase.EUPHORIA]:
            instance.add_episode_to_phase(
                phase=phase,
                episode_id=e1,
                source_id="book-1.txt",
                confidence=0.8,
            )

        # Book 2 covers remaining phases
        e2 = uuid.uuid4()
        for phase in [ArcPhase.DISTRESS, ArcPhase.PANIC, ArcPhase.REVULSION]:
            instance.add_episode_to_phase(
                phase=phase,
                episode_id=e2,
                source_id="book-2.txt",
                confidence=0.85,
            )

        assert "book-1.txt" in instance.source_coverage
        assert "book-2.txt" in instance.source_coverage
        # Both sources should have coverage scores
        assert instance.source_coverage["book-1.txt"] > 0
        assert instance.source_coverage["book-2.txt"] > 0


class TestPhaseCoverage:
    """Test PhaseCoverage model."""

    def test_phase_coverage_creation(self):
        """Test creating PhaseCoverage."""
        coverage = PhaseCoverage(phase=ArcPhase.CLIMAX)

        assert coverage.phase == ArcPhase.CLIMAX
        assert coverage.episode_ids == []
        assert coverage.source_ids == []
        assert coverage.confidence == 0.0

    def test_date_bounds(self):
        """Test tracking date bounds for a phase."""
        coverage = PhaseCoverage(phase=ArcPhase.BOOM)

        # Add episodes with dates
        coverage.earliest_date = datetime(1922, 1, 1)
        coverage.latest_date = datetime(1927, 12, 31)

        assert coverage.earliest_date.year == 1922
        assert coverage.latest_date.year == 1927
