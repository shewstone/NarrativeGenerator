"""Arc Instance model - concrete unfoldings of narrative arcs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from narrative_engine.models import ArcPhase, ArcType, CycleScale, utcnow


class CompositionStatus(str, Enum):
    """Status of arc instance composition."""

    COMPLETE = "complete"  # All phases covered
    GAPS = "gaps"  # Some phases missing
    FRAGMENTED = "fragmented"  # Multiple gaps, incomplete
    PENDING = "pending"  # Still being composed


class PhaseCoverage(BaseModel):
    """Coverage information for a single phase."""

    model_config = ConfigDict(frozen=False)

    phase: ArcPhase
    episode_ids: List[UUID] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)  # Which sources document this phase
    confidence: float = Field(default=0.0)  # Aggregate confidence across sources
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None

    @property
    def coverage_score(self) -> float:
        """Calculate coverage score based on sources and confidence.

        Confidence is the base signal: one well-attested source is real
        coverage (a sources/3 factor would leave every single-source phase
        permanently "under-covered"). Corroborating sources add a small
        bonus, capped at 1.0.
        """
        if not self.source_ids:
            return 0.0
        corroboration_bonus = 1 + 0.1 * (len(self.source_ids) - 1)
        return min(1.0, self.confidence * corroboration_bonus)


class ArcInstance(BaseModel):
    """A concrete unfolding of an arc across time and sources.

    Example: "HUBRIS_NEMESIS, Wilhelmine Germany, 1890–1918"
    - Phases 1-2 from Book A
    - Phases 3-5 from Book B
    - Per-phase coverage shows documentation gaps
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    arc_type: ArcType
    canonical_name: str  # "HUBRIS_NEMESIS, Wilhelmine Germany, 1890–1918"

    # Polity/institution scope this instance belongs to (every episode in
    # the composing cluster shares one, since composition partitions by
    # scope_id before clustering -- see compose_arc_instances_from_episodes).
    scope_id: Optional[str] = None
    scale: CycleScale = CycleScale.EPISODIC

    # Temporal bounds
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Phase composition
    phases: Dict[ArcPhase, PhaseCoverage] = Field(default_factory=dict)
    # Classified episodes can occasionally have no usable phase label. Keep
    # them attached to the instance with empty phase coverage so composition
    # never silently drops a source episode at persistence time.
    unphased_episode_ids: List[UUID] = Field(default_factory=list)

    # Source coverage tracking
    source_coverage: Dict[str, float] = Field(default_factory=dict)  # source_id -> coverage_ratio

    # Composition status
    status: CompositionStatus = CompositionStatus.PENDING
    coverage_gaps: List[str] = Field(default_factory=list)  # List of missing phases

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # Framework lineage
    framework_id: Optional[UUID] = None  # If imported from a framework

    @property
    def complete_phases(self) -> List[ArcPhase]:
        """Return phases with coverage above threshold."""
        return [phase for phase, coverage in self.phases.items() if coverage.coverage_score > 0.5]

    @property
    def overall_coverage(self) -> float:
        """Calculate overall coverage score."""
        if not self.phases:
            return 0.0
        scores = [p.coverage_score for p in self.phases.values()]
        return sum(scores) / len(scores)

    def get_phase_sources(self, phase: ArcPhase) -> List[str]:
        """Get all sources covering a specific phase."""
        if phase not in self.phases:
            return []
        return self.phases[phase].source_ids

    def add_episode_to_phase(
        self,
        phase: ArcPhase,
        episode_id: UUID,
        source_id: str,
        confidence: float,
        date: Optional[datetime] = None,
    ) -> None:
        """Add an episode to a phase, tracking provenance."""
        if phase not in self.phases:
            self.phases[phase] = PhaseCoverage(phase=phase)

        coverage = self.phases[phase]
        if episode_id not in coverage.episode_ids:
            coverage.episode_ids.append(episode_id)
        if source_id not in coverage.source_ids:
            coverage.source_ids.append(source_id)

        # Update confidence (running average)
        n = len(coverage.episode_ids)
        coverage.confidence = (coverage.confidence * (n - 1) + confidence) / n

        # Update date bounds
        if date:
            if coverage.earliest_date is None or date < coverage.earliest_date:
                coverage.earliest_date = date
            if coverage.latest_date is None or date > coverage.latest_date:
                coverage.latest_date = date

        self._recompute_source_coverage()

    def _recompute_source_coverage(self) -> None:
        """Recompute source_coverage: fraction of known phases each source documents."""
        total_phases = len(self.phases)
        counts: Dict[str, int] = {}
        for coverage in self.phases.values():
            for source_id in coverage.source_ids:
                counts[source_id] = counts.get(source_id, 0) + 1
        self.source_coverage = {source_id: count / total_phases for source_id, count in counts.items()}

    def identify_gaps(self, expected_phases: List[ArcPhase]) -> List[str]:
        """Identify which expected phases are missing or under-covered."""
        gaps = []
        for phase in expected_phases:
            if phase not in self.phases:
                gaps.append(f"Missing phase: {phase.value}")
            elif self.phases[phase].coverage_score < 0.5:
                gaps.append(f"Under-covered phase: {phase.value}")
        self.coverage_gaps = gaps
        return gaps

    def update_status(self) -> CompositionStatus:
        """Update composition status based on coverage."""
        if not self.phases:
            self.status = CompositionStatus.FRAGMENTED
        elif self.coverage_gaps:
            self.status = CompositionStatus.GAPS if len(self.coverage_gaps) <= 2 else CompositionStatus.FRAGMENTED
        else:
            self.status = CompositionStatus.COMPLETE
        return self.status
