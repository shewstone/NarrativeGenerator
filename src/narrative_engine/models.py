"""Core data models for the Narrative Engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ThesisConfidence(str, Enum):
    """Confidence level in a generated thesis."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ArcPhase(str, Enum):
    """Standard phases in narrative arcs."""

    SETUP = "setup"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    # Extended phases for financial/economic arcs
    BOOM = "boom"
    EUPHORIA = "euphoria"
    DISTRESS = "distress"
    PANIC = "panic"
    REVULSION = "revulsion"


class ArcType(str, Enum):
    """Taxonomy of narrative arcs."""

    RISE_AND_OVEREXTENSION = "rise_and_overextension"
    HUBRIS_NEMESIS = "hubris_nemesis"
    REFORM_THEN_REACTION = "reform_then_reaction"
    DECADENCE_AND_RENEWAL = "decadence_and_renewal"
    SIEGE_AND_COLLAPSE = "siege_and_collapse"
    SUCCESSION_CRISIS = "succession_crisis"
    CREDIT_BOOM_AND_BUST = "credit_boom_and_bust"
    GENERATIONAL_FORGETTING = "generational_forgetting"
    HERO_JOURNEY = "hero_journey"
    TRAGEDY = "tragedy"
    COMEDY = "comedy"
    REBIRTH = "rebirth"
    VOYAGE_RETURN = "voyage_return"
    RAGS_TO_RICHES = "rags_to_riches"


class CycleScale(str, Enum):
    """Fractal cycle scales."""

    CIVILIZATIONAL = "civilizational"  # ~centuries
    INSTITUTIONAL = "institutional"  # ~decades
    GENERATIONAL = "generational"  # ~20-25 years
    EPISODIC = "episodic"  # individual events


class Continuation(BaseModel):
    """A possible continuation/outcome with probability."""

    description: str
    probability: float
    supporting_analogs: int = 0


class Actor(BaseModel):
    """An actor participating in an episode."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    name: str
    role: str  # e.g., "protagonist", "antagonist", "institution"
    attributes: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)


class SourcePassage(BaseModel):
    """Reference to a source passage for provenance."""

    model_config = ConfigDict(frozen=True)

    work_id: str  # e.g., "kindleberger-mania-1978"
    passage_id: str
    text: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    historiographic_school: Optional[str] = None  # e.g., "Marxist", "Whig"


class Episode(BaseModel):
    """Atomic narrative unit: a bounded historical situation."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    title: str
    summary: str

    # Temporal
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    date_precision: str = "year"  # year, month, day

    # Setting
    location: Optional[str] = None
    setting_description: Optional[str] = None

    # Narrative structure
    actors: List[Actor] = Field(default_factory=list)
    initiating_conditions: List[str] = Field(default_factory=list)
    escalation_mechanics: List[str] = Field(default_factory=list)
    tension: Optional[str] = None
    resolution: Optional[str] = None
    consequences: List[str] = Field(default_factory=list)

    # Arc classification
    arc_type: Optional[ArcType] = None
    arc_phase: Optional[ArcPhase] = None
    phase_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    arc_rationale: Optional[str] = None

    # Multi-label support: episodes can instantiate multiple arcs
    secondary_arcs: List[tuple[ArcType, ArcPhase, float]] = Field(default_factory=list)

    # Provenance
    source_passages: List[SourcePassage] = Field(default_factory=list)
    extracted_from: List[str] = Field(default_factory=list)  # chunk IDs

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

    # Cycle membership
    parent_cycle_ids: Set[UUID] = Field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.id)

    def is_resolved(self) -> bool:
        """Check if episode has a resolution."""
        return self.resolution is not None and self.end_date is not None

    def get_phase_position(self) -> tuple[int, int] | None:
        """Return (current_phase, total_phases) if arc is defined."""
        if not self.arc_type or not self.arc_phase:
            return None
        # This would be populated from arc definition
        return None  # Placeholder


class ArcDefinition(BaseModel):
    """Definition of an archetypal arc with its phases."""

    model_config = ConfigDict(frozen=True)

    arc_type: ArcType
    name: str
    description: str
    phases: List[ArcPhase]  # Ordered list of phases
    phase_descriptions: Dict[ArcPhase, str] = Field(default_factory=dict)
    typical_duration: Optional[str] = None  # e.g., "2-5 years"

    # Historical base rates: from phase N, what typically follows
    transition_tendencies: Dict[ArcPhase, Dict[str, Any]] = Field(default_factory=dict)


class Cycle(BaseModel):
    """Recursive container for fractal cycle structure."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str
    scale: CycleScale
    description: Optional[str] = None

    # Temporal bounds
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Hierarchy
    parent_cycle_id: Optional[UUID] = None
    child_cycle_ids: Set[UUID] = Field(default_factory=set)
    episode_ids: Set[UUID] = Field(default_factory=set)

    # Cycle character
    dominant_arc_types: List[ArcType] = Field(default_factory=list)
    phase_estimate: Optional[ArcPhase] = None

    # Source framework (if bootstrapped)
    framework_source: Optional[str] = None  # e.g., "Turchin-secular-cycles"

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_depth(self) -> int:
        """Return depth in cycle hierarchy (root = 0)."""
        # This would be computed from graph traversal
        return 0  # Placeholder

    def contains_episode(self, episode_id: UUID) -> bool:
        """Check if cycle contains episode (directly or via children)."""
        return episode_id in self.episode_ids


class Thesis(BaseModel):
    """A generated forecast based on analog retrieval."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    query: str  # Present-day situation description
    query_date: datetime

    # Retrieved analogs
    analog_episode_ids: List[UUID] = Field(default_factory=list)
    analog_similarity_scores: List[float] = Field(default_factory=list)

    # Thesis content
    dominant_continuation: Optional[Continuation] = None
    alternative_continuations: List[tuple[str, float]] = Field(default_factory=list)  # (scenario, frequency)
    confidence: ThesisConfidence = ThesisConfidence.UNKNOWN
    watch_for_indicators: List[str] = Field(default_factory=list)
    key_uncertainties: List[str] = Field(default_factory=list)
    confidence_interval: Optional[tuple[float, float]] = None

    # LLM narrative synthesis (optional enhancement)
    narrative_synthesis: Optional[str] = None  # Human-readable interpretation

    # Timing
    estimated_duration: Optional[str] = None
    resolution_criteria: List[str] = Field(default_factory=list)

    # Citations
    cited_episodes: Dict[UUID, List[SourcePassage]] = Field(default_factory=dict)

    # Evaluation
    resolved: bool = False
    resolution_date: Optional[datetime] = None
    resolution_outcome: Optional[str] = None  # "accurate", "partial", "miss"
    brier_score: Optional[float] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str  # Model used (algorithm or LLM)
    taxonomy_version: str  # Arc taxonomy version


class ExtractionRecord(BaseModel):
    """Record of LLM extraction pipeline run."""

    id: UUID = Field(default_factory=uuid4)
    source_chunk_id: str
    pipeline_stage: str  # segmentation, extraction, classification, linking
    prompt_version: str
    model_used: str

    input: Dict[str, Any]
    output: Dict[str, Any]
    confidence: Optional[float] = None

    processing_time_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # For audit trail
    error_message: Optional[str] = None
    retry_count: int = 0
