"""Core data models for the Narrative Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now. All persisted timestamps are aware: the DB
    columns are timestamptz, and naive/aware mixing breaks round-trip
    equality (T3) besides being deprecated."""
    return datetime.now(timezone.utc)


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


class MechanismTag(str, Enum):
    """Turchin-style structural drivers.

    Controlled vocabulary for mechanisms that drive historical change.
    Used to tag episodes, condition phase transitions, and compute
    mechanism-density indices.
    """

    # Structural-demographic mechanisms (Turchin)
    ELITE_OVERPRODUCTION = "elite_overproduction"
    ELITE_INTRA_COMPETITION = "elite_intra_competition"
    POPULAR_IMMISERATION = "popular_immiseration"
    FISCAL_DISTRESS = "fiscal_distress"
    STATE_FRAGILITY = "state_fragility"
    GEOPOLITICAL_PRESSURE = "geopolitical_pressure"

    # Institutional mechanisms
    INSTITUTIONAL_DECAY = "institutional_decay"
    BUREAUCRATIC_SCLEROSIS = "bureaucratic_sclerosis"
    REGULATORY_CAPTURE = "regulatory_capture"
    CORRUPTION_SPIRAL = "corruption_spiral"

    # Economic mechanisms
    CREDIT_EXPANSION = "credit_expansion"
    DEBT_OVERHANG = "debt_overhang"
    CURRENCY_CRISIS = "currency_crisis"
    ASSET_BUBBLE = "asset_bubble"

    # Social mechanisms
    GENERATIONAL_FORGETTING = "generational_forgetting"
    COHESION_EROSION = "cohesion_erosion"
    IDENTITY_POLARIZATION = "identity_polarization"

    # Narrative mechanisms
    CULTURAL_DECADENCE = "cultural_decadence"
    HUBRIS_CULTURE = "hubris_culture"
    REFORM_RESISTANCE = "reform_resistance"


class ScopeKind(str, Enum):
    """Kinds of focal scope, from a person to a whole system.

    These describe *what the bounded subject is*, not how important it is.
    In particular, parties and factions are first-class scopes rather than
    being flattened into the polity that contains them.
    """

    PERSON = "person"
    DYAD = "dyad"
    GROUP = "group"
    FACTION = "faction"
    IDEA = "idea"
    DISCOURSE = "discourse"
    MOVEMENT = "movement"
    ORGANIZATION = "organization"
    PARTY = "party"
    INSTITUTION = "institution"
    POLITY = "polity"
    CIVILIZATION = "civilization"
    REGION = "region"
    SYSTEM = "system"


class SituationScale(str, Enum):
    """Analytical scale of a situation, independent of subject matter."""

    PERSONAL = "personal"
    INTERPERSONAL = "interpersonal"
    GROUP = "group"
    ORGANIZATION = "organization"
    INSTITUTION = "institution"
    POLITY = "polity"
    CIVILIZATION = "civilization"
    SYSTEM = "system"


class SituationDomain(str, Enum):
    """Facets through which the same change can be observed."""

    PSYCHOLOGICAL = "psychological"
    RELATIONAL = "relational"
    ORGANIZATIONAL = "organizational"
    POLITICAL = "political"
    ECONOMIC = "economic"
    CULTURAL = "cultural"
    TECHNOLOGICAL = "technological"
    ECOLOGICAL = "ecological"
    MILITARY = "military"
    RELIGIOUS = "religious"


class ChangePattern(str, Enum):
    """Scale-neutral modes of change.

    Unlike the legacy ``ArcType`` labels, these do not assume that the
    subject is a market, state, civilization, or literary protagonist.
    """

    EMERGENCE_AND_GATHERING = "emergence_and_gathering"
    EXPANSION_AND_CONSOLIDATION = "expansion_and_consolidation"
    SATURATION_AND_OVERREACH = "saturation_and_overreach"
    TENSION_AND_CONTESTATION = "tension_and_contestation"
    FRAGMENTATION_AND_RELEASE = "fragmentation_and_release"
    RETREAT_AND_PRESERVATION = "retreat_and_preservation"
    TURNING_AND_REORIENTATION = "turning_and_reorientation"
    RENEWAL_AND_INTEGRATION = "renewal_and_integration"
    SUCCESSION_AND_TRANSFER = "succession_and_transfer"


class MechanismFamily(str, Enum):
    """Broad causal families that travel across domains and scales."""

    AMPLIFICATION_FEEDBACK = "amplification_feedback"
    RESOURCE_STRAIN = "resource_strain"
    LEGITIMACY_EROSION = "legitimacy_erosion"
    COORDINATION_FAILURE = "coordination_failure"
    BOUNDARY_PRESSURE = "boundary_pressure"
    SUCCESSION_DYNAMICS = "succession_dynamics"
    INSTITUTIONAL_LOCK_IN = "institutional_lock_in"
    ADAPTATION_LEARNING = "adaptation_learning"
    CONTAGION_DIFFUSION = "contagion_diffusion"
    COOPERATION_ALIGNMENT = "cooperation_alignment"
    COMPETITION_DISPLACEMENT = "competition_displacement"
    MEMORY_LOSS = "memory_loss"


class SituationConfiguration(BaseModel):
    """Six continuous axes describing a situation without domain labels.

    Values run from -1 to +1; ``None`` means the source does not support an
    estimate. Positive poles are abundant capacity, integrated cohesion,
    acute pressure, accepted legitimacy, flexible adaptation, and
    concentrated agency respectively.
    """

    model_config = ConfigDict(frozen=True)

    capacity: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    cohesion: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    pressure: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    legitimacy: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    adaptability: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    agency: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class ClassificationState(str, Enum):
    """Whether an episode cleared the classification confidence floor.

    Classification is never a forced choice (design doc Sec 6.2 stage 4):
    if no canonical arc clears tau_class, the episode carries NO arc
    assignment and is marked UNCLASSIFIED rather than shoehorned into its
    least-bad label. Unclassified episodes stay out of the arc-conditioned
    analog base (Sec 6.5) and are re-attempted on taxonomy version bumps.
    """

    CLASSIFIED = "classified"
    UNCLASSIFIED = "unclassified"


class ThesisMode(str, Enum):
    """Forecast mode (design doc Sec 6.5.8).

    ARC_LESS is the visible degraded path: the query situation itself
    failed the classification floor, so there is no phase and no
    phase-completion forecast -- only bare structural nearest-neighbor
    retrieval with wider stated uncertainty. Silent shoehorning and silent
    failure are both prohibited; a visible degraded answer is the honest
    outcome.
    """

    ARC_BASED = "arc_based"
    ARC_LESS = "arc_less"


class CycleScale(str, Enum):
    """Fractal cycle scales."""

    CIVILIZATIONAL = "civilizational"  # ~centuries
    INSTITUTIONAL = "institutional"  # ~decades
    GENERATIONAL = "generational"  # ~20-25 years
    EPISODIC = "episodic"  # individual events


class EdgeKind(str, Enum):
    """Kinds of relationships between episodes.

    CORRECTION: Separate from link_status (attested vs inferred).
    Edge kind is orthogonal to evidentiary status.
    """

    CAUSES = "causes"  # Causal relationship
    PRECEDES = "precedes"  # Temporal precedence
    COMPOSES = "composes"  # Part of arc instance
    MEMBER_OF = "member_of"  # Part of cycle
    SAME_EVENT_AS = "same_event_as"  # Identity (surface embedding match)


class LinkStatus(str, Enum):
    """Evidentiary status of a link.

    Invariant: CAUSES edges must be ATTESTED (no inferred causal claims).
    """

    ATTESTED = "attested"  # Backed by textual evidence spans
    INFERRED = "inferred"  # Created by composition pass on structural grounds


class ReviewStatus(str, Enum):
    """Review state for inferred links.

    Inferred links above threshold go to human review queue.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO = "auto"  # Below review threshold; never entered the queue


class ArcAssignment(BaseModel):
    """A per-scope arc/phase reading for an episode.

    The same episode can carry a different reading in each cycle it
    belongs to (Sec 2): a trade war might be a REFORM_REACTION beat in one
    polity's institutional cycle and a RISE_OVEREXTENSION beat in the
    system cycle.
    """

    model_config = ConfigDict(frozen=True)

    arc_type: "ArcType"
    phase_index: int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: Optional[str] = None  # short; stored for audit


class CycleMembership(BaseModel):
    """Episode <-> cycle membership, many-to-many with a per-scope reading.

    The same episode can belong to multiple cycles (e.g. a trade war
    belongs to each participating polity's cycle AND the system-scoped
    cycle) with a different arc/phase reading in each -- `reading` carries
    that. `link_status` and `review_status` are orthogonal axes (Sec 4):
    link_status is *how we know* (attested vs. inferred), review_status is
    *whether a human has ratified it*. They must never collapse into one
    enum -- a composed-but-approved membership and an attested-but-pending
    one are both representable states.

    COMPOSES edges (episode -> arc-instance cycle) are a specialization of
    this for cycles with is_arc_instance=True; phase_coverage is only
    meaningful in that case.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    episode_id: UUID
    cycle_id: UUID

    reading: Optional[ArcAssignment] = None
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    phase_coverage: List[int] = Field(default_factory=list)

    link_status: LinkStatus = LinkStatus.ATTESTED
    review_status: ReviewStatus = ReviewStatus.AUTO

    created_at: datetime = Field(default_factory=utcnow)


class EpisodeLink(BaseModel):
    """Episode <-> episode edge: CAUSES, PRECEDES, SAME_EVENT_AS.

    Distinct from CycleMembership (episode <-> cycle). The causal-
    attestation invariant is enforced here at the model layer -- a
    validator, not prose -- mirroring the DB CHECK constraint
    (chk_causal_must_be_attested) so the invariant holds before a write
    ever reaches Postgres: CAUSES edges must be ATTESTED, since the
    composition pass has no textual evidence and can only ever produce
    inferred links.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    source_episode_id: UUID
    target_episode_id: UUID
    edge_kind: EdgeKind
    link_status: LinkStatus = LinkStatus.ATTESTED
    distance: Optional[float] = None  # semantic distance, for inferred links
    evidence: Optional[str] = None  # quote (attested) or similarity score (inferred)
    review_status: ReviewStatus = ReviewStatus.AUTO
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _causal_links_must_be_attested(self) -> "EpisodeLink":
        if self.edge_kind == EdgeKind.CAUSES and self.link_status == LinkStatus.INFERRED:
            raise ValueError(
                "CAUSES edges must be attested: no inferred causal claims "
                "(design doc Sec 4 invariant; mirrors DB CHECK "
                "chk_causal_must_be_attested)"
            )
        return self


class Continuation(BaseModel):
    """A possible continuation/outcome with probability."""

    description: str
    probability: float
    supporting_analogs: int = 0


class SourceDocumentStatus(str, Enum):
    """Lifecycle of a dropped source file (T7)."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DUPLICATE = "duplicate"  # same bytes already processed; never re-run


class SourceDocument(BaseModel):
    """A file dropped into the watch directory, tracked through processing.

    content_hash is the duplicate guard: the same bytes under any filename
    become a visible `duplicate` row pointing at the original instead of
    silently reprocessing (duplicate narrations corrupt analog counts --
    the failure class T6 mitigates downstream, prevented at the door here).
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    filename: str
    content_hash: str  # sha256 hex of raw bytes
    size_bytes: int = 0
    status: SourceDocumentStatus = SourceDocumentStatus.QUEUED
    error: Optional[str] = None
    chunks_created: int = 0
    # Live progress counter: how many chunks have completed extraction.
    # The watcher commits after every chunk so the dashboard's polling
    # session sees movement during long LLM runs.
    chunks_processed: int = 0
    episodes_created: int = 0
    # False when processing completed without an LLM key: ingested and
    # chunked, extraction pending -- visible degradation, never silent.
    extraction_ran: bool = False
    duplicate_of: Optional[UUID] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Scope(BaseModel):
    """A bounded subject that cycle trees and episodes belong to.

    The registry of scopes is versioned data, not ontology (Sec 9): whether
    "the West" is one scope or "China" spans dynastic breaks are hypotheses
    someone will revise. Canonical entries live in data/scope_registry.json;
    see narrative_engine.scopes for alias resolution.
    """

    model_config = ConfigDict(frozen=True)

    id: str  # slug: "us", "china", "intl_system"
    kind: ScopeKind
    name: str
    parent_scope_id: Optional[str] = None  # e.g. faction -> party -> polity
    aliases: List[str] = Field(default_factory=list)
    notes: Optional[str] = None  # where the boundary is contested, say so


class Actor(BaseModel):
    """An actor participating in an episode."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    name: str
    role: str  # raw extraction output, free text -- residue signal (T2)
    # Controlled-vocabulary structural position (extraction.roles.ActorRole
    # value) with the classifier's fit confidence. None when no role cleared
    # the fit floor (tau_role) -- no forced choice, same discipline as
    # tau_class. Only canonical_role may enter the structural render; the
    # free-text `role` never does.
    canonical_role: Optional[str] = None
    role_fit_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
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
    # Signed historical years preserve dates outside Python/PostgreSQL's CE
    # datetime range. Negative values mean BCE; year zero is not used.
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    date_precision: str = "year"  # year, month, day

    # Setting and scope (CORRECTION: scope is primary partition key)
    location: Optional[str] = None
    setting_description: Optional[str] = None
    scope_id: Optional[str] = None  # polity/institution scope, e.g., "us_national"
    # Raw focal-scope claims let extraction represent a previously unseen
    # party, faction, team, relationship, or person without inventing a
    # canonical registry id. ``scope_id`` is populated only when exact alias
    # resolution succeeds; composition safely falls back to scope_name.
    scope_name: Optional[str] = None
    scope_kind: Optional[ScopeKind] = None
    parent_scope_name: Optional[str] = None
    scope_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scope_evidence: Optional[str] = None
    scope_notes: Optional[str] = None

    # Narrative structure
    actors: List[Actor] = Field(default_factory=list)
    initiating_conditions: List[str] = Field(default_factory=list)
    escalation_mechanics: List[str] = Field(default_factory=list)
    tension: Optional[str] = None
    resolution: Optional[str] = None
    consequences: List[str] = Field(default_factory=list)

    # Structural drivers (design doc Sec 3.8): controlled-vocabulary tags for
    # the mechanisms at work in this episode, assigned during classification
    # alongside arc_type/arc_phase (same analytical-judgment stage).
    mechanism_tags: List[MechanismTag] = Field(default_factory=list)

    # Scale-neutral situation reading. This is the primary ontology; the
    # legacy arc fields below remain for compatibility and specialised views.
    change_pattern: Optional[ChangePattern] = None
    pattern_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pattern_rationale: Optional[str] = None
    situation_scale: Optional[SituationScale] = None
    domains: List[SituationDomain] = Field(default_factory=list)
    configuration: SituationConfiguration = Field(default_factory=SituationConfiguration)
    mechanism_families: List[MechanismFamily] = Field(default_factory=list)

    # Arc classification. arc_type/arc_phase are None when the episode is
    # UNCLASSIFIED (no forced choice -- see ClassificationState); confidence
    # and rationale are kept even then, for audit.
    arc_type: Optional[ArcType] = None
    arc_phase: Optional[ArcPhase] = None
    phase_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    arc_rationale: Optional[str] = None
    classification_state: ClassificationState = ClassificationState.CLASSIFIED

    # Multi-label support: episodes can instantiate multiple arcs
    secondary_arcs: List[tuple[ArcType, ArcPhase, float]] = Field(default_factory=list)

    # Provenance
    source_passages: List[SourcePassage] = Field(default_factory=list)
    extracted_from: List[str] = Field(default_factory=list)  # chunk IDs
    # When the source itself became available. Historical backtests must not
    # retrieve an episode extracted from a work published after their cutoff,
    # even when the episode describes an earlier event.
    source_published_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    version: int = 1

    # Cycle membership
    parent_cycle_ids: Set[UUID] = Field(default_factory=set)

    # Embeddings (CORRECTION v0.5, Sec 3.3a): two distinct roles, never swapped.
    # surface_embedding = raw title/summary text -> identity ("same happening?"),
    #   consumed by SAME_EVENT_AS resolution and arc composition.
    # structural_embedding = role-substituted narrative template -> analogy
    #   ("same shape?"), consumed by analog retrieval and discovery clustering.
    surface_embedding: Optional[List[float]] = None
    structural_embedding: Optional[List[float]] = None

    # Which (render template, embedding model) produced each vector (T4).
    # Vectors from different epochs live in different similarity spaces:
    # retrieval filters to the current epoch; composition treats a mismatch
    # as a missing signal. None whenever the vector is None.
    surface_embedding_epoch: Optional[str] = None
    structural_embedding_epoch: Optional[str] = None

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


class TransitionTendency(BaseModel):
    """A historical base rate for one phase transition, optionally
    conditioned on mechanism presence.

    E.g. "distress -> panic is more frequent when leverage_buildup was
    tagged in prior phases": base_weight is the unconditioned tendency,
    conditioned_weight is the tendency when conditioning_mechanisms were
    present. This is where Turchin-style falsifiable claims ("disintegrative
    follows elite_overproduction at lag L") live as scoreable data rather
    than prose.
    """

    model_config = ConfigDict(frozen=True)

    from_phase: ArcPhase
    to_phase: ArcPhase
    base_weight: float = Field(ge=0.0, le=1.0)
    conditioning_mechanisms: List[MechanismTag] = Field(default_factory=list)  # empty = unconditional
    conditioned_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ArcDefinition(BaseModel):
    """Definition of an archetypal arc with its phases."""

    model_config = ConfigDict(frozen=True)

    arc_type: ArcType
    name: str
    description: str
    phases: List[ArcPhase]  # Ordered list of phases
    phase_descriptions: Dict[ArcPhase, str] = Field(default_factory=dict)
    typical_duration: Optional[str] = None  # e.g., "2-5 years"

    # Historical base rates: from phase N, what typically follows.
    # list[TransitionTendency], not dict[phase, dict] -- a transition can
    # carry mechanism-conditioned weights, which a bare dict can't express
    # without an ad-hoc inner schema (Sec 4).
    transition_tendencies: List[TransitionTendency] = Field(default_factory=list)


class Cycle(BaseModel):
    """Recursive container for fractal cycle structure."""

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str
    scale: CycleScale
    description: Optional[str] = None

    # Polity/institution scope this cycle tree belongs to (CORRECTION: every
    # cycle tree is scoped, there is no global phase clock -- mirrors
    # Episode.scope_id). None only for cycles that predate scoping or
    # deliberately span all scopes (e.g. imported frameworks pre-binding).
    scope_id: Optional[str] = None

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

    # Arc instances (Sec 2) are episodic-scale cycles by convention,
    # populated by the composition pass -- this flag is what distinguishes
    # them from ordinary cycles for COMPOSES-vs-MEMBER_OF disambiguation.
    is_arc_instance: bool = False

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

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

    # Forecast mode (Sec 6.5.8): arc_less when the query situation failed
    # the classification floor -- no phase, so no phase-completion forecast.
    mode: ThesisMode = ThesisMode.ARC_BASED

    # Which scope-registry version was in effect at generation (Sec 4;
    # scope boundaries are versioned hypotheses, Sec 9).
    scope_registry_version: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
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
    created_at: datetime = Field(default_factory=utcnow)

    # For audit trail
    error_message: Optional[str] = None
    retry_count: int = 0
