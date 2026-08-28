"""Pydantic models for hybrid taxonomy system.

Supports both canonical (hand-coded) and discovered (emergent) arcs,
with soft membership and taxonomy versioning for experimentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from narrative_engine.models import utcnow


class TaxonomyType(str, Enum):
    """Type of taxonomy."""

    CANONICAL = "canonical"  # Hand-coded archetypes
    DISCOVERED = "discovered"  # Emergent from clustering
    HYBRID = "hybrid"  # Combined canonical + discovered


class TaxonomyStatus(str, Enum):
    """Status of a taxonomy version."""

    DRAFT = "draft"  # Still being developed
    ACTIVE = "active"  # Currently used for classification
    DEPRECATED = "deprecated"  # Older version, kept for comparison
    ARCHIVED = "archived"  # No longer used


class ArcTaxonomy(BaseModel):
    """A versioned taxonomy of narrative arcs.

    Allows experimentation with different arc sets and comparison
    between canonical, discovered, and hybrid taxonomies.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    name: str  # e.g., "v1-canonical", "v2-discovered-clusters", "v3-hybrid"
    description: Optional[str] = None

    taxonomy_type: TaxonomyType
    status: TaxonomyStatus = TaxonomyStatus.DRAFT

    # Versioning
    version: str = "1.0.0"
    parent_taxonomy_id: Optional[UUID] = None  # For tracking lineage

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    created_by: Optional[str] = None  # user or system

    # For discovered taxonomies: clustering parameters used
    discovery_params: Dict[str, Any] = Field(default_factory=dict)
    # e.g., {"algorithm": "hdbscan", "min_cluster_size": 5, "metric": "cosine"}

    def activate(self) -> None:
        """Mark this taxonomy as active."""
        self.status = TaxonomyStatus.ACTIVE
        self.updated_at = utcnow()

    def deprecate(self) -> None:
        """Mark this taxonomy as deprecated."""
        self.status = TaxonomyStatus.DEPRECATED
        self.updated_at = utcnow()


class CanonicalArc(BaseModel):
    """A hand-coded archetypal arc (migrated from ArcType enum).

    These are the "expert" arcs based on narrative theory,
    financial cycles, and historiography.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    slug: str  # e.g., "credit_boom_and_bust"
    name: str  # Display name: "Credit Boom and Bust"
    description: str

    # Phase definitions for this arc
    phases: List[str] = Field(default_factory=list)
    # e.g., ["boom", "euphoria", "distress", "panic", "revulsion"]

    # Theoretical basis
    theoretical_sources: List[str] = Field(default_factory=list)
    # e.g., ["Kindleberger-1978", "Minsky-1986"]

    # For similarity matching
    keywords: List[str] = Field(default_factory=list)
    example_episodes: List[str] = Field(default_factory=list)

    # Taxonomy membership
    taxonomy_ids: List[UUID] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def __hash__(self) -> int:
        return hash(self.id)


class DiscoveredArc(BaseModel):
    """An emergent arc discovered through clustering analysis.

    These arcs are learned from data, not hand-coded. Each represents
    a cluster of similar episodes in embedding space.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    cluster_id: int  # From clustering algorithm (e.g., HDBSCAN label)
    name: Optional[str] = None  # Auto-generated or manually labeled later
    description: Optional[str] = None  # LLM-generated summary of cluster

    # Centroid in embedding space (for similarity queries)
    embedding_centroid: Optional[List[float]] = None
    embedding_model: Optional[str] = None  # e.g., "sentence-transformers/all-MiniLM-L6-v2"

    # Cluster statistics
    member_count: int = 0
    silhouette_score: Optional[float] = None  # Cluster quality metric

    # Representative episodes (centroid + boundary cases)
    representative_episode_ids: List[UUID] = Field(default_factory=list)
    boundary_episode_ids: List[UUID] = Field(default_factory=list)

    # Discovered characteristics (from analyzing members)
    common_features: Dict[str, Any] = Field(default_factory=dict)
    # e.g., {"avg_duration_months": 18.5, "common_locations": ["US", "UK"]}

    # Relationship to canonical arcs (if mapped)
    canonical_arc_mappings: Dict[str, float] = Field(default_factory=dict)
    # e.g., {"credit_boom_and_bust": 0.75, "hubris_nemesis": 0.25}

    # Taxonomy membership
    taxonomy_ids: List[UUID] = Field(default_factory=list)

    # Discovery metadata
    discovered_at: datetime = Field(default_factory=utcnow)
    discovery_algorithm: Optional[str] = None
    discovery_params: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)


class ArcMembership(BaseModel):
    """Soft membership of an episode in an arc.

    Replaces the hard ArcType enum assignment with probabilistic
    membership. Episodes can belong to multiple arcs with varying
    confidence scores.
    """

    model_config = ConfigDict(frozen=True)

    episode_id: UUID
    arc_id: UUID
    arc_type: str  # "canonical" or "discovered"

    # Soft membership
    membership_score: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    confidence: float = Field(ge=0.0, le=1.0)

    # Phase position (if applicable)
    phase: Optional[str] = None
    phase_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # How was this assigned?
    assignment_method: str = "manual"  # "manual", "llm", "clustering", "hybrid"

    # For LLM-assigned: the rationale
    rationale: Optional[str] = None

    # For clustering-assigned: distance to centroid
    distance_to_centroid: Optional[float] = None

    # Versioning
    taxonomy_id: UUID
    assigned_at: datetime = Field(default_factory=utcnow)
    assigned_by: Optional[str] = None  # user, model name, or algorithm

    def __hash__(self) -> int:
        return hash((self.episode_id, self.arc_id))


class ArcComparison(BaseModel):
    """Comparison between two taxonomies or arc sets.

    Used for evaluating discovered taxonomies against canonical
    or comparing different versions.
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    baseline_taxonomy_id: UUID
    comparison_taxonomy_id: UUID

    # Overall similarity metrics
    episode_agreement_score: Optional[float] = None  # % episodes with same arc
    silhouette_comparison: Optional[float] = None  # clustering quality difference

    # Arc-level mappings
    arc_mappings: Dict[UUID, List[tuple[UUID, float]]] = Field(default_factory=dict)
    # canonical_arc_id -> [(discovered_arc_id, similarity), ...]

    # Evaluation metrics
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    thesis_accuracy: Optional[float] = None

    created_at: datetime = Field(default_factory=utcnow)
    evaluated_by: Optional[str] = None
