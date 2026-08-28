"""SQLAlchemy ORM models for taxonomy system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from narrative_engine.models import utcnow
from narrative_engine.storage.database import Base
from narrative_engine.taxonomy.models import TaxonomyStatus, TaxonomyType

# Association table for episode-arc membership (replaces hardcoded arc_type)
episode_arc_membership = Table(
    "episode_arc_membership",
    Base.metadata,
    Column(
        "episode_id",
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "arc_id",
        UUID(as_uuid=True),
        ForeignKey("arcs.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("arc_type", String(20), nullable=False),  # "canonical" or "discovered"
    Column("membership_score", Float, default=1.0),
    Column("confidence", Float, default=1.0),
    Column("phase", String(50), nullable=True),
    Column("phase_confidence", Float, default=0.0),
    Column("assignment_method", String(50), default="manual"),
    Column("rationale", Text, nullable=True),
    Column("distance_to_centroid", Float, nullable=True),
    Column("taxonomy_id", UUID(as_uuid=True), ForeignKey("taxonomies.id"), nullable=False),
    Column("assigned_at", DateTime, default=utcnow),
    Column("assigned_by", String(255), nullable=True),
)


class ArcTaxonomyORM(Base):
    """ORM model for ArcTaxonomy."""

    __tablename__ = "taxonomies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    taxonomy_type: Mapped[TaxonomyType] = mapped_column(
        Enum(TaxonomyType), nullable=False
    )
    status: Mapped[TaxonomyStatus] = mapped_column(
        Enum(TaxonomyStatus), default=TaxonomyStatus.DRAFT
    )

    version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    parent_taxonomy_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomies.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    discovery_params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Relationships
    arcs: Mapped[List["ArcORM"]] = relationship(
        "ArcORM",
        back_populates="taxonomy",
        foreign_keys="ArcORM.taxonomy_id",
    )


class ArcORM(Base):
    """ORM model for both CanonicalArc and DiscoveredArc.

    Uses a discriminator pattern to handle both types in one table,
    with type-specific fields stored in JSON.
    """

    __tablename__ = "arcs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Discriminator
    arc_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # "canonical" or "discovered"

    # Common fields
    slug: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embedding centroid (for discovered arcs)
    embedding_centroid: Mapped[Optional[List[float]]] = mapped_column(
        Vector(384), nullable=True  # Match episode embedding dimension
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Taxonomy membership
    taxonomy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomies.id"), nullable=False
    )

    # Canonical-specific fields (JSON)
    phases: Mapped[List[str]] = mapped_column(JSON, default=list)
    theoretical_sources: Mapped[List[str]] = mapped_column(JSON, default=list)
    keywords: Mapped[List[str]] = mapped_column(JSON, default=list)
    example_episodes: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Discovered-specific fields (JSON)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    silhouette_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    representative_episode_ids: Mapped[List[uuid.UUID]] = mapped_column(
        JSON, default=list
    )
    boundary_episode_ids: Mapped[List[uuid.UUID]] = mapped_column(JSON, default=list)
    common_features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    canonical_arc_mappings: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)

    # Discovery metadata
    discovered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    discovery_algorithm: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    discovery_params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Common metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    # Relationships
    taxonomy: Mapped["ArcTaxonomyORM"] = relationship(
        "ArcTaxonomyORM", back_populates="arcs"
    )

    def __repr__(self) -> str:
        return f"<ArcORM(id={self.id}, type={self.arc_type}, name={self.name})>"


class ArcComparisonORM(Base):
    """ORM model for ArcComparison."""

    __tablename__ = "arc_comparisons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    baseline_taxonomy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomies.id"), nullable=False
    )
    comparison_taxonomy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomies.id"), nullable=False
    )

    episode_agreement_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    silhouette_comparison: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    arc_mappings: Mapped[Dict[str, List[tuple[str, float]]]] = mapped_column(
        JSON, default=dict
    )

    retrieval_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retrieval_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thesis_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    evaluated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
