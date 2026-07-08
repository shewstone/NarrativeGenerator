"""SQLAlchemy ORM models for database storage."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MappedColumn,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from narrative_engine.models import ArcPhase, ArcType, CycleScale
from narrative_engine.storage.database import Base


# Association table for many-to-many relationships
episode_actor_association = Table(
    "episode_actor_association",
    Base.metadata,
    MappedColumn(
        "episode_id",
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    MappedColumn(
        "actor_id",
        UUID(as_uuid=True),
        ForeignKey("actors.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


cycle_episode_association = Table(
    "cycle_episode_association",
    Base.metadata,
    MappedColumn(
        "cycle_id",
        UUID(as_uuid=True),
        ForeignKey("cycles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    MappedColumn(
        "episode_id",
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class ActorORM(Base):
    """ORM model for Actor."""

    __tablename__ = "actors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    episodes: Mapped[List["EpisodeORM"]] = relationship(
        "EpisodeORM",
        secondary=episode_actor_association,
        back_populates="actors",
    )

    def __repr__(self) -> str:
        return f"<ActorORM(id={self.id}, name={self.name}, role={self.role})>"


class SourcePassageORM(Base):
    """ORM model for SourcePassage."""

    __tablename__ = "source_passages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    work_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    passage_id: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chapter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    historiographic_school: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Foreign key to episode
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Composite index for lookups
    __table_args__ = (Index("ix_source_passages_work_passage", "work_id", "passage_id"),)

    def __repr__(self) -> str:
        return f"<SourcePassageORM(work_id={self.work_id}, passage_id={self.passage_id})>"


class EpisodeORM(Base):
    """ORM model for Episode."""

    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # Temporal fields
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    date_precision: Mapped[str] = mapped_column(String(20), default="year")

    # Setting
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    setting_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Narrative structure
    initiating_conditions: Mapped[list] = mapped_column(JSON, default=list)
    escalation_mechanics: Mapped[list] = mapped_column(JSON, default=list)
    tension: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    consequences: Mapped[list] = mapped_column(JSON, default=list)

    # Arc classification
    arc_type: Mapped[Optional[ArcType]] = mapped_column(Enum(ArcType), nullable=True)
    arc_phase: Mapped[Optional[ArcPhase]] = mapped_column(Enum(ArcPhase), nullable=True)
    phase_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    arc_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    secondary_arcs: Mapped[list] = mapped_column(JSON, default=list)  # List of tuples

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Provenance
    extracted_from: Mapped[list] = mapped_column(JSON, default=list)  # chunk IDs

    # Relationships
    actors: Mapped[List["ActorORM"]] = relationship(
        "ActorORM",
        secondary=episode_actor_association,
        back_populates="episodes",
    )
    source_passages: Mapped[List["SourcePassageORM"]] = relationship(
        "SourcePassageORM",
        back_populates="episode",
        cascade="all, delete-orphan",
    )
    cycles: Mapped[List["CycleORM"]] = relationship(
        "CycleORM",
        secondary=cycle_episode_association,
        back_populates="episodes",
    )

    # Vector embedding for semantic search
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(768),  # Using 768-dim for sentence-transformers
        nullable=True,
    )

    # Indexes
    __table_args__ = (
        Index("ix_episodes_arc_type", "arc_type"),
        Index("ix_episodes_arc_phase", "arc_phase"),
        Index("ix_episodes_start_date", "start_date"),
        Index("ix_episodes_embedding", "embedding", postgresql_using="ivfflat"),
    )

    def __repr__(self) -> str:
        return f"<EpisodeORM(id={self.id}, title={self.title[:50]})>"


class CycleORM(Base):
    """ORM model for Cycle."""

    __tablename__ = "cycles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    scale: Mapped[CycleScale] = mapped_column(Enum(CycleScale), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Temporal bounds
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Hierarchy
    parent_cycle_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cycles.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Cycle character
    dominant_arc_types: Mapped[list] = mapped_column(JSON, default=list)
    phase_estimate: Mapped[Optional[ArcPhase]] = mapped_column(Enum(ArcPhase), nullable=True)
    framework_source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    parent: Mapped[Optional["CycleORM"]] = relationship(
        "CycleORM",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[List["CycleORM"]] = relationship(
        "CycleORM",
        back_populates="parent",
    )
    episodes: Mapped[List["EpisodeORM"]] = relationship(
        "EpisodeORM",
        secondary=cycle_episode_association,
        back_populates="cycles",
    )

    # Indexes
    __table_args__ = (
        Index("ix_cycles_scale", "scale"),
        Index("ix_cycles_parent_id", "parent_cycle_id"),
    )

    def __repr__(self) -> str:
        return f"<CycleORM(id={self.id}, name={self.name}, scale={self.scale})>"


class ThesisORM(Base):
    """ORM model for Thesis."""

    __tablename__ = "theses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Retrieved analogs (stored as JSON list of UUIDs)
    analog_episode_ids: Mapped[list] = mapped_column(JSON, default=list)
    analog_similarity_scores: Mapped[list] = mapped_column(JSON, default=list)

    # Thesis content
    dominant_continuation: Mapped[str] = mapped_column(Text, nullable=False)
    alternative_continuations: Mapped[list] = mapped_column(JSON, default=list)
    watch_for_indicators: Mapped[list] = mapped_column(JSON, default=list)
    confidence_interval: Mapped[Optional[tuple]] = mapped_column(JSON, nullable=True)
    estimated_duration: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resolution_criteria: Mapped[list] = mapped_column(JSON, default=list)

    # Citations (JSON mapping episode_id -> passage_ids)
    cited_episodes: Mapped[dict] = mapped_column(JSON, default=dict)

    # Evaluation
    resolved: Mapped[bool] = mapped_column(default=False)
    resolution_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolution_outcome: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(100), nullable=False)

    # Indexes
    __table_args__ = (
        Index("ix_theses_resolved", "resolved"),
        Index("ix_theses_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ThesisORM(id={self.id}, query_date={self.query_date})>"


class ExtractionRecordORM(Base):
    """ORM model for ExtractionRecord (audit trail)."""

    __tablename__ = "extraction_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_chunk_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pipeline_stage: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    input: Mapped[dict] = mapped_column(JSON, default=dict)
    output: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<ExtractionRecordORM(id={self.id}, stage={self.pipeline_stage})>"


class ArcDefinitionORM(Base):
    """ORM model for ArcDefinition (taxonomy)."""

    __tablename__ = "arc_definitions"

    arc_type: Mapped[ArcType] = mapped_column(Enum(ArcType), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    phases: Mapped[list] = mapped_column(JSON, nullable=False)  # Ordered list of ArcPhase
    phase_descriptions: Mapped[dict] = mapped_column(JSON, default=dict)
    typical_duration: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    transition_tendencies: Mapped[dict] = mapped_column(JSON, default=dict)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")

    def __repr__(self) -> str:
        return f"<ArcDefinitionORM(arc_type={self.arc_type}, name={self.name})>"
