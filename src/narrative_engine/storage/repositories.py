"""Repository pattern for database operations."""

from __future__ import annotations

from typing import List, Optional, Sequence
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from narrative_engine.models import Episode, Cycle, Actor, Thesis, ArcDefinition
from narrative_engine.storage.orm_models import (
    EpisodeORM,
    CycleORM,
    ActorORM,
    ThesisORM,
    ArcDefinitionORM,
)


class EpisodeRepository:
    """Repository for Episode CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, episode: Episode) -> Episode:
        """Create a new episode."""
        orm_episode = self._to_orm(episode)
        self.session.add(orm_episode)
        await self.session.flush()
        await self.session.refresh(orm_episode)
        return self._from_orm(orm_episode)

    async def get_by_id(self, episode_id: UUID) -> Optional[Episode]:
        """Get episode by ID with all relationships."""
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.id == episode_id)
            .options(
                joinedload(EpisodeORM.actors),
                joinedload(EpisodeORM.source_passages),
                joinedload(EpisodeORM.cycles),
            )
        )
        orm_episode = result.scalar_one_or_none()
        return self._from_orm(orm_episode) if orm_episode else None

    async def get_by_arc_type(
        self,
        arc_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Episode]:
        """Get episodes by arc type."""
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.arc_type == arc_type)
            .limit(limit)
            .offset(offset)
            .order_by(EpisodeORM.start_date)
        )
        return [self._from_orm(e) for e in result.scalars().all()]

    async def get_by_arc_phase(
        self,
        arc_phase: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Episode]:
        """Get episodes by arc phase."""
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.arc_phase == arc_phase)
            .limit(limit)
            .offset(offset)
            .order_by(EpisodeORM.phase_confidence.desc())
        )
        return [self._from_orm(e) for e in result.scalars().all()]

    async def update(self, episode: Episode) -> Episode:
        """Update an existing episode."""
        orm_episode = await self.session.get(EpisodeORM, episode.id)
        if not orm_episode:
            raise ValueError(f"Episode {episode.id} not found")

        self._update_orm(orm_episode, episode)
        await self.session.flush()
        await self.session.refresh(orm_episode)
        return self._from_orm(orm_episode)

    async def delete(self, episode_id: UUID) -> bool:
        """Delete an episode. Returns True if deleted."""
        orm_episode = await self.session.get(EpisodeORM, episode_id)
        if orm_episode:
            await self.session.delete(orm_episode)
            return True
        return False

    async def search_by_embedding(
        self,
        embedding: List[float],
        limit: int = 10,
    ) -> Sequence[tuple[Episode, float]]:
        """Semantic search using vector similarity."""
        from pgvector.sqlalchemy import cosine_distance

        result = await self.session.execute(
            select(EpisodeORM, cosine_distance(EpisodeORM.embedding, embedding).label("distance"))
            .where(EpisodeORM.embedding.is_not(None))
            .order_by("distance")
            .limit(limit)
        )

        return [(self._from_orm(row.EpisodeORM), float(row.distance)) for row in result.all()]

    async def count(self) -> int:
        """Get total episode count."""
        result = await self.session.execute(select(func.count(EpisodeORM.id)))
        return result.scalar() or 0

    def _to_orm(self, episode: Episode) -> EpisodeORM:
        """Convert Pydantic model to ORM."""
        from narrative_engine.storage.orm_models import SourcePassageORM

        return EpisodeORM(
            id=episode.id,
            title=episode.title,
            summary=episode.summary,
            start_date=episode.start_date,
            end_date=episode.end_date,
            date_precision=episode.date_precision,
            location=episode.location,
            setting_description=episode.setting_description,
            initiating_conditions=episode.initiating_conditions,
            escalation_mechanics=episode.escalation_mechanics,
            tension=episode.tension,
            resolution=episode.resolution,
            consequences=episode.consequences,
            arc_type=episode.arc_type,
            arc_phase=episode.arc_phase,
            phase_confidence=episode.phase_confidence,
            arc_rationale=episode.arc_rationale,
            secondary_arcs=episode.secondary_arcs,
            extracted_from=episode.extracted_from,
            version=episode.version,
            embedding=episode.embedding if hasattr(episode, "embedding") else None,
        )

    def _from_orm(self, orm: EpisodeORM) -> Episode:
        """Convert ORM to Pydantic model."""
        from narrative_engine.models import Actor, SourcePassage

        return Episode(
            id=orm.id,
            title=orm.title,
            summary=orm.summary,
            start_date=orm.start_date,
            end_date=orm.end_date,
            date_precision=orm.date_precision,
            location=orm.location,
            setting_description=orm.setting_description,
            actors=[
                Actor(
                    id=a.id,
                    name=a.name,
                    role=a.role,
                    attributes=a.attributes,
                )
                for a in (orm.actors or [])
            ],
            initiating_conditions=orm.initiating_conditions,
            escalation_mechanics=orm.escalation_mechanics,
            tension=orm.tension,
            resolution=orm.resolution,
            consequences=orm.consequences,
            arc_type=orm.arc_type,
            arc_phase=orm.arc_phase,
            phase_confidence=orm.phase_confidence,
            arc_rationale=orm.arc_rationale,
            secondary_arcs=orm.secondary_arcs,
            source_passages=[
                SourcePassage(
                    work_id=sp.work_id,
                    passage_id=sp.passage_id,
                    text=sp.text,
                    chapter=sp.chapter,
                    section=sp.section,
                    page=sp.page,
                    historiographic_school=sp.historiographic_school,
                )
                for sp in (orm.source_passages or [])
            ],
            extracted_from=orm.extracted_from,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            version=orm.version,
        )

    def _update_orm(self, orm: EpisodeORM, episode: Episode) -> None:
        """Update ORM from Pydantic model."""
        orm.title = episode.title
        orm.summary = episode.summary
        orm.start_date = episode.start_date
        orm.end_date = episode.end_date
        orm.location = episode.location
        orm.setting_description = episode.setting_description
        orm.initiating_conditions = episode.initiating_conditions
        orm.escalation_mechanics = episode.escalation_mechanics
        orm.tension = episode.tension
        orm.resolution = episode.resolution
        orm.consequences = episode.consequences
        orm.arc_type = episode.arc_type
        orm.arc_phase = episode.arc_phase
        orm.phase_confidence = episode.phase_confidence
        orm.arc_rationale = episode.arc_rationale
        orm.secondary_arcs = episode.secondary_arcs
        orm.version = episode.version + 1


class CycleRepository:
    """Repository for Cycle CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, cycle: Cycle) -> Cycle:
        """Create a new cycle."""
        orm_cycle = CycleORM(
            id=cycle.id,
            name=cycle.name,
            scale=cycle.scale,
            description=cycle.description,
            start_date=cycle.start_date,
            end_date=cycle.end_date,
            parent_cycle_id=cycle.parent_cycle_id,
            dominant_arc_types=cycle.dominant_arc_types,
            phase_estimate=cycle.phase_estimate,
            framework_source=cycle.framework_source,
        )
        self.session.add(orm_cycle)
        await self.session.flush()
        return cycle

    async def get_by_id(self, cycle_id: UUID) -> Optional[Cycle]:
        """Get cycle by ID."""
        result = await self.session.execute(
            select(CycleORM)
            .where(CycleORM.id == cycle_id)
            .options(
                joinedload(CycleORM.children),
                joinedload(CycleORM.episodes),
            )
        )
        orm_cycle = result.scalar_one_or_none()
        return self._from_orm(orm_cycle) if orm_cycle else None

    async def get_by_scale(
        self,
        scale: str,
        limit: int = 100,
    ) -> Sequence[Cycle]:
        """Get cycles by scale."""
        result = await self.session.execute(
            select(CycleORM).where(CycleORM.scale == scale).limit(limit)
        )
        return [self._from_orm(c) for c in result.scalars().all()]

    async def get_children(self, cycle_id: UUID) -> Sequence[Cycle]:
        """Get child cycles."""
        result = await self.session.execute(
            select(CycleORM).where(CycleORM.parent_cycle_id == cycle_id)
        )
        return [self._from_orm(c) for c in result.scalars().all()]

    async def add_episode(self, cycle_id: UUID, episode_id: UUID) -> None:
        """Add episode to cycle."""
        from narrative_engine.storage.orm_models import cycle_episode_association

        await self.session.execute(
            cycle_episode_association.insert().values(
                cycle_id=cycle_id,
                episode_id=episode_id,
            )
        )

    def _from_orm(self, orm: CycleORM) -> Cycle:
        """Convert ORM to Pydantic model."""
        return Cycle(
            id=orm.id,
            name=orm.name,
            scale=orm.scale,
            description=orm.description,
            start_date=orm.start_date,
            end_date=orm.end_date,
            parent_cycle_id=orm.parent_cycle_id,
            child_cycle_ids=set(),  # Would need separate query
            episode_ids=set(e.id for e in (orm.episodes or [])),
            dominant_arc_types=orm.dominant_arc_types,
            phase_estimate=orm.phase_estimate,
            framework_source=orm.framework_source,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )


class ThesisRepository:
    """Repository for Thesis CRUD and evaluation."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, thesis: Thesis) -> Thesis:
        """Create a new thesis."""
        orm_thesis = ThesisORM(
            id=thesis.id,
            query=thesis.query,
            query_date=thesis.query_date,
            analog_episode_ids=thesis.analog_episode_ids,
            analog_similarity_scores=thesis.analog_similarity_scores,
            dominant_continuation=thesis.dominant_continuation,
            alternative_continuations=thesis.alternative_continuations,
            watch_for_indicators=thesis.watch_for_indicators,
            confidence_interval=thesis.confidence_interval,
            estimated_duration=thesis.estimated_duration,
            resolution_criteria=thesis.resolution_criteria,
            cited_episodes=thesis.cited_episodes,
            model_version=thesis.model_version,
            taxonomy_version=thesis.taxonomy_version,
        )
        self.session.add(orm_thesis)
        await self.session.flush()
        return thesis

    async def get_by_id(self, thesis_id: UUID) -> Optional[Thesis]:
        """Get thesis by ID."""
        orm_thesis = await self.session.get(ThesisORM, thesis_id)
        return self._from_orm(orm_thesis) if orm_thesis else None

    async def get_unresolved(self, limit: int = 100) -> Sequence[Thesis]:
        """Get unresolved theses for monitoring."""
        result = await self.session.execute(
            select(ThesisORM)
            .where(ThesisORM.resolved == False)
            .order_by(ThesisORM.created_at.desc())
            .limit(limit)
        )
        return [self._from_orm(t) for t in result.scalars().all()]

    async def resolve(
        self,
        thesis_id: UUID,
        outcome: str,
        brier_score: Optional[float] = None,
    ) -> Optional[Thesis]:
        """Mark thesis as resolved with outcome."""
        from datetime import datetime

        orm_thesis = await self.session.get(ThesisORM, thesis_id)
        if not orm_thesis:
            return None

        orm_thesis.resolved = True
        orm_thesis.resolution_date = datetime.utcnow()
        orm_thesis.resolution_outcome = outcome
        orm_thesis.brier_score = brier_score

        await self.session.flush()
        return self._from_orm(orm_thesis)

    async def get_calibration_stats(self) -> dict:
        """Get Brier score statistics for calibration."""
        from sqlalchemy import func

        result = await self.session.execute(
            select(
                func.count(ThesisORM.id).label("total"),
                func.avg(ThesisORM.brier_score).label("avg_brier"),
                func.count(ThesisORM.brier_score).label("scored"),
            ).where(ThesisORM.resolved == True)
        )
        row = result.one()

        return {
            "total_resolved": row.total or 0,
            "with_brier_scores": row.scored or 0,
            "average_brier_score": float(row.avg_brier) if row.avg_brier else None,
        }

    def _from_orm(self, orm: ThesisORM) -> Thesis:
        """Convert ORM to Pydantic model."""
        return Thesis(
            id=orm.id,
            query=orm.query,
            query_date=orm.query_date,
            analog_episode_ids=orm.analog_episode_ids,
            analog_similarity_scores=orm.analog_similarity_scores,
            dominant_continuation=orm.dominant_continuation,
            alternative_continuations=orm.alternative_continuations,
            watch_for_indicators=orm.watch_for_indicators,
            confidence_interval=orm.confidence_interval,
            estimated_duration=orm.estimated_duration,
            resolution_criteria=orm.resolution_criteria,
            cited_episodes=orm.cited_episodes,
            resolved=orm.resolved,
            resolution_date=orm.resolution_date,
            resolution_outcome=orm.resolution_outcome,
            brier_score=orm.brier_score,
            created_at=orm.created_at,
            model_version=orm.model_version,
            taxonomy_version=orm.taxonomy_version,
        )


class RepositoryFactory:
    """Factory for creating repositories with shared session."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._episodes: Optional[EpisodeRepository] = None
        self._cycles: Optional[CycleRepository] = None
        self._theses: Optional[ThesisRepository] = None

    @property
    def episodes(self) -> EpisodeRepository:
        if self._episodes is None:
            self._episodes = EpisodeRepository(self.session)
        return self._episodes

    @property
    def cycles(self) -> CycleRepository:
        if self._cycles is None:
            self._cycles = CycleRepository(self.session)
        return self._cycles

    @property
    def theses(self) -> ThesisRepository:
        if self._theses is None:
            self._theses = ThesisRepository(self.session)
        return self._theses
