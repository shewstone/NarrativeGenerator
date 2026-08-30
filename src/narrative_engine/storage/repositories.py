"""Repository pattern for database operations."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from narrative_engine.logging_config import LogTimer, get_logger
from narrative_engine.models import (
    Actor,
    ArcAssignment,
    Continuation,
    Cycle,
    CycleMembership,
    Episode,
    EpisodeLink,
    Scope,
    SourceDocument,
    SourceDocumentStatus,
    Thesis,
)
from narrative_engine.storage.orm_models import (
    CycleMembershipORM,
    CycleORM,
    EpisodeLinkORM,
    EpisodeORM,
    ScopeORM,
    SourceDocumentORM,
    ThesisORM,
)

logger = get_logger(__name__)


class EpisodeRepository:
    """Repository for Episode CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, episode: Episode) -> Episode:
        """Create a new episode."""
        with LogTimer(logger, "episode_create", episode_id=str(episode.id)):
            try:
                orm_episode = self._to_orm(episode)
                self.session.add(orm_episode)
                await self.session.flush()
                logger.info(
                    "episode_created",
                    episode_id=str(episode.id),
                    title=episode.title,
                    arc_type=episode.arc_type.value if episode.arc_type else None,
                )
                # _to_orm persists every non-derived field (enforced by
                # tests/unit/test_roundtrip.py), so the input model IS the
                # stored state — no lossy read-back needed.
                return episode
            except Exception as e:
                logger.error(
                    "episode_create_failed",
                    episode_id=str(episode.id),
                    error=str(e),
                )
                raise

    async def get_by_id(self, episode_id: UUID) -> Optional[Episode]:
        """Get episode by ID with all relationships."""
        with LogTimer(logger, "episode_get_by_id", episode_id=str(episode_id)):
            try:
                result = await self.session.execute(
                    select(EpisodeORM)
                    .where(EpisodeORM.id == episode_id)
                    .options(
                        selectinload(EpisodeORM.actors),
                        selectinload(EpisodeORM.source_passages),
                        selectinload(EpisodeORM.cycles),
                    )
                )
                orm_episode = result.scalar_one_or_none()
                return self._from_orm(orm_episode) if orm_episode else None
            except Exception as e:
                logger.error(
                    "episode_get_by_id_failed",
                    episode_id=str(episode_id),
                    error=str(e),
                )
                raise

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
            .options(
                selectinload(EpisodeORM.actors),
                selectinload(EpisodeORM.source_passages),
                selectinload(EpisodeORM.cycles),
            )
            .limit(limit)
            .offset(offset)
            .order_by(EpisodeORM.start_date)
        )
        return [self._from_orm(e) for e in result.scalars().unique().all()]

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
            .options(
                selectinload(EpisodeORM.actors),
                selectinload(EpisodeORM.source_passages),
                selectinload(EpisodeORM.cycles),
            )
            .limit(limit)
            .offset(offset)
            .order_by(EpisodeORM.phase_confidence.desc())
        )
        return [self._from_orm(e) for e in result.scalars().unique().all()]

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
            # Sessions run with autoflush=False in tests; without an explicit
            # flush the delete stays pending and later reads still see the row.
            await self.session.flush()
            return True
        return False

    async def update_embedding(
        self,
        episode_id: UUID,
        embedding: List[float],
        kind: str = "structural",
    ) -> None:
        """Set one of an episode's embeddings, stamped with its epoch.

        kind is explicit ("structural" | "surface") because the two
        collections answer different questions and must never be swapped
        (design doc Sec 3.3a). This is the single write choke point for
        vectors: it must be impossible to store a vector without recording
        which (render, model) epoch produced it (T4).
        """
        from narrative_engine.retrieval.epochs import current_epoch

        orm_episode = await self.session.get(EpisodeORM, episode_id)
        if not orm_episode:
            raise ValueError(f"Episode {episode_id} not found")
        if kind == "structural":
            orm_episode.structural_embedding = embedding
            orm_episode.structural_embedding_epoch = current_epoch("structural")
        elif kind == "surface":
            orm_episode.surface_embedding = embedding
            orm_episode.surface_embedding_epoch = current_epoch("surface")
        else:
            raise ValueError(f"Unknown embedding kind: {kind}")
        await self.session.flush()

    async def search_by_embedding(
        self,
        embedding: List[float],
        limit: int = 10,
        include_unclassified: bool = False,
    ) -> Sequence[tuple[Episode, float]]:
        """Semantic search using vector similarity.

        Analogy retrieval use case only: queries structural_embedding, never
        surface_embedding (design doc Sec 3.3a). Identity resolution
        (composition, SAME_EVENT_AS) does its own surface-embedding
        comparison in-process rather than going through this method.

        Unclassified episodes (failed tau_class, Sec 6.2 stage 4) are
        excluded by default: they must stay out of the arc-conditioned
        analog base. Arc-less retrieval (Sec 6.5.8) passes
        include_unclassified=True, since bare structural nearest-neighbor
        matching doesn't condition on arc labels at all.

        Only vectors from the CURRENT structural epoch participate (T4):
        vectors produced by an older render template or embedding model live
        in a different similarity space, and comparing across spaces yields
        silently meaningless scores. A smaller honest analog base beats a
        larger corrupt one.
        """
        from narrative_engine.retrieval.epochs import current_epoch

        query = select(
            EpisodeORM,
            EpisodeORM.structural_embedding.cosine_distance(embedding).label("distance"),
        ).where(
            EpisodeORM.structural_embedding.is_not(None),
            EpisodeORM.structural_embedding_epoch == current_epoch("structural"),
        )

        if not include_unclassified:
            query = query.where(EpisodeORM.classification_state != "unclassified")

        result = await self.session.execute(query.order_by("distance").limit(limit))

        return [(self._from_orm(row.EpisodeORM), float(row.distance)) for row in result.all()]

    async def count(self) -> int:
        """Get total episode count."""
        result = await self.session.execute(select(func.count(EpisodeORM.id)))
        return result.scalar() or 0

    async def update_source_temporal_bounds(self, episode: Episode) -> None:
        """Persist source availability and its conservative event-date cap."""
        await self.session.execute(
            update(EpisodeORM)
            .where(EpisodeORM.id == episode.id)
            .values(
                start_date=episode.start_date,
                end_date=episode.end_date,
                start_year=episode.start_year,
                end_year=episode.end_year,
                source_published_at=episode.source_published_at,
            )
        )

    async def get_arc_types_for_chunks(self, chunk_ids: Sequence[str]) -> set:
        """Return classified arc types sourced from any of the given chunks."""
        wanted = set(chunk_ids)
        if not wanted:
            return set()
        result = await self.session.execute(
            select(EpisodeORM.arc_type, EpisodeORM.extracted_from).where(EpisodeORM.arc_type.is_not(None))
        )
        return {arc_type for arc_type, extracted_from in result.all() if wanted.intersection(extracted_from or [])}

    async def get_for_chunks(self, chunk_ids: Sequence[str]) -> List[Episode]:
        """Load episodes whose primary provenance chunk is in ``chunk_ids``."""
        wanted = list(dict.fromkeys(chunk_ids))
        if not wanted:
            return []
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.extracted_from[0].as_string().in_(wanted))
            .options(
                selectinload(EpisodeORM.actors),
                selectinload(EpisodeORM.source_passages),
            )
        )
        return [self._from_orm(orm) for orm in result.scalars().unique().all()]

    def _to_orm(self, episode: Episode) -> EpisodeORM:
        """Convert Pydantic model to ORM.

        HANDLED_FIELDS below is this converter's checklist: every
        Episode field is either mapped here or listed as derived/unpersisted
        with a justification. tests/unit/test_roundtrip.py enforces the
        checklist is complete.
        """
        from narrative_engine.storage.orm_models import ActorORM, SourcePassageORM

        return EpisodeORM(
            id=episode.id,
            title=episode.title,
            summary=episode.summary,
            start_date=episode.start_date,
            end_date=episode.end_date,
            start_year=episode.start_year,
            end_year=episode.end_year,
            date_precision=episode.date_precision,
            location=episode.location,
            setting_description=episode.setting_description,
            scope_id=episode.scope_id,
            scope_name=episode.scope_name,
            scope_kind=episode.scope_kind.value if episode.scope_kind else None,
            parent_scope_name=episode.parent_scope_name,
            scope_confidence=episode.scope_confidence,
            scope_evidence=episode.scope_evidence,
            scope_notes=episode.scope_notes,
            initiating_conditions=episode.initiating_conditions,
            escalation_mechanics=episode.escalation_mechanics,
            tension=episode.tension,
            resolution=episode.resolution,
            consequences=episode.consequences,
            mechanism_tags=episode.mechanism_tags,
            change_pattern=episode.change_pattern.value if episode.change_pattern else None,
            pattern_confidence=episode.pattern_confidence,
            pattern_rationale=episode.pattern_rationale,
            situation_scale=episode.situation_scale.value if episode.situation_scale else None,
            domains=[domain.value for domain in episode.domains],
            configuration=episode.configuration.model_dump(mode="json", exclude_none=True),
            mechanism_families=[family.value for family in episode.mechanism_families],
            arc_type=episode.arc_type,
            arc_phase=episode.arc_phase,
            phase_confidence=episode.phase_confidence,
            arc_rationale=episode.arc_rationale,
            secondary_arcs=episode.secondary_arcs,
            classification_state=episode.classification_state.value,
            extracted_from=episode.extracted_from,
            source_published_at=episode.source_published_at,
            version=episode.version,
            surface_embedding=episode.surface_embedding,
            structural_embedding=episode.structural_embedding,
            surface_embedding_epoch=episode.surface_embedding_epoch,
            structural_embedding_epoch=episode.structural_embedding_epoch,
            created_at=episode.created_at,
            updated_at=episode.updated_at,
            actors=[
                ActorORM(
                    id=a.id,
                    name=a.name,
                    role=a.role,
                    canonical_role=a.canonical_role,
                    role_fit_confidence=a.role_fit_confidence,
                    attributes=dict(a.attributes),
                )
                for a in episode.actors
            ],
            source_passages=[
                SourcePassageORM(
                    work_id=sp.work_id,
                    passage_id=sp.passage_id,
                    text=sp.text,
                    chapter=sp.chapter,
                    section=sp.section,
                    page=sp.page,
                    historiographic_school=sp.historiographic_school,
                )
                for sp in episode.source_passages
            ],
        )

    def _from_orm(self, orm: EpisodeORM) -> Episode:
        """Convert ORM to Pydantic model."""
        from narrative_engine.models import SourcePassage

        # Relationships are only read when eagerly loaded (selectinload);
        # touching an unloaded relationship on an async session raises
        # MissingGreenlet, so unloaded collections come back empty.
        unloaded = sa_inspect(orm).unloaded
        actors = orm.actors if "actors" not in unloaded else []
        source_passages = orm.source_passages if "source_passages" not in unloaded else []
        parent_cycle_ids = {c.id for c in orm.cycles} if "cycles" not in unloaded else set()

        return Episode(
            id=orm.id,
            title=orm.title,
            summary=orm.summary,
            start_date=orm.start_date,
            end_date=orm.end_date,
            start_year=orm.start_year,
            end_year=orm.end_year,
            date_precision=orm.date_precision,
            location=orm.location,
            setting_description=orm.setting_description,
            scope_id=orm.scope_id,
            scope_name=orm.scope_name,
            scope_kind=orm.scope_kind,
            parent_scope_name=orm.parent_scope_name,
            scope_confidence=orm.scope_confidence,
            scope_evidence=orm.scope_evidence,
            scope_notes=orm.scope_notes,
            actors=[
                Actor(
                    id=a.id,
                    name=a.name,
                    role=a.role,
                    canonical_role=a.canonical_role,
                    role_fit_confidence=a.role_fit_confidence,
                    attributes=a.attributes,
                )
                for a in actors
            ]
            if actors
            else [],
            initiating_conditions=orm.initiating_conditions,
            escalation_mechanics=orm.escalation_mechanics,
            tension=orm.tension,
            resolution=orm.resolution,
            consequences=orm.consequences,
            mechanism_tags=orm.mechanism_tags,
            change_pattern=orm.change_pattern,
            pattern_confidence=orm.pattern_confidence,
            pattern_rationale=orm.pattern_rationale,
            situation_scale=orm.situation_scale,
            domains=orm.domains,
            configuration=orm.configuration,
            mechanism_families=orm.mechanism_families,
            arc_type=orm.arc_type,
            arc_phase=orm.arc_phase,
            phase_confidence=orm.phase_confidence,
            arc_rationale=orm.arc_rationale,
            secondary_arcs=orm.secondary_arcs,
            classification_state=orm.classification_state,
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
                for sp in source_passages
            ]
            if source_passages
            else [],
            extracted_from=orm.extracted_from,
            source_published_at=orm.source_published_at,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            version=orm.version,
            parent_cycle_ids=parent_cycle_ids,
            surface_embedding=orm.surface_embedding,
            structural_embedding=orm.structural_embedding,
            surface_embedding_epoch=orm.surface_embedding_epoch,
            structural_embedding_epoch=orm.structural_embedding_epoch,
        )

    def _update_orm(self, orm: EpisodeORM, episode: Episode) -> None:
        """Update ORM from Pydantic model."""
        orm.title = episode.title
        orm.summary = episode.summary
        orm.start_date = episode.start_date
        orm.end_date = episode.end_date
        orm.start_year = episode.start_year
        orm.end_year = episode.end_year
        orm.location = episode.location
        orm.setting_description = episode.setting_description
        orm.scope_id = episode.scope_id
        orm.scope_name = episode.scope_name
        orm.scope_kind = episode.scope_kind.value if episode.scope_kind else None
        orm.parent_scope_name = episode.parent_scope_name
        orm.scope_confidence = episode.scope_confidence
        orm.scope_evidence = episode.scope_evidence
        orm.scope_notes = episode.scope_notes
        orm.initiating_conditions = episode.initiating_conditions
        orm.escalation_mechanics = episode.escalation_mechanics
        orm.tension = episode.tension
        orm.resolution = episode.resolution
        orm.consequences = episode.consequences
        orm.mechanism_tags = episode.mechanism_tags
        orm.change_pattern = episode.change_pattern.value if episode.change_pattern else None
        orm.pattern_confidence = episode.pattern_confidence
        orm.pattern_rationale = episode.pattern_rationale
        orm.situation_scale = episode.situation_scale.value if episode.situation_scale else None
        orm.domains = [domain.value for domain in episode.domains]
        orm.configuration = episode.configuration.model_dump(mode="json", exclude_none=True)
        orm.mechanism_families = [family.value for family in episode.mechanism_families]
        orm.arc_type = episode.arc_type
        orm.arc_phase = episode.arc_phase
        orm.phase_confidence = episode.phase_confidence
        orm.arc_rationale = episode.arc_rationale
        orm.secondary_arcs = episode.secondary_arcs
        orm.classification_state = episode.classification_state.value
        orm.source_published_at = episode.source_published_at
        orm.surface_embedding = episode.surface_embedding
        orm.structural_embedding = episode.structural_embedding
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
            scope_id=cycle.scope_id,
            start_date=cycle.start_date,
            end_date=cycle.end_date,
            parent_cycle_id=cycle.parent_cycle_id,
            dominant_arc_types=cycle.dominant_arc_types,
            phase_estimate=cycle.phase_estimate,
            framework_source=cycle.framework_source,
            is_arc_instance=cycle.is_arc_instance,
            created_at=cycle.created_at,
            updated_at=cycle.updated_at,
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
                selectinload(CycleORM.children),
                selectinload(CycleORM.episodes),
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
            select(CycleORM).where(CycleORM.scale == scale).options(selectinload(CycleORM.episodes)).limit(limit)
        )
        return [self._from_orm(c) for c in result.scalars().all()]

    async def get_children(self, cycle_id: UUID) -> Sequence[Cycle]:
        """Get child cycles."""
        result = await self.session.execute(
            select(CycleORM).where(CycleORM.parent_cycle_id == cycle_id).options(selectinload(CycleORM.episodes))
        )
        return [self._from_orm(c) for c in result.scalars().all()]

    async def add_episode(self, cycle_id: UUID, episode_id: UUID) -> None:
        """Add episode to cycle as a plain attested/auto membership.

        For memberships that need link_status/review_status/salience/
        phase_coverage set intentionally (e.g. composition's inferred
        COMPOSES links), use CycleMembershipRepository.create instead.
        """
        from narrative_engine.storage.orm_models import CycleMembershipORM

        self.session.add(CycleMembershipORM(episode_id=episode_id, cycle_id=cycle_id))
        await self.session.flush()

    def _from_orm(self, orm: CycleORM) -> Cycle:
        """Convert ORM to Pydantic model."""
        # Same unloaded-relationship discipline as EpisodeRepository._from_orm.
        unloaded = sa_inspect(orm).unloaded
        return Cycle(
            id=orm.id,
            name=orm.name,
            scale=orm.scale,
            description=orm.description,
            scope_id=orm.scope_id,
            start_date=orm.start_date,
            end_date=orm.end_date,
            parent_cycle_id=orm.parent_cycle_id,
            child_cycle_ids=({c.id for c in orm.children} if "children" not in unloaded else set()),
            episode_ids=({e.id for e in orm.episodes} if "episodes" not in unloaded else set()),
            dominant_arc_types=orm.dominant_arc_types,
            phase_estimate=orm.phase_estimate,
            framework_source=orm.framework_source,
            is_arc_instance=orm.is_arc_instance,
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
            # JSON columns can't take raw UUIDs/models; Pydantic coerces the
            # strings back on read in _from_orm.
            analog_episode_ids=[str(eid) for eid in thesis.analog_episode_ids],
            analog_similarity_scores=thesis.analog_similarity_scores,
            dominant_continuation=(
                thesis.dominant_continuation.model_dump(mode="json") if thesis.dominant_continuation else None
            ),
            alternative_continuations=thesis.alternative_continuations,
            confidence=thesis.confidence.value,
            watch_for_indicators=thesis.watch_for_indicators,
            key_uncertainties=thesis.key_uncertainties,
            narrative_synthesis=thesis.narrative_synthesis,
            confidence_interval=thesis.confidence_interval,
            estimated_duration=thesis.estimated_duration,
            resolution_criteria=thesis.resolution_criteria,
            cited_episodes={
                str(eid): [passage.model_dump(mode="json") for passage in passages]
                for eid, passages in thesis.cited_episodes.items()
            },
            resolved=thesis.resolved,
            resolution_date=thesis.resolution_date,
            resolution_outcome=thesis.resolution_outcome,
            brier_score=thesis.brier_score,
            mode=thesis.mode.value,
            scope_registry_version=thesis.scope_registry_version,
            created_at=thesis.created_at,
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
        from sqlalchemy import not_

        result = await self.session.execute(
            select(ThesisORM).where(not_(ThesisORM.resolved)).order_by(ThesisORM.created_at.desc()).limit(limit)
        )
        return [self._from_orm(t) for t in result.scalars().all()]

    async def resolve(
        self,
        thesis_id: UUID,
        outcome: str,
        brier_score: Optional[float] = None,
    ) -> Optional[Thesis]:
        """Mark thesis as resolved with outcome."""
        from narrative_engine.models import utcnow

        orm_thesis = await self.session.get(ThesisORM, thesis_id)
        if not orm_thesis:
            return None

        orm_thesis.resolved = True
        orm_thesis.resolution_date = utcnow()
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
            ).where(ThesisORM.resolved)
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
            dominant_continuation=(Continuation(**orm.dominant_continuation) if orm.dominant_continuation else None),
            alternative_continuations=orm.alternative_continuations,
            confidence=orm.confidence,
            watch_for_indicators=orm.watch_for_indicators,
            key_uncertainties=orm.key_uncertainties,
            narrative_synthesis=orm.narrative_synthesis,
            confidence_interval=orm.confidence_interval,
            estimated_duration=orm.estimated_duration,
            resolution_criteria=orm.resolution_criteria,
            cited_episodes=orm.cited_episodes,
            resolved=orm.resolved,
            resolution_date=orm.resolution_date,
            resolution_outcome=orm.resolution_outcome,
            brier_score=orm.brier_score,
            mode=orm.mode,
            scope_registry_version=orm.scope_registry_version,
            created_at=orm.created_at,
            model_version=orm.model_version,
            taxonomy_version=orm.taxonomy_version,
        )


class CycleMembershipRepository:
    """Repository for CycleMembership (episode <-> cycle) operations.

    Used for memberships that need link_status/review_status/salience/
    phase_coverage set intentionally -- in particular, the composition
    pass's inferred COMPOSES links (design doc Sec 6.2 stage 6).
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, membership: CycleMembership) -> CycleMembership:
        """Create a new cycle membership."""
        orm_membership = CycleMembershipORM(
            id=membership.id,
            episode_id=membership.episode_id,
            cycle_id=membership.cycle_id,
            reading=membership.reading.model_dump(mode="json") if membership.reading else None,
            salience=membership.salience,
            phase_coverage=membership.phase_coverage,
            link_status=membership.link_status.value,
            review_status=membership.review_status.value,
            created_at=membership.created_at,
        )
        self.session.add(orm_membership)
        await self.session.flush()
        return membership

    async def get_by_cycle(self, cycle_id: UUID) -> Sequence[CycleMembership]:
        """Get all memberships for a cycle."""
        result = await self.session.execute(select(CycleMembershipORM).where(CycleMembershipORM.cycle_id == cycle_id))
        return [self._from_orm(m) for m in result.scalars().all()]

    def _from_orm(self, orm: CycleMembershipORM) -> CycleMembership:
        return CycleMembership(
            id=orm.id,
            episode_id=orm.episode_id,
            cycle_id=orm.cycle_id,
            reading=ArcAssignment(**orm.reading) if orm.reading else None,
            salience=orm.salience,
            phase_coverage=orm.phase_coverage,
            link_status=orm.link_status,
            review_status=orm.review_status,
            created_at=orm.created_at,
        )


class ScopeRepository:
    """Repository for Scope rows (T5). The packaged registry
    (narrative_engine.scopes) is the resolution source of truth; this table
    exists so scope ids are queryable/joinable in SQL."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, scope: Scope) -> Scope:
        self.session.add(self._to_orm(scope))
        await self.session.flush()
        return scope

    async def get_by_id(self, scope_id: str) -> Optional[Scope]:
        orm_scope = await self.session.get(ScopeORM, scope_id)
        return self._from_orm(orm_scope) if orm_scope else None

    async def list_all(self) -> Sequence[Scope]:
        result = await self.session.execute(select(ScopeORM))
        return [self._from_orm(s) for s in result.scalars().all()]

    async def sync_from_registry(self, registry=None) -> int:
        """Upsert every packaged-registry scope into the table. Returns the
        number of scopes synced. Parents are inserted before children so the
        self-FK is satisfied."""
        if registry is None:
            from narrative_engine.scopes import get_registry

            registry = get_registry()

        # PostgreSQL is the production database.  A single ON CONFLICT
        # statement makes simultaneous API/worker startup safe: the previous
        # get-then-insert loop could race and fail one process with a unique
        # constraint violation while a new registry version was being synced.
        scopes = sorted(registry.all(), key=lambda scope: len(registry.lineage(scope.id)))
        if self.session.get_bind().dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            values = [
                {
                    "id": scope.id,
                    "kind": scope.kind,
                    "name": scope.name,
                    "parent_scope_id": scope.parent_scope_id,
                    "aliases": list(scope.aliases),
                    "notes": scope.notes,
                }
                for scope in scopes
            ]
            statement = pg_insert(ScopeORM).values(values)
            statement = statement.on_conflict_do_update(
                index_elements=[ScopeORM.id],
                set_={
                    "kind": statement.excluded.kind,
                    "name": statement.excluded.name,
                    "parent_scope_id": statement.excluded.parent_scope_id,
                    "aliases": statement.excluded.aliases,
                    "notes": statement.excluded.notes,
                },
            )
            await self.session.execute(statement)
            await self.session.flush()
            return len(scopes)

        # Keep the portable path for SQLite-backed unit tests. Parents must
        # exist before children; a boolean parent-first sort only handled one
        # level and breaks for faction -> party -> polity trees.
        for scope in scopes:
            existing = await self.session.get(ScopeORM, scope.id)
            if existing:
                existing.kind = scope.kind
                existing.name = scope.name
                existing.parent_scope_id = scope.parent_scope_id
                existing.aliases = list(scope.aliases)
                existing.notes = scope.notes
            else:
                self.session.add(self._to_orm(scope))
            await self.session.flush()
        return len(scopes)

    def _to_orm(self, scope: Scope) -> ScopeORM:
        return ScopeORM(
            id=scope.id,
            kind=scope.kind,
            name=scope.name,
            parent_scope_id=scope.parent_scope_id,
            aliases=list(scope.aliases),
            notes=scope.notes,
        )

    def _from_orm(self, orm: ScopeORM) -> Scope:
        return Scope(
            id=orm.id,
            kind=orm.kind,
            name=orm.name,
            parent_scope_id=orm.parent_scope_id,
            aliases=orm.aliases,
            notes=orm.notes,
        )


class SourceDocumentRepository:
    """Repository for SourceDocument rows (T7): the watcher's ledger."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, document: SourceDocument) -> SourceDocument:
        self.session.add(self._to_orm(document))
        await self.session.flush()
        return document

    async def get_by_id(self, document_id: UUID) -> Optional[SourceDocument]:
        orm_doc = await self.session.get(SourceDocumentORM, document_id)
        return self._from_orm(orm_doc) if orm_doc else None

    async def get_original_by_hash(self, content_hash: str) -> Optional[SourceDocument]:
        """The one non-duplicate row for this hash, if any."""
        result = await self.session.execute(
            select(SourceDocumentORM)
            .where(
                SourceDocumentORM.content_hash == content_hash,
                SourceDocumentORM.status != "duplicate",
            )
            .limit(1)
        )
        orm_doc = result.scalar_one_or_none()
        return self._from_orm(orm_doc) if orm_doc else None

    async def has_row_for(self, content_hash: str, filename: str) -> bool:
        """True if this exact (hash, filename) was already recorded --
        keeps the watcher from re-recording the same sitting file every
        scan, across restarts."""
        result = await self.session.execute(
            select(func.count(SourceDocumentORM.id)).where(
                SourceDocumentORM.content_hash == content_hash,
                SourceDocumentORM.filename == filename,
            )
        )
        return (result.scalar() or 0) > 0

    async def get_by_hash_and_filename(self, content_hash: str, filename: str) -> Optional[SourceDocument]:
        result = await self.session.execute(
            select(SourceDocumentORM)
            .where(
                SourceDocumentORM.content_hash == content_hash,
                SourceDocumentORM.filename == filename,
            )
            .order_by(SourceDocumentORM.created_at.desc())
            .limit(1)
        )
        orm_doc = result.scalar_one_or_none()
        return self._from_orm(orm_doc) if orm_doc else None

    async def claim_available(
        self,
        document_id: UUID,
        claim_token: UUID,
        lease_expires_at: datetime,
    ) -> bool:
        """Atomically claim queued work or replace an expired claim."""
        result = await self.session.execute(
            update(SourceDocumentORM)
            .where(
                SourceDocumentORM.id == document_id,
                or_(
                    SourceDocumentORM.status == "queued",
                    and_(
                        SourceDocumentORM.status == "processing",
                        or_(
                            SourceDocumentORM.lease_expires_at.is_(None),
                            SourceDocumentORM.lease_expires_at <= func.now(),
                        ),
                    ),
                ),
            )
            .values(
                status="processing",
                error=None,
                claim_token=claim_token,
                lease_expires_at=lease_expires_at,
                updated_at=func.now(),
            )
        )
        return getattr(result, "rowcount", 0) == 1

    async def renew_claim(
        self,
        document_id: UUID,
        claim_token: UUID,
        lease_expires_at: datetime,
    ) -> bool:
        result = await self.session.execute(
            update(SourceDocumentORM)
            .where(
                SourceDocumentORM.id == document_id,
                SourceDocumentORM.status == "processing",
                SourceDocumentORM.claim_token == claim_token,
            )
            .values(lease_expires_at=lease_expires_at, updated_at=func.now())
        )
        return getattr(result, "rowcount", 0) == 1

    async def update_claimed(self, document: SourceDocument, claim_token: UUID) -> bool:
        """Persist a checkpoint only while the caller still owns the claim."""
        values = {
            "status": document.status.value,
            "error": document.error,
            "chunks_created": document.chunks_created,
            "chunks_processed": document.chunks_processed,
            "episodes_created": document.episodes_created,
            "extraction_ran": document.extraction_ran,
            "duplicate_of": document.duplicate_of,
            "updated_at": func.now(),
        }
        if document.status != SourceDocumentStatus.PROCESSING:
            values.update(claim_token=None, lease_expires_at=None)
        result = await self.session.execute(
            update(SourceDocumentORM)
            .where(
                SourceDocumentORM.id == document.id,
                SourceDocumentORM.claim_token == claim_token,
            )
            .values(**values)
        )
        return getattr(result, "rowcount", 0) == 1

    async def list_all(self, limit: int = 200) -> Sequence[SourceDocument]:
        result = await self.session.execute(
            select(SourceDocumentORM).order_by(SourceDocumentORM.created_at.desc()).limit(limit)
        )
        return [self._from_orm(d) for d in result.scalars().all()]

    async def update(self, document: SourceDocument) -> SourceDocument:
        from narrative_engine.models import utcnow

        orm_doc = await self.session.get(SourceDocumentORM, document.id)
        if not orm_doc:
            raise ValueError(f"SourceDocument {document.id} not found")
        orm_doc.status = document.status.value
        orm_doc.error = document.error
        orm_doc.chunks_created = document.chunks_created
        orm_doc.chunks_processed = document.chunks_processed
        orm_doc.episodes_created = document.episodes_created
        orm_doc.extraction_ran = document.extraction_ran
        orm_doc.duplicate_of = document.duplicate_of
        orm_doc.updated_at = utcnow()
        await self.session.flush()
        document.updated_at = orm_doc.updated_at
        return document

    def _to_orm(self, document: SourceDocument) -> SourceDocumentORM:
        return SourceDocumentORM(
            id=document.id,
            filename=document.filename,
            content_hash=document.content_hash,
            size_bytes=document.size_bytes,
            status=document.status.value,
            error=document.error,
            chunks_created=document.chunks_created,
            chunks_processed=document.chunks_processed,
            episodes_created=document.episodes_created,
            extraction_ran=document.extraction_ran,
            duplicate_of=document.duplicate_of,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )

    def _from_orm(self, orm: SourceDocumentORM) -> SourceDocument:
        return SourceDocument(
            id=orm.id,
            filename=orm.filename,
            content_hash=orm.content_hash,
            size_bytes=orm.size_bytes,
            status=orm.status,
            error=orm.error,
            chunks_created=orm.chunks_created,
            chunks_processed=orm.chunks_processed,
            episodes_created=orm.episodes_created,
            extraction_ran=orm.extraction_ran,
            duplicate_of=orm.duplicate_of,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )


class EpisodeLinkRepository:
    """Repository for EpisodeLink (episode <-> episode) operations.

    Covers CAUSES, PRECEDES, SAME_EVENT_AS edges. Construction always goes
    through the EpisodeLink Pydantic model first, so the causal-attestation
    validator (Sec 4: CAUSES must be attested) runs before anything reaches
    the DB -- in addition to the DB's own chk_causal_must_be_attested CHECK.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, link: EpisodeLink) -> EpisodeLink:
        """Create a new episode link. Raises ValueError (via EpisodeLink's
        validator) if it's a CAUSES edge with link_status=inferred."""
        orm_link = EpisodeLinkORM(
            id=link.id,
            source_episode_id=link.source_episode_id,
            target_episode_id=link.target_episode_id,
            edge_kind=link.edge_kind.value,
            link_status=link.link_status.value,
            distance=link.distance,
            evidence=link.evidence,
            review_status=link.review_status.value,
            reviewed_by=link.reviewed_by,
            reviewed_at=link.reviewed_at,
            created_at=link.created_at,
        )
        self.session.add(orm_link)
        await self.session.flush()
        return link

    async def get_by_id(self, link_id: UUID) -> Optional[EpisodeLink]:
        """Get episode link by ID."""
        orm_link = await self.session.get(EpisodeLinkORM, link_id)
        return self._from_orm(orm_link) if orm_link else None

    def _from_orm(self, orm: EpisodeLinkORM) -> EpisodeLink:
        return EpisodeLink(
            id=orm.id,
            source_episode_id=orm.source_episode_id,
            target_episode_id=orm.target_episode_id,
            edge_kind=orm.edge_kind,
            link_status=orm.link_status,
            distance=orm.distance,
            evidence=orm.evidence,
            review_status=orm.review_status,
            reviewed_by=orm.reviewed_by,
            reviewed_at=orm.reviewed_at,
            created_at=orm.created_at,
        )


# ---------------------------------------------------------------------------
# Converter exclusion allowlists (T3, docs/tickets/T3-model-orm-roundtrip-tests.md).
#
# Fields listed here are deliberately NOT persisted by the converters above,
# each with a reason. tests/unit/test_roundtrip.py requires its maximal
# instances to explicitly set every field not listed here, and requires those
# fields to survive a create -> get round trip — so adding a model field
# without deciding its persistence story fails a unit test by name instead of
# silently writing nothing.
# ---------------------------------------------------------------------------

EPISODE_FIELDS_EXCLUDED = {
    # Derived from CycleMembership rows (owned by CycleMembershipRepository /
    # CycleRepository.add_episode); populated on read when the relationship
    # is eagerly loaded, never written from the Episode aggregate.
    "parent_cycle_ids",
}

CYCLE_FIELDS_EXCLUDED = {
    # Derived: children own the parent_cycle_id FK; memberships own episode
    # links. Populated on read when eagerly loaded, never written from Cycle.
    "child_cycle_ids",
    "episode_ids",
}

THESIS_FIELDS_EXCLUDED: set = set()

CYCLE_MEMBERSHIP_FIELDS_EXCLUDED: set = set()

EPISODE_LINK_FIELDS_EXCLUDED: set = set()

SCOPE_FIELDS_EXCLUDED: set = set()

SOURCE_DOCUMENT_FIELDS_EXCLUDED: set = set()


class RepositoryFactory:
    """Factory for creating repositories with shared session."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._episodes: Optional[EpisodeRepository] = None
        self._cycles: Optional[CycleRepository] = None
        self._theses: Optional[ThesisRepository] = None
        self._cycle_memberships: Optional[CycleMembershipRepository] = None
        self._episode_links: Optional[EpisodeLinkRepository] = None

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

    @property
    def cycle_memberships(self) -> CycleMembershipRepository:
        if self._cycle_memberships is None:
            self._cycle_memberships = CycleMembershipRepository(self.session)
        return self._cycle_memberships

    @property
    def episode_links(self) -> EpisodeLinkRepository:
        if self._episode_links is None:
            self._episode_links = EpisodeLinkRepository(self.session)
        return self._episode_links
