"""FastAPI app (T8, docs/tickets/T8-dashboard-and-review-ui.md).

Serves the dashboard, the processing-queue/arc-instance/review JSON
endpoints, and runs the drop-directory watcher (T7) as a lifespan task —
one always-on container.

NO AUTH: bind assumption is localhost/dev. Adding auth is a hard
precondition for any non-local deployment.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager, suppress
from importlib import resources
from typing import Annotated, AsyncGenerator, Optional
from uuid import UUID

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.composition.formation import (
    FORMED_ARC_MIN_EPISODES,
    FORMED_ARC_MIN_PHASES,
    arc_formation_gaps,
    arc_formation_status,
)
from narrative_engine.logging_config import get_logger
from narrative_engine.scopes import (
    DEFAULT_SCOPE_CONFIDENCE_FLOOR,
    get_registry,
    scope_partition_key,
)
from narrative_engine.storage.orm_models import (
    CycleMembershipORM,
    CycleORM,
    EpisodeLinkORM,
    EpisodeORM,
    ExtractionRecordORM,
    SourceDocumentORM,
    SourcePassageORM,
)
from narrative_engine.storage.repositories import SourceDocumentRepository

logger = get_logger(__name__)


def _pca_3d(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Deterministic, dependency-light 3D projection for exploration only."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if len(vectors) == 1 or not np.any(centered):
        return np.zeros((len(vectors), 3), dtype=vectors.dtype), [0.0, 0.0, 0.0]
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(3, len(singular_values))
    projected = u[:, :dimensions] * singular_values[:dimensions]
    projected = np.pad(projected, ((0, 0), (0, 3 - dimensions)))
    variance = singular_values**2
    total = float(variance.sum())
    explained = (variance[:dimensions] / total).tolist() if total else []
    return projected, [float(value) for value in explained] + [0.0] * (3 - dimensions)


def _cosine_similarities(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return normalized @ normalized.T


def _episode_start_year(episode: EpisodeORM) -> Optional[int]:
    """Return the signed chronology key, including pre-migration CE rows."""
    if episode.start_year is not None:
        return episode.start_year
    return episode.start_date.year if episode.start_date else None


def _episode_end_year(episode: EpisodeORM) -> Optional[int]:
    if episode.end_year is not None:
        return episode.end_year
    return episode.end_date.year if episode.end_date else None


def _episode_temporal_key(episode: EpisodeORM) -> tuple:
    year = _episode_start_year(episode)
    return (
        year is None,
        year or 0,
        episode.start_date.isoformat() if episode.start_date else "",
        episode.created_at.isoformat(),
    )


def _episode_scope_path(episode: EpisodeORM) -> list[str]:
    """Resolve the most useful human-readable scope lineage for the UI."""
    scope_registry = get_registry()
    registered_lineage = scope_registry.lineage(episode.scope_id or episode.scope_name)
    if registered_lineage:
        return [scope.name for scope in reversed(registered_lineage)]

    parent_lineage = scope_registry.lineage(episode.parent_scope_name)
    scope_path = [scope.name for scope in reversed(parent_lineage)]
    if episode.scope_name and episode.scope_name not in scope_path:
        scope_path.append(episode.scope_name)
    if not scope_path:
        scope_path = [label for label in (episode.parent_scope_name, episode.scope_name) if label]
    return scope_path


def _source_work_id(chunk_id: str) -> str:
    """Collapse the ingestion chunk suffix to a stable source-work key."""
    return re.sub(r"_[0-9]+$", "", chunk_id)


def _arc_quality(
    membership_episodes: list[tuple[CycleMembershipORM, EpisodeORM]],
) -> dict:
    """Derive evidence maturity without discarding composition candidates."""
    active = [
        (membership, episode)
        for membership, episode in membership_episodes
        if membership.review_status != "rejected"
    ]
    episode_count = len({episode.id for _, episode in active})
    phases = {episode.arc_phase.value for _, episode in active if episode.arc_phase}
    source_works = {
        _source_work_id(chunk_id)
        for _, episode in active
        for chunk_id in (episode.extracted_from or [])
        if chunk_id
    }
    reviewed_count = sum(membership.review_status == "approved" for membership, _ in active)
    pending_count = sum(membership.review_status == "pending" for membership, _ in active)
    status = arc_formation_status(episode_count, len(phases))
    return {
        "formation_status": status,
        "formation_gaps": arc_formation_gaps(episode_count, len(phases)),
        "episode_count": episode_count,
        "phase_count": len(phases),
        "source_count": len(source_works),
        "cross_source": len(source_works) >= 2,
        "reviewed_count": reviewed_count,
        "pending_count": pending_count,
        "human_reviewed": bool(active) and reviewed_count == len(active),
    }


def _episode_payload(
    episode: EpisodeORM,
    episode_memberships: list[tuple[CycleMembershipORM, CycleORM]] | None = None,
    coordinates: tuple[float, float, float] | None = None,
    arc_quality_by_cycle: dict[UUID, dict] | None = None,
) -> dict:
    """Serialize one episode for the dashboard's coordinated views.

    Coordinates are optional so the chronological views can load every
    episode without paying the memory and CPU cost of a full PCA projection.
    """
    payload = {
        "id": str(episode.id),
        "title": episode.title,
        "summary": episode.summary,
        "arc_type": episode.arc_type.value if episode.arc_type else None,
        "phase": episode.arc_phase.value if episode.arc_phase else None,
        "confidence": float(episode.phase_confidence or 0.0),
        "classification_state": episode.classification_state,
        "start_date": episode.start_date.isoformat() if episode.start_date else None,
        "end_date": episode.end_date.isoformat() if episode.end_date else None,
        "start_year": _episode_start_year(episode),
        "end_year": _episode_end_year(episode),
        "scope_id": episode.scope_id,
        "scope_name": episode.scope_name,
        "scope_kind": episode.scope_kind,
        "parent_scope_name": episode.parent_scope_name,
        "scope_confidence": episode.scope_confidence,
        "scope_evidence": episode.scope_evidence,
        "scope_notes": episode.scope_notes,
        "scope_path": _episode_scope_path(episode),
        "location": episode.location,
        "change_pattern": episode.change_pattern,
        "pattern_confidence": float(episode.pattern_confidence or 0.0),
        "pattern_rationale": episode.pattern_rationale,
        "situation_scale": episode.situation_scale,
        "domains": list(episode.domains or []),
        "configuration": dict(episode.configuration or {}),
        "mechanism_families": list(episode.mechanism_families or []),
        "mechanisms": list(episode.mechanism_tags or []),
        "source_chunks": list(episode.extracted_from or []),
        "source_published_at": (
            episode.source_published_at.isoformat() if episode.source_published_at else None
        ),
        "arc_instances": [
            {
                "id": str(cycle.id),
                "name": cycle.name,
                "link_status": membership.link_status,
                "review_status": membership.review_status,
                **((arc_quality_by_cycle or {}).get(cycle.id) or {}),
            }
            for membership, cycle in (episode_memberships or [])
        ],
    }
    if coordinates is not None:
        payload.update(zip(("x", "y", "z"), coordinates, strict=True))
    return payload


async def _arc_quality_for_cycles(
    session: AsyncSession,
    cycle_ids: set[UUID] | list[UUID],
) -> dict[UUID, dict]:
    """Load complete evidence statistics for the requested arc candidates."""
    if not cycle_ids:
        return {}
    rows = (
        await session.execute(
            select(CycleMembershipORM, EpisodeORM)
            .join(EpisodeORM, EpisodeORM.id == CycleMembershipORM.episode_id)
            .where(
                CycleMembershipORM.cycle_id.in_(cycle_ids),
                CycleMembershipORM.review_status != "rejected",
            )
        )
    ).all()
    by_cycle: dict[UUID, list[tuple[CycleMembershipORM, EpisodeORM]]] = {}
    for membership, episode in rows:
        by_cycle.setdefault(membership.cycle_id, []).append((membership, episode))
    return {
        cycle_id: _arc_quality(by_cycle.get(cycle_id, []))
        for cycle_id in cycle_ids
    }


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Session dependency; tests override this with their fixture session."""
    from narrative_engine.storage.database import db_manager

    async with db_manager.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class ReviewDecision(BaseModel):
    decision: str  # "approved" | "rejected"


def create_app(start_watcher: Optional[bool] = None) -> FastAPI:
    if start_watcher is None:
        start_watcher = os.getenv("NE_WATCH_ENABLED", "true").lower() == "true"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher_task = None
        from narrative_engine.storage.database import db_manager
        from narrative_engine.storage.repositories import ScopeRepository

        # Keep the queryable SQL mirror aligned with the packaged, versioned
        # hierarchy before extraction/composition starts.
        async with db_manager.session() as session:
            await ScopeRepository(session).sync_from_registry()
            await session.commit()
        if start_watcher:
            from narrative_engine.watcher import watch_loop

            watcher_task = asyncio.create_task(watch_loop())
        try:
            yield
        finally:
            if watcher_task:
                watcher_task.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher_task
            await db_manager.close()

    app = FastAPI(title="Narrative Engine", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return resources.files("narrative_engine.api").joinpath("static/dashboard.html").read_text()

    @app.get("/api/health")
    async def health(session: SessionDep) -> dict:
        async def count(stmt):
            return (await session.execute(stmt)).scalar() or 0

        active_membership = CycleMembershipORM.review_status != "rejected"
        arc_evidence = (
            select(
                CycleORM.id.label("cycle_id"),
                func.count(
                    func.distinct(
                        case((active_membership, CycleMembershipORM.episode_id), else_=None)
                    )
                ).label("episode_count"),
                func.count(
                    func.distinct(
                        case(
                            (
                                and_(active_membership, EpisodeORM.arc_phase.is_not(None)),
                                EpisodeORM.arc_phase,
                            ),
                            else_=None,
                        )
                    )
                ).label("phase_count"),
            )
            .outerjoin(CycleMembershipORM, CycleMembershipORM.cycle_id == CycleORM.id)
            .outerjoin(EpisodeORM, EpisodeORM.id == CycleMembershipORM.episode_id)
            .where(CycleORM.is_arc_instance)
            .group_by(CycleORM.id)
            .subquery()
        )
        formed_condition = and_(
            arc_evidence.c.episode_count >= FORMED_ARC_MIN_EPISODES,
            arc_evidence.c.phase_count >= FORMED_ARC_MIN_PHASES,
        )
        arc_counts = (
            await session.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((formed_condition, 1), else_=0)).label("formed"),
                    func.sum(case((formed_condition, 0), else_=1)).label("candidates"),
                ).select_from(arc_evidence)
            )
        ).one()
        episode_counts = (
            await session.execute(
                select(
                    func.count(EpisodeORM.id).label("total"),
                    func.sum(
                        case(
                            (
                                (EpisodeORM.start_year.is_not(None))
                                | (EpisodeORM.start_date.is_not(None)),
                                1,
                            ),
                            else_=0,
                        )
                    ).label("dated"),
                    func.sum(
                        case((EpisodeORM.source_published_at.is_not(None), 1), else_=0)
                    ).label("source_dated"),
                    func.sum(case((EpisodeORM.scope_id.is_not(None), 1), else_=0)).label(
                        "scope_resolved"
                    ),
                    func.sum(case((EpisodeORM.arc_phase.is_not(None), 1), else_=0)).label(
                        "phased"
                    ),
                    func.sum(
                        case((EpisodeORM.start_year >= 1990, 1), else_=0)
                    ).label("recent"),
                )
            )
        ).one()
        episode_total = int(episode_counts.total or 0)
        return {
            "status": "ok",
            "documents": await count(select(func.count(SourceDocumentORM.id))),
            "episodes": episode_total,
            "arc_instances": int(arc_counts.total or 0),
            "formed_arcs": int(arc_counts.formed or 0),
            "arc_candidates": int(arc_counts.candidates or 0),
            "pending_reviews": (
                await count(
                    select(func.count(CycleMembershipORM.id)).where(CycleMembershipORM.review_status == "pending")
                )
            )
            + (await count(select(func.count(EpisodeLinkORM.id)).where(EpisodeLinkORM.review_status == "pending"))),
            "quality": {
                "dated_episodes": int(episode_counts.dated or 0),
                "undated_episodes": episode_total - int(episode_counts.dated or 0),
                "source_dated_episodes": int(episode_counts.source_dated or 0),
                "source_date_missing": episode_total - int(episode_counts.source_dated or 0),
                "episodes_since_1990": int(episode_counts.recent or 0),
                "scope_resolved_episodes": int(episode_counts.scope_resolved or 0),
                "scope_unresolved_episodes": episode_total
                - int(episode_counts.scope_resolved or 0),
                "phased_episodes": int(episode_counts.phased or 0),
                "unphased_episodes": episode_total - int(episode_counts.phased or 0),
                "source_backed_episodes": await count(
                    select(func.count(func.distinct(SourcePassageORM.episode_id)))
                ),
                "extraction_audit_records": await count(
                    select(func.count(ExtractionRecordORM.id))
                ),
                "episode_links": await count(select(func.count(EpisodeLinkORM.id))),
                "formation_threshold": {
                    "episodes": FORMED_ARC_MIN_EPISODES,
                    "phases": FORMED_ARC_MIN_PHASES,
                },
            },
        }

    @app.get("/api/documents")
    async def documents(session: SessionDep) -> list:
        repo = SourceDocumentRepository(session)
        return [
            {
                "id": str(d.id),
                "filename": d.filename,
                "status": d.status.value,
                "size_bytes": d.size_bytes,
                "chunks_created": d.chunks_created,
                "chunks_processed": d.chunks_processed,
                "episodes_created": d.episodes_created,
                "extraction_ran": d.extraction_ran,
                "duplicate_of": str(d.duplicate_of) if d.duplicate_of else None,
                "error": d.error,
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
            }
            for d in await repo.list_all()
        ]

    @app.get("/api/arc-instances")
    async def arc_instances(
        session: SessionDep,
        limit: int = Query(100, ge=1, le=2500),
    ) -> list:
        from narrative_engine.composition.pipeline import _infer_expected_phases
        from narrative_engine.models import ArcType

        cycles = (
            (
                await session.execute(
                    select(CycleORM)
                    .where(CycleORM.is_arc_instance)
                    .order_by(CycleORM.created_at.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not cycles:
            return []

        cycle_ids = [c.id for c in cycles]
        memberships = (
            (
                await session.execute(
                    select(CycleMembershipORM).where(
                        CycleMembershipORM.cycle_id.in_(cycle_ids),
                        CycleMembershipORM.review_status != "rejected",
                    )
                )
            )
            .scalars()
            .all()
        )
        episode_ids = {m.episode_id for m in memberships}
        episodes = {}
        if episode_ids:
            rows = (await session.execute(select(EpisodeORM).where(EpisodeORM.id.in_(episode_ids)))).scalars().all()
            episodes = {e.id: e for e in rows}

        by_cycle: dict = {}
        for m in memberships:
            by_cycle.setdefault(m.cycle_id, []).append(m)

        payload = []
        for cycle in cycles:
            arc_value = None
            if cycle.dominant_arc_types:
                arc_value = cycle.dominant_arc_types[0]
            elif cycle.name and "," in cycle.name:
                arc_value = cycle.name.split(",")[0].strip()
            expected_phases = []
            with suppress(ValueError, TypeError):
                expected_phases = [p.value for p in _infer_expected_phases(ArcType(arc_value))]

            members = []
            for m in sorted(
                by_cycle.get(cycle.id, []),
                key=lambda m: _episode_temporal_key(episodes[m.episode_id])
                if m.episode_id in episodes
                else (True, 0, "", ""),
            ):
                episode = episodes.get(m.episode_id)
                if episode is None:
                    continue
                members.append(
                    {
                        "id": str(episode.id),
                        "title": episode.title,
                        "phase": episode.arc_phase.value if episode.arc_phase else None,
                        "start_date": episode.start_date.isoformat() if episode.start_date else None,
                        "end_date": episode.end_date.isoformat() if episode.end_date else None,
                        "start_year": _episode_start_year(episode),
                        "end_year": _episode_end_year(episode),
                        "link_status": m.link_status,
                        "review_status": m.review_status,
                        "membership_id": str(m.id),
                    }
                )

            covered = {m["phase"] for m in members if m["phase"]}
            quality = _arc_quality(
                [
                    (membership, episodes[membership.episode_id])
                    for membership in by_cycle.get(cycle.id, [])
                    if membership.episode_id in episodes
                ]
            )
            payload.append(
                {
                    "id": str(cycle.id),
                    "name": cycle.name,
                    "arc_type": arc_value,
                    "scope_id": cycle.scope_id,
                    "start_date": cycle.start_date.isoformat() if cycle.start_date else None,
                    "end_date": cycle.end_date.isoformat() if cycle.end_date else None,
                    "expected_phases": expected_phases,
                    "covered_phases": sorted(covered),
                    "episodes": members,
                    **quality,
                }
            )
        return payload

    @app.get("/api/episodes")
    async def episodes(
        session: SessionDep,
        limit: int = Query(2500, ge=1, le=5000),
    ) -> dict:
        """Return the lightweight chronological corpus used by the main UI.

        Unlike ``/api/arc-space``, this endpoint deliberately excludes
        embeddings, PCA coordinates, and nearest-neighbour calculations.  It
        therefore supports complete timelines without making first paint pay
        the quadratic similarity-matrix cost.
        """
        total = (await session.execute(select(func.count(EpisodeORM.id)))).scalar() or 0
        rows = (
            (
                await session.execute(
                    select(EpisodeORM)
                    .order_by(EpisodeORM.start_year.asc().nullslast(), EpisodeORM.created_at)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return {"nodes": [], "total": total, "truncated": False}

        selected_ids = {episode.id for episode in rows}
        memberships = (
            await session.execute(
                select(CycleMembershipORM, CycleORM)
                .join(CycleORM, CycleMembershipORM.cycle_id == CycleORM.id)
                .where(
                    CycleORM.is_arc_instance,
                    CycleMembershipORM.episode_id.in_(selected_ids),
                    CycleMembershipORM.review_status != "rejected",
                )
            )
        ).all()
        memberships_by_episode: dict = {}
        for membership, cycle in memberships:
            memberships_by_episode.setdefault(membership.episode_id, []).append((membership, cycle))
        arc_quality_by_cycle = await _arc_quality_for_cycles(
            session,
            {cycle.id for _, cycle in memberships},
        )

        return {
            "nodes": [
                _episode_payload(
                    episode,
                    memberships_by_episode.get(episode.id, []),
                    arc_quality_by_cycle=arc_quality_by_cycle,
                )
                for episode in rows
            ],
            "total": total,
            "truncated": total > len(rows),
        }

    @app.get("/api/arc-space")
    async def arc_space(
        session: SessionDep,
        k: int = Query(3, ge=1, le=10),
        limit: int = Query(500, ge=1, le=1000),
    ) -> dict:
        """Project structural vectors and explain graph relationships.

        PCA coordinates are an exploratory view only. Similarity values and
        neighbor selection always come from the original embedding space.
        """
        from narrative_engine.retrieval.epochs import current_epoch

        episodes = (
            (
                await session.execute(
                    select(EpisodeORM)
                    .where(
                        EpisodeORM.structural_embedding.is_not(None),
                        EpisodeORM.structural_embedding_epoch == current_epoch("structural"),
                    )
                    .order_by(EpisodeORM.start_year.asc().nullslast(), EpisodeORM.created_at)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        if not episodes:
            return {
                "projection": {
                    "algorithm": "pca",
                    "source_dimensions": 384,
                    "display_dimensions": 3,
                    "explained_variance": [0.0, 0.0, 0.0],
                    "embedding_epoch": current_epoch("structural"),
                    "warning": "Projection distance is approximate; similarity uses original vectors.",
                },
                "nodes": [],
                "edges": [],
            }

        vectors = np.asarray([np.asarray(episode.structural_embedding, dtype=np.float32) for episode in episodes])
        coordinates, explained = _pca_3d(vectors)
        similarities = _cosine_similarities(vectors)
        episode_ids = [episode.id for episode in episodes]
        episode_index = {episode_id: index for index, episode_id in enumerate(episode_ids)}
        selected_ids = set(episode_ids)
        by_id = {episode.id: episode for episode in episodes}

        memberships = (
            await session.execute(
                select(CycleMembershipORM, CycleORM)
                .join(CycleORM, CycleMembershipORM.cycle_id == CycleORM.id)
                .where(
                    CycleORM.is_arc_instance,
                    CycleMembershipORM.episode_id.in_(selected_ids),
                    CycleMembershipORM.review_status != "rejected",
                )
            )
        ).all()
        memberships_by_episode: dict = {}
        memberships_by_cycle: dict = {}
        for membership, cycle in memberships:
            memberships_by_episode.setdefault(membership.episode_id, []).append((membership, cycle))
            memberships_by_cycle.setdefault(cycle.id, []).append((membership, cycle))
        arc_quality_by_cycle = await _arc_quality_for_cycles(session, set(memberships_by_cycle))

        nodes = [
            _episode_payload(
                episode,
                memberships_by_episode.get(episode.id, []),
                (
                    float(coordinates[index, 0]),
                    float(coordinates[index, 1]),
                    float(coordinates[index, 2]),
                ),
                arc_quality_by_cycle,
            )
            for index, episode in enumerate(episodes)
        ]
        scope_registry = get_registry()

        edges = []
        seen_neighbors = set()
        neighbor_count = min(k, max(0, len(episodes) - 1))
        for source_index, source in enumerate(episodes):
            candidates = np.argsort(-similarities[source_index])
            neighbors = [i for i in candidates if i != source_index][:neighbor_count]
            for target_index in neighbors:
                pair = tuple(sorted((source_index, int(target_index))))
                if pair in seen_neighbors:
                    continue
                seen_neighbors.add(pair)
                target = episodes[target_index]
                source_scope_key = scope_partition_key(source.scope_id, source.scope_name)
                target_scope_key = scope_partition_key(target.scope_id, target.scope_name)
                shared = sorted(set(source.mechanism_tags or []) & set(target.mechanism_tags or []))
                surface_similarity = None
                if source.surface_embedding is not None and target.surface_embedding is not None:
                    surface_pair = np.asarray(
                        [source.surface_embedding, target.surface_embedding],
                        dtype=np.float32,
                    )
                    surface_similarity = float(_cosine_similarities(surface_pair)[0, 1])
                structural_similarity = float(similarities[source_index, target_index])
                edges.append(
                    {
                        "source": str(source.id),
                        "target": str(target.id),
                        "kind": "structural_neighbor",
                        "structural_similarity": structural_similarity,
                        "surface_similarity": surface_similarity,
                        "shared_mechanisms": shared,
                        "explanation": {
                            "summary": (
                                f"Structural cosine similarity {structural_similarity:.3f}"
                                + (
                                    f" with shared mechanisms: {', '.join(shared)}"
                                    if shared
                                    else "; no shared controlled mechanism tags"
                                )
                            ),
                            "same_arc_type": source.arc_type == target.arc_type,
                            "same_change_pattern": (
                                source.change_pattern == target.change_pattern and source.change_pattern is not None
                            ),
                            "same_scope": bool(source_scope_key and source_scope_key == target_scope_key),
                            "projection_is_approximate": True,
                        },
                    }
                )

        # An episode belongs to the trajectory of its focal subject and any
        # explicitly supported containing subject. This lets Han, Ming, Qing,
        # and a movement inside China appear as distinct facets of a larger
        # Chinese trajectory without flattening their stored focal identities.
        episodes_by_scope: dict[str, dict[UUID, tuple[EpisodeORM, str, str]]] = {}
        for episode in episodes:
            if episode.scope_confidence is not None and episode.scope_confidence < DEFAULT_SCOPE_CONFIDENCE_FLOOR:
                continue
            if _episode_start_year(episode) is None or not episode.change_pattern:
                continue
            initial_candidates = [
                (episode.scope_id or episode.scope_name, "focal"),
                (episode.parent_scope_name, "parent"),
            ]
            candidates: list[tuple[str | None, str]] = []
            for label, relation in initial_candidates:
                candidates.append((label, relation))
                candidates.extend((scope.id, "ancestor") for scope in scope_registry.lineage(label)[1:])
            for label, relation in candidates:
                key = scope_partition_key(None, label)
                if key and label:
                    canonical_scope = scope_registry.get(key)
                    display_label = canonical_scope.name if canonical_scope else label
                    episodes_by_scope.setdefault(key, {})[episode.id] = (episode, relation, display_label)

        for scope_entries in episodes_by_scope.values():
            ordered_scope_episodes = sorted(
                scope_entries.values(),
                key=lambda item: _episode_temporal_key(item[0]),
            )
            for (source, source_relation, scope_label), (target, target_relation, _) in zip(
                ordered_scope_episodes,
                ordered_scope_episodes[1:],
                strict=False,
            ):
                edges.append(
                    {
                        "source": str(source.id),
                        "target": str(target.id),
                        "kind": "scope_sequence",
                        "scope_name": scope_label,
                        "source_scope_relation": source_relation,
                        "target_scope_relation": target_relation,
                        "source_pattern": source.change_pattern,
                        "target_pattern": target.change_pattern,
                        "link_status": "attested",
                        "review_status": "auto",
                        "explanation": {
                            "summary": (
                                f"Chronological observations within {scope_label}; "
                                "the focal facets may differ and this is not a causal claim"
                            ),
                            "projection_is_approximate": True,
                        },
                    }
                )

        for cycle_memberships in memberships_by_cycle.values():
            ordered = sorted(
                cycle_memberships,
                key=lambda item: _episode_temporal_key(by_id[item[0].episode_id]),
            )
            for (source_membership, cycle), (target_membership, _) in zip(ordered, ordered[1:], strict=False):
                source = by_id[source_membership.episode_id]
                target = by_id[target_membership.episode_id]
                edges.append(
                    {
                        "source": str(source.id),
                        "target": str(target.id),
                        "kind": "arc_sequence",
                        "arc_id": str(cycle.id),
                        "arc_name": cycle.name,
                        "formation_status": arc_quality_by_cycle.get(cycle.id, {}).get(
                            "formation_status", "candidate"
                        ),
                        "link_status": target_membership.link_status,
                        "review_status": target_membership.review_status,
                        "structural_similarity": float(
                            similarities[episode_index[source.id], episode_index[target.id]]
                        ),
                        "shared_mechanisms": sorted(
                            set(source.mechanism_tags or []) & set(target.mechanism_tags or [])
                        ),
                        "explanation": {
                            "summary": (
                                f"Chronological progression within {cycle.name}: "
                                f"{source.arc_phase.value if source.arc_phase else 'unknown'} → "
                                f"{target.arc_phase.value if target.arc_phase else 'unknown'}"
                            ),
                            "evidence_status": target_membership.link_status,
                            "review_status": target_membership.review_status,
                        },
                    }
                )

        links = (
            (
                await session.execute(
                    select(EpisodeLinkORM).where(
                        EpisodeLinkORM.source_episode_id.in_(selected_ids),
                        EpisodeLinkORM.target_episode_id.in_(selected_ids),
                        EpisodeLinkORM.review_status != "rejected",
                    )
                )
            )
            .scalars()
            .all()
        )
        for link in links:
            edges.append(
                {
                    "source": str(link.source_episode_id),
                    "target": str(link.target_episode_id),
                    "kind": link.edge_kind,
                    "confidence": (max(0.0, 1.0 - float(link.distance)) if link.distance is not None else None),
                    "link_status": link.link_status,
                    "review_status": link.review_status,
                    "evidence": link.evidence,
                    "explanation": {
                        "summary": link.evidence or f"{link.edge_kind} relationship",
                        "evidence_status": link.link_status,
                        "review_status": link.review_status,
                    },
                }
            )

        return {
            "projection": {
                "algorithm": "pca",
                "source_dimensions": int(vectors.shape[1]),
                "display_dimensions": 3,
                "explained_variance": explained,
                "embedding_epoch": current_epoch("structural"),
                "warning": "Projection distance is approximate; similarity uses original vectors.",
            },
            "nodes": nodes,
            "edges": edges,
        }

    @app.get("/api/review-queue")
    async def review_queue(session: SessionDep) -> dict:
        memberships = (
            await session.execute(
                select(CycleMembershipORM, CycleORM.name, EpisodeORM.title)
                .join(CycleORM, CycleMembershipORM.cycle_id == CycleORM.id)
                .join(EpisodeORM, CycleMembershipORM.episode_id == EpisodeORM.id)
                .where(CycleMembershipORM.review_status == "pending")
                .limit(100)
            )
        ).all()
        links = (
            (await session.execute(select(EpisodeLinkORM).where(EpisodeLinkORM.review_status == "pending").limit(100)))
            .scalars()
            .all()
        )
        episode_ids = {link.source_episode_id for link in links} | {link.target_episode_id for link in links}
        titles = {}
        if episode_ids:
            rows = (
                await session.execute(select(EpisodeORM.id, EpisodeORM.title).where(EpisodeORM.id.in_(episode_ids)))
            ).all()
            titles = {row.id: row.title for row in rows}

        return {
            "memberships": [
                {
                    "id": str(m.CycleMembershipORM.id),
                    "cycle": m.name,
                    "episode": m.title,
                    "link_status": m.CycleMembershipORM.link_status,
                }
                for m in memberships
            ],
            "links": [
                {
                    "id": str(link.id),
                    "edge_kind": link.edge_kind,
                    "link_status": link.link_status,
                    "source": titles.get(link.source_episode_id, "?"),
                    "target": titles.get(link.target_episode_id, "?"),
                    "evidence": link.evidence,
                }
                for link in links
            ],
        }

    async def _apply_review(orm_row, decision: str, session: AsyncSession) -> dict:
        if decision not in ("approved", "rejected"):
            raise HTTPException(422, "decision must be 'approved' or 'rejected'")
        orm_row.review_status = decision
        await session.flush()
        await session.commit()
        return {"id": str(orm_row.id), "review_status": decision}

    @app.post("/api/documents/{document_id}/retry")
    async def retry_document(document_id: UUID, session: SessionDep) -> dict:
        """Queue a row so the watcher resumes it on the next scan.

        Retryable: failed rows, and completed rows whose extraction never
        ran (ingested before an LLM key was configured — re-picking them is
        exactly the "key arrived later" path). Fully-extracted work is not
        silently redone, and duplicates were rejected on purpose. Existing
        chunk progress stays intact so retries resume rather than duplicate.
        """
        row = await session.get(SourceDocumentORM, document_id)
        if row is None:
            raise HTTPException(404, "document not found")
        retryable = row.status == "failed" or (row.status == "completed" and not row.extraction_ran)
        if not retryable:
            raise HTTPException(
                409,
                f"not retryable (status={row.status}, extraction_ran={row.extraction_ran})",
            )
        row.status = "queued"
        row.error = None
        await session.flush()
        await session.commit()
        return {"id": str(document_id), "retried": True}

    @app.post("/api/review/membership/{membership_id}")
    async def review_membership(
        membership_id: UUID,
        body: ReviewDecision,
        session: SessionDep,
    ) -> dict:
        row = await session.get(CycleMembershipORM, membership_id)
        if row is None:
            raise HTTPException(404, "membership not found")
        return await _apply_review(row, body.decision, session)

    @app.post("/api/review/link/{link_id}")
    async def review_link(
        link_id: UUID,
        body: ReviewDecision,
        session: SessionDep,
    ) -> dict:
        row = await session.get(EpisodeLinkORM, link_id)
        if row is None:
            raise HTTPException(404, "link not found")
        return await _apply_review(row, body.decision, session)

    return app


app = create_app()
