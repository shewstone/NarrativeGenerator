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
from contextlib import asynccontextmanager
from importlib import resources
from typing import AsyncGenerator, Optional
from uuid import UUID

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.logging_config import get_logger
from narrative_engine.storage.orm_models import (
    CycleMembershipORM,
    CycleORM,
    EpisodeLinkORM,
    EpisodeORM,
    SourceDocumentORM,
)
from narrative_engine.storage.repositories import SourceDocumentRepository

logger = get_logger(__name__)


def _pca_3d(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Deterministic, dependency-light 3D projection for exploration only."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if len(vectors) == 1 or not np.any(centered):
        return np.zeros((len(vectors), 3)), [0.0, 0.0, 0.0]
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


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Session dependency; tests override this with their fixture session."""
    from narrative_engine.storage.database import db_manager

    async with db_manager.session() as session:
        yield session


class ReviewDecision(BaseModel):
    decision: str  # "approved" | "rejected"


def create_app(start_watcher: Optional[bool] = None) -> FastAPI:
    if start_watcher is None:
        start_watcher = os.getenv("NE_WATCH_ENABLED", "true").lower() == "true"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher_task = None
        if start_watcher:
            from narrative_engine.watcher import watch_loop

            watcher_task = asyncio.create_task(watch_loop())
        yield
        if watcher_task:
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Narrative Engine", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return (
            resources.files("narrative_engine.api")
            .joinpath("static/dashboard.html")
            .read_text()
        )

    @app.get("/api/health")
    async def health(session: AsyncSession = Depends(get_session)) -> dict:
        async def count(stmt):
            return (await session.execute(stmt)).scalar() or 0

        return {
            "status": "ok",
            "documents": await count(select(func.count(SourceDocumentORM.id))),
            "episodes": await count(select(func.count(EpisodeORM.id))),
            "arc_instances": await count(
                select(func.count(CycleORM.id)).where(CycleORM.is_arc_instance)
            ),
            "pending_reviews": (
                await count(
                    select(func.count(CycleMembershipORM.id)).where(
                        CycleMembershipORM.review_status == "pending"
                    )
                )
            )
            + (
                await count(
                    select(func.count(EpisodeLinkORM.id)).where(
                        EpisodeLinkORM.review_status == "pending"
                    )
                )
            ),
        }

    @app.get("/api/documents")
    async def documents(session: AsyncSession = Depends(get_session)) -> list:
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
    async def arc_instances(session: AsyncSession = Depends(get_session)) -> list:
        from narrative_engine.composition.pipeline import _infer_expected_phases
        from narrative_engine.models import ArcType

        cycles = (
            (
                await session.execute(
                    select(CycleORM)
                    .where(CycleORM.is_arc_instance)
                    .order_by(CycleORM.created_at.desc())
                    .limit(100)
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
                        CycleMembershipORM.cycle_id.in_(cycle_ids)
                    )
                )
            )
            .scalars()
            .all()
        )
        episode_ids = {m.episode_id for m in memberships}
        episodes = {}
        if episode_ids:
            rows = (
                (
                    await session.execute(
                        select(EpisodeORM).where(EpisodeORM.id.in_(episode_ids))
                    )
                )
                .scalars()
                .all()
            )
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
            try:
                expected_phases = [
                    p.value for p in _infer_expected_phases(ArcType(arc_value))
                ]
            except (ValueError, TypeError):
                pass

            members = []
            for m in sorted(
                by_cycle.get(cycle.id, []),
                key=lambda m: (
                    episodes[m.episode_id].start_date is None,
                    episodes[m.episode_id].start_date,
                )
                if m.episode_id in episodes
                else (True, None),
            ):
                episode = episodes.get(m.episode_id)
                if episode is None:
                    continue
                members.append(
                    {
                        "id": str(episode.id),
                        "title": episode.title,
                        "phase": episode.arc_phase.value if episode.arc_phase else None,
                        "start_date": episode.start_date.isoformat()
                        if episode.start_date
                        else None,
                        "end_date": episode.end_date.isoformat()
                        if episode.end_date
                        else None,
                        "link_status": m.link_status,
                        "review_status": m.review_status,
                        "membership_id": str(m.id),
                    }
                )

            covered = {m["phase"] for m in members if m["phase"]}
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
                }
            )
        return payload

    @app.get("/api/arc-space")
    async def arc_space(
        k: int = Query(3, ge=1, le=10),
        limit: int = Query(500, ge=1, le=1000),
        session: AsyncSession = Depends(get_session),
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
                        EpisodeORM.structural_embedding_epoch
                        == current_epoch("structural"),
                    )
                    .order_by(EpisodeORM.start_date, EpisodeORM.created_at)
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

        vectors = np.asarray(
            [np.asarray(episode.structural_embedding, dtype=float) for episode in episodes]
        )
        coordinates, explained = _pca_3d(vectors)
        similarities = _cosine_similarities(vectors)
        episode_ids = [episode.id for episode in episodes]
        selected_ids = set(episode_ids)
        by_id = {episode.id: episode for episode in episodes}

        memberships = (
            (
                await session.execute(
                    select(CycleMembershipORM, CycleORM)
                    .join(CycleORM, CycleMembershipORM.cycle_id == CycleORM.id)
                    .where(
                        CycleORM.is_arc_instance,
                        CycleMembershipORM.episode_id.in_(selected_ids),
                        CycleMembershipORM.review_status != "rejected",
                    )
                )
            )
            .all()
        )
        memberships_by_episode: dict = {}
        memberships_by_cycle: dict = {}
        for membership, cycle in memberships:
            memberships_by_episode.setdefault(membership.episode_id, []).append(
                (membership, cycle)
            )
            memberships_by_cycle.setdefault(cycle.id, []).append((membership, cycle))

        nodes = []
        for index, episode in enumerate(episodes):
            episode_memberships = memberships_by_episode.get(episode.id, [])
            nodes.append(
                {
                    "id": str(episode.id),
                    "title": episode.title,
                    "summary": episode.summary,
                    "x": float(coordinates[index, 0]),
                    "y": float(coordinates[index, 1]),
                    "z": float(coordinates[index, 2]),
                    "arc_type": episode.arc_type.value if episode.arc_type else None,
                    "phase": episode.arc_phase.value if episode.arc_phase else None,
                    "confidence": float(episode.phase_confidence or 0.0),
                    "classification_state": episode.classification_state,
                    "start_date": episode.start_date.isoformat() if episode.start_date else None,
                    "end_date": episode.end_date.isoformat() if episode.end_date else None,
                    "scope_id": episode.scope_id,
                    "location": episode.location,
                    "mechanisms": list(episode.mechanism_tags or []),
                    "source_chunks": list(episode.extracted_from or []),
                    "arc_instances": [
                        {
                            "id": str(cycle.id),
                            "name": cycle.name,
                            "link_status": membership.link_status,
                            "review_status": membership.review_status,
                        }
                        for membership, cycle in episode_memberships
                    ],
                }
            )

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
                shared = sorted(
                    set(source.mechanism_tags or []) & set(target.mechanism_tags or [])
                )
                surface_similarity = None
                if source.surface_embedding is not None and target.surface_embedding is not None:
                    surface_pair = np.asarray(
                        [source.surface_embedding, target.surface_embedding], dtype=float
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
                            "same_scope": bool(
                                source.scope_id
                                and target.scope_id
                                and source.scope_id == target.scope_id
                            ),
                            "projection_is_approximate": True,
                        },
                    }
                )

        for cycle_memberships in memberships_by_cycle.values():
            ordered = sorted(
                cycle_memberships,
                key=lambda item: (
                    by_id[item[0].episode_id].start_date is None,
                    by_id[item[0].episode_id].start_date,
                    by_id[item[0].episode_id].created_at,
                ),
            )
            for (source_membership, cycle), (target_membership, _) in zip(
                ordered, ordered[1:], strict=False
            ):
                source = by_id[source_membership.episode_id]
                target = by_id[target_membership.episode_id]
                edges.append(
                    {
                        "source": str(source.id),
                        "target": str(target.id),
                        "kind": "arc_sequence",
                        "arc_id": str(cycle.id),
                        "arc_name": cycle.name,
                        "link_status": target_membership.link_status,
                        "review_status": target_membership.review_status,
                        "structural_similarity": float(
                            similarities[
                                episode_ids.index(source.id), episode_ids.index(target.id)
                            ]
                        ),
                        "shared_mechanisms": sorted(
                            set(source.mechanism_tags or [])
                            & set(target.mechanism_tags or [])
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
                    "confidence": (
                        max(0.0, 1.0 - float(link.distance))
                        if link.distance is not None
                        else None
                    ),
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
    async def review_queue(session: AsyncSession = Depends(get_session)) -> dict:
        memberships = (
            (
                await session.execute(
                    select(CycleMembershipORM, CycleORM.name, EpisodeORM.title)
                    .join(CycleORM, CycleMembershipORM.cycle_id == CycleORM.id)
                    .join(EpisodeORM, CycleMembershipORM.episode_id == EpisodeORM.id)
                    .where(CycleMembershipORM.review_status == "pending")
                    .limit(100)
                )
            )
            .all()
        )
        links = (
            (
                await session.execute(
                    select(EpisodeLinkORM).where(EpisodeLinkORM.review_status == "pending").limit(100)
                )
            )
            .scalars()
            .all()
        )
        episode_ids = {l.source_episode_id for l in links} | {
            l.target_episode_id for l in links
        }
        titles = {}
        if episode_ids:
            rows = (
                await session.execute(
                    select(EpisodeORM.id, EpisodeORM.title).where(
                        EpisodeORM.id.in_(episode_ids)
                    )
                )
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
                    "id": str(l.id),
                    "edge_kind": l.edge_kind,
                    "link_status": l.link_status,
                    "source": titles.get(l.source_episode_id, "?"),
                    "target": titles.get(l.target_episode_id, "?"),
                    "evidence": l.evidence,
                }
                for l in links
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
    async def retry_document(
        document_id: UUID, session: AsyncSession = Depends(get_session)
    ) -> dict:
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
        retryable = row.status == "failed" or (
            row.status == "completed" and not row.extraction_ran
        )
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
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        row = await session.get(CycleMembershipORM, membership_id)
        if row is None:
            raise HTTPException(404, "membership not found")
        return await _apply_review(row, body.decision, session)

    @app.post("/api/review/link/{link_id}")
    async def review_link(
        link_id: UUID,
        body: ReviewDecision,
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        row = await session.get(EpisodeLinkORM, link_id)
        if row is None:
            raise HTTPException(404, "link not found")
        return await _apply_review(row, body.decision, session)

    return app


app = create_app()
