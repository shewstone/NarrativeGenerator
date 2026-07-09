"""Analog episode retrieval combining vector similarity and graph traversal."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.models import (
    ArcPhase,
    ArcType,
    ClassificationState,
    CycleScale,
    Episode,
    MechanismTag,
)
from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from narrative_engine.storage.repositories import CycleRepository, EpisodeRepository

logger = structlog.get_logger()


@dataclass
class RetrievedAnalog:
    """A retrieved episode with similarity scores and reasoning."""

    episode: Episode
    semantic_similarity: float  # Vector cosine similarity
    arc_match_score: float  # 1.0 if same arc type, 0.5 if related, 0.0 otherwise
    phase_compatibility: float  # How well phases align for forecasting
    cycle_context_score: float  # Same cycle scale boosts relevance
    mechanism_match_score: float  # Jaccard overlap of mechanism_tags (Sec 3.8)
    combined_score: float  # Weighted combination of above
    retrieval_method: str  # "vector", "graph", or "hybrid"
    reasoning: str  # Why this analog was selected


class AnalogRetrievalEngine:
    """Retrieve historical analogs for a query episode."""

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_weight: float = 0.4,
        arc_weight: float = 0.2,
        phase_weight: float = 0.15,
        cycle_weight: float = 0.15,
        mechanism_weight: float = 0.1,
    ) -> None:
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_weight = vector_weight
        self.arc_weight = arc_weight
        self.phase_weight = phase_weight
        self.cycle_weight = cycle_weight
        self.mechanism_weight = mechanism_weight
        self.logger = structlog.get_logger()

    async def retrieve_analogs(
        self,
        query_episode: Episode,
        session: AsyncSession,
        k: int = 10,
        min_similarity: float = 0.7,
        same_arc_type: Optional[bool] = None,
        same_cycle_scale: Optional[CycleScale] = None,
    ) -> List[RetrievedAnalog]:
        """Retrieve k best historical analogs for a query episode.

        Combines vector similarity with structured filtering for relevance.
        """
        self.logger.info(
            "Starting analog retrieval",
            query_title=query_episode.title,
            k=k,
        )

        # Arc-less fallback (Sec 6.5.8): if the query situation itself has no
        # arc (failed tau_class), there is no phase and no phase-completion
        # forecast -- retrieval degrades to bare structural nearest-neighbor
        # matching. Unclassified corpus episodes are admissible analogs in
        # this mode (nothing conditions on arc labels); in arc-based mode
        # they are excluded from the analog base entirely.
        arc_less = (
            query_episode.arc_type is None
            or query_episode.classification_state == ClassificationState.UNCLASSIFIED
        )
        if arc_less:
            self.logger.info(
                "Query is unclassified; retrieval degrading to arc-less mode",
                query_title=query_episode.title,
            )

        # Step 1: Generate query embedding (structural -- analogy signal, Sec 3.3a)
        query_embedding = self.embedding_generator.generate_structural_embedding(query_episode)

        # Step 2: Vector search for candidates
        episode_repo = EpisodeRepository(session)

        # Get candidates via vector search (returns more than k for filtering)
        candidates = await episode_repo.search_by_embedding(
            query_embedding,
            limit=k * 3,  # Get extra for filtering
            include_unclassified=arc_less,
        )

        self.logger.info(f"Vector search returned {len(candidates)} candidates")

        # Step 3: Score and rank candidates
        scored_analogs: List[RetrievedAnalog] = []

        for episode, distance in candidates:
            # search_by_embedding returns cosine DISTANCE (0 = identical);
            # convert to similarity before thresholding -- filtering the raw
            # distance against min_similarity kept the most dissimilar
            # candidates and dropped the best ones.
            vector_sim = 1.0 - distance

            # Skip if below minimum similarity
            if vector_sim < min_similarity:
                continue

            # Skip the query episode itself
            if episode.id == query_episode.id:
                continue

            analog = await self._score_analog(
                query_episode=query_episode,
                candidate=episode,
                vector_similarity=vector_sim,
                session=session,
                arc_less=arc_less,
            )

            scored_analogs.append(analog)

        # Step 4: Sort by combined score and return top k
        scored_analogs.sort(key=lambda x: x.combined_score, reverse=True)

        self.logger.info(
            "Analog retrieval complete",
            retrieved=len(scored_analogs[:k]),
            top_score=scored_analogs[0].combined_score if scored_analogs else 0,
        )

        return scored_analogs[:k]

    async def _score_analog(
        self,
        query_episode: Episode,
        candidate: Episode,
        vector_similarity: float,
        session: AsyncSession,
        arc_less: bool = False,
    ) -> RetrievedAnalog:
        """Score a candidate analog across multiple dimensions.

        In arc-less mode (Sec 6.5.8) the query has no arc/phase, so the
        arc, phase, and mechanism components have nothing to condition on:
        combined_score is bare structural similarity with cycle context as
        soft context, and the component scores are recorded for
        transparency but carry no weight.
        """

        # Arc type match
        arc_match = self._compute_arc_match(
            query_episode.arc_type,
            candidate.arc_type,
        )

        # Phase compatibility (for forecasting utility)
        phase_compat = self._compute_phase_compatibility(
            query_episode.arc_phase,
            candidate.arc_phase,
            query_episode.arc_type,
            candidate.arc_type,
        )

        # Cycle context (episodes in same cycle scale are more comparable)
        cycle_score = await self._compute_cycle_context_score(
            query_episode,
            candidate,
            session,
        )

        # Mechanism overlap (design doc Sec 3.8): shared structural drivers
        mechanism_score = self._compute_mechanism_overlap(
            query_episode.mechanism_tags,
            candidate.mechanism_tags,
        )

        if arc_less:
            # Bare structural nearest-neighbor with cycle-state as soft
            # context only; weights renormalized over the two live signals.
            live_weight = self.vector_weight + self.cycle_weight
            combined = (
                self.vector_weight * vector_similarity
                + self.cycle_weight * cycle_score
            ) / live_weight
        else:
            # Combined score (weighted)
            combined = (
                self.vector_weight * vector_similarity
                + self.arc_weight * arc_match
                + self.phase_weight * phase_compat
                + self.cycle_weight * cycle_score
                + self.mechanism_weight * mechanism_score
            )

        # Generate reasoning
        reasoning_parts = []
        if arc_less:
            reasoning_parts.append("Arc-less structural match (query unclassified)")
        if not arc_less and arc_match > 0.8 and candidate.arc_type:
            reasoning_parts.append(f"Same arc type: {candidate.arc_type.value}")
        if not arc_less and phase_compat > 0.8 and candidate.arc_phase:
            reasoning_parts.append(f"Similar phase: {candidate.arc_phase.value}")
        if vector_similarity > 0.85:
            reasoning_parts.append("High semantic similarity")
        shared_mechanisms = set(query_episode.mechanism_tags) & set(candidate.mechanism_tags)
        if shared_mechanisms:
            reasoning_parts.append(
                f"Shared mechanisms: {', '.join(sorted(m.value for m in shared_mechanisms))}"
            )

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "General similarity"

        return RetrievedAnalog(
            episode=candidate,
            semantic_similarity=vector_similarity,
            arc_match_score=arc_match,
            phase_compatibility=phase_compat,
            cycle_context_score=cycle_score,
            mechanism_match_score=mechanism_score,
            combined_score=combined,
            retrieval_method="vector_arc_less" if arc_less else "hybrid",
            reasoning=reasoning,
        )

    def _compute_arc_match(
        self,
        query_arc: Optional[ArcType],
        candidate_arc: Optional[ArcType],
    ) -> float:
        """Score arc type match."""
        if not query_arc or not candidate_arc:
            return 0.5  # Neutral if unknown

        if query_arc == candidate_arc:
            return 1.0  # Exact match

        # Related arc types (could expand this taxonomy)
        related_arcs = {
            ArcType.CREDIT_BOOM_AND_BUST: [
                ArcType.RISE_AND_OVEREXTENSION,
                ArcType.HUBRIS_NEMESIS,
            ],
            ArcType.HUBRIS_NEMESIS: [
                ArcType.TRAGEDY,
                ArcType.RISE_AND_OVEREXTENSION,
            ],
            ArcType.HERO_JOURNEY: [
                ArcType.REBIRTH,
                ArcType.VOYAGE_RETURN,
            ],
        }

        if candidate_arc in related_arcs.get(query_arc, []):
            return 0.6  # Related

        return 0.1  # Different arc types

    def _compute_mechanism_overlap(
        self,
        query_tags: List[MechanismTag],
        candidate_tags: List[MechanismTag],
    ) -> float:
        """Score mechanism-tag overlap (design doc Sec 3.8) via Jaccard
        similarity. Neutral if either side has no tags -- most episodes
        won't have been re-tagged yet, so absence isn't evidence against
        a match (same neutral-default convention as actor-overlap in
        composition/identity.py).
        """
        if not query_tags or not candidate_tags:
            return 0.5

        query_set = set(query_tags)
        candidate_set = set(candidate_tags)

        union = query_set | candidate_set
        if not union:
            return 0.5

        return len(query_set & candidate_set) / len(union)

    def _compute_phase_compatibility(
        self,
        query_phase: Optional[ArcPhase],
        candidate_phase: Optional[ArcPhase],
        query_arc: Optional[ArcType],
        candidate_arc: Optional[ArcType],
    ) -> float:
        """Score phase compatibility for forecasting.

        Key insight: We want analogs at similar phases so we can
        observe how they transitioned (phase completion).
        """
        if not query_phase or not candidate_phase:
            return 0.5  # Neutral if unknown

        if query_arc and candidate_arc and query_arc == candidate_arc:
            # Same arc: phase matching is very important
            if query_phase == candidate_phase:
                return 1.0  # Same phase - ideal for forecasting

            # Sequential phases are also useful (shows progression)
            phase_order = self._get_phase_order(query_arc)
            if phase_order:
                with suppress(ValueError):
                    query_idx = phase_order.index(query_phase)
                    cand_idx = phase_order.index(candidate_phase)
                    distance = abs(query_idx - cand_idx)
                    if distance == 1:
                        return 0.7  # Adjacent phase
                    elif distance == 2:
                        return 0.4  # Nearby phase

        # Different arcs: just check if phases are similar categories
        query_category = self._categorize_phase(query_phase)
        cand_category = self._categorize_phase(candidate_phase)

        if query_category == cand_category:
            return 0.6

        return 0.3

    def _get_phase_order(self, arc_type: ArcType) -> List[ArcPhase]:
        """Get ordered phases for an arc type."""
        from narrative_engine.extraction.config import DEFAULT_ARC_TAXONOMY

        arc_info = DEFAULT_ARC_TAXONOMY.get(arc_type.value, {})
        phase_names = arc_info.get("phases", [])

        # Convert to ArcPhase enums
        phases = []
        for name in phase_names:
            with suppress(ValueError):
                phases.append(ArcPhase(name))

        return phases

    def _categorize_phase(self, phase: ArcPhase) -> str:
        """Categorize phase into early/middle/late."""
        early_phases = [ArcPhase.SETUP, ArcPhase.BOOM]
        middle_phases = [
            ArcPhase.RISING_ACTION,
            ArcPhase.EUPHORIA,
            ArcPhase.CLIMAX,
            ArcPhase.DISTRESS,
        ]
        late_phases = [
            ArcPhase.FALLING_ACTION,
            ArcPhase.PANIC,
            ArcPhase.RESOLUTION,
            ArcPhase.REVULSION,
        ]

        if phase in early_phases:
            return "early"
        elif phase in middle_phases:
            return "middle"
        elif phase in late_phases:
            return "late"
        return "unknown"

    async def _compute_cycle_context_score(
        self,
        query: Episode,
        candidate: Episode,
        session: AsyncSession,
    ) -> float:
        """Score cycle context similarity.

        Episodes in the same cycle scale are more directly comparable.
        """
        CycleRepository(session)

        # Get cycles for both episodes
        # This would require episode.cycles relationship to be loaded
        # For now, simplified approach

        # If both are in institutional cycles, that's more comparable than
        # one being civilizational and one being episodic

        # Simplified: assume good context match
        return 0.8

    async def retrieve_by_causal_chain(
        self,
        episode_id: UUID,
        session: AsyncSession,
        direction: str = "forward",  # "forward" or "backward"
        depth: int = 2,
    ) -> List[Episode]:
        """Retrieve episodes in causal chain (graph traversal).

        Follow CAUSES / CAUSED_BY edges to find downstream/upstream episodes.
        """
        # This would use the graph database
        # For now, placeholder returning empty list
        self.logger.info(
            "Causal chain retrieval",
            episode_id=episode_id,
            direction=direction,
            depth=depth,
        )
        return []

    async def retrieve_temporal_neighbors(
        self,
        episode: Episode,
        session: AsyncSession,
        years_before: int = 10,
        years_after: int = 10,
    ) -> List[Episode]:
        """Retrieve episodes temporally close to query."""
        from datetime import timedelta

        from sqlalchemy import select

        from narrative_engine.storage.orm_models import EpisodeORM

        if not episode.start_date:
            return []

        start_window = episode.start_date - timedelta(days=years_before * 365)
        end_window = episode.start_date + timedelta(days=years_after * 365)

        result = await session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.start_date >= start_window)
            .where(EpisodeORM.start_date <= end_window)
            .where(EpisodeORM.id != episode.id)
        )

        episodes = []
        for _orm_episode in result.scalars().all():
            # Convert to Pydantic model (would need repository method)
            pass

        return episodes
