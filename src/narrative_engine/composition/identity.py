"""Arc identity resolution - determining if episodes belong to the same arc instance.

Implements the fixed-order staged pipeline from design doc Sec 6.2 stage 6:
1. Hard filter: scope_id partition (done by the caller before this module).
2. Hard-ish filter: actor-entity overlap above threshold.
3. Soft signal: surface-embedding similarity (raw text, NOT structural).
4. Per-scale temporal gap threshold + arc_type agreement.
5. Phase-sequence continuity check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.models import Actor, ArcPhase, ArcType, Episode
from narrative_engine.storage.orm_models import EpisodeORM

_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_actor_name(name: str) -> str:
    """Alias-normalized actor name: lowercased, punctuation stripped,
    whitespace collapsed.

    Interim substitute for entity resolution (design doc Sec 6.2 stage 6
    item 2, "acceptable interim debt" until full entity resolution lands).
    Actor.id is a random UUID minted per extraction, so two independently
    extracted mentions of the same actor never share an id -- gating on id
    made the actor-overlap stage reject nearly every real match.
    """
    normalized = _NON_WORD_RE.sub("", name.strip().lower())
    return _WHITESPACE_RE.sub(" ", normalized).strip()


@dataclass
class IdentityScore:
    """Composite identity score for arc matching.
    
    CORRECTION: Uses surface embeddings (raw text similarity) for identity,
    NOT structural embeddings which are for analog retrieval.
    """
    
    # Hard filters (prerequisites)
    scope_match: bool  # Same polity/institution scope (hard filter)
    
    # Soft signals
    temporal_score: float  # 0-1, per-scale adjusted threshold
    actor_overlap_score: float  # 0-1, entity-resolved actor overlap
    surface_embedding_similarity: float  # 0-1, raw text similarity (NOT structural)
    phase_sequence_score: float  # 0-1, valid phase progression
    
    # Composite score (weighted combination)
    overall_score: float
    
    # Confidence and decision
    confidence: float  # 0-1, certainty in the score
    is_match: bool  # True if overall_score >= threshold
    
    # Reasons for decision (for human review)
    match_reasons: List[str]
    mismatch_reasons: List[str]


class ArcIdentityResolver:
    """Multi-factor identity resolution for arc composition.
    
    CORRECTION: Partition by scope_id BEFORE temporal clustering.
    Use surface embeddings (raw text), NOT structural embeddings.
    
    Determines if two episodes belong to the same concrete arc instance.
    """
    
    def __init__(
        self,
        # CORRECTION: Per-scale temporal thresholds, not global
        temporal_thresholds: Optional[Dict[str, timedelta]] = None,
        actor_weight: float = 0.30,
        temporal_weight: float = 0.30,
        surface_embedding_weight: float = 0.25,
        phase_weight: float = 0.15,
        match_threshold: float = 0.70,
        high_confidence_threshold: float = 0.85,
        actor_overlap_threshold: float = 0.1,
        min_evidence_signals: int = 2,
    ):
        # CORRECTION: Thresholds are hypotheses - tune against composition fixture
        # Starting points (user suggestion): episodic ~1-2y, institutional ~5y,
        # generational ~10y, civilizational ~30-50y
        self.temporal_thresholds = temporal_thresholds or {
            "episodic": timedelta(days=365 * 2),       # ~2 years (panic episodes)
            "institutional": timedelta(days=365 * 5),   # ~5 years (debt buildup)
            "generational": timedelta(days=365 * 10),  # ~10 years (secular cycles)
            "civilizational": timedelta(days=365 * 40), # ~40 years (long arcs)
        }
        # NOTE: These are untuned hypotheses. Validate against composition fixture
        # before production use.
        self.actor_weight = actor_weight
        self.temporal_weight = temporal_weight
        self.surface_embedding_weight = surface_embedding_weight
        self.phase_weight = phase_weight
        self.match_threshold = match_threshold
        self.high_confidence_threshold = high_confidence_threshold
        # Stage 2 hard-ish filter (design doc Sec 6.2 stage 6): actor overlap
        # below this rejects the pair outright, unless neutral (no actor data).
        self.actor_overlap_threshold = actor_overlap_threshold
        # Evidence floor: every gate treats missing data as a neutral pass
        # (interim-debt convention), which left the DEGENERATE case open --
        # an episode pair with no dates, no actors, no embeddings, and no
        # phases passed every gate, making the worst-documented episodes
        # the easiest to merge. is_match now additionally requires at least
        # this many signals computed from data actually present on both
        # sides. Identity is an evidence claim; absence of evidence must
        # not behave like weak evidence for it.
        self.min_evidence_signals = min_evidence_signals

    def calculate_identity_score(
        self,
        episode_a: Episode,
        episode_b: Episode,
        scale: str = "institutional",
    ) -> IdentityScore:
        """Calculate identity score between two episodes.

        CORRECTION: Assumes scope_id already matched (hard partition -- stage
        1, checked by the caller before this method runs).

        Implements the fixed-order staged pipeline from design doc Sec 6.2
        stage 6, NOT a single weighted-sum-vs-threshold: `is_match` is a hard
        cascade (actor-overlap gate -> per-scale temporal gate -> arc_type
        agreement -> phase-sequence continuity gate). `overall_score` remains
        a weighted diagnostic/ranking value (used by find_candidate_matches,
        suggest_split_points, requires_human_review) but no longer determines
        is_match on its own -- that conflated the stage-3 soft signal
        (surface-embedding similarity) into a hard cutoff, which is exactly
        what made this method a parallel weighted sum instead of a staged
        filter pipeline.

        Args:
            scale: cycle scale this comparison is scoped to (Sec 6.2 stage 6
                item 4: per-scale gap thresholds, not a single global one).
                Defaults to "institutional" to preserve existing callers'
                behavior; CompositionPipeline passes the scale explicitly.
        """
        match_reasons = []
        mismatch_reasons = []

        # CORRECTION: Scope match is prerequisite (hard filter)
        # This should be checked BEFORE calling this method
        scope_match = True  # Caller ensures same scope

        # Stage 4: per-scale temporal threshold
        temporal_score = self._calculate_temporal_score(episode_a, episode_b, scale)
        if temporal_score > 0.8:
            match_reasons.append(f"High temporal proximity ({temporal_score:.2f})")
        elif temporal_score < 0.3:
            mismatch_reasons.append(f"Low temporal proximity ({temporal_score:.2f})")

        # Stage 2: actor overlap (entity-resolved, not string match)
        actor_score = self._calculate_actor_overlap(episode_a, episode_b)
        actor_gate = self._passes_actor_gate(episode_a, episode_b, actor_score)
        if actor_score > 0.5:
            match_reasons.append(f"Actor entity overlap ({actor_score:.2f})")
        elif not actor_gate:
            mismatch_reasons.append("No actor entity overlap")

        # Stage 3: surface embedding similarity (raw text, NOT structural) --
        # a SOFT signal per the doc: feeds the score, never gates is_match.
        surface_embedding_score = self._calculate_surface_embedding_similarity(episode_a, episode_b)
        if surface_embedding_score > 0.8:
            match_reasons.append(f"High surface similarity ({surface_embedding_score:.2f})")
        elif surface_embedding_score < 0.5 and surface_embedding_score > 0:
            mismatch_reasons.append(f"Low surface similarity ({surface_embedding_score:.2f})")

        # Stage 5: phase sequence continuity
        phase_score = self._calculate_phase_sequence(episode_a, episode_b)
        phase_gate = phase_score > 0.0  # 0.0 == strictly out of order (hard reject)
        if phase_score > 0.8:
            match_reasons.append(f"Sequential phases ({episode_a.arc_phase.value} → {episode_b.arc_phase.value})")
        elif phase_score < 0.3:
            mismatch_reasons.append(f"Non-sequential phases ({episode_a.arc_phase.value} vs {episode_b.arc_phase.value})")

        # Stage 4 (arc_type agreement half): unknown arc_type on either side
        # is neutral (doesn't block); a known mismatch is a hard reject.
        arc_type_gate = (
            episode_a.arc_type == episode_b.arc_type
            if episode_a.arc_type and episode_b.arc_type
            else True
        )
        if not arc_type_gate:
            mismatch_reasons.append(
                f"Different arc types ({episode_a.arc_type} vs {episode_b.arc_type})"
            )

        # Temporal gate: 0.0 means beyond the per-scale threshold (with
        # overflow allowance) -- a hard reject, not just a low score.
        temporal_gate = temporal_score > 0.0

        # Evidence floor: count signals computed from data present on BOTH
        # sides (as opposed to neutral 0.5 defaults for missing data).
        temporal_concrete = bool(episode_a.end_date and episode_b.start_date)
        actor_concrete = bool(episode_a.actors and episode_b.actors)
        surface_concrete = (
            episode_a.surface_embedding is not None
            and episode_b.surface_embedding is not None
        )
        phase_concrete = bool(episode_a.arc_phase and episode_b.arc_phase)
        evidence_signals = sum(
            [temporal_concrete, actor_concrete, surface_concrete, phase_concrete]
        )
        evidence_gate = evidence_signals >= self.min_evidence_signals
        if not evidence_gate:
            mismatch_reasons.append(
                f"Insufficient identity evidence ({evidence_signals} concrete "
                f"signals < {self.min_evidence_signals} required)"
            )

        # Calculate weighted composite score
        # CORRECTION: No location weight, added surface embedding
        overall_score = (
            temporal_score * self.temporal_weight +
            actor_score * self.actor_weight +
            surface_embedding_score * self.surface_embedding_weight +
            phase_score * self.phase_weight
        )

        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(
            episode_a, episode_b, temporal_score, actor_score, surface_embedding_score
        )

        # Stage 6 identity decision: hard cascade, not overall_score vs.
        # threshold. All four gates must pass, AND enough of them must be
        # grounded in actual data (evidence floor above) -- gates that all
        # passed by neutral default are not an identity finding.
        is_match = (
            actor_gate and temporal_gate and arc_type_gate and phase_gate and evidence_gate
        )

        return IdentityScore(
            scope_match=scope_match,
            temporal_score=temporal_score,
            actor_overlap_score=actor_score,
            surface_embedding_similarity=surface_embedding_score,
            phase_sequence_score=phase_score,
            overall_score=overall_score,
            confidence=confidence,
            is_match=is_match,
            match_reasons=match_reasons,
            mismatch_reasons=mismatch_reasons,
        )

    def _passes_actor_gate(self, a: Episode, b: Episode, actor_score: Optional[float] = None) -> bool:
        """Stage 2 hard-ish filter: actor overlap above threshold.

        Neutral (no actor data on either side) never blocks -- alias-
        normalized name match is "acceptable interim debt" per the design
        doc until full entity resolution lands (Sec 6.2 stage 6 item 2).
        """
        if not a.actors or not b.actors:
            return True
        score = actor_score if actor_score is not None else self._calculate_actor_overlap(a, b)
        return score >= self.actor_overlap_threshold

    def _calculate_temporal_score(self, a: Episode, b: Episode, scale: str = "institutional") -> float:
        """Calculate temporal proximity score (0-1) with per-scale threshold.

        CORRECTION: Different thresholds for episodic vs institutional vs generational arcs.
        """
        if not a.end_date or not b.start_date:
            return 0.5  # Neutral if dates unknown

        # CORRECTION: Select threshold based on the scale this comparison is
        # scoped to (arc instances are episodic-scale by convention, Sec 2;
        # other cycle-membership resolution may pass other scales).
        threshold = self.temporal_thresholds.get(scale, self.temporal_thresholds["institutional"])

        # Gap between episodes (negative = overlap)
        gap = b.start_date - a.end_date
        
        if gap.total_seconds() < 0:
            # Episodes overlap - maximum temporal connection
            return 1.0
        
        # Exponential decay based on gap
        gap_days = gap.days
        threshold_days = threshold.days
        
        if gap_days > threshold_days * 2:  # Allow some overflow
            return 0.0
        
        # Exponential decay: score = exp(-3 * gap / threshold)
        import math
        score = math.exp(-3.0 * gap_days / threshold_days)
        return score
    
    def _calculate_actor_overlap(self, a: Episode, b: Episode) -> float:
        """Calculate Jaccard similarity of actor sets, matched by
        alias-normalized name (see normalize_actor_name) rather than
        Actor.id, since independently extracted episodes never share ids
        for the same real-world actor.
        """
        if not a.actors or not b.actors:
            return 0.5  # Neutral if no actors

        actors_a = {normalize_actor_name(actor.name) for actor in a.actors}
        actors_b = {normalize_actor_name(actor.name) for actor in b.actors}
        
        if not actors_a or not actors_b:
            return 0.0
        
        intersection = len(actors_a & actors_b)
        union = len(actors_a | actors_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    # CORRECTION: Removed _calculate_location_match - location is not identity signal
    # Location feeds into scope resolution, not direct matching
    
    def _calculate_surface_embedding_similarity(self, a: Episode, b: Episode) -> float:
        """Calculate surface embedding similarity (raw text, NOT structural).

        CORRECTION: Uses surface embeddings (raw summaries) for identity.
        Structural embeddings (arc patterns) are for analog retrieval, NOT identity.
        No fallback to structural_embedding: that fallback was the exact bug
        v0.5 corrected -- it would happily score identity on the place/date-
        blind structural signal, merging unrelated same-shaped episodes.
        """
        a_surface = a.surface_embedding
        b_surface = b.surface_embedding

        if a_surface is None or b_surface is None:
            return 0.5  # Neutral if no surface embeddings
        
        # Cosine similarity
        import numpy as np
        
        a_vec = np.array(a_surface)
        b_vec = np.array(b_surface)
        
        norm_a = np.linalg.norm(a_vec)
        norm_b = np.linalg.norm(b_vec)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = np.dot(a_vec, b_vec) / (norm_a * norm_b)
        
        # Normalize to 0-1
        return (similarity + 1) / 2
    
    def _calculate_phase_sequence(self, a: Episode, b: Episode) -> float:
        """Calculate phase sequence continuity score."""
        if not a.arc_phase or not b.arc_phase:
            return 0.5  # Neutral if unknown
        
        # Define phase order for common arcs
        phase_orders = {
            ArcType.CREDIT_BOOM_AND_BUST: [
                ArcPhase.BOOM, ArcPhase.EUPHORIA, ArcPhase.DISTRESS, 
                ArcPhase.PANIC, ArcPhase.REVULSION
            ],
            ArcType.HUBRIS_NEMESIS: [
                ArcPhase.SETUP, ArcPhase.RISING_ACTION, ArcPhase.CLIMAX,
                ArcPhase.FALLING_ACTION, ArcPhase.RESOLUTION
            ],
        }
        
        phase_order = phase_orders.get(a.arc_type, [])
        
        if not phase_order:
            return 0.5  # Neutral for unknown arc types
        
        try:
            idx_a = phase_order.index(a.arc_phase)
            idx_b = phase_order.index(b.arc_phase)
        except ValueError:
            return 0.5  # Neutral if phases not in order
        
        # Ideal: b comes immediately after a
        expected_gap = 1
        actual_gap = idx_b - idx_a
        
        if actual_gap == expected_gap:
            return 1.0  # Perfect sequence
        elif actual_gap > 0:
            # Valid sequence but with gaps
            return max(0.0, 1.0 - (actual_gap - expected_gap) * 0.3)
        else:
            # b comes before a (out of order)
            return 0.0
    
    def _calculate_confidence(
        self,
        a: Episode,
        b: Episode,
        temporal_score: float,
        actor_score: float,
        embedding_score: float,
    ) -> float:
        """Calculate confidence in the identity score."""
        # Confidence is higher when we have more data
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more signals
        if a.start_date and b.start_date and a.end_date and b.end_date:
            confidence += 0.15
        
        if a.actors and b.actors:
            confidence += 0.15
        
        if a.location and b.location:
            confidence += 0.10
        
        if embedding_score != 0.5:  # Embeddings exist
            confidence += 0.10
        
        return min(confidence, 1.0)
    
    def requires_human_review(self, score: IdentityScore) -> bool:
        """Determine if this match needs human review.
        
        Review triggered when:
        - Borderline score (close to threshold)
        - Low confidence
        - Contradictory signals (high score but low confidence)
        """
        # Borderline cases
        if abs(score.overall_score - self.match_threshold) < 0.1:
            return True
        
        # Low confidence
        if score.confidence < 0.6:
            return True
        
        # Contradictory: match but low confidence
        if score.is_match and score.confidence < 0.7:
            return True
        
        # Contradictory: high score but mismatched signals
        if score.overall_score > 0.8:
            mismatch_count = len(score.mismatch_reasons)
            if mismatch_count >= 2:
                return True
        
        return False
    
    async def find_candidate_matches(
        self,
        session: AsyncSession,
        episode: Episode,
        arc_type: ArcType,
        limit: int = 20,
    ) -> List[Tuple[Episode, IdentityScore]]:
        """Find candidate episodes that might match the given episode.
        
        Uses efficient database filtering before computing full identity scores.
        """
        # Build query with temporal filter
        query = select(EpisodeORM).where(
            EpisodeORM.arc_type == arc_type
        ).where(
            EpisodeORM.id != episode.id
        )
        
        # Temporal filter (episodes within threshold)
        if episode.end_date:
            date_range = self.temporal_thresholds.get("institutional", timedelta(days=365)) * 2
            query = query.where(
                EpisodeORM.start_date >= episode.end_date - date_range
            ).where(
                EpisodeORM.start_date <= episode.end_date + date_range
            )
        
        # Add limit
        query = query.limit(limit)
        
        result = await session.execute(query)
        candidates = result.scalars().all()
        
        # Calculate identity scores
        scored_matches = []
        for candidate_orm in candidates:
            # Convert to Pydantic model
            candidate = self.episode_from_orm(candidate_orm)
            score = self.calculate_identity_score(episode, candidate)
            scored_matches.append((candidate, score))
        
        # Sort by score descending
        scored_matches.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scored_matches
    
    def episode_from_orm(self, orm: EpisodeORM) -> Episode:
        """Convert ORM to Pydantic (fields relevant to identity + composition).

        Shared by CompositionPipeline so ORM<->Pydantic conversion for
        identity-relevant fields lives in one place.
        """
        from narrative_engine.models import Actor, SourcePassage
        
        return Episode(
            id=orm.id,
            title=orm.title,
            summary=orm.summary or "",
            start_date=orm.start_date,
            end_date=orm.end_date,
            location=orm.location,
            scope_id=orm.scope_id,
            actors=[
                Actor(id=a.id, name=a.name, role=a.role, attributes=a.attributes)
                for a in (orm.actors or [])
            ],
            arc_type=orm.arc_type,
            arc_phase=orm.arc_phase,
            phase_confidence=orm.phase_confidence,
            extracted_from=orm.extracted_from,
            surface_embedding=orm.surface_embedding,
            structural_embedding=orm.structural_embedding,
        )


class DisambiguationEngine:
    """Detect and resolve potential false merges.
    
    CORRECTION: Uses composition fixture for validation.
    Build fixture BEFORE tuning thresholds — otherwise tune blind.
    """
    
    def __init__(self, resolver: ArcIdentityResolver):
        self.resolver = resolver
    
    def detect_false_merge_risk(
        self,
        cluster: Sequence[Episode],
    ) -> Dict[str, any]:
        """Analyze a cluster for signs of false merge.
        
        Returns risk assessment with confidence and recommended action.
        """
        if len(cluster) < 2:
            return {"risk": "low", "reason": "Single episode cluster"}
        
        risks = []
        
        # Check for location discontinuity
        locations = set(e.location for e in cluster if e.location)
        if len(locations) > 2:
            risks.append(f"Multiple locations: {locations}")
        
        # Check for temporal gaps
        sorted_eps = sorted(cluster, key=lambda e: e.start_date or datetime.min)
        for i in range(len(sorted_eps) - 1):
            gap = (sorted_eps[i+1].start_date or datetime.min) - (sorted_eps[i].end_date or datetime.min)
            if gap > timedelta(days=365 * 3):  # 3+ year gaps are suspicious
                risks.append(f"Large temporal gap: {gap.days} days between episodes")
        
        # Check for actor discontinuity (alias-normalized name match --
        # Actor.id is a per-extraction UUID, see normalize_actor_name)
        actor_continuity_scores = []
        for i in range(len(sorted_eps) - 1):
            current_actors = {normalize_actor_name(a.name) for a in sorted_eps[i].actors}
            next_actors = {normalize_actor_name(a.name) for a in sorted_eps[i+1].actors}
            
            if current_actors and next_actors:
                overlap = len(current_actors & next_actors) / len(current_actors | next_actors)
                actor_continuity_scores.append(overlap)
        
        avg_actor_continuity = sum(actor_continuity_scores) / len(actor_continuity_scores) if actor_continuity_scores else 0
        
        if avg_actor_continuity < 0.2 and len(cluster) > 2:
            risks.append(f"Low actor continuity ({avg_actor_continuity:.2f})")
        
        # Assess overall risk
        risk_level = "low"
        if len(risks) >= 2:
            risk_level = "high"
        elif len(risks) == 1:
            risk_level = "medium"
        
        return {
            "risk": risk_level,
            "risk_factors": risks,
            "recommendation": (
                "manual_review" if risk_level == "high" else
                "proceed_with_caution" if risk_level == "medium" else
                "proceed"
            ),
            "actor_continuity": avg_actor_continuity,
        }
    
    def suggest_split_points(
        self,
        cluster: Sequence[Episode],
    ) -> List[int]:
        """Suggest where to split a cluster if it's a false merge.
        
        Returns indices where cluster should be split.
        """
        if len(cluster) < 3:
            return []
        
        split_points = []
        sorted_eps = sorted(cluster, key=lambda e: e.start_date or datetime.min)
        
        for i in range(len(sorted_eps) - 1):
            score = self.resolver.calculate_identity_score(
                sorted_eps[i], sorted_eps[i+1]
            )
            
            # Low score between adjacent episodes suggests split
            if score.overall_score < 0.3:
                split_points.append(i + 1)
        
        return split_points
