"""Thesis generation: synthesize forecasts from historical analogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

import structlog

from narrative_engine.extraction.config import CURRENT_TAXONOMY_VERSION
from narrative_engine.models import (
    ClassificationState,
    Continuation,
    Episode,
    Thesis,
    ThesisConfidence,
    ThesisMode,
    utcnow,
)
from narrative_engine.retrieval.analog_retrieval import RetrievedAnalog
from narrative_engine.scopes import scope_registry_version

logger = structlog.get_logger()


@dataclass
class AnalogEvidence:
    """Evidence from a single historical analog."""

    analog: RetrievedAnalog
    relevance_score: float  # How relevant to the query
    continuation: str  # What happened next in this analog
    probability_weight: float  # Confidence weight for this analog


@dataclass
class ThesisDraft:
    """Draft thesis before final synthesis."""

    query_summary: str
    dominant_arc: Optional[str]
    current_phase: Optional[str]
    analogs: List[RetrievedAnalog]
    evidence: List[AnalogEvidence]
    raw_continuations: List[Continuation]
    confidence_breakdown: Dict[str, float]
    key_uncertainties: List[str]


class ThesisGenerator:
    """Generate forecasts from historical analogs."""

    def __init__(
        self,
        min_analogs: int = 3,
        max_analogs: int = 10,
        confidence_threshold: float = 0.6,
        max_per_source: int = 3,
        llm_client=None,
    ) -> None:
        self.min_analogs = min_analogs
        self.max_analogs = max_analogs
        self.confidence_threshold = confidence_threshold
        # Defense-in-depth against single-book narrative dominance (T6):
        # even when identity resolution misses duplicates, one source can't
        # supply more than this many analogs to a thesis.
        self.max_per_source = max_per_source
        self.llm_client = llm_client
        self.logger = structlog.get_logger()

    def generate(
        self,
        query_episode: Episode,
        analogs: List[RetrievedAnalog],
    ) -> Thesis:
        """Generate thesis from query and historical analogs.

        Key insight: Historical analogs don't give point predictions,
        they give distributions of possible continuations weighted by
        structural similarity and outcome probability.
        """
        self.logger.info(
            "Generating thesis",
            query=query_episode.title,
            analogs=len(analogs),
        )

        # Arc-less degraded mode (Sec 6.5.8): the query situation failed the
        # classification floor, so there is no phase and no phase-completion
        # forecast. The thesis is still emitted -- silent failure and silent
        # shoehorning are both prohibited -- but visibly degraded: wider
        # stated uncertainty, no arc-conditioned probability weights.
        mode = self._determine_mode(query_episode)

        # Filter and rank analogs
        filtered = self._filter_analogs(analogs)

        if len(filtered) < self.min_analogs:
            self.logger.warning(f"Insufficient analogs ({len(filtered)} < {self.min_analogs})")
            return self._create_uncertain_thesis(query_episode, filtered, mode)

        # Extract continuations from analogs
        evidence = self._extract_evidence(filtered, mode)

        # Generate alternative futures (continuations)
        continuations = self._generate_continuations(evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, continuations)

        # Arc-less theses cap at LOW: branch frequencies rest on bare
        # structural similarity with no phase-conditioned transition
        # statistics behind them (Sec 6.5.8, "wider stated uncertainty").
        if mode == ThesisMode.ARC_LESS and confidence in (
            ThesisConfidence.HIGH,
            ThesisConfidence.MEDIUM,
        ):
            confidence = ThesisConfidence.LOW

        # Generate watch conditions
        watch_conditions = self._generate_watch_conditions(evidence)

        # Generate key uncertainties
        uncertainties = self._identify_uncertainties(evidence)
        if mode == ThesisMode.ARC_LESS:
            uncertainties.insert(
                0,
                "Situation fits no known arc (unclassified): forecast rests on "
                "structural similarity only, with no phase-completion statistics",
            )

        # Dedup disclosure (T6): if retrieval collapsed duplicate narrations
        # of the same happening, say so -- branch frequencies are counts,
        # and the reader must know what was and wasn't counted.
        collapsed_narrations = sum(len(a.duplicate_ids) for a in filtered)
        if collapsed_narrations:
            uncertainties.append(
                f"{collapsed_narrations} duplicate narration(s) of the same "
                "underlying events were collapsed before counting analog frequencies"
            )

        thesis = Thesis(
            id=uuid4(),
            query=self._formulate_query(query_episode),
            query_date=utcnow(),
            analog_episode_ids=[e.analog.episode.id for e in evidence],
            analog_similarity_scores=[e.analog.combined_score for e in evidence],
            dominant_continuation=continuations[0] if continuations else None,
            alternative_continuations=[(c.description, c.probability) for c in continuations[1:]],
            confidence=confidence,
            watch_for_indicators=watch_conditions,
            key_uncertainties=uncertainties,
            mode=mode,
            model_version="thesis-v1.0",
            taxonomy_version=CURRENT_TAXONOMY_VERSION,
            scope_registry_version=scope_registry_version(),
            narrative_synthesis=None,
        )

        self.logger.info(
            "Thesis generated",
            thesis_id=str(thesis.id),
            confidence=thesis.confidence.value,
            has_narrative=False,
        )

        return thesis

    async def generate_async(
        self,
        query_episode: Episode,
        analogs: List[RetrievedAnalog],
    ) -> Thesis:
        """Generate the quantitative thesis and optionally synthesize it with an LLM."""
        thesis = self.generate(query_episode, analogs)
        if thesis.dominant_continuation is None:
            return thesis

        client = self.llm_client or self._create_llm_client_if_configured()
        if client is None:
            return thesis

        from narrative_engine.thesis.prompts import THESIS_NARRATIVE_PROMPT, format_analogs

        alternatives = "; ".join(description for description, _probability in thesis.alternative_continuations[:3])
        prompt = THESIS_NARRATIVE_PROMPT.format(
            dominant_continuation=thesis.dominant_continuation.description,
            probability=thesis.dominant_continuation.probability,
            alternatives=alternatives or "None identified",
            confidence=thesis.confidence.value,
            uncertainties="; ".join(thesis.key_uncertainties[:3]) or "None identified",
            analogs=format_analogs(analogs[:5]),
        )
        try:
            result = await client.complete_with_json(prompt)
            thesis.narrative_synthesis = result.get("narrative_summary") or result.get("key_pattern")
        except Exception as e:
            self.logger.warning("Failed to generate narrative synthesis", error=str(e))
        return thesis

    @staticmethod
    def _create_llm_client_if_configured():
        import os

        if not any(os.getenv(name) for name in ("NE_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            return None
        from narrative_engine.extraction.client import ExtractionPipeline

        return ExtractionPipeline().client

    def _determine_mode(self, query_episode: Episode) -> ThesisMode:
        """Arc-less iff the query situation carries no arc (failed tau_class
        or was never classified) -- Sec 6.5.8."""
        if query_episode.arc_type is None or query_episode.classification_state == ClassificationState.UNCLASSIFIED:
            return ThesisMode.ARC_LESS
        return ThesisMode.ARC_BASED

    def _filter_analogs(
        self,
        analogs: List[RetrievedAnalog],
    ) -> List[RetrievedAnalog]:
        """Filter analogs by quality and relevance."""
        filtered = [
            a
            for a in analogs
            if a.combined_score >= self.confidence_threshold
            and a.episode.resolution is not None  # Need resolved analogs
        ]

        # Sort by combined score
        filtered.sort(key=lambda x: x.combined_score, reverse=True)

        # Per-source cap (T6): branch frequencies are counts over analogs,
        # so one heavily-narrated source must not dominate them. Episodes
        # without provenance are uncapped (nothing to cap on).
        capped: List[RetrievedAnalog] = []
        per_source: Dict[str, int] = {}
        for analog in filtered:
            source = analog.episode.extracted_from[0] if analog.episode.extracted_from else None
            if source is not None:
                if per_source.get(source, 0) >= self.max_per_source:
                    continue
                per_source[source] = per_source.get(source, 0) + 1
            capped.append(analog)

        return capped[: self.max_analogs]

    def _extract_evidence(
        self,
        analogs: List[RetrievedAnalog],
        mode: ThesisMode = ThesisMode.ARC_BASED,
    ) -> List[AnalogEvidence]:
        """Extract evidence from filtered analogs.

        In arc-less mode the query has no arc, so arc_match_score has
        nothing to condition on -- weighting by it would zero out every
        analog. Weights fall back to bare semantic similarity (Sec 6.5.8:
        no arc-conditioned branch weights).
        """
        evidence = []

        for analog in analogs:
            # What happened next in this analog?
            continuation = self._extract_continuation(analog.episode)

            if continuation:
                weight = (
                    analog.semantic_similarity
                    if mode == ThesisMode.ARC_LESS
                    else analog.semantic_similarity * analog.arc_match_score
                )
                evidence.append(
                    AnalogEvidence(
                        analog=analog,
                        relevance_score=analog.combined_score,
                        continuation=continuation,
                        probability_weight=weight,
                    )
                )

        return evidence

    def _extract_continuation(self, episode: Episode) -> Optional[str]:
        """Extract what happened next from a resolved episode."""
        if episode.resolution and episode.consequences:
            return f"{episode.resolution} → {', '.join(episode.consequences[:2])}"
        elif episode.resolution:
            return episode.resolution
        return None

    def _generate_continuations(
        self,
        evidence: List[AnalogEvidence],
    ) -> List[Continuation]:
        """Generate weighted continuations from evidence.

        Cluster similar outcomes and weight by analog confidence.
        """
        if not evidence:
            return []

        # Simple clustering by continuation text similarity
        clusters: Dict[str, List[AnalogEvidence]] = {}

        for e in evidence:
            # Find matching cluster (simplified: exact match)
            found = False
            for key in clusters:
                if self._similar_continuation(key, e.continuation):
                    clusters[key].append(e)
                    found = True
                    break

            if not found:
                clusters[e.continuation] = [e]

        # Calculate probabilities for each cluster
        continuations = []
        total_weight = sum(sum(e.probability_weight for e in cluster) for cluster in clusters.values())

        for desc, cluster in clusters.items():
            weight = sum(e.probability_weight for e in cluster)
            prob = weight / total_weight if total_weight > 0 else 1.0 / len(clusters)

            continuations.append(
                Continuation(
                    description=desc,
                    probability=round(prob, 2),
                    supporting_analogs=len(cluster),
                )
            )

        # Sort by probability
        continuations.sort(key=lambda x: x.probability, reverse=True)

        return continuations

    def _similar_continuation(self, a: str, b: str) -> bool:
        """Check if two continuations describe similar outcomes."""
        # Simplified: check for shared keywords
        # Production: use embeddings or LLM similarity
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())

        overlap = len(a_words & b_words)
        return overlap >= 2  # At least 2 shared significant words

    def _calculate_confidence(
        self,
        evidence: List[AnalogEvidence],
        continuations: List[Continuation],
    ) -> ThesisConfidence:
        """Calculate overall confidence in the thesis."""

        # Factors:
        # 1. Number of supporting analogs
        # 2. Agreement among analogs (entropy of continuations)
        # 3. Quality of analogs (average relevance score)

        n_analogs = len(evidence)
        if n_analogs < self.min_analogs:
            return ThesisConfidence.LOW

        # Check for agreement
        if len(continuations) > 0:
            top_prob = continuations[0].probability
            if top_prob >= 0.7:
                agreement = "high"
            elif top_prob >= 0.5:
                agreement = "medium"
            else:
                agreement = "low"
        else:
            agreement = "low"

        # Average relevance
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)

        # Determine confidence level
        if n_analogs >= 5 and agreement == "high" and avg_relevance >= 0.8:
            return ThesisConfidence.HIGH
        elif n_analogs >= 3 and agreement in ["medium", "high"] and avg_relevance >= 0.6:
            return ThesisConfidence.MEDIUM
        else:
            return ThesisConfidence.LOW

    def _generate_watch_conditions(
        self,
        evidence: List[AnalogEvidence],
    ) -> List[str]:
        """Generate indicators to watch based on analog patterns.

        Prefers mechanism_tags (design doc Sec 3.8, controlled vocabulary)
        over free-text escalation_mechanics when an analog has been tagged --
        falls back to escalation_mechanics for episodes extracted before
        mechanism tagging existed.
        """
        conditions = []

        for e in evidence[:5]:  # Top 5 analogs
            episode = e.analog.episode
            if episode.mechanism_tags:
                conditions.extend(tag.value.replace("_", " ") for tag in episode.mechanism_tags[:2])
            elif episode.escalation_mechanics:
                conditions.extend(episode.escalation_mechanics[:2])

        # Deduplicate and prioritize
        unique_conditions = list(dict.fromkeys(conditions))  # Preserve order

        return unique_conditions[:8]  # Limit to 8 conditions

    def _identify_uncertainties(
        self,
        evidence: List[AnalogEvidence],
    ) -> List[str]:
        """Identify key uncertainties that could invalidate the thesis."""
        uncertainties = []

        # Check for contradictory analogs
        continuations = [e.continuation for e in evidence]
        unique_outcomes = len(set(continuations))

        if unique_outcomes > 3:
            uncertainties.append(f"High outcome variance ({unique_outcomes} distinct patterns)")

        # Check for contextual differences
        contexts = set()
        for e in evidence:
            if e.analog.episode.location:
                contexts.add(e.analog.episode.location)

        if len(contexts) > 3:
            uncertainties.append(f"Diverse geographic contexts: {', '.join(list(contexts)[:3])}")

        # Low confidence analogs
        low_conf = [e for e in evidence if e.relevance_score < 0.7]
        if low_conf:
            uncertainties.append(f"{len(low_conf)} analogs with lower relevance (< 0.7)")

        return uncertainties[:5]  # Limit to top 5

    def _formulate_query(self, episode: Episode) -> str:
        """Formulate natural language query from episode."""
        arc = episode.arc_type.value if episode.arc_type else "unknown"
        phase = episode.arc_phase.value if episode.arc_phase else "unknown"

        return f"What is the likely outcome of '{episode.title}' ({arc}, {phase} phase)?"

    def _create_uncertain_thesis(
        self,
        query_episode: Episode,
        analogs: List[RetrievedAnalog],
        mode: ThesisMode = ThesisMode.ARC_BASED,
    ) -> Thesis:
        """Create thesis when insufficient analogs available."""
        uncertainties = ["Insufficient historical analogs for reliable forecast"]
        if mode == ThesisMode.ARC_LESS:
            uncertainties.insert(
                0,
                "Situation fits no known arc (unclassified): forecast rests on "
                "structural similarity only, with no phase-completion statistics",
            )
        return Thesis(
            id=uuid4(),
            query=self._formulate_query(query_episode),
            query_date=utcnow(),
            analog_episode_ids=[a.episode.id for a in analogs],
            analog_similarity_scores=[a.combined_score for a in analogs],
            dominant_continuation=None,
            alternative_continuations=[],
            confidence=ThesisConfidence.UNKNOWN,
            watch_for_indicators=[],
            key_uncertainties=uncertainties,
            mode=mode,
            model_version="thesis-v1.0",
            taxonomy_version=CURRENT_TAXONOMY_VERSION,
            scope_registry_version=scope_registry_version(),
            narrative_synthesis=None,
        )
