"""Tests for the as-built alignment fixes (design doc v0.7 reconciliation).

Covers:
1. Scope partition: unscoped episodes never pool together in composition.
2. Structural render: outcome fields excluded; mechanisms included.
3. tau_class floor: low-confidence classifications become unclassified.
4. Evidence floor: sparse episode pairs cannot pass the identity gates.
5. Arc-less thesis mode: unclassified queries produce visibly degraded theses.
6. Data-layer masking + baselines for the masked-ending harness.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from narrative_engine.composition.identity import ArcIdentityResolver
from narrative_engine.composition.pipeline import compose_arc_instances_from_episodes
from narrative_engine.evaluation.baselines import (
    BareLLMBaseline,
    PersistenceBaseline,
)
from narrative_engine.evaluation.masking import mask_corpus_at, mask_episode_at
from narrative_engine.extraction.config import ExtractionPipelineConfig
from narrative_engine.extraction.pipeline import ExtractionOrchestrator
from narrative_engine.models import (
    ArcPhase,
    ArcType,
    ClassificationState,
    Episode,
    MechanismTag,
    ThesisConfidence,
    ThesisMode,
)
from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from narrative_engine.retrieval.analog_retrieval import RetrievedAnalog
from narrative_engine.thesis.generator import ThesisGenerator


def make_episode(**overrides) -> Episode:
    defaults = dict(
        title="Test Episode",
        summary="A test episode",
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
    )
    defaults.update(overrides)
    return Episode(**defaults)


class TestScopePartition:
    """Fix 1: unscoped episodes must not pool into one shared partition."""

    def _mergeable_pair(self, scope_a, scope_b):
        return [
            make_episode(
                title="Boom",
                scope_id=scope_a,
                start_date=datetime(1925, 1, 1),
                end_date=datetime(1928, 1, 1),
                arc_phase=ArcPhase.BOOM,
            ),
            make_episode(
                title="Panic",
                scope_id=scope_b,
                start_date=datetime(1929, 10, 1),
                end_date=datetime(1929, 11, 1),
                arc_phase=ArcPhase.EUPHORIA,
            ),
        ]

    def test_same_scope_pair_merges(self):
        episodes = self._mergeable_pair("us_national", "us_national")
        instances = compose_arc_instances_from_episodes(
            episodes, ArcType.CREDIT_BOOM_AND_BUST
        )
        assert len(instances) == 1

    def test_unscoped_pair_never_merges(self):
        episodes = self._mergeable_pair(None, None)
        instances = compose_arc_instances_from_episodes(
            episodes, ArcType.CREDIT_BOOM_AND_BUST
        )
        # Identical except for scope_id=None: must surface as two
        # singleton instances, not one merged instance.
        assert len(instances) == 2

    def test_unscoped_episodes_not_dropped(self):
        episodes = self._mergeable_pair(None, "us_national")
        instances = compose_arc_instances_from_episodes(
            episodes, ArcType.CREDIT_BOOM_AND_BUST
        )
        total_episode_ids = {
            eid
            for inst in instances
            for cov in inst.phases.values()
            for eid in cov.episode_ids
        }
        assert len(instances) == 2
        assert total_episode_ids == {e.id for e in episodes}


class TestStructuralRenderMasking:
    """Fix 2: outcome fields never enter the structural embedding text."""

    def test_render_excludes_resolution_and_consequences(self):
        episode = make_episode(
            resolution="Market bottomed in 1932, down 89% from peak",
            consequences=["Great Depression begins", "New Deal reforms"],
            tension="speculative excess vs fundamentals",
        )
        rendered = EmbeddingGenerator().render_structural_template(episode)
        assert "1932" not in rendered
        assert "Great Depression" not in rendered
        assert "New Deal" not in rendered
        # Non-outcome fields still render.
        assert "Tension: speculative excess vs fundamentals" in rendered

    def test_render_includes_mechanism_tags(self):
        episode = make_episode(
            mechanism_tags=[MechanismTag.CREDIT_EXPANSION, MechanismTag.ASSET_BUBBLE],
        )
        rendered = EmbeddingGenerator().render_structural_template(episode)
        assert "Mechanisms: credit_expansion, asset_bubble" in rendered

    def test_query_and_corpus_render_symmetrically(self):
        """A resolved corpus episode and an unresolved query episode with the
        same situation must render identically (query/corpus symmetry)."""
        common = dict(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.DISTRESS,
            initiating_conditions=["credit expansion"],
            tension="leverage vs liquidity",
        )
        resolved = make_episode(resolution="it crashed", consequences=["ruin"], **common)
        unresolved = make_episode(resolution=None, consequences=[], **common)
        generator = EmbeddingGenerator()
        assert generator.render_structural_template(
            resolved
        ) == generator.render_structural_template(unresolved)


class TestTauClassFloor:
    """Fix 3a: classification is not a forced choice."""

    def _orchestrator(self, classify_result):
        pipeline = AsyncMock()
        pipeline.classify.return_value = classify_result
        return ExtractionOrchestrator(
            pipeline=pipeline,
            config=ExtractionPipelineConfig(),  # floor = 0.5
        )

    @pytest.mark.asyncio
    async def test_low_confidence_becomes_unclassified(self):
        orchestrator = self._orchestrator(
            {
                "arc_type": "hubris_nemesis",
                "arc_phase": "setup",
                "phase_confidence": 0.3,
                "rationale": "weak fit",
                "secondary_arcs": [["tragedy", "setup", 0.2]],
            }
        )
        episode = make_episode(arc_type=None)
        await orchestrator._classify_episode(episode)

        assert episode.classification_state == ClassificationState.UNCLASSIFIED
        assert episode.arc_type is None
        assert episode.arc_phase is None
        assert episode.secondary_arcs == []
        # Confidence and rationale kept for audit.
        assert episode.phase_confidence == 0.3
        assert episode.arc_rationale == "weak fit"

    @pytest.mark.asyncio
    async def test_confident_classification_kept(self):
        orchestrator = self._orchestrator(
            {
                "arc_type": "credit_boom_and_bust",
                "arc_phase": "panic",
                "phase_confidence": 0.9,
                "rationale": "clear fit",
            }
        )
        episode = make_episode(arc_type=None)
        await orchestrator._classify_episode(episode)

        assert episode.classification_state == ClassificationState.CLASSIFIED
        assert episode.arc_type == ArcType.CREDIT_BOOM_AND_BUST
        assert episode.arc_phase == ArcPhase.PANIC

    @pytest.mark.asyncio
    async def test_no_usable_label_becomes_unclassified(self):
        orchestrator = self._orchestrator(
            {"arc_type": "not_a_real_arc", "phase_confidence": 0.9}
        )
        episode = make_episode(arc_type=None)
        await orchestrator._classify_episode(episode)

        assert episode.classification_state == ClassificationState.UNCLASSIFIED
        assert episode.arc_type is None


class TestEvidenceFloor:
    """Fix 4: absence of data must not behave like evidence for identity."""

    def test_sparse_pair_cannot_match(self):
        resolver = ArcIdentityResolver()
        # No dates, no actors, no embeddings, no phases: every gate would
        # pass by neutral default -- the evidence floor must reject it.
        a = make_episode(title="Sparse A")
        b = make_episode(title="Sparse B")
        score = resolver.calculate_identity_score(a, b)
        assert not score.is_match
        assert any("Insufficient identity evidence" in r for r in score.mismatch_reasons)

    def test_concrete_pair_can_match(self):
        resolver = ArcIdentityResolver()
        a = make_episode(
            start_date=datetime(1925, 1, 1),
            end_date=datetime(1928, 1, 1),
            arc_phase=ArcPhase.BOOM,
        )
        b = make_episode(
            start_date=datetime(1928, 6, 1),
            end_date=datetime(1929, 11, 1),
            arc_phase=ArcPhase.EUPHORIA,
        )
        score = resolver.calculate_identity_score(a, b, scale="episodic")
        assert score.is_match


class TestArcLessThesisMode:
    """Fix 3b: unclassified queries produce a visibly degraded thesis."""

    def _analogs(self, n=4):
        analogs = []
        for i in range(n):
            episode = make_episode(
                title=f"Analog {i}",
                resolution=f"outcome pattern {i % 2}",
                consequences=[f"consequence {i % 2}"],
            )
            analogs.append(
                RetrievedAnalog(
                    episode=episode,
                    semantic_similarity=0.9,
                    arc_match_score=0.0,  # nothing to match for arc-less query
                    phase_compatibility=0.5,
                    cycle_context_score=0.5,
                    mechanism_match_score=0.5,
                    combined_score=0.8,
                    retrieval_method="vector_arc_less",
                    reasoning="test",
                )
            )
        return analogs

    def test_unclassified_query_produces_arc_less_thesis(self):
        query = make_episode(
            arc_type=None,
            classification_state=ClassificationState.UNCLASSIFIED,
        )
        thesis = ThesisGenerator().generate(query, self._analogs())

        assert thesis.mode == ThesisMode.ARC_LESS
        assert thesis.confidence in (ThesisConfidence.LOW, ThesisConfidence.UNKNOWN)
        assert any("fits no known arc" in u for u in thesis.key_uncertainties)
        # Arc-less weights fall back to semantic similarity, so the
        # analogs still produce continuations (not all zeroed out).
        assert thesis.dominant_continuation is not None

    def test_classified_query_produces_arc_based_thesis(self):
        query = make_episode(arc_phase=ArcPhase.DISTRESS)
        analogs = self._analogs()
        for a in analogs:
            a.arc_match_score = 1.0
        thesis = ThesisGenerator().generate(query, analogs)
        assert thesis.mode == ThesisMode.ARC_BASED

    def test_uncertain_thesis_carries_mode(self):
        query = make_episode(arc_type=None)
        thesis = ThesisGenerator().generate(query, [])  # no analogs at all
        assert thesis.mode == ThesisMode.ARC_LESS


class TestDataLayerMasking:
    """Fix 5: masked-ending support masks at the data layer."""

    CUTOFF = datetime(1928, 1, 1)

    def test_post_cutoff_episode_dropped(self):
        episode = make_episode(start_date=datetime(1930, 1, 1))
        assert mask_episode_at(episode, self.CUTOFF) is None

    def test_resolved_before_cutoff_untouched(self):
        episode = make_episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            resolution="panic resolved by Morgan pool",
        )
        masked = mask_episode_at(episode, self.CUTOFF)
        assert masked is episode
        assert masked.resolution is not None

    def test_ongoing_at_cutoff_masked(self):
        episode = make_episode(
            start_date=datetime(1925, 1, 1),
            end_date=datetime(1932, 7, 1),
            resolution="market bottomed in 1932",
            consequences=["Great Depression"],
        )
        masked = mask_episode_at(episode, self.CUTOFF)
        assert masked is not None
        assert masked.resolution is None
        assert masked.consequences == []
        assert masked.end_date is None
        # Original untouched (copies, not mutation).
        assert episode.resolution is not None

    def test_mask_corpus(self):
        episodes = [
            make_episode(start_date=datetime(1907, 10, 1), end_date=datetime(1907, 11, 1)),
            make_episode(start_date=datetime(1925, 1, 1), end_date=datetime(1932, 7, 1),
                         resolution="crash"),
            make_episode(start_date=datetime(1930, 1, 1)),
        ]
        masked = mask_corpus_at(episodes, self.CUTOFF)
        assert len(masked) == 2
        assert all(
            e.resolution is None or (e.end_date and e.end_date <= self.CUTOFF)
            for e in masked
        )


class TestBaselines:
    """Fix 5: baselines exist so backtest scores have comparison points."""

    def test_persistence_baseline(self):
        query = make_episode(tension="leverage vs liquidity")
        prediction = PersistenceBaseline().predict(query)
        assert prediction.baseline_name == "persistence"
        assert "leverage vs liquidity" in prediction.predicted_continuation

    @pytest.mark.asyncio
    async def test_bare_llm_baseline_parses_json(self):
        client = MagicMock()
        client.complete = AsyncMock(
            return_value='{"continuation": "credit contracts", "probability": 0.7, "rationale": "history"}'
        )
        prediction = await BareLLMBaseline(llm_client=client).predict(make_episode())
        assert prediction.predicted_continuation == "credit contracts"
        assert prediction.probability == 0.7

    @pytest.mark.asyncio
    async def test_bare_llm_baseline_handles_garbage(self):
        client = MagicMock()
        client.complete = AsyncMock(return_value="not json at all")
        prediction = await BareLLMBaseline(llm_client=client).predict(make_episode())
        assert prediction.baseline_name == "bare_llm"
        assert prediction.predicted_continuation == "not json at all"
