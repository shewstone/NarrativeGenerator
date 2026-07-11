"""Unit tests for thesis generation."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from narrative_engine.models import (
    ArcPhase,
    ArcType,
    Continuation,
    Episode,
    MechanismTag,
    ThesisConfidence,
)
from narrative_engine.retrieval.analog_retrieval import RetrievedAnalog
from narrative_engine.thesis.generator import AnalogEvidence, ThesisGenerator


class TestThesisGenerator:
    """Tests for thesis generation."""

    def test_create_generator(self):
        """Test creating thesis generator."""
        generator = ThesisGenerator(min_analogs=2, max_analogs=5)
        assert generator.min_analogs == 2
        assert generator.max_analogs == 5

    def test_filter_analogs(self):
        """Test analog filtering."""
        generator = ThesisGenerator(min_analogs=2, confidence_threshold=0.7)

        # Create mock analogs
        episode1 = Episode(
            id=uuid4(),
            title="1929 Crash",
            summary="Stock market crash",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            resolution="Market bottomed",
        )

        analog1 = RetrievedAnalog(
            episode=episode1,
            semantic_similarity=0.9,
            arc_match_score=1.0,
            phase_compatibility=1.0,
            cycle_context_score=0.8,
            mechanism_match_score=0.5,
            combined_score=0.95,
            retrieval_method="hybrid",
            reasoning="High similarity",
        )

        filtered = generator._filter_analogs([analog1])
        # Should include analog with score >= 0.7
        assert len(filtered) == 1
        assert filtered[0].combined_score == 0.95

    def test_extract_continuation(self):
        """Test extracting continuation from episode."""
        generator = ThesisGenerator()

        episode = Episode(
            id=uuid4(),
            title="Test",
            summary="Test summary",
            resolution="Crisis resolved",
            consequences=["Recovery began", "Reforms enacted"],
        )

        continuation = generator._extract_continuation(episode)
        assert "Crisis resolved" in continuation
        assert "Recovery began" in continuation

    def test_similar_continuation(self):
        """Test continuation similarity detection."""
        generator = ThesisGenerator()

        assert (
            generator._similar_continuation("Market crashed and recovered", "Stock market crashed then recovered")
            is True
        )

        assert generator._similar_continuation("Market crashed", "Peaceful transition") is False

    def test_generate_continuations(self):
        """Test generating continuations from evidence."""
        generator = ThesisGenerator()

        episode = Episode(
            id=uuid4(),
            title="1929 Crash",
            summary="Crash",
            resolution="Recovery within 2 years",
        )

        analog = RetrievedAnalog(
            episode=episode,
            semantic_similarity=0.9,
            arc_match_score=1.0,
            phase_compatibility=1.0,
            cycle_context_score=0.8,
            mechanism_match_score=0.5,
            combined_score=0.9,
            retrieval_method="hybrid",
            reasoning="Similar",
        )

        evidence = [
            AnalogEvidence(
                analog=analog,
                relevance_score=0.9,
                continuation="Recovery within 2 years",
                probability_weight=0.9,
            )
        ]

        continuations = generator._generate_continuations(evidence)
        assert len(continuations) == 1
        assert continuations[0].description == "Recovery within 2 years"

    def test_calculate_confidence_high(self):
        """Test confidence calculation for strong evidence."""
        generator = ThesisGenerator(min_analogs=3)

        episode = Episode(
            id=uuid4(),
            title="Test",
            summary="Test",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
        )

        analogs = [
            RetrievedAnalog(
                episode=episode,
                semantic_similarity=0.9,
                arc_match_score=0.9,
                phase_compatibility=0.9,
                cycle_context_score=0.9,
                mechanism_match_score=0.5,
                combined_score=0.9,
                retrieval_method="hybrid",
                reasoning="Strong",
            )
            for _ in range(5)
        ]

        evidence = [
            AnalogEvidence(
                analog=a,
                relevance_score=0.85,
                continuation="Recovery",
                probability_weight=0.8,
            )
            for a in analogs
        ]

        continuations = [Continuation(description="Recovery", probability=0.8, supporting_analogs=5)]

        confidence = generator._calculate_confidence(evidence, continuations)
        assert confidence in [ThesisConfidence.HIGH, ThesisConfidence.MEDIUM]

    def test_generate_populates_thesis_from_mechanism_tags(self):
        """generate() populates analog_episode_ids/watch_for_indicators, and
        prefers mechanism_tags over escalation_mechanics for watch conditions
        when an analog has been tagged (Sec 3.8)."""
        query_episode = Episode(
            id=uuid4(),
            title="Query episode",
            summary="Present-day situation",
        )

        analog_episodes = [
            Episode(
                id=uuid4(),
                title=f"Analog {i}",
                summary="Historical analog",
                resolution="Market bottomed",
                consequences=["Recovery began"],
                escalation_mechanics=["Panic selling"],
                mechanism_tags=[MechanismTag.CREDIT_EXPANSION, MechanismTag.ASSET_BUBBLE],
            )
            for i in range(3)
        ]

        analogs = [
            RetrievedAnalog(
                episode=ep,
                semantic_similarity=0.9,
                arc_match_score=0.9,
                phase_compatibility=0.9,
                cycle_context_score=0.9,
                mechanism_match_score=1.0,
                combined_score=0.9,
                retrieval_method="hybrid",
                reasoning="Strong match",
            )
            for ep in analog_episodes
        ]

        generator = ThesisGenerator(min_analogs=2)
        thesis = generator.generate(query_episode, analogs)

        assert sorted(thesis.analog_episode_ids) == sorted(ep.id for ep in analog_episodes)
        assert len(thesis.analog_similarity_scores) == len(analog_episodes)
        assert "credit expansion" in thesis.watch_for_indicators
        assert "Panic selling" not in thesis.watch_for_indicators

    @pytest.mark.asyncio
    async def test_generate_async_uses_configured_llm_for_narrative(self):
        query = Episode(title="Query", summary="Present situation")
        episodes = [
            Episode(
                title=f"Analog {index}",
                summary="Historical situation",
                resolution="Institutions reformed",
                consequences=["Stability returned"],
            )
            for index in range(3)
        ]
        analogs = [
            RetrievedAnalog(
                episode=episode,
                semantic_similarity=0.9,
                arc_match_score=0.9,
                phase_compatibility=0.9,
                cycle_context_score=0.9,
                mechanism_match_score=0.5,
                combined_score=0.9,
                retrieval_method="hybrid",
                reasoning="Strong match",
            )
            for episode in episodes
        ]
        client = AsyncMock()
        client.complete_with_json.return_value = {"narrative_summary": "Historical analogs favor institutional reform."}

        thesis = await ThesisGenerator(min_analogs=2, llm_client=client).generate_async(query, analogs)

        assert thesis.narrative_synthesis == "Historical analogs favor institutional reform."
        client.complete_with_json.assert_awaited_once()


class TestAnalogEvidence:
    """Tests for analog evidence data class."""

    def test_create_evidence(self):
        """Test creating analog evidence."""
        episode = Episode(
            id=uuid4(),
            title="Test",
            summary="Test",
        )

        analog = RetrievedAnalog(
            episode=episode,
            semantic_similarity=0.8,
            arc_match_score=0.7,
            phase_compatibility=0.9,
            cycle_context_score=0.6,
            mechanism_match_score=0.5,
            combined_score=0.75,
            retrieval_method="vector",
            reasoning="Similar arc",
        )

        evidence = AnalogEvidence(
            analog=analog,
            relevance_score=0.8,
            continuation="Market recovered",
            probability_weight=0.56,
        )

        assert evidence.relevance_score == 0.8
        assert evidence.continuation == "Market recovered"
