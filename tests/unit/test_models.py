"""Unit tests for core data models."""

import pytest
from datetime import datetime
from uuid import UUID, uuid4

from narrative_engine.models import (
    Actor,
    ArcDefinition,
    ArcPhase,
    ArcType,
    Cycle,
    CycleScale,
    Episode,
    SourcePassage,
    Thesis,
    ExtractionRecord,
)


class TestActor:
    """Tests for Actor model."""

    def test_actor_creation(self) -> None:
        """Test basic actor creation."""
        actor = Actor(name="Napoleon", role="protagonist")
        assert actor.name == "Napoleon"
        assert actor.role == "protagonist"
        assert isinstance(actor.id, UUID)
        assert actor.attributes == {}

    def test_actor_with_attributes(self) -> None:
        """Test actor with custom attributes."""
        actor = Actor(
            name="Federal Reserve",
            role="institution",
            attributes={"established": 1913, "type": "central_bank"},
        )
        assert actor.attributes["established"] == 1913

    def test_actor_immutable(self) -> None:
        """Test that actors are immutable (frozen)."""
        actor = Actor(name="Test", role="test")
        with pytest.raises(Exception):
            actor.name = "Changed"


class TestArcPhase:
    """Tests for ArcPhase enum."""

    def test_standard_phases(self) -> None:
        """Test standard narrative phases exist."""
        assert ArcPhase.SETUP.value == "setup"
        assert ArcPhase.CLIMAX.value == "climax"
        assert ArcPhase.RESOLUTION.value == "resolution"

    def test_financial_phases(self) -> None:
        """Test financial-specific phases exist."""
        assert ArcPhase.BOOM.value == "boom"
        assert ArcPhase.EUPHORIA.value == "euphoria"
        assert ArcPhase.PANIC.value == "panic"


class TestArcType:
    """Tests for ArcType enum."""

    def test_financial_arcs(self) -> None:
        """Test financial crisis arc types."""
        assert ArcType.CREDIT_BOOM_AND_BUST.value == "credit_boom_and_bust"
        assert ArcType.RISE_AND_OVEREXTENSION.value == "rise_and_overextension"

    def test_classical_arcs(self) -> None:
        """Test classical narrative arc types."""
        assert ArcType.HERO_JOURNEY.value == "hero_journey"
        assert ArcType.TRAGEDY.value == "tragedy"


class TestEpisode:
    """Tests for Episode model."""

    def test_basic_episode_creation(self) -> None:
        """Test creating a minimal episode."""
        episode = Episode(
            title="1929 Stock Market Crash",
            summary="The collapse of the US stock market in October 1929",
        )
        assert episode.title == "1929 Stock Market Crash"
        assert isinstance(episode.id, UUID)
        assert episode.actors == []
        assert episode.is_resolved() is False

    def test_episode_with_dates(self) -> None:
        """Test episode with temporal bounds."""
        episode = Episode(
            title="Weimar Hyperinflation",
            summary="German hyperinflation 1921-1923",
            start_date=datetime(1921, 1, 1),
            end_date=datetime(1923, 12, 31),
        )
        assert episode.start_date.year == 1921
        assert episode.end_date.year == 1923

    def test_episode_with_arc_classification(self) -> None:
        """Test episode with arc classification."""
        episode = Episode(
            title="Dot-com Bubble",
            summary="Tech stock bubble 1995-2000",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.EUPHORIA,
            phase_confidence=0.85,
        )
        assert episode.arc_type == ArcType.CREDIT_BOOM_AND_BUST
        assert episode.arc_phase == ArcPhase.EUPHORIA
        assert episode.phase_confidence == pytest.approx(0.85)

    def test_episode_with_actors(self) -> None:
        """Test episode with actors."""
        fed = Actor(name="Federal Reserve", role="institution")
        investor = Actor(name="Retail Investors", role="crowd")

        episode = Episode(title="Test Episode", summary="Test", actors=[fed, investor])
        assert len(episode.actors) == 2
        assert episode.actors[0].name == "Federal Reserve"

    def test_episode_resolved(self) -> None:
        """Test episode resolution detection."""
        # Unresolved episode
        episode = Episode(title="Ongoing Crisis", summary="Not over yet")
        assert episode.is_resolved() is False

        # Resolved episode
        episode.resolution = "Market stabilized after intervention"
        episode.end_date = datetime(2008, 12, 31)
        assert episode.is_resolved() is True

    def test_episode_source_provenance(self) -> None:
        """Test episode with source passages."""
        passage = SourcePassage(
            work_id="kindleberger-1978",
            passage_id="ch3-p45",
            text="Manias typically follow a pattern...",
            page=45,
            historiographic_school="Keynesian",
        )

        episode = Episode(title="Test", summary="Test", source_passages=[passage])
        assert len(episode.source_passages) == 1
        assert episode.source_passages[0].historiographic_school == "Keynesian"


class TestArcDefinition:
    """Tests for ArcDefinition model."""

    def test_arc_definition_creation(self) -> None:
        """Test creating an arc definition."""
        arc_def = ArcDefinition(
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            name="Credit Boom and Bust",
            description="The Minsky-Kindleberger financial cycle",
            phases=[
                ArcPhase.BOOM,
                ArcPhase.EUPHORIA,
                ArcPhase.DISTRESS,
                ArcPhase.PANIC,
                ArcPhase.REVULSION,
            ],
            typical_duration="3-7 years",
        )
        assert len(arc_def.phases) == 5
        assert arc_def.phases[0] == ArcPhase.BOOM
        assert arc_def.phases[-1] == ArcPhase.REVULSION


class TestCycle:
    """Tests for Cycle model."""

    def test_cycle_creation(self) -> None:
        """Test basic cycle creation."""
        cycle = Cycle(
            name="Fourth Turning",
            scale=CycleScale.GENERATIONAL,
            description="Strauss-Howe generational crisis",
            framework_source="Strauss-Howe-1997",
        )
        assert cycle.scale == CycleScale.GENERATIONAL
        assert cycle.framework_source == "Strauss-Howe-1997"

    def test_cycle_hierarchy(self) -> None:
        """Test cycle parent-child relationships."""
        civilizational = Cycle(name="Western Decline", scale=CycleScale.CIVILIZATIONAL)
        institutional = Cycle(
            name="Post-WWII Order",
            scale=CycleScale.INSTITUTIONAL,
            parent_cycle_id=civilizational.id,
        )

        assert institutional.parent_cycle_id == civilizational.id
        assert civilizational.id in [institutional.parent_cycle_id]


class TestThesis:
    """Tests for Thesis model."""

    def test_thesis_creation(self) -> None:
        """Test creating a thesis."""
        thesis = Thesis(
            query="Current market conditions in 2024",
            query_date=datetime(2024, 1, 1),
            dominant_continuation="Continued expansion with late-cycle risks",
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )
        assert thesis.query_date.year == 2024
        assert thesis.resolved is False
        assert thesis.brier_score is None

    def test_thesis_with_analogs(self) -> None:
        """Test thesis with retrieved analogs."""
        episode_ids = [uuid4(), uuid4(), uuid4()]
        thesis = Thesis(
            query="Test query",
            query_date=datetime.now(),
            dominant_continuation="Expansion",
            analog_episode_ids=episode_ids,
            analog_similarity_scores=[0.92, 0.87, 0.84],
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )
        assert len(thesis.analog_episode_ids) == 3
        assert len(thesis.analog_similarity_scores) == 3


class TestExtractionRecord:
    """Tests for ExtractionRecord model."""

    def test_extraction_record_creation(self) -> None:
        """Test extraction record for audit trail."""
        record = ExtractionRecord(
            source_chunk_id="chunk-123",
            pipeline_stage="classification",
            prompt_version="v1.2.0",
            model_used="gpt-4",
            input={"text": "Historical event description..."},
            output={"arc_type": "credit_boom_and_bust", "confidence": 0.89},
            confidence=0.89,
            processing_time_ms=1250,
        )
        assert record.pipeline_stage == "classification"
        assert record.confidence == pytest.approx(0.89)


class TestIntegration:
    """Integration tests for model relationships."""

    def test_episode_to_cycle_membership(self) -> None:
        """Test linking episodes to cycles."""
        episode = Episode(title="Test", summary="Test")
        cycle = Cycle(name="Test Cycle", scale=CycleScale.EPISODIC)

        # Add episode to cycle
        episode.parent_cycle_ids.add(cycle.id)
        cycle.episode_ids.add(episode.id)

        assert cycle.contains_episode(episode.id)
        assert cycle.id in episode.parent_cycle_ids

    def test_thesis_with_episodes(self) -> None:
        """Test thesis referencing episodes."""
        episode = Episode(
            title="1929 Crash",
            summary="Stock market collapse",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
        )

        thesis = Thesis(
            query="Will 2024 see a crash?",
            query_date=datetime(2024, 1, 1),
            dominant_continuation="Soft landing",
            analog_episode_ids=[episode.id],
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )

        assert episode.id in thesis.analog_episode_ids
