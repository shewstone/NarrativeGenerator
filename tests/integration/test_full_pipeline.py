"""Integration test for full narrative engine pipeline."""

from datetime import datetime
from uuid import uuid4

import pytest

from narrative_engine.models import (
    ArcPhase,
    ArcType,
    Cycle,
    CycleScale,
    Episode,
)
from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from narrative_engine.storage.repositories import RepositoryFactory
from narrative_engine.thesis.generator import ThesisGenerator


@pytest.mark.asyncio
async def test_full_pipeline(engine, session):
    """Test complete pipeline: create episodes → embed → retrieve → generate thesis."""

    # 1. Create historical episodes (simulated database)
    factory = RepositoryFactory(session)

    # Episode 1: 1929 Crash
    episode_1929 = Episode(
        id=uuid4(),
        title="1929 Stock Market Crash",
        summary="The Great Crash, Black Tuesday, market collapsed",
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.PANIC,
        start_date=datetime(1929, 10, 24),
        end_date=datetime(1929, 10, 29),
        location="United States",
        initiating_conditions=["Speculation", "Margin buying"],
        escalation_mechanics=["Panic selling", "Bank runs"],
        tension="Market collapse",
        resolution="Market bottomed, Great Depression followed",
        consequences=["Great Depression", "Bank failures", "25% unemployment"],
        extracted_from=["source-1929"],
    )

    # Episode 2: 2008 Financial Crisis
    episode_2008 = Episode(
        id=uuid4(),
        title="2008 Global Financial Crisis",
        summary="Subprime mortgage crisis, Lehman Brothers collapse",
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.PANIC,
        start_date=datetime(2008, 9, 15),
        end_date=datetime(2009, 3, 1),
        location="Global",
        initiating_conditions=["Subprime mortgages", "CDO/CDS complexity"],
        escalation_mechanics=["Bank failures", "Credit freeze"],
        tension="Global financial meltdown",
        resolution="TARP, QE, gradual recovery",
        consequences=["Great Recession", "Regulatory reform", "Slow recovery"],
        extracted_from=["source-2008"],
    )

    # Store episodes
    created_1929 = await factory.episodes.create(episode_1929)
    created_2008 = await factory.episodes.create(episode_2008)

    # 2. Generate embeddings
    embedder = EmbeddingGenerator()

    embedding_1929 = embedder.generate_structural_embedding(created_1929)
    embedding_2008 = embedder.generate_structural_embedding(created_2008)

    assert len(embedding_1929) == 384
    assert len(embedding_2008) == 384

    # Update episodes with embeddings
    await factory.episodes.update_embedding(created_1929.id, embedding_1929)
    await factory.episodes.update_embedding(created_2008.id, embedding_2008)

    # 3. Create query episode (current situation)
    query_episode = Episode(
        id=uuid4(),
        title="2024 Market Conditions",
        summary="High valuations, AI bubble concerns, rising rates",
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.DISTRESS,
        start_date=datetime(2024, 1, 1),
        location="Global",
        initiating_conditions=["High valuations", "Rate hikes"],
        escalation_mechanics=["Liquidity tightening"],
        tension="Market uncertainty",
    )

    query_embedding = embedder.generate_structural_embedding(query_episode)

    # 4. Retrieve analogs
    await factory.episodes.create(query_episode)
    await factory.episodes.update_embedding(query_episode.id, query_embedding)

    analogs = await factory.episodes.search_by_embedding(query_embedding, limit=5)

    assert len(analogs) >= 2

    # 5. Generate thesis
    generator = ThesisGenerator(min_analogs=2)

    # Convert to RetrievedAnalog objects
    from narrative_engine.retrieval.analog_retrieval import RetrievedAnalog

    retrieved_analogs = [
        RetrievedAnalog(
            episode=episode,
            semantic_similarity=0.85,
            arc_match_score=1.0,
            phase_compatibility=0.8,
            cycle_context_score=0.9,
            combined_score=0.88,
            retrieval_method="hybrid",
            reasoning="Same arc type",
        )
        for episode, _ in analogs
    ]

    thesis = generator.generate(query_episode, retrieved_analogs)

    # Assertions
    assert thesis is not None
    assert thesis.dominant_continuation is not None
    assert thesis.confidence is not None
    assert len(thesis.watch_conditions) > 0

    # Verify pattern matching
    assert thesis.dominant_continuation.probability > 0
    assert thesis.dominant_continuation.probability <= 1.0


@pytest.mark.asyncio
async def test_episode_analog_search_by_arc_type(engine, session):
    """Test retrieving episodes by arc type similarity."""

    factory = RepositoryFactory(session)

    # Create episodes with different arc types
    episodes = [
        Episode(
            id=uuid4(),
            title=f"Episode {i}",
            summary="Test",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST if i % 2 == 0 else ArcType.HERO_JOURNEY,
            arc_phase=ArcPhase.PANIC,
        )
        for i in range(5)
    ]

    for ep in episodes:
        await factory.episodes.create(ep)

    # Search by arc type
    credit_episodes = await factory.episodes.get_by_arc_type(ArcType.CREDIT_BOOM_AND_BUST.value)

    # Should find credit boom episodes
    assert len(credit_episodes) >= 2
    for ep in credit_episodes:
        assert ep.arc_type == ArcType.CREDIT_BOOM_AND_BUST


@pytest.mark.asyncio
async def test_cycle_episode_relationship(engine, session):
    """Test creating cycles and associating episodes."""

    factory = RepositoryFactory(session)

    # Create a cycle
    cycle = Cycle(
        id=uuid4(),
        name="Fourth Turning: 1946-1964",
        scale=CycleScale.GENERATIONAL,
        start_date=datetime(1946, 1, 1),
        end_date=datetime(1964, 12, 31),
        description="Post-WWII High: American High",
    )

    created_cycle = await factory.cycles.create(cycle)

    # Create episodes within cycle
    episodes = [
        Episode(
            id=uuid4(),
            title="Post-war Boom",
            summary="Economic prosperity",
            arc_type=ArcType.RISE_AND_OVEREXTENSION,
            arc_phase=ArcPhase.BOOM,
            start_date=datetime(1950, 1, 1),
        ),
        Episode(
            id=uuid4(),
            title="Sputnik Moment",
            summary="Space race begins",
            arc_type=ArcType.HERO_JOURNEY,
            arc_phase=ArcPhase.RISING_ACTION,
            start_date=datetime(1957, 10, 4),
        ),
    ]

    for ep in episodes:
        created_ep = await factory.episodes.create(ep)
        await factory.cycles.add_episode(created_cycle.id, created_ep.id)

    # Retrieve cycle with episodes
    cycle_with_episodes = await factory.cycles.get_by_id(created_cycle.id)

    assert cycle_with_episodes is not None
    assert len(cycle_with_episodes.episode_ids) == 2
