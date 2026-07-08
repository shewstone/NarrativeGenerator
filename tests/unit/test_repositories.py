"""Unit tests for repository layer."""

import pytest
from datetime import datetime
from uuid import uuid4

from narrative_engine.models import Episode, Cycle, Actor, ArcType, ArcPhase, CycleScale, Thesis
from narrative_engine.storage.repositories import (
    EpisodeRepository,
    CycleRepository,
    ThesisRepository,
    RepositoryFactory,
)


class TestEpisodeRepository:
    """Tests for EpisodeRepository."""
    
    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            title="1929 Stock Market Crash",
            summary="The collapse of the US stock market in October 1929",
            start_date=datetime(1929, 10, 1),
            end_date=datetime(1929, 10, 31),
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            phase_confidence=0.95,
        )
    
    @pytest.mark.asyncio
    async def test_create_episode(self, db_session, sample_episode):
        """Test creating an episode."""
        repo = EpisodeRepository(db_session)
        
        created = await repo.create(sample_episode)
        
        assert created.id is not None
        assert created.title == sample_episode.title
        assert created.arc_type == ArcType.CREDIT_BOOM_AND_BUST
    
    @pytest.mark.asyncio
    async def test_get_by_id(self, db_session, sample_episode):
        """Test retrieving episode by ID."""
        repo = EpisodeRepository(db_session)
        
        created = await repo.create(sample_episode)
        retrieved = await repo.get_by_id(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session):
        """Test retrieving non-existent episode."""
        repo = EpisodeRepository(db_session)
        
        retrieved = await repo.get_by_id(uuid4())
        
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_update_episode(self, db_session, sample_episode):
        """Test updating an episode."""
        repo = EpisodeRepository(db_session)
        
        created = await repo.create(sample_episode)
        created.title = "Updated Title"
        
        updated = await repo.update(created)
        
        assert updated.title == "Updated Title"
        assert updated.version > created.version
    
    @pytest.mark.asyncio
    async def test_delete_episode(self, db_session, sample_episode):
        """Test deleting an episode."""
        repo = EpisodeRepository(db_session)
        
        created = await repo.create(sample_episode)
        deleted = await repo.delete(created.id)
        
        assert deleted is True
        assert await repo.get_by_id(created.id) is None
    
    @pytest.mark.asyncio
    async def test_get_by_arc_type(self, db_session):
        """Test filtering by arc type."""
        repo = EpisodeRepository(db_session)
        
        # Create episodes with different arc types
        for i in range(3):
            await repo.create(Episode(
                title=f"Credit Boom {i}",
                summary="Test",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            ))
        
        await repo.create(Episode(
            title="Hero Journey",
            summary="Test",
            arc_type=ArcType.HERO_JOURNEY,
        ))
        
        results = await repo.get_by_arc_type(ArcType.CREDIT_BOOM_AND_BUST.value, limit=10)
        
        assert len(results) == 3
        assert all(e.arc_type == ArcType.CREDIT_BOOM_AND_BUST for e in results)
    
    @pytest.mark.asyncio
    async def test_count(self, db_session):
        """Test counting episodes."""
        repo = EpisodeRepository(db_session)
        
        initial_count = await repo.count()
        
        await repo.create(Episode(title="Test 1", summary="Test"))
        await repo.create(Episode(title="Test 2", summary="Test"))
        
        final_count = await repo.count()
        
        assert final_count == initial_count + 2


class TestCycleRepository:
    """Tests for CycleRepository."""
    
    @pytest.fixture
    def sample_cycle(self):
        """Create a sample cycle for testing."""
        return Cycle(
            name="Fourth Turning",
            scale=CycleScale.GENERATIONAL,
            description="Strauss-Howe generational crisis",
            framework_source="Strauss-Howe-1997",
        )
    
    @pytest.mark.asyncio
    async def test_create_cycle(self, db_session, sample_cycle):
        """Test creating a cycle."""
        repo = CycleRepository(db_session)
        
        created = await repo.create(sample_cycle)
        
        assert created.id is not None
        assert created.name == sample_cycle.name
        assert created.scale == CycleScale.GENERATIONAL
    
    @pytest.mark.asyncio
    async def test_get_by_scale(self, db_session):
        """Test filtering by scale."""
        repo = CycleRepository(db_session)
        
        await repo.create(Cycle(name="Gen 1", scale=CycleScale.GENERATIONAL))
        await repo.create(Cycle(name="Gen 2", scale=CycleScale.GENERATIONAL))
        await repo.create(Cycle(name="Inst 1", scale=CycleScale.INSTITUTIONAL))
        
        results = await repo.get_by_scale(CycleScale.GENERATIONAL.value, limit=10)
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_get_children(self, db_session):
        """Test getting child cycles."""
        repo = CycleRepository(db_session)
        
        parent = await repo.create(Cycle(name="Parent", scale=CycleScale.CIVILIZATIONAL))
        child = await repo.create(Cycle(
            name="Child",
            scale=CycleScale.INSTITUTIONAL,
            parent_cycle_id=parent.id,
        ))
        
        children = await repo.get_children(parent.id)
        
        assert len(children) == 1
        assert children[0].id == child.id


class TestThesisRepository:
    """Tests for ThesisRepository."""
    
    @pytest.fixture
    def sample_thesis(self):
        """Create a sample thesis for testing."""
        return Thesis(
            query="Will 2024 see a market crash?",
            query_date=datetime(2024, 1, 1),
            dominant_continuation="Soft landing likely",
            analog_episode_ids=[uuid4(), uuid4()],
            analog_similarity_scores=[0.92, 0.87],
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )
    
    @pytest.mark.asyncio
    async def test_create_thesis(self, db_session, sample_thesis):
        """Test creating a thesis."""
        repo = ThesisRepository(db_session)
        
        created = await repo.create(sample_thesis)
        
        assert created.id is not None
        assert created.query == sample_thesis.query
        assert created.resolved is False
    
    @pytest.mark.asyncio
    async def test_resolve_thesis(self, db_session, sample_thesis):
        """Test resolving a thesis."""
        repo = ThesisRepository(db_session)
        
        created = await repo.create(sample_thesis)
        resolved = await repo.resolve(
            created.id,
            outcome="accurate",
            brier_score=0.15,
        )
        
        assert resolved is not None
        assert resolved.resolved is True
        assert resolved.resolution_outcome == "accurate"
        assert resolved.brier_score == pytest.approx(0.15)
    
    @pytest.mark.asyncio
    async def test_get_unresolved(self, db_session):
        """Test getting unresolved theses."""
        repo = ThesisRepository(db_session)
        
        # Create resolved thesis
        resolved = Thesis(
            query="Resolved",
            query_date=datetime.now(),
            dominant_continuation="Test",
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )
        created_resolved = await repo.create(resolved)
        await repo.resolve(created_resolved.id, "accurate")
        
        # Create unresolved thesis
        unresolved = Thesis(
            query="Unresolved",
            query_date=datetime.now(),
            dominant_continuation="Test",
            model_version="gpt-4",
            taxonomy_version="v0.1.0",
        )
        await repo.create(unresolved)
        
        unresolved_list = await repo.get_unresolved()
        
        assert len(unresolved_list) == 1
        assert unresolved_list[0].query == "Unresolved"
    
    @pytest.mark.asyncio
    async def test_calibration_stats(self, db_session):
        """Test calibration statistics."""
        repo = ThesisRepository(db_session)
        
        # Create and resolve theses with Brier scores
        for i, brier in enumerate([0.1, 0.2, 0.3]):
            thesis = Thesis(
                query=f"Query {i}",
                query_date=datetime.now(),
                dominant_continuation="Test",
                model_version="gpt-4",
                taxonomy_version="v0.1.0",
            )
            created = await repo.create(thesis)
            await repo.resolve(created.id, "accurate", brier)
        
        stats = await repo.get_calibration_stats()
        
        assert stats["total_resolved"] == 3
        assert stats["with_brier_scores"] == 3
        assert stats["average_brier_score"] == pytest.approx(0.2)


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""
    
    @pytest.mark.asyncio
    async def test_factory_creates_repositories(self, db_session):
        """Test factory creates repository instances."""
        factory = RepositoryFactory(db_session)
        
        assert factory.episodes is not None
        assert factory.cycles is not None
        assert factory.theses is not None
        
        # Should return same instance on repeated access
        assert factory.episodes is factory.episodes


class TestRepositoryIntegration:
    """Integration tests for repository interactions."""
    
    @pytest.mark.asyncio
    async def test_episode_cycle_relationship(self, db_session):
        """Test adding episode to cycle."""
        episode_repo = EpisodeRepository(db_session)
        cycle_repo = CycleRepository(db_session)
        
        # Create entities
        episode = await episode_repo.create(Episode(
            title="Test Episode",
            summary="Test",
        ))
        cycle = await cycle_repo.create(Cycle(
            name="Test Cycle",
            scale=CycleScale.GENERATIONAL,
        ))
        
        # Add relationship
        await cycle_repo.add_episode(cycle.id, episode.id)
        await db_session.commit()
        
        # Verify
        cycle_with_episodes = await cycle_repo.get_by_id(cycle.id)
        assert episode.id in cycle_with_episodes.episode_ids


# Fixtures are provided by tests/conftest.py
