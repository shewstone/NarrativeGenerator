"""Integration tests for database operations."""

import pytest
from datetime import datetime

from narrative_engine.models import Episode, Cycle, ArcType, ArcPhase, CycleScale
from narrative_engine.storage.database import DatabaseManager
from narrative_engine.storage.config import DatabaseConfig
from narrative_engine.storage.repositories import RepositoryFactory


class TestDatabaseOperations:
    """Integration tests for database CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_and_retrieve_episode(self, db_session):
        """Test creating and retrieving an episode from database."""
        factory = RepositoryFactory(db_session)
        
        # Create episode
        episode = Episode(
            title="1929 Crash",
            summary="Stock market crash",
            start_date=datetime(1929, 10, 24),
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            phase_confidence=0.95,
        )
        
        created = await factory.episodes.create(episode)
        
        # Retrieve
        retrieved = await factory.episodes.get_by_id(created.id)
        
        assert retrieved is not None
        assert retrieved.title == "1929 Crash"
        assert retrieved.arc_type == ArcType.CREDIT_BOOM_AND_BUST
    
    @pytest.mark.asyncio
    async def test_update_episode(self, db_session):
        """Test updating an episode."""
        factory = RepositoryFactory(db_session)
        
        episode = await factory.episodes.create(Episode(
            title="Original Title",
            summary="Test summary",
        ))
        
        episode.title = "Updated Title"
        updated = await factory.episodes.update(episode)
        
        assert updated.title == "Updated Title"
        assert updated.version == 2
    
    @pytest.mark.asyncio
    async def test_delete_episode(self, db_session):
        """Test deleting an episode."""
        factory = RepositoryFactory(db_session)
        
        episode = await factory.episodes.create(Episode(
            title="To Delete",
            summary="Will be deleted",
        ))
        
        deleted = await factory.episodes.delete(episode.id)
        assert deleted is True
        
        # Verify deletion
        retrieved = await factory.episodes.get_by_id(episode.id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_filter_by_arc_type(self, db_session):
        """Test filtering episodes by arc type."""
        factory = RepositoryFactory(db_session)
        
        # Create episodes with different arc types
        await factory.episodes.create(Episode(
            title="Credit Boom 1",
            summary="Test",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        ))
        await factory.episodes.create(Episode(
            title="Credit Boom 2",
            summary="Test",
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        ))
        await factory.episodes.create(Episode(
            title="Hero Journey",
            summary="Test",
            arc_type=ArcType.HERO_JOURNEY,
        ))
        
        # Filter
        results = await factory.episodes.get_by_arc_type(
            ArcType.CREDIT_BOOM_AND_BUST.value
        )
        
        assert len(results) == 2
        assert all(e.arc_type == ArcType.CREDIT_BOOM_AND_BUST for e in results)
    
    @pytest.mark.asyncio
    async def test_cycle_crud(self, db_session):
        """Test cycle CRUD operations."""
        factory = RepositoryFactory(db_session)
        
        cycle = await factory.cycles.create(Cycle(
            name="Test Cycle",
            scale=CycleScale.GENERATIONAL,
            description="A test cycle",
        ))
        
        retrieved = await factory.cycles.get_by_id(cycle.id)
        
        assert retrieved is not None
        assert retrieved.name == "Test Cycle"
        assert retrieved.scale == CycleScale.GENERATIONAL
    
    @pytest.mark.asyncio
    async def test_cycle_hierarchy(self, db_session):
        """Test cycle parent-child relationships."""
        factory = RepositoryFactory(db_session)
        
        # Create parent
        parent = await factory.cycles.create(Cycle(
            name="Parent Cycle",
            scale=CycleScale.CIVILIZATIONAL,
        ))
        
        # Create child
        child = await factory.cycles.create(Cycle(
            name="Child Cycle",
            scale=CycleScale.INSTITUTIONAL,
            parent_cycle_id=parent.id,
        ))
        
        # Get children
        children = await factory.cycles.get_children(parent.id)
        
        assert len(children) == 1
        assert children[0].id == child.id
    
    @pytest.mark.asyncio
    async def test_thesis_crud(self, db_session):
        """Test thesis CRUD operations."""
        factory = RepositoryFactory(db_session)
        
        thesis = await factory.theses.create(
            Thesis(
                query="Will markets crash?",
                query_date=datetime(2024, 1, 1),
                dominant_continuation="Soft landing",
                model_version="gpt-4",
                taxonomy_version="v0.1.0",
            )
        )
        
        # Resolve
        resolved = await factory.theses.resolve(
            thesis.id,
            outcome="accurate",
            brier_score=0.15,
        )
        
        assert resolved is not None
        assert resolved.resolved is True
        assert resolved.resolution_outcome == "accurate"
    
    @pytest.mark.asyncio
    async def test_count_operations(self, db_session):
        """Test count operations."""
        factory = RepositoryFactory(db_session)
        
        initial = await factory.episodes.count()
        
        await factory.episodes.create(Episode(title="Test 1", summary="Test"))
        await factory.episodes.create(Episode(title="Test 2", summary="Test"))
        
        final = await factory.episodes.count()
        
        assert final == initial + 2


class TestDatabaseManager:
    """Integration tests for DatabaseManager."""
    
    @pytest.mark.asyncio
    async def test_database_health_check(self):
        """Test database health check."""
        config = DatabaseConfig.from_env().with_test_db()
        manager = DatabaseManager(config)
        
        # Create tables
        await manager.create_tables()
        
        # Check health
        healthy = await manager.health_check()
        assert healthy is True
        
        # Cleanup
        await manager.drop_tables()
        await manager.close()
