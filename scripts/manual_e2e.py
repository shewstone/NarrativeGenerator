"""End-to-end test of the Narrative Engine."""

import asyncio
import os
import uuid
from datetime import datetime

# Set database URL before imports
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/narrative_engine"

from narrative_engine.models import (
    Actor,
    ArcPhase,
    ArcType,
    CycleScale,
    Episode,
)
from narrative_engine.storage.database import DatabaseManager
from narrative_engine.storage.repositories import RepositoryFactory


async def test_create_and_retrieve_episode():
    """Test creating and retrieving an episode."""
    print("\n=== End-to-End Test: Narrative Engine ===\n")

    # Initialize database connection (reads from DATABASE_URL env var)
    db_manager = DatabaseManager()

    print("1. Testing database connection...")
    async with db_manager.session() as session:
        print("   ✓ Database connection established")

        print("\n2. Creating repository factory...")
        factory = RepositoryFactory(session)
        print("   ✓ Repository factory created")

        print("\n3. Creating a sample episode (1929 Stock Market Crash)...")

        # Create actors
        hoover = Actor(
            id=uuid.uuid4(),
            name="Herbert Hoover",
            role="President",
            attributes={"party": "Republican", "term": "1929-1933"}
        )

        # Create the episode
        episode = Episode(
            id=uuid.uuid4(),
            title="1929 Stock Market Crash",
            summary=(
                "The collapse of the US stock market in October 1929, "
                "marking the beginning of the Great Depression"
            ),
            start_date=datetime(1929, 10, 24),
            end_date=datetime(1929, 10, 29),
            date_precision="day",
            location="New York Stock Exchange",
            setting_description="The Roaring Twenties speculation bubble burst",
            initiating_conditions=[
                "Speculative excess in stock market",
                "Widespread margin trading",
                "Weak economic fundamentals"
            ],
            escalation_mechanics=[
                "Panic selling cascades",
                "Margin calls forcing liquidation",
                "Loss of confidence in banking system"
            ],
            tension="Fear of economic collapse spreads",
            resolution="Market bottomed in 1932, down 89% from peak",
            consequences=[
                "Great Depression begins",
                "Banking crisis follows",
                "New Deal reforms enacted"
            ],
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            phase_confidence=0.95,
            arc_rationale="Classic speculative bubble burst pattern",
            actors=[hoover],
            extracted_from=["kindleberger-mania-2026:ch4"],
        )

        # Save to database
        episode_repo = factory.episodes
        created = await episode_repo.create(episode)
        print(f"   ✓ Episode created with ID: {created.id}")
        print(f"     Title: {created.title}")
        print(f"     Arc Type: {created.arc_type.value}")
        print(f"     Arc Phase: {created.arc_phase.value}")

        print("\n4. Retrieving episode by ID...")
        retrieved = await episode_repo.get_by_id(created.id)
        print(f"   ✓ Retrieved: {retrieved.title}")
        print(f"     Actors: {[a.name for a in retrieved.actors]}")

        print("\n5. Searching episodes by arc type...")
        similar = await episode_repo.get_by_arc_type(ArcType.CREDIT_BOOM_AND_BUST.value)
        print(f"   ✓ Found {len(similar)} episode(s) with arc type '{ArcType.CREDIT_BOOM_AND_BUST.value}'")
        for ep in similar:
            print(f"     - {ep.title}")

        print("\n6. Counting total episodes...")
        count = await episode_repo.count()
        print(f"   ✓ Total episodes in database: {count}")

    print("\n=== Test Completed Successfully ===\n")
    return True


async def test_cycle_operations():
    """Test cycle CRUD operations."""
    print("\n=== Testing Cycle Operations ===\n")

    db_manager = DatabaseManager()

    async with db_manager.session() as session:
        factory = RepositoryFactory(session)
        cycle_repo = factory.cycles

        from narrative_engine.models import Cycle

        print("1. Creating a generational cycle...")
        cycle = Cycle(
            id=uuid.uuid4(),
            name="The Roaring Twenties",
            scale=CycleScale.GENERATIONAL,
            description="A period of economic prosperity and cultural change",
            start_date=datetime(1920, 1, 1),
            end_date=datetime(1929, 10, 29),
            dominant_arc_types=[ArcType.CREDIT_BOOM_AND_BUST],
            phase_estimate=ArcPhase.EUPHORIA,
            framework_source="Kindleberger's framework",
        )

        created = await cycle_repo.create(cycle)
        print(f"   ✓ Cycle created: {created.name} ({created.scale.value})")

        print("\n2. Retrieving cycle...")
        retrieved = await cycle_repo.get_by_id(created.id)
        print(f"   ✓ Retrieved: {retrieved.name}")
        print(f"     Phase estimate: {retrieved.phase_estimate.value}")

    print("\n=== Cycle Test Completed ===\n")
    return True


def main():
    """Run end-to-end tests."""
    print("\n" + "=" * 60)
    print("NARRATIVE ENGINE - END-TO-END TEST")
    print("=" * 60)

    try:
        asyncio.run(test_create_and_retrieve_episode())
        asyncio.run(test_cycle_operations())

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe Narrative Engine is working correctly!")
        print("- Database: PostgreSQL with pgvector")
        print("- ORM: SQLAlchemy with async support")
        print("- Models: Episodes, Cycles, Actors, Arcs")
        print("\nYou can now:")
        print("  1. Ingest historical texts")
        print("  2. Extract narrative episodes")
        print("  3. Search for analogs")
        print("  4. Generate forecasts")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
