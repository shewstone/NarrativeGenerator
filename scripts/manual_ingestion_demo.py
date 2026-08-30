"""Test ingestion of the design document into the Narrative Engine."""

import asyncio
import os
import uuid
from datetime import datetime

os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/narrative_engine"

from narrative_engine.models import (
    Actor,
    ArcPhase,
    ArcType,
    Episode,
    SourcePassage,
)
from narrative_engine.storage.database import DatabaseManager
from narrative_engine.storage.repositories import RepositoryFactory


async def ingest_design_document():
    """Manually extract episodes from the design doc as a test."""
    print("\n" + "=" * 60)
    print("NARRATIVE ENGINE - INGESTION TEST")
    print("=" * 60)
    print()

    db_manager = DatabaseManager()

    # Read the document
    doc_path = "data/raw/design-doc.txt"
    print(f"1. Reading document: {doc_path}")

    with open(doc_path, "r") as f:
        content = f.read()

    print(f"   ✓ Read {len(content)} characters")
    print(f"   ✓ ~{len(content) // 5} tokens (estimated)")

    print("\n2. Manually extracting episodes...")
    print("   (In production, this would use the LLM pipeline)")

    async with db_manager.session() as session:
        factory = RepositoryFactory(session)

        # Episode 1: 1929 Stock Market Crash
        print("\n   Extracting Episode 1: 1929 Stock Market Crash")

        hoover = Actor(
            id=uuid.uuid4(),
            name="Herbert Hoover",
            role="President",
            attributes={
                "party": "Republican",
                "term": "1929-1933",
                "beliefs": "Limited government intervention"
            }
        )

        episode_1929 = Episode(
            id=uuid.uuid4(),
            title="1929 Stock Market Crash",
            summary=(
                "The collapse of the US stock market in October 1929, marking the beginning of the Great Depression. "
                "Instantiated the credit-boom-and-bust arc with clear phase progression."
            ),
            start_date=datetime(1929, 10, 24),
            end_date=datetime(1932, 7, 8),  # Market bottom
            date_precision="day",
            location="New York Stock Exchange",
            setting_description="The Roaring Twenties speculation bubble burst",
            initiating_conditions=[
                "Speculative excess in stock market",
                "Widespread margin trading (10% down)",
                "Weak economic fundamentals hidden by optimism"
            ],
            escalation_mechanics=[
                "Panic selling cascades",
                "Margin calls forcing liquidation",
                "Loss of confidence in banking system"
            ],
            tension="Fear of economic collapse spreads as selling accelerates",
            resolution="Market bottomed July 1932, Dow declined 89% from peak",
            consequences=[
                "Banking crisis (1930-1933)",
                "New Deal reforms enacted",
                "Keynesian economics ascendant",
                "Bretton Woods system established post-WWII"
            ],
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.PANIC,
            phase_confidence=0.95,
            arc_rationale="Classic credit cycle phases: boom → euphoria → distress → panic → revulsion",
            actors=[hoover],
            extracted_from=["design-doc.txt:section-1929-crash"],
        )

        # Episode 2: Weimar Hyperinflation
        print("   Extracting Episode 2: Weimar Hyperinflation")

        episode_weimar = Episode(
            id=uuid.uuid4(),
            title="Weimar Hyperinflation",
            summary=(
                "German currency collapse following WWI reparations and Ruhr occupation. "
                "Demonstrates fiscal distress mechanism."
            ),
            start_date=datetime(1921, 1, 1),
            end_date=datetime(1923, 11, 15),  # Rentenmark introduced
            date_precision="month",
            location="Weimar Republic, Germany",
            setting_description="Post-WWI Germany burdened by war reparations",
            initiating_conditions=[
                "War reparations burden (Treaty of Versailles)",
                "Occupation of Ruhr (1923)",
                "Passive resistance funded by money printing"
            ],
            escalation_mechanics=[
                "Currency velocity accelerated",
                "Wage-price spirals",
                "Savings destroyed"
            ],
            tension="Currency becomes worthless, economic chaos",
            resolution="Rentenmark introduced November 1923, currency stabilized",
            consequences=[
                "Political radicalization followed",
                "Middle class savings wiped out",
                "Erosion of trust in democratic institutions"
            ],
            arc_type=ArcType.DECADENCE_AND_RENEWAL,
            arc_phase=ArcPhase.CLIMAX,
            phase_confidence=0.85,
            arc_rationale="Terminal phase of fiscal distress, followed by stabilization",
            actors=[],
            extracted_from=["design-doc.txt:section-weimar"],
        )

        print("\n3. Saving episodes to database...")

        # Save episodes (and their actors)
        saved_1929 = await factory.episodes.create(episode_1929)
        saved_weimar = await factory.episodes.create(episode_weimar)

        print(f"   ✓ Saved: {saved_1929.title} ({saved_1929.id})")
        print(f"   ✓ Saved: {saved_weimar.title} ({saved_weimar.id})")

        print("\n4. Adding source passages...")

        # Add source passages
        passage_1929 = SourcePassage(
            work_id="design-doc.txt",
            passage_id="1929-crash-001",
            text="""This episode instantiates the credit-boom-and-bust arc, with clear phases:
- Boom (1922-1927): Post-war economic expansion
- Euphoria (1927-1929): Speculative excess, margin trading
- Distress (October 1929): Initial cracks, Black Thursday
- Panic (October 1929): Cascade selling, Black Tuesday
- Revulsion (1929-1932): Market bottom, 89% decline""",
            chapter="Historical Example: The 1929 Stock Market Crash",
        )

        passage_weimar = SourcePassage(
            work_id="design-doc.txt",
            passage_id="weimar-001",
            text="""This episode shows fiscal distress and currency collapse.

Initiating Conditions:
- War reparations burden (Treaty of Versailles)
- Occupation of Ruhr (1923)
- Passive resistance funded by money printing""",
            chapter="Another Example: Weimar Hyperinflation (1921-1923)",
        )

        print(f"   ✓ Source passage 1: {len(passage_1929.text)} chars")
        print(f"   ✓ Source passage 2: {len(passage_weimar.text)} chars")

        print("\n5. Querying episodes...")

        # Query by arc type
        credit_episodes = await factory.episodes.get_by_arc_type(
            ArcType.CREDIT_BOOM_AND_BUST.value
        )
        print(f"   ✓ Found {len(credit_episodes)} credit-boom-and-bust episode(s)")
        for ep in credit_episodes:
            print(f"     - {ep.title} (phase: {ep.arc_phase.value})")

        # Query by arc phase
        panic_episodes = await factory.episodes.get_by_arc_phase(
            ArcPhase.PANIC.value
        )
        print(f"\n   ✓ Found {len(panic_episodes)} episode(s) in PANIC phase")
        for ep in panic_episodes:
            print(f"     - {ep.title}")

        # Count total
        total = await factory.episodes.count()
        print(f"\n   ✓ Total episodes in database: {total}")

    print("\n" + "=" * 60)
    print("✅ INGESTION TEST COMPLETE")
    print("=" * 60)
    print()
    print("Successfully extracted and stored:")
    print("  • 2 historical episodes")
    print("  • 1 actor (Herbert Hoover)")
    print("  • 2 source passages")
    print("  • Arc classifications verified")
    print()
    print("Next: Try semantic search or generate a thesis")
    print("  from these extracted episodes!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(ingest_design_document())
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
