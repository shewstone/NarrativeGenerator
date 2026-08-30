"""Rebuild every persisted arc instance from the current episode corpus.

Use after taxonomy, scope-registry, or corpus-wide corrections.  Reconciliation
reuses exact instances and removes stale snapshots, so repeated runs are safe.
"""

from __future__ import annotations

import asyncio

from narrative_engine.composition.pipeline import CompositionPipeline
from narrative_engine.logging_config import configure_logging
from narrative_engine.models import ArcType
from narrative_engine.storage.database import db_manager


async def main() -> None:
    configure_logging(level="WARNING")
    total = 0
    async with db_manager.session() as session:
        composer = CompositionPipeline(session)
        for arc_type in ArcType:
            instances = await composer.compose_arc_instances(arc_type)
            persisted = await composer.reconcile_instances(instances, arc_type)
            total += len(persisted)
            print(f"{arc_type.value}: {len(persisted)} instances")
    await db_manager.close()
    print(f"total: {total} instances")


if __name__ == "__main__":
    asyncio.run(main())
