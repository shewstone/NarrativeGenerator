"""Resolve previously unscoped episodes against the current scope registry.

The operation is intentionally one-way and conservative: it only fills a
missing ``scope_id`` when the retained raw ``scope_name`` is an exact registry
name or alias. Existing classifications are never rewritten. Run without
``--apply`` to preview the number of changes.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from narrative_engine.logging_config import configure_logging
from narrative_engine.scopes import get_registry, resolve_scope
from narrative_engine.storage.database import db_manager
from narrative_engine.storage.orm_models import EpisodeORM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="persist exact registry matches (default: report only)",
    )
    return parser.parse_args()


async def backfill(*, apply: bool) -> tuple[int, dict[str, int]]:
    registry = get_registry()
    matches: dict[str, int] = {}
    matched = 0

    async with db_manager.session() as session:
        result = await session.execute(
            select(EpisodeORM).where(
                EpisodeORM.scope_id.is_(None),
                EpisodeORM.scope_name.is_not(None),
            )
        )
        for episode in result.scalars():
            scope_id = resolve_scope(episode.scope_name)
            if scope_id is None:
                continue
            matched += 1
            matches[scope_id] = matches.get(scope_id, 0) + 1
            if not apply:
                continue

            canonical = registry.get(scope_id)
            episode.scope_id = scope_id
            if canonical is not None:
                episode.scope_kind = canonical.kind
                if canonical.parent_scope_id:
                    parent = registry.get(canonical.parent_scope_id)
                    episode.parent_scope_name = parent.name if parent else episode.parent_scope_name

        if not apply:
            await session.rollback()

    return matched, matches


async def main() -> None:
    args = parse_args()
    configure_logging(level="WARNING")
    matched, matches = await backfill(apply=args.apply)
    action = "updated" if args.apply else "would update"
    print(f"{action}: {matched} episodes across {len(matches)} scopes")
    for scope_id, count in sorted(matches.items(), key=lambda item: (-item[1], item[0])):
        print(f"{scope_id}: {count}")
    await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
