"""Regression tests for canonical taxonomy creation."""

from unittest.mock import AsyncMock

import pytest

from narrative_engine.models import ArcType
from narrative_engine.taxonomy.models import TaxonomyStatus
from narrative_engine.taxonomy.service import ArcDiscoveryService


@pytest.mark.asyncio
async def test_canonical_taxonomy_persists_every_arc_type():
    repository = AsyncMock()
    repository.create_taxonomy.side_effect = lambda taxonomy: taxonomy
    service = ArcDiscoveryService(repository)

    taxonomy = await service.create_canonical_taxonomy()

    created = [call.args[0] for call in repository.create_canonical_arc.await_args_list]
    assert {arc.slug for arc in created} == {arc_type.value for arc_type in ArcType}
    assert all(arc.taxonomy_ids == [taxonomy.id] for arc in created)
    assert all(phase == phase.strip() for arc in created for phase in arc.phases)
    repository.update_taxonomy_status.assert_awaited_once_with(taxonomy.id, TaxonomyStatus.ACTIVE)
    assert taxonomy.status == TaxonomyStatus.ACTIVE
