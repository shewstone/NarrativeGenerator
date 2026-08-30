"""Tests for the pure, DB-free composition algorithm.

Covers the design doc Sec 6.2 stage 6 staged pipeline directly, plus the
same positive/negative cases as the composition fixture (Sec 6.6) --
duplicated here as fast unit tests, with the fixture-level integration
test added separately (test_fixture.py) as the actual regression gate.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from narrative_engine.composition import compose_arc_instances_from_episodes
from narrative_engine.composition.identity import ArcIdentityResolver
from narrative_engine.composition.pipeline import (
    CompositionPipeline,
    _cluster_within_scope,
)
from narrative_engine.models import (
    Actor,
    ArcPhase,
    ArcType,
    CycleScale,
    Episode,
    ScopeKind,
    SituationScale,
)
from narrative_engine.storage.orm_models import CycleMembershipORM, CycleORM
from narrative_engine.storage.repositories import EpisodeRepository


def _episode(**overrides) -> Episode:
    defaults = {
        "id": uuid4(),
        "title": "Episode",
        "summary": "Summary",
        "arc_type": ArcType.CREDIT_BOOM_AND_BUST,
        "scope_id": "us_national",
    }
    defaults.update(overrides)
    return Episode(**defaults)


class TestScopePartition:
    """Stage 1: hard filter, episodes in different scopes never compare."""

    def test_different_scopes_never_merge_even_if_otherwise_identical(self):
        a = _episode(
            start_date=datetime(1929, 1, 1),
            end_date=datetime(1929, 2, 1),
            arc_phase=ArcPhase.BOOM,
            scope_id="us_national",
        )
        b = _episode(
            start_date=datetime(1929, 2, 1),
            end_date=datetime(1929, 3, 1),
            arc_phase=ArcPhase.EUPHORIA,
            scope_id="uk_national",
        )

        instances = compose_arc_instances_from_episodes([a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 2
        assert {i.canonical_name for i in instances} != set()

    def test_none_scope_only_merges_with_none_scope(self):
        a = _episode(
            start_date=datetime(1929, 1, 1),
            end_date=datetime(1929, 2, 1),
            arc_phase=ArcPhase.BOOM,
            scope_id=None,
        )
        b = _episode(
            start_date=datetime(1929, 2, 1),
            end_date=datetime(1929, 3, 1),
            arc_phase=ArcPhase.EUPHORIA,
            scope_id="us_national",
        )

        instances = compose_arc_instances_from_episodes([a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 2


class TestTemporalGate:
    """Stage 4: per-scale gap threshold, hard reject beyond 2x threshold."""

    def test_episodic_scale_rejects_gap_beyond_threshold(self):
        a = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=ArcPhase.PANIC,
        )
        b = _episode(
            start_date=datetime(1922, 1, 1),
            end_date=datetime(1927, 1, 1),
            arc_phase=ArcPhase.BOOM,
        )

        instances = compose_arc_instances_from_episodes(
            [a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST, scale=CycleScale.EPISODIC
        )

        assert len(instances) == 2

    def test_civilizational_scale_is_more_permissive_than_episodic(self):
        # Same 14-year gap that fails at episodic (2y) scale should pass
        # at civilizational (40y) scale.
        a = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=ArcPhase.BOOM,
        )
        b = _episode(
            start_date=datetime(1922, 1, 1),
            end_date=datetime(1927, 1, 1),
            arc_phase=ArcPhase.EUPHORIA,
        )

        instances = compose_arc_instances_from_episodes(
            [a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST, scale=CycleScale.CIVILIZATIONAL
        )

        assert len(instances) == 1

    def test_movement_scope_uses_generational_timing_without_actor_continuity(self):
        first = _episode(
            title="Movement organizes",
            summary="A suffrage coalition establishes national organizations",
            arc_type=ArcType.REFORM_THEN_REACTION,
            arc_phase=ArcPhase.SETUP,
            start_date=datetime(1900, 1, 1),
            end_date=datetime(1900, 12, 31),
            scope_id="uk_womens_suffrage_movement",
            scope_kind=ScopeKind.MOVEMENT,
            situation_scale=SituationScale.GROUP,
            actors=[Actor(name="Early organizers", role="organizer")],
        )
        second = _episode(
            title="Movement escalates",
            summary="A later generation adopts militant suffrage tactics",
            arc_type=ArcType.REFORM_THEN_REACTION,
            arc_phase=ArcPhase.RISING_ACTION,
            start_date=datetime(1910, 1, 1),
            end_date=datetime(1910, 12, 31),
            scope_id="uk_womens_suffrage_movement",
            scope_kind=ScopeKind.MOVEMENT,
            situation_scale=SituationScale.GROUP,
            actors=[Actor(name="Later organizers", role="organizer")],
        )

        instances = compose_arc_instances_from_episodes(
            [first, second],
            arc_type=ArcType.REFORM_THEN_REACTION,
        )

        assert len(instances) == 1
        assert instances[0].scale == CycleScale.GENERATIONAL

    def test_signed_bce_years_participate_in_temporal_composition(self):
        actor = Actor(name="Roman Senate", role="governing institution")
        a = _episode(
            start_date=None,
            end_date=None,
            start_year=-500,
            end_year=-499,
            arc_phase=ArcPhase.BOOM,
            scope_id="rome",
            actors=[actor],
        )
        b = _episode(
            start_date=None,
            end_date=None,
            start_year=-498,
            end_year=-497,
            arc_phase=ArcPhase.EUPHORIA,
            scope_id="rome",
            actors=[actor],
        )

        instances = compose_arc_instances_from_episodes([a, b], ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 1
        assert instances[0].scope_id == "rome"
        assert "500 BCE–497 BCE" in instances[0].canonical_name


class TestPhaseSequenceGate:
    """Stage 5: out-of-order phases hard-reject the merge."""

    def test_out_of_order_phases_split(self):
        # PANIC then BOOM (going backwards) should not merge even though
        # temporally adjacent and same scope/arc_type.
        a = _episode(
            start_date=datetime(1929, 1, 1),
            end_date=datetime(1929, 2, 1),
            arc_phase=ArcPhase.PANIC,
        )
        b = _episode(
            start_date=datetime(1929, 3, 1),
            end_date=datetime(1929, 4, 1),
            arc_phase=ArcPhase.BOOM,
        )

        instances = compose_arc_instances_from_episodes([a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 2


class TestArcTypeAgreement:
    def test_different_arc_types_are_filtered_before_clustering(self):
        a = _episode(
            start_date=datetime(1929, 1, 1),
            end_date=datetime(1929, 2, 1),
            arc_phase=ArcPhase.BOOM,
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        )
        b = _episode(
            start_date=datetime(1929, 2, 1),
            end_date=datetime(1929, 3, 1),
            arc_phase=ArcPhase.RISING_ACTION,
            arc_type=ArcType.HUBRIS_NEMESIS,
        )

        # compose_arc_instances_from_episodes filters to the requested
        # arc_type up front, so the HUBRIS_NEMESIS episode is excluded
        # entirely from a CREDIT_BOOM_AND_BUST composition run.
        instances = compose_arc_instances_from_episodes([a, b], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 1
        assert instances[0].phases.get(ArcPhase.BOOM) is not None


class TestNoSilentDrops:
    def test_singleton_episode_still_produces_an_instance(self):
        a = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=ArcPhase.PANIC,
        )

        instances = compose_arc_instances_from_episodes([a], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 1
        assert a.id in instances[0].phases[ArcPhase.PANIC].episode_ids

    def test_unphased_episode_is_retained_for_persistence(self):
        episode = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=None,
        )

        instances = compose_arc_instances_from_episodes([episode], arc_type=ArcType.CREDIT_BOOM_AND_BUST)

        assert instances[0].unphased_episode_ids == [episode.id]


class TestPersistenceReconciliation:
    @pytest.mark.asyncio
    async def test_orm_conversion_preserves_raw_scope_and_signed_year(self, db_session):
        episode = _episode(
            start_date=None,
            end_date=None,
            start_year=-1200,
            end_year=-1190,
            arc_phase=ArcPhase.BOOM,
            scope_id=None,
            scope_name="Temudjin",
            scope_kind="person",
            scope_confidence=0.9,
        )
        await EpisodeRepository(db_session).create(episode)

        instances = await CompositionPipeline(db_session).compose_arc_instances(
            ArcType.CREDIT_BOOM_AND_BUST
        )

        assert len(instances) == 1
        assert instances[0].scope_id == "genghis_khan"
        assert "1200 BCE–1190 BCE" in instances[0].canonical_name

    @pytest.mark.asyncio
    async def test_repeated_composition_reuses_existing_instance(self, db_session):
        episode = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=ArcPhase.PANIC,
        )
        await EpisodeRepository(db_session).create(episode)
        pipeline = CompositionPipeline(db_session)

        instances = await pipeline.compose_arc_instances(ArcType.CREDIT_BOOM_AND_BUST)
        first = await pipeline.reconcile_instances(instances, ArcType.CREDIT_BOOM_AND_BUST)
        second = await pipeline.reconcile_instances(instances, ArcType.CREDIT_BOOM_AND_BUST)

        cycle_count = (await db_session.execute(select(func.count(CycleORM.id)))).scalar_one()
        membership_count = (await db_session.execute(select(func.count(CycleMembershipORM.id)))).scalar_one()
        assert cycle_count == 1
        assert membership_count == 1
        assert second[0].id == first[0].id

    @pytest.mark.asyncio
    async def test_unphased_episode_gets_membership(self, db_session):
        episode = _episode(
            start_date=datetime(1907, 10, 1),
            end_date=datetime(1907, 11, 1),
            arc_phase=None,
        )
        await EpisodeRepository(db_session).create(episode)
        pipeline = CompositionPipeline(db_session)

        instances = await pipeline.compose_arc_instances(ArcType.CREDIT_BOOM_AND_BUST)
        await pipeline.reconcile_instances(instances, ArcType.CREDIT_BOOM_AND_BUST)

        membership = (await db_session.execute(select(CycleMembershipORM))).scalar_one()
        assert membership.episode_id == episode.id
        assert membership.phase_coverage == []


class TestClusterWithinScope:
    """Direct tests of the sequential merge helper."""

    def test_empty_input_returns_no_clusters(self):
        resolver = ArcIdentityResolver()
        assert _cluster_within_scope([], resolver, CycleScale.EPISODIC) == []
