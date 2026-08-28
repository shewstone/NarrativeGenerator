"""Regression tests for ArcIdentityResolver / DisambiguationEngine bug fixes.

Both bugs here were latent AttributeError/NameError crashes: the code paths
were never exercised (find_candidate_matches referenced a non-existent
`self.temporal_threshold` singular attribute; detect_false_merge_risk
referenced an undefined `avg_continuity` name). These tests just need to
reach the previously-crashing lines without raising.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from narrative_engine.composition.identity import (
    ArcIdentityResolver,
    DisambiguationEngine,
)
from narrative_engine.models import Actor, ArcType, Episode
from narrative_engine.storage.orm_models import EpisodeORM
from narrative_engine.storage.repositories import EpisodeRepository


def _episode(**overrides) -> Episode:
    defaults = {
        "id": uuid4(),
        "title": "Episode",
        "summary": "Summary",
        "arc_type": ArcType.CREDIT_BOOM_AND_BUST,
    }
    defaults.update(overrides)
    return Episode(**defaults)


class FakeAsyncSession:
    """Minimal async session stand-in: returns no rows for any execute()."""

    async def execute(self, _query):
        class _Result:
            def scalars(self):
                return self

            def all(self):
                return []

        return _Result()


class TestFindCandidateMatches:
    """find_candidate_matches previously crashed with AttributeError on
    `self.temporal_threshold` (singular) which doesn't exist on the resolver.
    """

    @pytest.mark.asyncio
    async def test_does_not_raise_with_end_date(self):
        resolver = ArcIdentityResolver()
        episode = _episode(end_date=datetime(1929, 10, 24))

        matches = await resolver.find_candidate_matches(
            session=FakeAsyncSession(),
            episode=episode,
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        )

        assert matches == []


class TestOrmConversion:
    @pytest.mark.asyncio
    async def test_preserves_embedding_epochs_without_lazy_loading(self, db_session):
        episode = _episode(
            surface_embedding=[1.0] * 384,
            surface_embedding_epoch="surface-v1",
            structural_embedding=[0.5] * 384,
            structural_embedding_epoch="structural-v2",
        )
        await EpisodeRepository(db_session).create(episode)
        orm = await db_session.get(EpisodeORM, episode.id)

        converted = ArcIdentityResolver().episode_from_orm(orm)

        assert converted.surface_embedding_epoch == "surface-v1"
        assert converted.structural_embedding_epoch == "structural-v2"
        assert converted.actors == []


class TestDetectFalseMergeRisk:
    """detect_false_merge_risk previously crashed with NameError on
    `avg_continuity` (undefined; should be `avg_actor_continuity`) whenever
    low actor continuity across a cluster of 3+ episodes triggered that branch.
    """

    def test_low_actor_continuity_branch_does_not_raise(self):
        resolver = ArcIdentityResolver()
        engine = DisambiguationEngine(resolver)

        # Three episodes, each with disjoint actors -> low actor continuity,
        # and len(cluster) > 2, which is what triggers the buggy branch.
        cluster = [
            _episode(
                start_date=datetime(1920, 1, 1),
                end_date=datetime(1920, 6, 1),
                actors=[Actor(name="A", role="RISING_POWER")],
            ),
            _episode(
                start_date=datetime(1921, 1, 1),
                end_date=datetime(1921, 6, 1),
                actors=[Actor(name="B", role="RISING_POWER")],
            ),
            _episode(
                start_date=datetime(1922, 1, 1),
                end_date=datetime(1922, 6, 1),
                actors=[Actor(name="C", role="RISING_POWER")],
            ),
        ]

        result = engine.detect_false_merge_risk(cluster)

        assert "actor_continuity" in result
        assert any("Low actor continuity" in r for r in result["risk_factors"])

    def test_missing_and_timezone_aware_dates_do_not_crash(self):
        engine = DisambiguationEngine(ArcIdentityResolver())
        cluster = [
            _episode(start_date=None, end_date=None),
            _episode(
                start_date=datetime(1920, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(1920, 2, 1, tzinfo=timezone.utc),
            ),
            _episode(
                start_date=datetime(1921, 1, 1),
                end_date=datetime(1921, 2, 1),
            ),
        ]

        result = engine.detect_false_merge_risk(cluster)

        assert "risk" in result

    def test_identity_score_accepts_mixed_timezone_dates(self):
        resolver = ArcIdentityResolver()
        earlier = _episode(
            start_date=datetime(1920, 1, 1),
            end_date=datetime(1920, 2, 1),
        )
        later = _episode(
            start_date=datetime(1920, 3, 1, tzinfo=timezone.utc),
            end_date=datetime(1920, 4, 1, tzinfo=timezone.utc),
        )

        score = resolver.calculate_identity_score(earlier, later)

        assert 0.0 <= score.temporal_score <= 1.0
