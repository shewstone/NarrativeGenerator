"""Scope registry and resolver tests (T5, docs/tickets/T5-scope-registry.md)."""

from datetime import datetime, timedelta

import pytest

from narrative_engine.composition.pipeline import compose_arc_instances_from_episodes
from narrative_engine.models import Actor, ArcPhase, ArcType, Episode, Scope
from narrative_engine.scopes import (
    ScopeRegistry,
    get_registry,
    resolve_scope,
    scope_registry_version,
)
from narrative_engine.storage.repositories import ScopeRepository


class TestResolver:
    def test_alias_variants_resolve_to_one_id(self):
        assert resolve_scope("United States") == "us"
        assert resolve_scope("USA") == "us"
        assert resolve_scope("U.S.") == "us"
        assert resolve_scope("the United States of America") == "us"
        assert resolve_scope("us") == "us"

    def test_case_and_punctuation_insensitive(self):
        assert resolve_scope("UNITED STATES") == "us"
        assert resolve_scope("wilhelmine germany") == "germany"
        assert resolve_scope("Austria-Hungary") == "austria_hungary"

    def test_party_resolves_within_parent_polity(self):
        assert resolve_scope("CCP") == "chinese_communist_party"
        assert [scope.id for scope in get_registry().lineage("CCP")][:2] == [
            "chinese_communist_party",
            "china",
        ]

    def test_unknown_returns_none_never_guesses(self):
        # No fuzzy matching: a wrong scope silently poisons the composition
        # partition; an unresolved one falls to the visible singleton path.
        assert resolve_scope("Atlantis") is None
        assert resolve_scope("Uni") is None
        assert resolve_scope("") is None
        assert resolve_scope(None) is None

    def test_registry_is_versioned(self):
        assert scope_registry_version().startswith("scope-v")

    def test_no_alias_collisions(self):
        # ScopeRegistry.load() raises on collision; loading at all is the test.
        registry = get_registry()
        assert len(registry.all()) >= 20

    def test_nested_faction_party_polity_lineage(self):
        registry = ScopeRegistry(
            version="test-v1",
            scopes=[
                Scope(id="us", kind="polity", name="United States"),
                Scope(
                    id="example_party",
                    kind="party",
                    name="Example Party",
                    parent_scope_id="us",
                ),
                Scope(
                    id="reform_faction",
                    kind="faction",
                    name="Reform Faction",
                    parent_scope_id="example_party",
                ),
            ],
        )

        assert [scope.id for scope in registry.lineage("Reform Faction")] == [
            "reform_faction",
            "example_party",
            "us",
        ]
        assert registry.is_within("Reform Faction", "United States")
        assert {scope.id for scope in registry.descendants("us")} == {
            "example_party",
            "reform_faction",
        }

    def test_invalid_scope_hierarchy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown parent"):
            ScopeRegistry(
                version="test-v1",
                scopes=[
                    Scope(
                        id="orphan",
                        kind="faction",
                        name="Orphan Faction",
                        parent_scope_id="missing",
                    )
                ],
            )


class TestCompositionPartitionNormalization:
    def _episode(self, scope_label: str, title: str, start: datetime) -> Episode:
        return Episode(
            title=title,
            summary=f"{title} summary",
            scope_id=scope_label,
            arc_type=ArcType.CREDIT_BOOM_AND_BUST,
            arc_phase=ArcPhase.BOOM if "boom" in title else ArcPhase.PANIC,
            start_date=start,
            end_date=start + timedelta(days=90),
            actors=[Actor(name="Wall Street", role="Financier")],
            extracted_from=["src-a"],
        )

    def test_alias_labeled_episodes_land_in_one_partition(self):
        """'US' and 'United States' used to be two partitions (false split)."""
        boom = self._episode("US", "credit boom", datetime(1927, 1, 1))
        panic = self._episode("United States", "panic", datetime(1929, 10, 1))

        instances = compose_arc_instances_from_episodes([boom, panic], ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 1
        merged = instances[0]
        covered_episode_ids = {eid for cov in merged.phases.values() for eid in cov.episode_ids}
        assert covered_episode_ids == {boom.id, panic.id}

    def test_unresolved_labels_do_not_merge_with_each_other(self):
        """Two distinct unknown labels stay distinct partitions."""
        a = self._episode("Atlantis", "credit boom", datetime(1927, 1, 1))
        b = self._episode("Mu", "panic", datetime(1929, 10, 1))

        instances = compose_arc_instances_from_episodes([a, b], ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 2

    def test_new_subgroup_names_form_their_own_partition(self):
        a = self._episode("", "credit boom", datetime(1927, 1, 1))
        b = self._episode("", "panic", datetime(1929, 10, 1))
        for episode in (a, b):
            episode.scope_id = None
            episode.scope_name = "Reform Faction"
            episode.scope_confidence = 0.9

        instances = compose_arc_instances_from_episodes([a, b], ArcType.CREDIT_BOOM_AND_BUST)

        assert len(instances) == 1
        assert instances[0].scope_id == "Reform Faction"


class TestScopeRepositorySync:
    @pytest.mark.asyncio
    async def test_sync_from_registry_upserts_all(self, db_session):
        repo = ScopeRepository(db_session)

        count = await repo.sync_from_registry()
        assert count == len(get_registry().all())

        us = await repo.get_by_id("us")
        assert us is not None
        assert us.kind == "polity"
        assert "USA" in us.aliases

        # Idempotent: second sync neither errors nor duplicates.
        count_again = await repo.sync_from_registry()
        assert count_again == count
        assert len(await repo.list_all()) == count
