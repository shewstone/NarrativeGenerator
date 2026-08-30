"""Scope registry and resolver tests (T5, docs/tickets/T5-scope-registry.md)."""

from datetime import datetime, timedelta

import pytest

from narrative_engine.composition.pipeline import compose_arc_instances_from_episodes
from narrative_engine.models import Actor, ArcPhase, ArcType, Episode, Scope
from narrative_engine.scopes import (
    ScopeRegistry,
    get_registry,
    resolve_scope,
    scope_partition_key,
    scope_registry_version,
    suggest_scopes,
)
from narrative_engine.storage.repositories import ScopeRepository


class TestResolver:
    def test_candidate_retrieval_surfaces_non_exact_movement_alias(self):
        suggestions = suggest_scopes(
            "British suffragists",
            kind="movement",
            parent_name="United Kingdom",
        )

        assert any(candidate.scope.id == "uk_womens_suffrage_movement" for candidate in suggestions)

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

    def test_unicode_and_gutenberg_diacritics_share_an_exact_key(self):
        assert scope_partition_key(None, "Abd-er-Rahmān III") == scope_partition_key(
            None,
            "Abd-er-Rahm[=a]n III",
        )

    def test_party_resolves_within_parent_polity(self):
        assert resolve_scope("CCP") == "chinese_communist_party"
        assert [scope.id for scope in get_registry().lineage("CCP")][:2] == [
            "chinese_communist_party",
            "china",
        ]

    def test_modern_country_parties_remain_distinct_nested_scopes(self):
        assert [scope.id for scope in get_registry().lineage("BJP")] == [
            "bharatiya_janata_party",
            "india",
            "south_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("ANC")] == [
            "african_national_congress",
            "south_africa",
            "southern_africa",
            "africa",
        ]
        assert [scope.id for scope in get_registry().lineage("LDP")] == [
            "liberal_democratic_party_japan",
            "japan",
            "sinosphere",
        ]
        assert [scope.id for scope in get_registry().lineage("Siloviki")] == [
            "siloviki",
            "russia",
        ]
        assert resolve_scope("India") == "india"

    def test_observed_movement_resolves_inside_regional_trajectory(self):
        assert [scope.id for scope in get_registry().lineage("Propaganda Movement")] == [
            "filipino_reform_movement",
            "philippines",
            "southeast_asia",
        ]

    def test_mesoamerican_facets_retain_their_own_scope(self):
        assert [scope.id for scope in get_registry().lineage("Maya")][:2] == [
            "maya",
            "mesoamerica",
        ]

    def test_inca_dynasty_is_nested_inside_civilization(self):
        assert [scope.id for scope in get_registry().lineage("Inca dynasty")] == [
            "inca_dynasty",
            "inca",
            "andean",
        ]

    def test_historical_transliteration_variants_share_a_dynasty(self):
        assert resolve_scope("Romanoff dynasty") == "romanov_dynasty"
        assert resolve_scope("Romanov dynasty") == "romanov_dynasty"

    def test_japanese_clan_is_nested_inside_its_regime(self):
        assert [scope.id for scope in get_registry().lineage("Tokugawa family")] == [
            "tokugawa_clan",
            "tokugawa_shogunate",
            "japan",
            "sinosphere",
        ]

    def test_new_gap_batch_groups_have_regional_lineage(self):
        assert [scope.id for scope in get_registry().lineage("Maori")] == [
            "maori",
            "new_zealand",
            "oceania",
        ]

    def test_central_asian_polities_form_regional_lineages(self):
        assert [scope.id for scope in get_registry().lineage("Bokhara")] == [
            "bukhara",
            "central_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Golden Horde")] == [
            "golden_horde",
            "mongol_empire",
            "central_asia",
        ]

    def test_southeast_asian_regimes_retain_country_and_region(self):
        assert [scope.id for scope in get_registry().lineage("Ayuthia")] == [
            "ayutthaya",
            "thailand",
            "southeast_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Alompra dynasty")] == [
            "konbaung_dynasty",
            "burma",
            "southeast_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Acheen")] == [
            "aceh_sultanate",
            "sumatra",
            "southeast_asia",
        ]

    def test_new_african_and_european_subgroups_are_nested(self):
        assert [scope.id for scope in get_registry().lineage("Zulu people")] == [
            "zulu_kingdom",
            "southern_africa",
            "africa",
        ]
        assert [scope.id for scope in get_registry().lineage("Hussites")] == [
            "hussite_movement",
            "bohemia",
            "central_eastern_europe",
            "western",
        ]
        assert resolve_scope("John Huss") == "jan_hus"
        assert [scope.id for scope in get_registry().lineage("Roumania")] == [
            "romania",
            "balkans",
            "central_eastern_europe",
            "western",
        ]

    def test_latest_gap_batches_consolidate_durable_subgroups(self):
        assert [scope.id for scope in get_registry().lineage("Utraquist Party")][:3] == [
            "utraquists",
            "hussite_movement",
            "bohemia",
        ]
        assert [scope.id for scope in get_registry().lineage("House of Árpád")][:2] == [
            "arpad_dynasty",
            "hungary",
        ]
        assert [scope.id for scope in get_registry().lineage("Sultanate of Achin")][:2] == [
            "aceh_sultanate",
            "sumatra",
        ]
        assert [scope.id for scope in get_registry().lineage("Transvaal Republic")] == [
            "south_african_republic",
            "southern_africa",
            "africa",
        ]
        assert [scope.id for scope in get_registry().lineage("Cetewayo")][:2] == [
            "cetshwayo",
            "zulu_kingdom",
        ]

    def test_movement_evidence_resolves_inside_its_historical_context(self):
        assert [scope.id for scope in get_registry().lineage("Ti-ping movement")][:2] == [
            "taiping_movement",
            "china",
        ]
        assert [scope.id for scope in get_registry().lineage("Chartism")][:2] == [
            "chartist_movement",
            "uk",
        ]
        assert [scope.id for scope in get_registry().lineage("Paris Commune")][:2] == [
            "paris_commune",
            "france",
        ]
        assert [scope.id for scope in get_registry().lineage("Maji Maji uprising")][:2] == [
            "maji_maji_movement",
            "german_east_africa",
        ]
        assert [scope.id for scope in get_registry().lineage("Woman Suffrage Movement")][:2] == [
            "uk_womens_suffrage_movement",
            "uk",
        ]
        assert [scope.id for scope in get_registry().lineage("BLM movement")][:2] == [
            "black_lives_matter",
            "us",
        ]

    def test_byzantine_and_ottoman_scopes_have_civilizational_parents(self):
        assert [scope.id for scope in get_registry().lineage("Byzantine Empire")] == [
            "byzantium",
            "hellenic",
        ]
        assert [scope.id for scope in get_registry().lineage("Ottoman Empire")] == [
            "ottoman",
            "islamicate",
        ]

    def test_corpus_promotions_keep_people_and_subgroups_nested(self):
        assert [scope.id for scope in get_registry().lineage("Temudjin")] == [
            "genghis_khan",
            "mongol_empire",
            "central_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Othmân")] == [
            "osman_i",
            "ottoman",
            "islamicate",
        ]
        assert [scope.id for scope in get_registry().lineage("Committee of Union and Progress")] == [
            "committee_union_progress",
            "young_turks",
            "ottoman",
            "islamicate",
        ]

    def test_corpus_promotions_preserve_dynastic_subscopes(self):
        assert [scope.id for scope in get_registry().lineage("Sung dynasty")] == [
            "song_dynasty",
            "china",
            "sinosphere",
        ]
        assert [scope.id for scope in get_registry().lineage("Kwaresmian Empire")] == [
            "khwarezmian_empire",
            "central_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Jelal ud din")] == [
            "jalal_al_din_mingburnu",
            "khwarezmian_empire",
            "central_asia",
        ]
        assert [scope.id for scope in get_registry().lineage("Sassanid Persia")] == [
            "sassanid_empire",
            "persia",
        ]

    def test_indigenous_subgroups_are_nested_without_flattening(self):
        assert [scope.id for scope in get_registry().lineage("Oglala Sioux")] == [
            "oglala",
            "lakota",
            "sioux",
            "indigenous_north_america",
            "north_america",
        ]
        assert [scope.id for scope in get_registry().lineage("Northern Cheyenne")] == [
            "northern_cheyenne",
            "cheyenne",
            "indigenous_north_america",
            "north_america",
        ]
        assert [scope.id for scope in get_registry().lineage("Sioux nation")] == [
            "sioux",
            "indigenous_north_america",
            "north_america",
        ]
        assert [scope.id for scope in get_registry().lineage("King Movement")] == [
            "maori_king_movement",
            "maori",
            "new_zealand",
            "oceania",
        ]
        assert [scope.id for scope in get_registry().lineage("Mexican pantheon")][:2] == [
            "nahua_religion",
            "mesoamerica",
        ]

    def test_unknown_returns_none_never_guesses(self):
        # No fuzzy matching: a wrong scope silently poisons the composition
        # partition; an unresolved one falls to the visible singleton path.
        assert resolve_scope("Atlantis") is None
        assert resolve_scope("Uni") is None
        assert resolve_scope("") is None
        assert resolve_scope(None) is None

    def test_each_unresolved_scope_is_logged_once(self, monkeypatch):
        registry = ScopeRegistry(
            version="test-v1",
            scopes=[Scope(id="known", kind="polity", name="Known")],
        )
        calls = []
        monkeypatch.setattr(
            "narrative_engine.scopes.logger.info",
            lambda event, **fields: calls.append((event, fields)),
        )

        assert registry.resolve("Atlantis") is None
        assert registry.resolve("ATLANTIS!") is None

        assert calls == [("scope_unresolved", {"raw": "Atlantis"})]

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
        assert merged.scope_id == "us"
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
