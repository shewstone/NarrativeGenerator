"""Embedding epoch tests (T4, docs/tickets/T4-embedding-render-versioning.md).

Stale vectors must be invisible, not silently wrong: retrieval filters to
the current structural epoch, composition treats epoch mismatch as a
missing signal, and the re-embed job restores staleness to zero
idempotently.
"""


import pytest

from narrative_engine.composition.identity import ArcIdentityResolver
from narrative_engine.models import Episode
from narrative_engine.retrieval.epochs import (
    STRUCTURAL_EMBEDDING_EPOCH,
    SURFACE_EMBEDDING_EPOCH,
    current_epoch,
)
from narrative_engine.retrieval.reembed import reembed_stale, stale_fraction
from narrative_engine.storage.repositories import EpisodeRepository


def _episode(**kwargs) -> Episode:
    defaults = {"title": "Test", "summary": "Test episode"}
    defaults.update(kwargs)
    return Episode(**defaults)


class FakeEmbedder:
    """Deterministic stand-in for EmbeddingGenerator (no model download)."""

    def generate_surface_embedding(self, episode):
        return [0.5] * 384

    def generate_structural_embedding(self, episode):
        return [0.25] * 384


class TestEpochStamping:
    @pytest.mark.asyncio
    async def test_update_embedding_stamps_current_epoch(self, db_session):
        repo = EpisodeRepository(db_session)
        episode = _episode()
        await repo.create(episode)

        await repo.update_embedding(episode.id, [0.1] * 384, kind="structural")
        await repo.update_embedding(episode.id, [0.2] * 384, kind="surface")

        fetched = await repo.get_by_id(episode.id)
        assert fetched.structural_embedding_epoch == STRUCTURAL_EMBEDDING_EPOCH
        assert fetched.surface_embedding_epoch == SURFACE_EMBEDDING_EPOCH

    def test_current_epoch_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            current_epoch("sideways")


class TestRetrievalEpochFilter:
    @pytest.mark.asyncio
    async def test_stale_vectors_are_invisible(self, db_session):
        repo = EpisodeRepository(db_session)

        current = _episode(title="Current epoch")
        await repo.create(current)
        await repo.update_embedding(current.id, [0.1] * 384, kind="structural")

        # Stale: vector present, pre-v0.7 epoch (as the backfill migration
        # leaves them). Persisted via create, which passes epochs through.
        stale = _episode(
            title="Stale epoch",
            structural_embedding=[0.1] * 384,
            structural_embedding_epoch="render-v0.6.0+all-MiniLM-L6-v2",
        )
        await repo.create(stale)

        # NULL-epoch vector (never stamped) must be invisible too.
        unstamped = _episode(title="Unstamped", structural_embedding=[0.1] * 384)
        await repo.create(unstamped)

        results = await repo.search_by_embedding([0.1] * 384, limit=10)
        ids = {episode.id for episode, _ in results}

        assert current.id in ids
        assert stale.id not in ids
        assert unstamped.id not in ids


class TestCompositionEpochDiscipline:
    def test_epoch_mismatch_is_neutral_not_similar(self):
        resolver = ArcIdentityResolver()
        a = _episode(
            surface_embedding=[1.0] * 384,
            surface_embedding_epoch="all-MiniLM-L6-v2",
        )
        # Identical vector but different epoch: the 1.0 cosine similarity is
        # meaningless and must NOT read as evidence of identity.
        b = _episode(
            surface_embedding=[1.0] * 384,
            surface_embedding_epoch="some-other-model",
        )
        assert resolver._calculate_surface_embedding_similarity(a, b) == 0.5

        b_same = _episode(
            surface_embedding=[1.0] * 384,
            surface_embedding_epoch="all-MiniLM-L6-v2",
        )
        assert resolver._calculate_surface_embedding_similarity(a, b_same) > 0.9

    def test_epoch_mismatch_does_not_count_toward_evidence_floor(self):
        resolver = ArcIdentityResolver(min_evidence_signals=2)
        base = {
            "scope_id": "us",
            "arc_type": None,
            "arc_phase": None,
            "surface_embedding": [1.0] * 384,
        }
        # Only two potentially-concrete signals here: surface + actors=none,
        # dates=none, phases=none. With matching epochs surface counts (1
        # signal -> still below floor); the point is mismatch must not count
        # MORE than match. Compare the reported evidence via is_match paths.
        a = _episode(**base, surface_embedding_epoch="all-MiniLM-L6-v2")
        b_mismatch = _episode(**base, surface_embedding_epoch="other")
        b_match = _episode(**base, surface_embedding_epoch="all-MiniLM-L6-v2")

        score_mismatch = resolver.calculate_identity_score(a, b_mismatch)
        score_match = resolver.calculate_identity_score(a, b_match)

        floor_msg = "Insufficient identity evidence"
        assert any(floor_msg in r for r in score_mismatch.mismatch_reasons)
        # Sanity: with matching epochs the surface signal is concrete, so the
        # evidence count is strictly higher (visible as 1 vs 0 in the message).
        mismatch_msg = next(r for r in score_mismatch.mismatch_reasons if floor_msg in r)
        match_msgs = [r for r in score_match.mismatch_reasons if floor_msg in r]
        assert "(0 concrete" in mismatch_msg
        assert match_msgs and "(1 concrete" in match_msgs[0]


class TestReembedJob:
    @pytest.mark.asyncio
    async def test_reembed_restamps_and_is_idempotent(self, db_session):
        repo = EpisodeRepository(db_session)

        stale = _episode(
            title="Pre-v0.7",
            structural_embedding=[0.9] * 384,
            structural_embedding_epoch="render-v0.6.0+all-MiniLM-L6-v2",
            surface_embedding=[0.9] * 384,
            surface_embedding_epoch="all-MiniLM-L6-v2",
        )
        never_embedded = _episode(title="No vectors yet")
        await repo.create(stale)
        await repo.create(never_embedded)

        assert await stale_fraction(db_session) == 1.0

        result = await reembed_stale(db_session, embedder=FakeEmbedder())
        assert result["updated"] == 2
        assert await stale_fraction(db_session) == 0.0

        # Both now retrievable at the current epoch.
        results = await repo.search_by_embedding([0.25] * 384, limit=10)
        assert {e.id for e, _ in results} == {stale.id, never_embedded.id}

        # Idempotent: second run touches nothing.
        again = await reembed_stale(db_session, embedder=FakeEmbedder())
        assert again == {"stale": 0, "updated": 0}

    @pytest.mark.asyncio
    async def test_dry_run_counts_without_writing(self, db_session):
        repo = EpisodeRepository(db_session)
        await repo.create(_episode(title="No vectors"))

        result = await reembed_stale(db_session, embedder=FakeEmbedder(), dry_run=True)
        assert result == {"stale": 1, "updated": 0}
        assert await stale_fraction(db_session) == 1.0
