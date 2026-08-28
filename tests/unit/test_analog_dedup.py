"""Analog dedup tests (T6, docs/tickets/T6-analog-dedup.md).

Branch frequencies are counts over analogs; duplicate narrations of one
happening must not count as independent evidence.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from narrative_engine.models import (
    EdgeKind,
    Episode,
    EpisodeLink,
    LinkStatus,
)
from narrative_engine.retrieval.analog_retrieval import (
    AnalogRetrievalEngine,
    RetrievedAnalog,
)
from narrative_engine.storage.repositories import (
    EpisodeLinkRepository,
    EpisodeRepository,
)
from narrative_engine.thesis.generator import ThesisGenerator

UTC = timezone.utc


def _episode(**kwargs) -> Episode:
    defaults = {"title": "Analog", "summary": "Analog summary"}
    defaults.update(kwargs)
    return Episode(**defaults)


def _analog(episode: Episode, score: float) -> RetrievedAnalog:
    return RetrievedAnalog(
        episode=episode,
        semantic_similarity=score,
        arc_match_score=1.0,
        phase_compatibility=0.8,
        cycle_context_score=0.5,
        mechanism_match_score=0.5,
        combined_score=score,
        retrieval_method="vector",
        reasoning="test",
    )


class TestSameEventCollapse:
    @pytest.mark.asyncio
    async def test_linked_episodes_collapse_to_best_scored(self, db_session):
        engine = AnalogRetrievalEngine.__new__(AnalogRetrievalEngine)
        engine.same_event_similarity_threshold = 0.85
        engine.embedding_generator = None  # heuristic path not exercised here
        import structlog

        engine.logger = structlog.get_logger()

        repo = EpisodeRepository(db_session)
        kindleberger_1929 = _episode(title="1929 per Kindleberger")
        galbraith_1929 = _episode(title="1929 per Galbraith")
        distinct = _episode(title="1907 Panic")
        for e in (kindleberger_1929, galbraith_1929, distinct):
            await repo.create(e)

        await EpisodeLinkRepository(db_session).create(
            EpisodeLink(
                source_episode_id=kindleberger_1929.id,
                target_episode_id=galbraith_1929.id,
                edge_kind=EdgeKind.SAME_EVENT_AS,
                link_status=LinkStatus.INFERRED,
                evidence="surface similarity 0.93",
            )
        )

        analogs = [
            _analog(kindleberger_1929, 0.9),
            _analog(galbraith_1929, 0.8),
            _analog(distinct, 0.7),
        ]

        deduped = await engine._collapse_same_event_duplicates(analogs, db_session)

        ids = {a.episode.id for a in deduped}
        assert ids == {kindleberger_1929.id, distinct.id}
        representative = next(a for a in deduped if a.episode.id == kindleberger_1929.id)
        assert representative.duplicate_ids == [galbraith_1929.id]


class TestSameEventHeuristic:
    def _engine(self):
        # Real EmbeddingGenerator instance is lazy: similarity() only uses
        # numpy, so no model download happens here.
        return AnalogRetrievalEngine()

    def _pair(self, **overrides):
        base = {
            "scope_id": "US",
            "arc_type": "credit_boom_and_bust",
            "start_date": datetime(1929, 9, 1, tzinfo=UTC),
            "end_date": datetime(1929, 12, 1, tzinfo=UTC),
            "surface_embedding": [1.0] * 8,
            "surface_embedding_epoch": "all-MiniLM-L6-v2",
        }
        a = _episode(title="A", **base)
        b_fields = dict(base)
        b_fields.update(overrides)
        b = _episode(title="B", **b_fields)
        return a, b

    def test_full_agreement_merges(self):
        a, b = self._pair(scope_id="United States")  # alias resolves to same scope
        assert self._engine()._same_event_heuristic(a, b) is True

    def test_near_miss_decoy_does_not_merge(self):
        # 1907 panic vs 1929 crash: same scope, same arc type, disjoint spans.
        a, b = self._pair(
            start_date=datetime(1907, 10, 1, tzinfo=UTC),
            end_date=datetime(1907, 12, 1, tzinfo=UTC),
        )
        assert self._engine()._same_event_heuristic(a, b) is False

    def test_missing_data_never_merges(self):
        for missing in ("scope_id", "arc_type", "start_date", "surface_embedding"):
            a, b = self._pair(**{missing: None})
            assert self._engine()._same_event_heuristic(a, b) is False, missing

    def test_epoch_mismatch_never_merges(self):
        a, b = self._pair(surface_embedding_epoch="some-other-model")
        assert self._engine()._same_event_heuristic(a, b) is False

    def test_low_similarity_never_merges(self):
        a, b = self._pair(surface_embedding=[1.0] * 4 + [-1.0] * 4)
        assert self._engine()._same_event_heuristic(a, b) is False


class TestThesisSourceCapAndDisclosure:
    def _resolved_analog(self, source: str, score: float = 0.9) -> RetrievedAnalog:
        episode = _episode(
            title=f"Episode {uuid4().hex[:6]}",
            resolution="Market crashed, then recovered",
            end_date=datetime(1930, 1, 1, tzinfo=UTC),
            extracted_from=[source],
        )
        return _analog(episode, score)

    def test_single_source_cannot_dominate(self):
        generator = ThesisGenerator(min_analogs=1, max_per_source=3)
        analogs = [self._resolved_analog("one-big-book") for _ in range(6)] + [
            self._resolved_analog("other-book")
        ]

        filtered = generator._filter_analogs(analogs)

        from_big_book = [
            a for a in filtered if a.episode.extracted_from == ["one-big-book"]
        ]
        assert len(from_big_book) == 3
        assert len(filtered) == 4

    def test_collapsed_duplicates_are_disclosed(self):
        generator = ThesisGenerator(min_analogs=1)
        analogs = [self._resolved_analog(f"book-{i}") for i in range(3)]
        analogs[0].duplicate_ids = [uuid4(), uuid4()]

        query = _episode(title="Query", arc_type="credit_boom_and_bust")
        thesis = generator.generate(query, analogs)

        assert any("duplicate narration" in u for u in thesis.key_uncertainties)

    def test_no_disclosure_when_nothing_collapsed(self):
        generator = ThesisGenerator(min_analogs=1)
        analogs = [self._resolved_analog(f"book-{i}") for i in range(3)]

        query = _episode(title="Query", arc_type="credit_boom_and_bust")
        thesis = generator.generate(query, analogs)

        assert not any("duplicate narration" in u for u in thesis.key_uncertainties)
