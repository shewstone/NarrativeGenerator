"""End-to-end masked-ending harness test (T1; design doc Sec 6.6).

Proves the snapshot -> retrieve -> forecast -> score loop runs, that
masked data is physically unreachable (post-cutoff episodes never enter
the DB; ongoing episodes carry no outcomes), and that the persistence
baseline is scored alongside every thesis.
"""

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from narrative_engine.evaluation.harness import run_backtest
from narrative_engine.models import ArcPhase, ArcType, Episode
from narrative_engine.storage.orm_models import EpisodeORM

UTC = timezone.utc
CUTOFF = datetime(1930, 1, 1, tzinfo=UTC)


class DeterministicEmbedder:
    """Arc/phase-keyed vectors: same-arc episodes are near neighbors.

    No model download; retrieval quality is not under test here, the loop
    and its leakage guarantees are.
    """

    def _vector(self, seed: str):
        digest = hashlib.sha256(seed.encode()).digest()
        base = [(b - 128) / 128 for b in digest] * 12  # 384 dims
        return base[:384]

    def generate_structural_embedding(self, episode: Episode):
        key = f"{episode.arc_type}|structural"
        return self._vector(key)

    def generate_surface_embedding(self, episode: Episode):
        return self._vector(f"{episode.title}|surface")

    def similarity(self, a, b):
        import numpy as np

        va, vb = np.array(a), np.array(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _resolved_analog(i: int, year: int) -> Episode:
    return Episode(
        id=uuid4(),
        title=f"Historical credit episode {i}",
        summary="A leveraged boom collapsed into panic and contraction",
        scope_id="us",
        start_date=datetime(year, 1, 1, tzinfo=UTC),
        end_date=datetime(year + 1, 6, 1, tzinfo=UTC),
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.PANIC,
        phase_confidence=0.9,
        tension="leverage against liquidity",
        resolution="Market crashed and a severe contraction followed",
        consequences=["bank failures"],
        extracted_from=[f"book-{i}"],
        source_published_at=datetime(year + 1, 12, 31, tzinfo=UTC),
    )


def _corpus():
    analogs = [_resolved_analog(i, 1815 + i * 20) for i in range(5)]
    test_case = Episode(
        id=uuid4(),
        title="Late-twenties leveraged boom",
        summary="Margin-financed equity boom late in the cycle",
        scope_id="us",
        start_date=datetime(1928, 1, 1, tzinfo=UTC),
        end_date=datetime(1932, 7, 1, tzinfo=UTC),  # resolves AFTER cutoff
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.EUPHORIA,
        phase_confidence=0.9,
        tension="leverage against liquidity",
        resolution="Market crashed and a severe depression followed",
        consequences=["Great Depression"],
        extracted_from=["book-test"],
    )
    post_cutoff = Episode(
        id=uuid4(),
        title="Mid-thirties recovery episode",
        summary="Post-crash policy recovery",
        scope_id="us",
        start_date=datetime(1935, 1, 1, tzinfo=UTC),
        end_date=datetime(1937, 1, 1, tzinfo=UTC),
        arc_type=ArcType.CREDIT_BOOM_AND_BUST,
        arc_phase=ArcPhase.BOOM,
        resolution="Recovery stalled in a policy-induced recession",
        extracted_from=["book-late"],
    )
    return analogs, test_case, post_cutoff


class TestMaskedEndingHarness:
    @pytest.mark.asyncio
    async def test_full_loop_runs_and_scores(self, db_session):
        analogs, test_case, post_cutoff = _corpus()
        corpus = analogs + [test_case, post_cutoff]

        report = await run_backtest(
            db_session, corpus, CUTOFF, embedder=DeterministicEmbedder(), k=8
        )

        # Five publication-dated analogs. The unknown-date query and
        # post-cutoff episode never enter the strict analog snapshot.
        assert report.corpus_size == 5
        assert len(report.cases) == 1

        case = report.cases[0]
        assert case.title == "Late-twenties leveraged boom"
        assert case.status == "scored"
        assert case.brier is not None and 0.0 <= case.brier <= 1.0
        assert case.accuracy in {"accurate", "partial", "miss"}
        assert 0.0 <= case.persistence_brier <= 1.0

        summary = report.summary()
        assert summary["source_dates_required"] is True
        assert summary["scored"] == 1
        assert summary["mean_persistence_brier"] is not None
        assert summary["skill_vs_persistence"] is not None

    @pytest.mark.asyncio
    async def test_masked_data_is_physically_unreachable(self, db_session):
        analogs, test_case, post_cutoff = _corpus()
        corpus = analogs + [test_case, post_cutoff]

        await run_backtest(
            db_session, corpus, CUTOFF, embedder=DeterministicEmbedder(), k=8
        )

        # Canary 1: the post-cutoff episode never entered the database.
        count = (
            await db_session.execute(
                select(func.count(EpisodeORM.id)).where(
                    EpisodeORM.start_date > CUTOFF
                )
            )
        ).scalar()
        assert count == 0

        # Canary 2: the unknown-publication-date test case cannot enter the
        # strict analog snapshot (it is still used as the held-out query).
        row = await db_session.get(EpisodeORM, test_case.id)
        assert row is None

        # Canary 3: every resolved row in the snapshot resolved pre-cutoff.
        resolved = (
            (
                await db_session.execute(
                    select(EpisodeORM).where(EpisodeORM.resolution.is_not(None))
                )
            )
            .scalars()
            .all()
        )
        assert resolved, "expected pre-cutoff resolved analogs in the snapshot"
        assert all(r.end_date is not None and r.end_date <= CUTOFF for r in resolved)
