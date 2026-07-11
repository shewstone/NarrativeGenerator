"""Masked-ending backtest harness (T1; design doc Sec 6.6).

Assembles the loop around the data-layer masking primitives:

    snapshot corpus at cutoff -> retrieve analogs -> generate theses ->
    score against ground truth -> compare against baselines.

Test cases are episodes that SPAN the cutoff (started before, resolved
after): at the cutoff they look exactly like a present-day query
(resolution=None by masking), but the true continuation is known and
scoreable.

Leakage canaries are enforced HERE as hard errors, not left to test code:
a backtest that silently read post-cutoff data is worse than no backtest,
because it produces a confident wrong number (Sec 6.6: "leakage remains
the single biggest threat to believing your own results").

Known residual leakage, stated honestly: masking controls what the SYSTEM
sees, not what the LLM remembers, and historians' prose was written with
hindsight. Score analog-selection quality separately from narrative
plausibility; prefer obscure test cases and post-training-cutoff events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from narrative_engine.evaluation.backtest import BacktestEngine
from narrative_engine.evaluation.baselines import PersistenceBaseline
from narrative_engine.evaluation.masking import mask_corpus_at, mask_episode_at
from narrative_engine.evaluation.metrics import BrierScore
from narrative_engine.logging_config import get_logger
from narrative_engine.models import Episode
from narrative_engine.retrieval.analog_retrieval import AnalogRetrievalEngine
from narrative_engine.retrieval.reembed import stale_fraction
from narrative_engine.storage.repositories import EpisodeRepository
from narrative_engine.thesis.generator import ThesisGenerator

logger = get_logger(__name__)


class LeakageError(RuntimeError):
    """Post-cutoff information reached the forecasting path."""


@dataclass
class CaseResult:
    episode_id: str
    title: str
    mode: str
    status: str  # "scored" | "no_forecast"
    predicted: Optional[str]
    actual: str
    brier: Optional[float]
    accuracy: Optional[str]
    persistence_brier: float


@dataclass
class HarnessReport:
    cutoff: datetime
    corpus_size: int
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def scored(self) -> List[CaseResult]:
        return [c for c in self.cases if c.status == "scored"]

    def summary(self) -> dict:
        scored = self.scored
        thesis_briers = [c.brier for c in scored]
        persistence_briers = [c.persistence_brier for c in self.cases]
        mean_thesis = sum(thesis_briers) / len(thesis_briers) if thesis_briers else None
        mean_persistence = sum(persistence_briers) / len(persistence_briers) if persistence_briers else None
        skill = None
        if mean_thesis is not None and mean_persistence:
            skill = 1.0 - (mean_thesis / mean_persistence)
        return {
            "cutoff": self.cutoff.isoformat(),
            "corpus_size": self.corpus_size,
            "cases": len(self.cases),
            "scored": len(scored),
            "no_forecast": len(self.cases) - len(scored),
            "mean_thesis_brier": mean_thesis,
            "mean_persistence_brier": mean_persistence,
            # >0: the machinery beats "things continue". <=0: the structure
            # is not paying rent (Sec 6.6).
            "skill_vs_persistence": skill,
        }


def _assert_no_leakage(analogs, cutoff: datetime) -> None:
    """Hard canary: every analog must be knowable at the cutoff."""
    for analog in analogs:
        episode = analog.episode
        if episode.start_date is not None and episode.start_date > cutoff:
            raise LeakageError(f"Analog {episode.title!r} starts after cutoff {cutoff:%Y-%m-%d}")
        if episode.resolution is not None and (episode.end_date is None or episode.end_date > cutoff):
            raise LeakageError(
                f"Analog {episode.title!r} carries a resolution not knowable " f"at cutoff {cutoff:%Y-%m-%d}"
            )


async def load_masked_corpus(
    session: AsyncSession,
    corpus: Sequence[Episode],
    cutoff: datetime,
    embedder,
) -> int:
    """Snapshot the corpus at `cutoff` into the session's database.

    Post-cutoff episodes are dropped and ongoing episodes stripped of
    outcomes BEFORE anything touches the DB, so downstream code physically
    cannot read them (Sec 6.6: data layer, not prompt layer).
    """
    masked = mask_corpus_at(corpus, cutoff)
    repo = EpisodeRepository(session)
    for episode in masked:
        await repo.create(episode)
        await repo.update_embedding(episode.id, embedder.generate_surface_embedding(episode), kind="surface")
        await repo.update_embedding(
            episode.id,
            embedder.generate_structural_embedding(episode),
            kind="structural",
        )
    return len(masked)


async def run_backtest(
    session: AsyncSession,
    corpus: Sequence[Episode],
    cutoff: datetime,
    embedder=None,
    k: int = 10,
    min_analogs: int = 3,
) -> HarnessReport:
    """Run the full masked-ending loop over one corpus snapshot.

    `corpus` is the UNMASKED ground-truth corpus; this function performs
    the masking itself so no caller can accidentally hand the forecasting
    path an unmasked snapshot.
    """
    if embedder is None:
        from narrative_engine.retrieval.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator()

    corpus_size = await load_masked_corpus(session, corpus, cutoff, embedder)

    # A backtest over mixed embedding epochs is uninterpretable (T4).
    fraction = await stale_fraction(session)
    if fraction > 0:
        raise LeakageError(
            f"Corpus contains {fraction:.1%} stale-epoch embeddings; re-embed " "before running a backtest"
        )

    # Test cases: episodes spanning the cutoff with a known true outcome.
    test_cases = [
        e
        for e in corpus
        if e.start_date is not None
        and e.start_date <= cutoff
        and e.resolution is not None
        and e.end_date is not None
        and e.end_date > cutoff
    ]
    logger.info(
        "harness_snapshot",
        cutoff=str(cutoff.date()),
        corpus_size=corpus_size,
        test_cases=len(test_cases),
    )

    retrieval = AnalogRetrievalEngine(embedding_generator=embedder)
    generator = ThesisGenerator(min_analogs=min_analogs)
    scorer = BacktestEngine()
    persistence = PersistenceBaseline()

    report = HarnessReport(cutoff=cutoff, corpus_size=corpus_size)

    for original in test_cases:
        masked_query = mask_episode_at(original, cutoff)
        assert masked_query is not None and masked_query.resolution is None

        analogs = await retrieval.retrieve_analogs(masked_query, session, k=k)
        _assert_no_leakage(analogs, cutoff)

        thesis = await generator.generate_async(masked_query, analogs)

        # Persistence baseline, scored with the same matching rule.
        baseline = persistence.predict(masked_query)
        matched, _ = scorer._match_outcome(original.resolution, [baseline.predicted_continuation])
        persistence_brier = BrierScore.calculate(baseline.probability, 1 if matched else 0).score

        if thesis.dominant_continuation is None:
            # Visible degraded outcome, never silently skipped (Sec 6.5.8).
            report.cases.append(
                CaseResult(
                    episode_id=str(original.id),
                    title=original.title,
                    mode=thesis.mode.value,
                    status="no_forecast",
                    predicted=None,
                    actual=original.resolution,
                    brier=None,
                    accuracy=None,
                    persistence_brier=persistence_brier,
                )
            )
            continue

        scored = scorer.run_backtest(
            thesis=thesis,
            actual_outcome=original.resolution,
            resolution_date=original.end_date,
        )
        report.cases.append(
            CaseResult(
                episode_id=str(original.id),
                title=original.title,
                mode=thesis.mode.value,
                status="scored",
                predicted=scored.predicted_continuation,
                actual=original.resolution,
                brier=scored.brier_score,
                accuracy=scored.accuracy,
                persistence_brier=persistence_brier,
            )
        )

    logger.info("harness_complete", **report.summary())
    return report
