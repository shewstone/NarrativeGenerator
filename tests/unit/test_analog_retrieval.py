"""Unit tests for analog retrieval scoring helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from narrative_engine.models import ArcPhase, CycleScale, Episode, MechanismTag
from narrative_engine.retrieval.analog_retrieval import AnalogRetrievalEngine


class TestMechanismOverlap:
    """Tests for mechanism-tag overlap scoring (design doc Sec 3.8)."""

    def test_no_tags_is_neutral(self):
        engine = AnalogRetrievalEngine()
        assert engine._compute_mechanism_overlap([], []) == 0.5
        assert engine._compute_mechanism_overlap([MechanismTag.CREDIT_EXPANSION], []) == 0.5

    def test_identical_tags_score_one(self):
        engine = AnalogRetrievalEngine()
        tags = [MechanismTag.CREDIT_EXPANSION, MechanismTag.ASSET_BUBBLE]
        assert engine._compute_mechanism_overlap(tags, tags) == 1.0

    def test_partial_overlap_is_jaccard(self):
        engine = AnalogRetrievalEngine()
        query = [MechanismTag.CREDIT_EXPANSION, MechanismTag.ASSET_BUBBLE]
        candidate = [MechanismTag.CREDIT_EXPANSION, MechanismTag.FISCAL_DISTRESS]
        # shared={CREDIT_EXPANSION}, union has 3 -> 1/3
        assert engine._compute_mechanism_overlap(query, candidate) == 1 / 3

    def test_disjoint_tags_score_zero(self):
        engine = AnalogRetrievalEngine()
        query = [MechanismTag.CREDIT_EXPANSION]
        candidate = [MechanismTag.FISCAL_DISTRESS]
        assert engine._compute_mechanism_overlap(query, candidate) == 0.0


class TestCycleContext:
    @pytest.mark.asyncio
    async def test_matching_scope_scale_and_phase_scores_one(self):
        engine = AnalogRetrievalEngine()
        query = Episode(title="Query", summary="Q")
        candidate = Episode(title="Candidate", summary="C")
        cycle = SimpleNamespace(
            id="cycle-1",
            scope_id="us",
            scale=CycleScale.INSTITUTIONAL,
            phase_estimate=ArcPhase.DISTRESS,
        )
        result = MagicMock()
        result.all.return_value = [(cycle, query.id), (cycle, candidate.id)]
        session = AsyncMock()
        session.execute.return_value = result

        score = await engine._compute_cycle_context_score(query, candidate, session)

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_different_cycle_context_scores_zero(self):
        engine = AnalogRetrievalEngine()
        query = Episode(title="Query", summary="Q")
        candidate = Episode(title="Candidate", summary="C")
        result = MagicMock()
        result.all.return_value = [
            (
                SimpleNamespace(
                    id="a",
                    scope_id="us",
                    scale=CycleScale.INSTITUTIONAL,
                    phase_estimate=ArcPhase.DISTRESS,
                ),
                query.id,
            ),
            (
                SimpleNamespace(
                    id="b",
                    scope_id="china",
                    scale=CycleScale.CIVILIZATIONAL,
                    phase_estimate=ArcPhase.BOOM,
                ),
                candidate.id,
            ),
        ]
        session = AsyncMock()
        session.execute.return_value = result

        score = await engine._compute_cycle_context_score(query, candidate, session)

        assert score == 0.0
