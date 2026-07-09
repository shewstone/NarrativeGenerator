"""Evaluation and backtesting framework.

Brier scores, calibration, and masked-ending tests.
"""

from narrative_engine.evaluation.backtest import (
    BacktestEngine,
    HistoricalDataset,
)
from narrative_engine.evaluation.baselines import (
    BareLLMBaseline,
    BaselinePrediction,
    PersistenceBaseline,
)
from narrative_engine.evaluation.masking import (
    mask_corpus_at,
    mask_episode_at,
)
from narrative_engine.evaluation.metrics import (
    BrierScore,
    CalibrationAnalyzer,
)

__all__ = [
    "BacktestEngine",
    "HistoricalDataset",
    "BareLLMBaseline",
    "BaselinePrediction",
    "PersistenceBaseline",
    "mask_corpus_at",
    "mask_episode_at",
    "BrierScore",
    "CalibrationAnalyzer",
]
