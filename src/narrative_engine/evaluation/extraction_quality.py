"""Small, source-span-based benchmark for the live extraction pipeline.

The analog and composition fixtures test downstream machinery. This module
measures the upstream questions those fixtures assume: were the events found,
were their boundaries source-backed, and were dates/scopes normalized?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class GoldEpisode:
    start_char: int
    end_char: int
    scope_id: Optional[str]
    start_year: Optional[int]


@dataclass(frozen=True)
class PredictedEpisode:
    start_char: int
    end_char: int
    scope_id: Optional[str]
    start_year: Optional[int]


@dataclass(frozen=True)
class ExtractionScore:
    expected_events: int
    predicted_events: int
    matched_events: int
    event_precision: float
    event_recall: float
    mean_boundary_iou: float
    scope_accuracy: float
    date_accuracy: float


def span_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    """Intersection-over-union for half-open character spans."""
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def score_extraction(
    expected: Sequence[GoldEpisode],
    predicted: Sequence[PredictedEpisode],
    *,
    match_iou: float = 0.35,
) -> ExtractionScore:
    """Greedily match source spans, then score normalized fields."""
    candidates = sorted(
        (
            (
                span_iou(
                    (gold.start_char, gold.end_char),
                    (guess.start_char, guess.end_char),
                ),
                gold_index,
                guess_index,
            )
            for gold_index, gold in enumerate(expected)
            for guess_index, guess in enumerate(predicted)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, gold_index, guess_index in candidates:
        if overlap < match_iou:
            break
        if gold_index in used_gold or guess_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(guess_index)
        matches.append((gold_index, guess_index, overlap))

    matched = len(matches)
    scope_pairs = [
        (expected[gold_index].scope_id, predicted[guess_index].scope_id)
        for gold_index, guess_index, _ in matches
        if expected[gold_index].scope_id is not None
    ]
    date_pairs = [
        (expected[gold_index].start_year, predicted[guess_index].start_year)
        for gold_index, guess_index, _ in matches
        if expected[gold_index].start_year is not None
    ]
    return ExtractionScore(
        expected_events=len(expected),
        predicted_events=len(predicted),
        matched_events=matched,
        event_precision=matched / len(predicted) if predicted else 0.0,
        event_recall=matched / len(expected) if expected else 0.0,
        mean_boundary_iou=(
            sum(overlap for _, _, overlap in matches) / matched if matched else 0.0
        ),
        scope_accuracy=(
            sum(expected_id == predicted_id for expected_id, predicted_id in scope_pairs)
            / len(scope_pairs)
            if scope_pairs
            else 0.0
        ),
        date_accuracy=(
            sum(expected_year == predicted_year for expected_year, predicted_year in date_pairs)
            / len(date_pairs)
            if date_pairs
            else 0.0
        ),
    )
