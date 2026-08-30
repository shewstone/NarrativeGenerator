from narrative_engine.evaluation.extraction_quality import (
    GoldEpisode,
    PredictedEpisode,
    score_extraction,
    span_iou,
)


def test_span_iou_uses_half_open_boundaries():
    assert span_iou((0, 10), (5, 15)) == 5 / 15
    assert span_iou((0, 5), (5, 10)) == 0.0


def test_extraction_score_matches_spans_before_normalized_fields():
    expected = [
        GoldEpisode(0, 20, "china", 1911),
        GoldEpisode(20, 40, "china", 1949),
    ]
    predicted = [
        PredictedEpisode(0, 19, "china", 1911),
        PredictedEpisode(21, 40, "us", 1950),
        PredictedEpisode(50, 60, None, None),
    ]

    score = score_extraction(expected, predicted)

    assert score.matched_events == 2
    assert score.event_recall == 1.0
    assert score.event_precision == 2 / 3
    assert score.scope_accuracy == 0.5
    assert score.date_accuracy == 0.5
