"""Per-scale temporal threshold sweep against the composition fixture (T1).

Design doc Sec 6.2 stage 6: the per-scale gap thresholds are
"fixture-tuned hypotheses, not constants... thresholds are set by fixture
performance, not intuition." This script grid-searches the episodic-scale
gap threshold and prints the precision/recall frontier so the value in
ArcIdentityResolver is justified by data.

Usage:
    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from composition_fixture import CompositionFixture  # noqa: E402

from narrative_engine.composition import compose_arc_instances_from_episodes  # noqa: E402
from narrative_engine.composition.identity import ArcIdentityResolver  # noqa: E402

# Candidate episodic-scale gap thresholds, in years.
EPISODIC_GAP_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def evaluate(gap_years: float) -> dict:
    from datetime import timedelta

    resolver = ArcIdentityResolver()
    # Override the episodic threshold for this sweep point (the resolver
    # stores per-scale thresholds as timedeltas).
    resolver.temporal_thresholds["episodic"] = timedelta(days=gap_years * 365)

    fixture = CompositionFixture()
    positives_ok = 0
    positives = fixture.get_positive_cases()
    for episodes, _ in positives:
        instances = compose_arc_instances_from_episodes(
            episodes, arc_type=episodes[0].arc_type, resolver=resolver
        )
        if len(instances) == 1:
            positives_ok += 1

    negatives_ok = 0
    negatives = fixture.get_negative_cases()
    for episodes, _, _ in negatives:
        instances = compose_arc_instances_from_episodes(
            episodes, arc_type=episodes[0].arc_type, resolver=resolver
        )
        if len(instances) >= 2:
            negatives_ok += 1

    return {
        "gap_years": gap_years,
        "positive_recall": positives_ok / len(positives),
        "negative_precision": negatives_ok / len(negatives),
    }


def main() -> None:
    print(f"{'gap (y)':>8} | {'positive recall':>16} | {'decoy rejection':>16}")
    print("-" * 48)
    frontier = []
    for gap in EPISODIC_GAP_CANDIDATES:
        result = evaluate(gap)
        frontier.append(result)
        print(
            f"{result['gap_years']:>8} | {result['positive_recall']:>16.2f} "
            f"| {result['negative_precision']:>16.2f}"
        )

    perfect = [r for r in frontier if r["positive_recall"] == 1.0 and r["negative_precision"] == 1.0]
    if perfect:
        lo, hi = perfect[0]["gap_years"], perfect[-1]["gap_years"]
        print(f"\nBoth-perfect range: {lo}-{hi} years.")
        print("The production default should sit inside this range and cite this sweep.")
    else:
        print("\nNo threshold satisfies both criteria: the fixture has found a real")
        print("tension -- grow the fixture or revisit the non-temporal gates.")


if __name__ == "__main__":
    main()
