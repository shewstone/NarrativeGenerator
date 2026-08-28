"""Composition fixture for validating identity resolution thresholds.

Contains hand-built Arc Instances that the composition pass MUST recover,
plus deliberate near-miss decoys that it must NOT merge.

Use this to tune thresholds BEFORE deploying composition pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple
from uuid import uuid4

from narrative_engine.models import (
    ArcPhase,
    ArcType,
    Episode,
)


class CompositionFixture:
    """Test fixture for Arc Instance composition validation.

    Each case includes:
    - Input episodes (from various "sources")
    - Expected: Should they merge into one Arc Instance?
    - Rationale: Why they should or shouldn't merge

    PASS criteria:
    - Recover all POSITIVE cases (merge correctly)
    - Reject all NEGATIVE cases (don't false-merge)
    - Tune thresholds until both criteria met
    """

    @staticmethod
    def get_positive_cases() -> List[Tuple[List[Episode], str]]:
        """Cases that SHOULD merge into single Arc Instance."""
        cases = []

        # Case 1: 1929 Crash across two books
        case_1929 = [
            Episode(
                id=uuid4(),
                title="1920s Economic Expansion",
                summary="Post-war boom and stock market growth",
                start_date=datetime(1922, 1, 1),
                end_date=datetime(1927, 9, 1),
                location="United States",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.BOOM,
                extracted_from=["galbraith-1929.txt"],
            ),
            Episode(
                id=uuid4(),
                title="Speculative Euphoria",
                summary="Margin trading and irrational exuberance",
                start_date=datetime(1927, 9, 1),
                end_date=datetime(1929, 9, 1),
                location="New York",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.EUPHORIA,
                extracted_from=["galbraith-1929.txt"],
            ),
            Episode(
                id=uuid4(),
                title="Black Tuesday and Crash",
                summary="Market panic and initial collapse",
                start_date=datetime(1929, 10, 24),
                end_date=datetime(1929, 10, 29),
                location="Wall Street",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.PANIC,
                extracted_from=["kindleberger-mania.txt"],
            ),
        ]
        cases.append((case_1929, "1929 Crash: multi-book, continuous arc"))

        def _ep(title, start, end, scope, phase, source, arc=ArcType.CREDIT_BOOM_AND_BUST):
            return Episode(
                id=uuid4(),
                title=title,
                summary=f"{title} (fixture)",
                start_date=start,
                end_date=end,
                scope_id=scope,
                arc_type=arc,
                arc_phase=phase,
                extracted_from=[source],
            )

        # Case 2: Weimar hyperinflation, two books covering different phases
        cases.append((
            [
                _ep("Reparations Squeeze", datetime(1921, 5, 1), datetime(1922, 6, 1),
                    "germany", ArcPhase.DISTRESS, "fergusson-money.txt"),
                _ep("Mark Collapse Accelerates", datetime(1922, 7, 1), datetime(1923, 8, 1),
                    "germany", ArcPhase.PANIC, "fergusson-money.txt"),
                _ep("Currency Death and Rentenmark", datetime(1923, 9, 1), datetime(1924, 1, 1),
                    "germany", ArcPhase.REVULSION, "taylor-weimar.txt"),
            ],
            "Weimar hyperinflation: phases split across two books",
        ))

        # Case 3: South Sea Bubble, three sources
        cases.append((
            [
                _ep("Debt Conversion Scheme Launched", datetime(1719, 9, 1), datetime(1720, 1, 1),
                    "uk", ArcPhase.BOOM, "carswell-southsea.txt"),
                _ep("Subscription Frenzy", datetime(1720, 2, 1), datetime(1720, 6, 1),
                    "uk", ArcPhase.EUPHORIA, "mackay-delusions.txt"),
                _ep("Bubble Act Backfires and Collapse", datetime(1720, 7, 1), datetime(1720, 12, 1),
                    "uk", ArcPhase.PANIC, "dale-first-crash.txt"),
            ],
            "South Sea Bubble: one arc across three sources",
        ))

        # Case 4: Japanese bubble, boom book + aftermath book
        cases.append((
            [
                _ep("Endaka Easy Money Boom", datetime(1986, 1, 1), datetime(1988, 12, 1),
                    "japan", ArcPhase.BOOM, "wood-bubble-economy.txt"),
                _ep("Land and Equity Euphoria", datetime(1989, 1, 1), datetime(1989, 12, 29),
                    "japan", ArcPhase.EUPHORIA, "wood-bubble-economy.txt"),
                _ep("BOJ Tightening and Break", datetime(1990, 1, 1), datetime(1990, 10, 1),
                    "japan", ArcPhase.DISTRESS, "koo-balance-sheet.txt"),
                _ep("Lost Decade Onset", datetime(1991, 1, 1), datetime(1992, 6, 1),
                    "japan", ArcPhase.REVULSION, "koo-balance-sheet.txt"),
            ],
            "Japan bubble: boom-side and aftermath-side books stitch",
        ))

        # Case 5: GFC across three sources
        cases.append((
            [
                _ep("Subprime Origination Machine", datetime(2005, 1, 1), datetime(2006, 12, 1),
                    "us_national", ArcPhase.BOOM, "lewis-big-short.txt"),
                _ep("Housing Peak and First Cracks", datetime(2007, 1, 1), datetime(2007, 8, 1),
                    "us_national", ArcPhase.DISTRESS, "lewis-big-short.txt"),
                _ep("Lehman Weekend and Freeze", datetime(2008, 9, 1), datetime(2008, 10, 15),
                    "us_national", ArcPhase.PANIC, "sorkin-tbtf.txt"),
                _ep("Deleveraging and TARP Aftermath", datetime(2008, 10, 16), datetime(2009, 6, 1),
                    "us_national", ArcPhase.REVULSION, "blinder-music-stopped.txt"),
            ],
            "2008 GFC: three books, four phases, one unfolding",
        ))

        # Case 6: Tulip mania, two sources
        cases.append((
            [
                _ep("Bulb Futures Frenzy", datetime(1636, 10, 1), datetime(1637, 1, 15),
                    "netherlands", ArcPhase.EUPHORIA, "mackay-delusions.txt"),
                _ep("February Collapse of the Colleges", datetime(1637, 2, 1), datetime(1637, 5, 1),
                    "netherlands", ArcPhase.PANIC, "goldgar-tulipmania.txt"),
            ],
            "Tulip mania: euphoria and collapse from different sources",
        ))

        return cases

    @staticmethod
    def get_negative_cases() -> List[Tuple[List[Episode], str, str]]:
        """Cases that should NOT merge (deliberate near-misses)."""
        cases = []

        # Case N1: 1907 Panic vs 1920s Boom (same scope, close-ish time)
        panic_1907 = [
            Episode(
                id=uuid4(),
                title="1907 Bankers' Panic",
                summary="Financial crisis",
                start_date=datetime(1907, 10, 1),
                end_date=datetime(1907, 11, 1),
                location="New York",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.PANIC,
                extracted_from=["bruner-carr-1907.txt"],
            ),
        ]

        boom_1920s = [
            Episode(
                id=uuid4(),
                title="Roaring Twenties Expansion",
                summary="Post-WWI boom",
                start_date=datetime(1922, 1, 1),
                end_date=datetime(1927, 1, 1),
                location="United States",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.BOOM,
                extracted_from=["galbraith-1929.txt"],
            ),
            Episode(
                id=uuid4(),
                title="1929 Market Crash",
                summary="Great Crash begins",
                start_date=datetime(1929, 10, 24),
                end_date=datetime(1929, 10, 29),
                location="Wall Street",
                scope_id="us_national",
                arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                arc_phase=ArcPhase.PANIC,
                extracted_from=["galbraith-1929.txt"],
            ),
        ]

        cases.append((
            panic_1907 + boom_1920s,
            "1907 Panic vs 1920s Boom: same scope, different instances",
            "temporal_overmerge"
        ))

        def _ep(title, start, end, scope, phase, source, arc=ArcType.CREDIT_BOOM_AND_BUST):
            return Episode(
                id=uuid4(),
                title=title,
                summary=f"{title} (fixture)",
                start_date=start,
                end_date=end,
                scope_id=scope,
                arc_type=arc,
                arc_phase=phase,
                extracted_from=[source],
            )

        # Case N2: dot-com vs GFC -- same scope, same arc, 7-year gap.
        cases.append((
            [
                _ep("Dot-com Peak and Crash", datetime(1999, 6, 1), datetime(2000, 10, 1),
                    "us_national", ArcPhase.PANIC, "cassidy-dotcon.txt"),
                _ep("Subprime Boom", datetime(2005, 1, 1), datetime(2007, 6, 1),
                    "us_national", ArcPhase.BOOM, "lewis-big-short.txt"),
            ],
            "Dot-com vs GFC: same scope+arc, distinct cycles years apart",
            "temporal_overmerge",
        ))

        # Case N3: cross-scope decoy -- near-identical booms, different polities.
        # The scope partition must split these BY CONSTRUCTION (Sec 6.2
        # stage 6.1); if this ever merges, the hard filter itself broke.
        cases.append((
            [
                _ep("Late-80s Credit Boom", datetime(1988, 1, 1), datetime(1989, 12, 1),
                    "japan", ArcPhase.EUPHORIA, "wood-bubble-economy.txt"),
                _ep("Late-80s Credit Boom", datetime(1988, 1, 1), datetime(1989, 12, 1),
                    "us_national", ArcPhase.EUPHORIA, "grant-money-mind.txt"),
            ],
            "Same-shaped simultaneous booms in different scopes",
            "scope_partition_breach",
        ))

        # Case N4: same year, same arc, different scopes (1873 twin panics).
        cases.append((
            [
                _ep("Vienna Exchange Crash", datetime(1873, 5, 1), datetime(1873, 11, 1),
                    "austria_hungary", ArcPhase.PANIC, "gruenderzeit-hist.txt"),
                _ep("Jay Cooke Failure Panic", datetime(1873, 9, 1), datetime(1874, 2, 1),
                    "us_national", ArcPhase.PANIC, "white-railroaded.txt"),
            ],
            "1873 twin panics: temporally identical, scope-distinct instances",
            "scope_partition_breach",
        ))

        # Case N5: phase-order decoy -- revulsion then a NEW boom within the
        # temporal threshold. Phase-sequence continuity must split them:
        # boom-after-revulsion is the next cycle, not a later beat.
        cases.append((
            [
                _ep("Post-Crash Liquidation", datetime(1932, 1, 1), datetime(1933, 3, 1),
                    "us_national", ArcPhase.REVULSION, "galbraith-1929.txt"),
                _ep("New Deal Recovery Rally", datetime(1934, 1, 1), datetime(1936, 12, 1),
                    "us_national", ArcPhase.BOOM, "ahamed-lords-finance.txt"),
            ],
            "Revulsion then next-cycle boom inside the gap threshold",
            "phase_sequence_overmerge",
        ))

        # Case N6: unscoped near-duplicates -- the v0.7 rule: unscoped
        # episodes never merge, they surface as singletons.
        cases.append((
            [
                _ep("Unattributed Credit Boom", datetime(1925, 1, 1), datetime(1926, 6, 1),
                    None, ArcPhase.BOOM, "fragment-a.txt"),
                _ep("Unattributed Credit Boom II", datetime(1926, 7, 1), datetime(1927, 6, 1),
                    None, ArcPhase.EUPHORIA, "fragment-b.txt"),
            ],
            "Unscoped episodes: singleton rule forbids merging",
            "unscoped_pooling",
        ))

        return cases


async def validate_composition_pipeline(
    pipeline_func: Callable,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run composition fixture against a pipeline implementation."""

    fixture = CompositionFixture()

    results = {
        "positive_passed": 0,
        "positive_total": 0,
        "negative_passed": 0,
        "negative_total": 0,
        "recommendations": [],
    }

    # Test positive cases (should merge)
    for episodes, description in fixture.get_positive_cases():
        results["positive_total"] += 1
        instances = await pipeline_func(episodes)

        if len(instances) == 1:
            results["positive_passed"] += 1
            if verbose:
                print(f"✅ PASS: {description}")
        else:
            if verbose:
                print(f"❌ FAIL: {description} - fragmented into {len(instances)} instances")
            results["recommendations"].append(f"Loosen threshold for: {description}")

    # Test negative cases (should NOT merge)
    for episodes, description, failure_mode in fixture.get_negative_cases():
        results["negative_total"] += 1
        instances = await pipeline_func(episodes)

        # Expect 2 separate instances
        if len(instances) >= 2:
            results["negative_passed"] += 1
            if verbose:
                print(f"✅ PASS: {description}")
        else:
            if verbose:
                print(f"❌ FAIL: {description} - over-merged into {len(instances)} instances")
            results["recommendations"].append(f"Tighten {failure_mode} threshold")

    return results
