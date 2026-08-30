"""Evidence thresholds for promoting an extracted arc candidate.

Composition intentionally retains singleton clusters so no classified episode
is silently discarded.  Retention is not the same as evidentiary maturity:
only clusters with enough observations and narrative movement are presented as
formed arcs in the API and dashboard.
"""

from __future__ import annotations

from typing import Final

FORMED_ARC_MIN_EPISODES: Final = 3
FORMED_ARC_MIN_PHASES: Final = 2


def arc_formation_status(episode_count: int, phase_count: int) -> str:
    """Return ``formed`` only when both evidence thresholds are met."""
    if episode_count >= FORMED_ARC_MIN_EPISODES and phase_count >= FORMED_ARC_MIN_PHASES:
        return "formed"
    return "candidate"


def arc_formation_gaps(episode_count: int, phase_count: int) -> list[str]:
    """Explain what evidence a candidate still needs to become formed."""
    gaps = []
    if episode_count < FORMED_ARC_MIN_EPISODES:
        gaps.append(f"needs {FORMED_ARC_MIN_EPISODES - episode_count} more episode(s)")
    if phase_count < FORMED_ARC_MIN_PHASES:
        gaps.append(f"needs {FORMED_ARC_MIN_PHASES - phase_count} more distinct phase(s)")
    return gaps
