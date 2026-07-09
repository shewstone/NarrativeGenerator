"""Data-layer corpus masking for masked-ending backtests.

Design doc Sec 6.6: "corpus snapshot truncated at year T (resolutions and
post-T episodes masked at the data layer, not the prompt layer)". Masking
at the prompt layer is not credible -- the model can be steered around a
prompt instruction, and downstream code can accidentally read the unmasked
fields. These helpers produce masked *copies* of episodes so everything
downstream of the snapshot (retrieval, thesis generation, scoring inputs)
physically cannot see post-cutoff information in the structured fields.

Known residual leakage, stated honestly (Sec 6.6 "leakage control"):

- Episode summaries are prose written by historians who knew the ending;
  a summary like "the boom that ended in the 1929 crash" leaks through the
  surface embedding. Data-layer masking cannot fix authorial hindsight --
  that is what pre-cutoff sources and post-training-cutoff test cases are
  for.
- The LLM itself knows how famous episodes ended. Masking controls what
  the *system* sees, not what the model remembers; score analog-selection
  quality separately from narrative plausibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from narrative_engine.models import Episode


def mask_episode_at(episode: Episode, cutoff: datetime) -> Optional[Episode]:
    """Return the episode as it would be known at `cutoff`, or None.

    - Starts after cutoff: not yet knowable -> None (dropped from corpus).
    - Resolved on/before cutoff: fully knowable -> returned unchanged.
    - Ongoing at cutoff (spans it, or has no end date): returned as a copy
      with outcome fields masked -- resolution=None, consequences=[],
      end_date=None -- exactly the shape a present-day episode has.

    Episodes with no start_date cannot be placed relative to the cutoff;
    they are conservatively masked rather than dropped, since dropping
    them would silently shrink the analog base on a data-quality accident.
    """
    if episode.start_date is not None and episode.start_date > cutoff:
        return None

    resolved_before_cutoff = (
        episode.end_date is not None
        and episode.end_date <= cutoff
        and episode.start_date is not None
    )
    if resolved_before_cutoff:
        return episode

    return episode.model_copy(
        update={
            "resolution": None,
            "consequences": [],
            "end_date": None,
        }
    )


def mask_corpus_at(
    episodes: Sequence[Episode], cutoff: datetime
) -> List[Episode]:
    """Mask a whole corpus snapshot at `cutoff` (see mask_episode_at)."""
    masked: List[Episode] = []
    for episode in episodes:
        result = mask_episode_at(episode, cutoff)
        if result is not None:
            masked.append(result)
    return masked
