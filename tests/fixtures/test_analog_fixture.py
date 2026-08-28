"""Analog-fixture gate (T1): does place-blind structural matching work at all?

For every hand-built cross-era pair, the partner must rank in the top-k
structural neighbors among all fixture episodes (59 alternatives including
30 same-era distractors). pair_recall@5 and MRR are printed on every run so
the history is greppable; the recall floor gates CI and BLOCKS
embedding-model and render-template upgrades (Sec 6.3).

The floor is a ratchet, not an aspiration: set from the observed baseline
minus slack, raised as the render improves. A floor that fails from day
one teaches everyone to ignore the gate.
"""

import numpy as np
import pytest

from narrative_engine.retrieval.embeddings import EmbeddingGenerator
from tests.fixtures.analog_fixture import (
    ANALOG_PAIRS,
    DISTRACTORS,
    FIXTURE_VERSION,
    all_fixture_episodes,
)

# Ratchet floor. Baseline observed 2026-08-28 with render-v0.9.0 +
# all-MiniLM-L6-v2: pair_recall@5 = 0.900, MRR = 0.753 (60 rankings, 59
# alternatives each). Floor = baseline - 0.05 slack; raise it when the
# render improves, never lower it to make a change pass.
RECALL_AT_5_FLOOR = 0.85
K = 5


@pytest.fixture(scope="module")
def similarity_matrix():
    episodes = all_fixture_episodes()
    generator = EmbeddingGenerator()
    renders = [generator.render_structural_template(e) for e in episodes]
    vectors = np.array(generator.generate_batch(renders))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    return episodes, normalized @ normalized.T


@pytest.mark.fixture_gate
class TestAnalogFixtureGate:
    def test_pair_recall_at_k(self, similarity_matrix):
        episodes, sims = similarity_matrix
        index_of = {e.id: i for i, e in enumerate(episodes)}

        ranks = []
        misses = []
        for a, b, rationale in ANALOG_PAIRS:
            for query, target in ((a, b), (b, a)):
                qi, ti = index_of[query.id], index_of[target.id]
                order = np.argsort(-sims[qi])
                # rank of target among all episodes except the query itself
                rank = int(np.where(order == ti)[0][0])
                ranks.append(rank)
                if rank > K:
                    misses.append((query.title, target.title, rank, rationale))

        recall_at_k = sum(1 for r in ranks if r <= K) / len(ranks)
        mrr = float(np.mean([1.0 / r for r in ranks]))

        print(
            f"\n[analog-fixture {FIXTURE_VERSION}] "
            f"pairs={len(ANALOG_PAIRS)} distractors={len(DISTRACTORS)} "
            f"pair_recall@{K}={recall_at_k:.3f} MRR={mrr:.3f}"
        )
        for query_title, target_title, rank, rationale in misses:
            print(f"  MISS rank={rank}: {query_title!r} -> {target_title!r} ({rationale})")

        assert recall_at_k >= RECALL_AT_5_FLOOR, (
            f"pair_recall@{K}={recall_at_k:.3f} fell below the ratchet floor "
            f"{RECALL_AT_5_FLOOR}. The render/embedding change you just made "
            "degraded cross-era analog matching (Sec 6.3: this gate blocks "
            "embedding-model and render-template upgrades)."
        )

    def test_fixture_shape(self):
        assert len(ANALOG_PAIRS) >= 30
        assert len(DISTRACTORS) >= 30
        ids = [e.id for e in all_fixture_episodes()]
        assert len(ids) == len(set(ids))
