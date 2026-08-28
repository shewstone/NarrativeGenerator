"""Fast tests for embedding math and bounded caching."""

import pytest

from narrative_engine.retrieval.embeddings import EmbeddingCache, EmbeddingGenerator


class TestSimilarity:
    def test_zero_vectors_have_finite_zero_similarity(self):
        generator = EmbeddingGenerator()

        assert generator.similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert generator.compute_similarities(
            [1.0, 0.0], [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        ) == pytest.approx([1.0, 0.0, 0.0])

    def test_dimension_mismatch_is_explicit(self):
        with pytest.raises(ValueError, match="dimensions differ"):
            EmbeddingGenerator().similarity([1.0], [1.0, 2.0])


class TestEmbeddingCache:
    def test_lru_bound_evicts_oldest_entry(self):
        cache = EmbeddingCache(max_size=2)
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        assert cache.get("a") == [1.0]

        cache.set("c", [3.0])

        assert cache.get("b") is None
        assert cache.get("a") == [1.0]
        assert cache.get("c") == [3.0]
        assert cache.get_stats()["size"] == 2
