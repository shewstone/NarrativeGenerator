"""Vector embedding generation for semantic search."""

from __future__ import annotations

from typing import List, Optional

import structlog
from sentence_transformers import SentenceTransformer

from narrative_engine.models import Episode

logger = structlog.get_logger()


class EmbeddingGenerator:
    """Generate vector embeddings for episodes using sentence-transformers."""
    
    # Default model: good balance of quality and speed
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None
        self.logger = structlog.get_logger()
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self.logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self.logger.info(
                "Model loaded",
                embedding_dim=self._model.get_sentence_embedding_dimension(),
            )
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [e.tolist() for e in embeddings]
    
    def generate_for_episode(self, episode: Episode) -> List[float]:
        """Generate embedding optimized for episode analog retrieval.
        
        Key insight: Embed the abstract narrative structure, not just raw text.
        This improves cross-domain analogy matching.
        """
        # Construct structured representation
        components = [
            f"Title: {episode.title}",
            f"Summary: {episode.summary}",
        ]
        
        if episode.arc_type:
            components.append(f"Arc: {episode.arc_type.value}")
        
        if episode.arc_phase:
            components.append(f"Phase: {episode.arc_phase.value}")
        
        if episode.actors:
            actor_roles = ", ".join([
                f"{a.role}:{a.name}" for a in episode.actors[:5]  # Top 5 actors
            ])
            components.append(f"Actors: {actor_roles}")
        
        if episode.initiating_conditions:
            conditions = "; ".join(episode.initiating_conditions[:3])
            components.append(f"Initiating conditions: {conditions}")
        
        if episode.escalation_mechanics:
            mechanics = "; ".join(episode.escalation_mechanics[:3])
            components.append(f"Escalation: {mechanics}")
        
        if episode.tension:
            components.append(f"Tension: {episode.tension}")
        
        # Join into single string for embedding
        text = "\n".join(components)
        
        return self.generate(text)
    
    def generate_for_query(self, query: str) -> List[float]:
        """Generate embedding for a search query.
        
        Queries are often shorter and less structured than episodes,
        so we use them as-is but could enhance with query expansion.
        """
        return self.generate(query)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        import numpy as np
        
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def compute_similarities(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute similarities between query and multiple candidates."""
        return [self.similarity(query_embedding, cand) for cand in candidate_embeddings]


class EmbeddingCache:
    """Simple in-memory cache for embeddings (production: use Redis)."""
    
    def __init__(self) -> None:
        self._cache: dict = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        self._cache[key] = embedding
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
