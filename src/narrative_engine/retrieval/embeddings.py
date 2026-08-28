"""Vector embedding generation for semantic search."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

import structlog
from sentence_transformers import SentenceTransformer

from narrative_engine.models import Episode
from narrative_engine.retrieval.epochs import EMBEDDING_MODEL_NAME

logger = structlog.get_logger()


class EmbeddingGenerator:
    """Generate vector embeddings for episodes using sentence-transformers."""

    # Pinned model (Sec 6.3); single source of truth lives in epochs.py so
    # epoch identifiers can't drift from the model actually used.
    DEFAULT_MODEL = EMBEDDING_MODEL_NAME

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
                embedding_dim=self.embedding_dim,
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        model = self.model
        get_dimension = getattr(model, "get_embedding_dimension", None)
        dimension = get_dimension() if get_dimension is not None else model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Embedding model {self.model_name!r} did not report a dimension")
        return int(dimension)

    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [e.tolist() for e in embeddings]

    def render_structural_template(self, episode: Episode) -> str:
        """Deterministically render the episode's abstract narrative shape
        (design doc Sec 3.3): scale-neutral pattern/configuration, actor roles
        (not names), mechanism families, and the sequence of
        conditions/mechanics/tension -- never raw title/summary text, actor
        names, focal-scope names, location, or dates. Legacy arc fields remain
        as a secondary compatibility signal.

        This is what makes the structural embedding place/date-blind (so
        Athens-Sparta can embed near Wilhelmine Germany-Britain): title and
        summary are natural-language prose anchored to the specific
        happening, and actor names are proper nouns, so including either
        re-introduces the exact identity signal the structural/surface
        split (Sec 3.3a) exists to keep separate.

        Outcome fields (resolution, consequences) are ALWAYS excluded, for
        two reasons. (1) Query/corpus symmetry: a present-day query episode
        has resolution=None by definition, so embedding outcomes for corpus
        episodes but not queries makes every retrieval score systematically
        biased by a field only one side has. (2) Leakage: retrieval is
        "given the situation so far, what followed?" -- if the embedding
        encodes how the story ended, backtests partially retrieve analogs
        BY their endings, which is the masking failure Sec 6.6 exists to
        prevent. Outcomes still reach theses via the analogs' stored
        resolution/consequences fields after retrieval (Sec 6.5 step 5);
        they just never enter the similarity signal.
        """
        lines: List[str] = []

        if episode.change_pattern:
            lines.append(f"Change pattern: {episode.change_pattern.value}")

        if episode.situation_scale:
            lines.append(f"Scale: {episode.situation_scale.value}")

        if episode.domains:
            lines.append(f"Domains: {', '.join(domain.value for domain in episode.domains)}")

        configuration = episode.configuration.model_dump(exclude_none=True)
        if configuration:
            serialized = ", ".join(f"{dimension}={value:+.2f}" for dimension, value in configuration.items())
            lines.append(f"Configuration: {serialized}")

        if episode.mechanism_families:
            lines.append("Mechanism families: " + ", ".join(family.value for family in episode.mechanism_families))

        if episode.arc_type:
            lines.append(f"Legacy arc: {episode.arc_type.value}")

        if episode.arc_phase:
            lines.append(f"Phase: {episode.arc_phase.value}")

        if episode.actors:
            # Controlled-vocabulary tokens ONLY (T2): canonical_role where
            # the fit floor was cleared, one shared unresolved token where it
            # wasn't. The free-text role is prose ("President", "J.P.
            # Morgan's bank") and must never reach the analogy signal.
            from narrative_engine.extraction.roles import UNRESOLVED_ACTOR_TOKEN

            seen = set()
            roles = []
            for actor in episode.actors:
                token = actor.canonical_role or UNRESOLVED_ACTOR_TOKEN
                if token not in seen:
                    seen.add(token)
                    roles.append(token)
            lines.append(f"Actor roles: {', '.join(roles)}")

        if episode.mechanism_tags:
            # Sec 3.3 template's MECHANISMS line: controlled-vocabulary
            # structural drivers, serialized in tag order.
            lines.append(f"Mechanisms: {', '.join(tag.value for tag in episode.mechanism_tags)}")

        scrub = self._identity_scrubber(episode)

        if episode.initiating_conditions:
            lines.append("Initiating conditions:")
            lines.extend(f"- {scrub(c)}" for c in episode.initiating_conditions)

        if episode.escalation_mechanics:
            lines.append("Escalation mechanics:")
            lines.extend(f"- {scrub(m)}" for m in episode.escalation_mechanics)

        if episode.tension:
            lines.append(f"Tension: {scrub(episode.tension)}")

        return "\n".join(lines)

    def _identity_scrubber(self, episode: Episode):
        """Build a deterministic scrub function for this episode's free-text
        lines (T2): actor names -> their role token, the episode's
        place/scope names -> <PLACE>, years and month names -> <DATE>.

        The render is deterministic code, not an LLM (Sec 6.2 stage 3), so
        the place/date-blind guarantee is enforced HERE, not hoped for in
        the extraction prompt. Replacement, not deletion: a resolved actor
        mention keeps its structural meaning under its role token.
        """
        import re

        from narrative_engine.extraction.roles import UNRESOLVED_ACTOR_TOKEN

        replacements: List[tuple] = []

        for actor in episode.actors:
            if actor.name:
                token = actor.canonical_role or UNRESOLVED_ACTOR_TOKEN
                replacements.append((actor.name, token))

        place_names: List[str] = []
        if episode.location:
            place_names.append(episode.location)
        if episode.scope_id:
            from narrative_engine.scopes import get_registry, resolve_scope

            scope_id = resolve_scope(episode.scope_id) or None
            scope = get_registry().get(scope_id) if scope_id else None
            place_names.append(episode.scope_id)
            if scope:
                place_names.extend([scope.name, *scope.aliases])
        for place in place_names:
            replacements.append((place, "<PLACE>"))
        if episode.scope_name:
            replacements.append((episode.scope_name, "<FOCAL_SCOPE>"))
        if episode.parent_scope_name:
            replacements.append((episode.parent_scope_name, "<PARENT_SCOPE>"))

        # Longest-first so "United States of America" wins over "America".
        # Lookaround word anchors, not \b: aliases like "U.S." end in
        # punctuation where \b would fail, and unanchored matching rewrites
        # substrings ("us" inside "Trust").
        replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

        def _compile(source: str, target: str):
            # Place replacements absorb a preceding article so "the United
            # States" and "Austria-Hungary" both normalize to bare <PLACE> --
            # article variance is identity residue too.
            article = r"(?:the\s+)?" if target == "<PLACE>" else ""
            return re.compile(rf"(?<!\w){article}{re.escape(source)}(?!\w)", re.IGNORECASE)

        compiled = [(_compile(source, target), target) for source, target in replacements if source]

        year_pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")
        month_pattern = re.compile(
            r"\b(January|February|March|April|May|June|July|August|" r"September|October|November|December)\b",
            re.IGNORECASE,
        )

        def scrub(text: str) -> str:
            for pattern, target in compiled:
                text = pattern.sub(target, text)
            text = year_pattern.sub("<DATE>", text)
            text = month_pattern.sub("<DATE>", text)
            return text

        return scrub

    def generate_structural_embedding(self, episode: Episode) -> List[float]:
        """Generate the structural embedding: analogy signal, NOT identity.

        Embeds render_structural_template's place/date-blind rendering of
        the episode's abstract narrative structure. This is what makes
        cross-domain analogy matching work (Athens-Sparta embeds near
        Germany-Britain). That same place/date-blindness makes it WRONG for
        identity resolution ("is this the same happening?") -- use
        generate_surface_embedding for that (design doc Sec 3.3a).

        Consumers: analog retrieval (AnalogRetrievalEngine), discovery
        clustering. Never: SAME_EVENT_AS resolution, arc composition.
        """
        return self.generate(self.render_structural_template(episode))

    def generate_surface_embedding(self, episode: Episode) -> List[float]:
        """Generate the surface embedding: identity signal, NOT analogy.

        Raw title + summary text only -- no role substitution, no arc/phase
        labels. Captures "is this the same happening?" (two sources
        describing the same crash), not "is this the same shape?" (design
        doc Sec 3.3a).

        Consumers: SAME_EVENT_AS resolution, arc composition. Never: analog
        retrieval, discovery clustering.
        """
        text = f"{episode.title}\n{episode.summary}"
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

        v1 = np.asarray(embedding1, dtype=np.float32)
        v2 = np.asarray(embedding2, dtype=np.float32)
        if v1.shape != v2.shape:
            raise ValueError(f"Embedding dimensions differ: {v1.shape!r} != {v2.shape!r}")
        denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denominator == 0.0:
            return 0.0

        return float(np.dot(v1, v2) / denominator)

    def compute_similarities(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
    ) -> List[float]:
        """Compute similarities between query and multiple candidates."""
        if not candidate_embeddings:
            return []

        import numpy as np

        query = np.asarray(query_embedding, dtype=np.float32)
        candidates = np.asarray(candidate_embeddings, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1:] != query.shape:
            raise ValueError("Candidate embeddings must all match the query embedding dimension")

        query_norm = float(np.linalg.norm(query))
        candidate_norms = np.linalg.norm(candidates, axis=1)
        denominators = candidate_norms * query_norm
        similarities = np.divide(
            candidates @ query,
            denominators,
            out=np.zeros(len(candidates), dtype=np.float32),
            where=denominators != 0,
        )
        return similarities.tolist()


class EmbeddingCache:
    """Bounded in-memory LRU cache for embeddings (production: use Redis)."""

    def __init__(self, max_size: int = 1024) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

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
