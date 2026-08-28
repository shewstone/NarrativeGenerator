"""Embedding epoch identifiers (T4, docs/tickets/T4-embedding-render-versioning.md).

An epoch names the (render template, embedding model) combination that
produced a stored vector. Vectors from different epochs live in different
similarity spaces and must never be compared: retrieval filters to the
current epoch, and composition treats an epoch mismatch as a missing
signal. Design refs: Sec 6.3 (pinned model, batch re-embed), Sec 11.4
(the v0.7 render change invalidated all prior structural embeddings).

Deliberately dependency-free so storage/composition code can import epoch
constants without pulling in sentence-transformers.
"""

from typing import Literal

# Bumped whenever render_structural_template's output changes for the same
# input (Sec 6.2 stage 3: deterministic render, versioned).
# v0.7.0 = outcome-free render (resolution/consequences excluded).
# v0.8.0 = controlled-role tokens + proper-noun/date scrub (T2): free-text
#          roles and identity markers no longer reach the analogy signal.
# v0.9.0 = scale-neutral change pattern/configuration/mechanism families;
#          focal subgroup/idea names are scrubbed from the analogy signal.
CURRENT_RENDER_VERSION = "render-v0.9.0"

# The pinned sentence-transformers model (Sec 6.3). Single source of truth:
# EmbeddingGenerator.DEFAULT_MODEL is derived from this.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_ID = EMBEDDING_MODEL_NAME.rsplit("/", 1)[-1]

# Structural vectors depend on render template AND model; surface vectors
# embed raw title/summary text, so only the model matters.
STRUCTURAL_EMBEDDING_EPOCH = f"{CURRENT_RENDER_VERSION}+{EMBEDDING_MODEL_ID}"
SURFACE_EMBEDDING_EPOCH = EMBEDDING_MODEL_ID

EmbeddingKind = Literal["structural", "surface"]


def current_epoch(kind: EmbeddingKind) -> str:
    """The epoch a freshly generated vector of this kind belongs to."""
    if kind == "structural":
        return STRUCTURAL_EMBEDDING_EPOCH
    if kind == "surface":
        return SURFACE_EMBEDDING_EPOCH
    raise ValueError(f"Unknown embedding kind: {kind}")
