"""Smart chunking that respects narrative boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from narrative_engine.ingestion.models import (
    IngestionConfig,
    ParsedDocument,
    StructuralElement,
)
from narrative_engine.observability import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A chunk of text ready for extraction."""

    content: str
    chunk_id: str
    source_work_id: str
    source_format: str

    # Structural context
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    element_type: str = "chunk"

    # Position in document
    chunk_index: int = 0
    total_chunks: int = 0
    char_start: int = 0
    char_end: int = 0

    # Token estimates
    estimated_tokens: int = 0

    # Metadata
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Estimate tokens (rough: 1 token ≈ 0.75 words)
        word_count = len(self.content.split())
        self.estimated_tokens = int(word_count / 0.75)


class SmartChunker:
    """Chunks documents while respecting narrative boundaries.

    Strategy:
    1. Prefer chapter/section boundaries
    2. Target chunk size: 2k-8k tokens
    3. Overlap between chunks: ~200 tokens
    4. Never split in the middle of a paragraph if possible
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig.default()

    def chunk_document(self, parsed: ParsedDocument) -> List[Chunk]:
        """Chunk a parsed document into extraction-ready chunks."""
        logger.info(
            "chunking_document",
            work_id=parsed.metadata.work_id,
            total_words=len(parsed.content.split()),
        )

        chunks = []
        chunk_index = 0

        for element in parsed.structural_elements:
            element_chunks = self._chunk_element(element, parsed.metadata, chunk_index)
            chunks.extend(element_chunks)
            chunk_index += len(element_chunks)

        # Update total chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        logger.info(
            "document_chunked",
            work_id=parsed.metadata.work_id,
            chunks_created=len(chunks),
        )

        return chunks

    def _chunk_element(
        self,
        element: StructuralElement,
        metadata,
        start_index: int,
    ) -> List[Chunk]:
        """Chunk a single structural element."""
        content = element.content.strip()
        if not content:
            return []

        # If element is small enough, keep as single chunk
        word_count = len(content.split())
        estimated_tokens = int(word_count / 0.75)

        if estimated_tokens <= self.config.max_chunk_size:
            return [
                Chunk(
                    content=content,
                    chunk_id=f"{metadata.work_id}_{start_index}",
                    source_work_id=metadata.work_id,
                    source_format=metadata.source_format.value,
                    chapter_title=element.title if element.element_type == "chapter" else None,
                    section_title=element.title if element.element_type == "section" else None,
                    element_type=element.element_type,
                    chunk_index=start_index,
                )
            ]

        # Element is too large, need to split
        return self._split_content(
            content,
            metadata,
            element,
            start_index,
        )

    def _split_content(
        self,
        content: str,
        metadata,
        element: StructuralElement,
        start_index: int,
    ) -> List[Chunk]:
        """Split large content into chunks respecting paragraph boundaries."""
        # Split on double newlines (paragraphs)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = start_index

        for paragraph in paragraphs:
            para_word_count = len(paragraph.split())

            # If adding this paragraph exceeds max size, finalize current chunk
            if (
                current_chunk
                and current_word_count + para_word_count > self.config.target_chunk_size
            ):
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_id=f"{metadata.work_id}_{chunk_index}",
                        source_work_id=metadata.work_id,
                        source_format=metadata.source_format.value,
                        chapter_title=element.title if element.element_type == "chapter" else None,
                        section_title=element.title if element.element_type == "section" else None,
                        element_type=element.element_type,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                # Keep last paragraph for overlap (if configured)
                if self.config.overlap_tokens > 0:
                    overlap_words = int(self.config.overlap_tokens * 0.75)
                    overlap_para = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_para, paragraph]
                    current_word_count = len(overlap_para.split()) + para_word_count
                else:
                    current_chunk = [paragraph]
                    current_word_count = para_word_count
            else:
                current_chunk.append(paragraph)
                current_word_count += para_word_count

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_id=f"{metadata.work_id}_{chunk_index}",
                    source_work_id=metadata.work_id,
                    source_format=metadata.source_format.value,
                    chapter_title=element.title if element.element_type == "chapter" else None,
                    section_title=element.title if element.element_type == "section" else None,
                    element_type=element.element_type,
                    chunk_index=chunk_index,
                )
            )

        return chunks

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: 1 token ≈ 0.75 words (for English)
        word_count = len(text.split())
        return int(word_count / 0.75)

    def validate_chunk(self, chunk: Chunk) -> bool:
        """Validate a chunk meets requirements."""
        if not chunk.content.strip():
            logger.warning("empty_chunk", chunk_id=chunk.chunk_id)
            return False

        if chunk.estimated_tokens < self.config.min_chunk_size:
            logger.warning(
                "chunk_too_small",
                chunk_id=chunk.chunk_id,
                tokens=chunk.estimated_tokens,
            )
            # Still valid, just small

        if chunk.estimated_tokens > self.config.max_chunk_size * 1.2:  # Allow 20% overflow
            logger.warning(
                "chunk_too_large",
                chunk_id=chunk.chunk_id,
                tokens=chunk.estimated_tokens,
            )
            return False

        return True
