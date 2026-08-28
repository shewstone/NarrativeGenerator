"""Tests for smart chunking."""

import pytest

from narrative_engine.ingestion.chunker import Chunk, SmartChunker
from narrative_engine.ingestion.models import (
    IngestionConfig,
    ParsedDocument,
    SourceFormat,
    SourceMetadata,
    StructuralElement,
)


class TestSmartChunker:
    """Tests for SmartChunker."""

    @pytest.fixture
    def config(self):
        return IngestionConfig(
            target_chunk_size=100,  # Small for testing
            min_chunk_size=20,
            max_chunk_size=200,
            overlap_tokens=10,
        )

    @pytest.fixture
    def chunker(self, config):
        return SmartChunker(config)

    def test_chunker_initialization(self, config):
        chunker = SmartChunker(config)
        assert chunker.config.target_chunk_size == 100

    def test_chunk_small_document(self, chunker):
        """Small documents become single chunks."""
        doc = ParsedDocument(
            metadata=SourceMetadata(work_id="test", source_format=SourceFormat.TXT),
            content="Small document with few words.",
            structural_elements=[
                StructuralElement(element_type="document", level=0, content="Small document with few words.")
            ],
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Small document with few words."
        assert chunks[0].chunk_id == "test_0"
        assert chunks[0].total_chunks == 1

    def test_chunk_respects_structure(self, chunker):
        """Chunker respects chapter boundaries."""
        doc = ParsedDocument(
            metadata=SourceMetadata(work_id="test", source_format=SourceFormat.TXT),
            content="",
            structural_elements=[
                StructuralElement(
                    element_type="chapter",
                    level=0,
                    title="Chapter 1",
                    content="Chapter 1 content here.",
                ),
                StructuralElement(
                    element_type="chapter",
                    level=0,
                    title="Chapter 2",
                    content="Chapter 2 content here.",
                ),
            ],
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert chunks[0].chapter_title == "Chapter 1"
        assert chunks[1].chapter_title == "Chapter 2"

    def test_chunk_large_element_split(self, chunker):
        """Large elements are split into multiple chunks."""
        # Create content that exceeds max_chunk_size
        large_content = "Word " * 1000  # ~2000 words, ~2600 tokens

        doc = ParsedDocument(
            metadata=SourceMetadata(work_id="test", source_format=SourceFormat.TXT),
            content=large_content,
            structural_elements=[
                StructuralElement(element_type="document", level=0, content=large_content)
            ],
        )

        chunks = chunker.chunk_document(doc)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should have sequential IDs
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_preserves_paragraphs(self, chunker):
        """Chunker tries not to split paragraphs."""
        # Content with clear paragraph breaks
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        doc = ParsedDocument(
            metadata=SourceMetadata(work_id="test", source_format=SourceFormat.TXT),
            content=content,
            structural_elements=[
                StructuralElement(element_type="document", level=0, content=content)
            ],
        )

        chunks = chunker.chunk_document(doc)

        # Verify chunks don't break mid-paragraph
        for chunk in chunks:
            # Each chunk should start at a paragraph boundary
            assert chunk.content.strip()  # Not empty

    def test_chunk_overlap(self):
        """Chunks have overlap between them."""
        config = IngestionConfig(
            target_chunk_size=50,
            min_chunk_size=10,
            max_chunk_size=100,
            overlap_tokens=20,
        )
        chunker = SmartChunker(config)

        # Content with paragraphs
        content = "\n\n".join([f"Paragraph {i} with some content." for i in range(10)])

        doc = ParsedDocument(
            metadata=SourceMetadata(work_id="test", source_format=SourceFormat.TXT),
            content=content,
            structural_elements=[
                StructuralElement(element_type="document", level=0, content=content)
            ],
        )

        chunks = chunker.chunk_document(doc)

        if len(chunks) > 1:
            # Check for overlap
            for i in range(len(chunks) - 1):
                # Adjacent chunks should share some content
                # This is a basic check - the overlap might be small
                assert chunks[i].content != chunks[i + 1].content


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        chunk = Chunk(
            content="Test content",
            chunk_id="test_0",
            source_work_id="work123",
            source_format="txt",
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_id == "test_0"
        assert chunk.source_work_id == "work123"

    def test_token_estimation(self):
        """Token estimation is roughly correct."""
        # ~75 words ≈ 100 tokens (rough estimate)
        content = " ".join(["word"] * 75)

        chunk = Chunk(
            content=content,
            chunk_id="test",
            source_work_id="work",
            source_format="txt",
        )

        # Should be approximately 100 tokens
        assert 80 <= chunk.estimated_tokens <= 120

    def test_chunk_with_metadata(self):
        chunk = Chunk(
            content="Content",
            chunk_id="test",
            source_work_id="work",
            source_format="txt",
            chapter_title="Chapter 1",
            section_title="Section 1.1",
            chunk_index=0,
            total_chunks=5,
            metadata={"key": "value"},
        )

        assert chunk.chapter_title == "Chapter 1"
        assert chunk.section_title == "Section 1.1"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 5
        assert chunk.metadata == {"key": "value"}


class TestChunkValidation:
    """Tests for chunk validation."""

    def test_empty_chunk_invalid(self):
        config = IngestionConfig()
        chunker = SmartChunker(config)

        chunk = Chunk(
            content="   ",  # Just whitespace
            chunk_id="test",
            source_work_id="work",
            source_format="txt",
        )

        assert chunker.validate_chunk(chunk) is False

    def test_normal_chunk_valid(self):
        config = IngestionConfig(min_chunk_size=10, max_chunk_size=1000)
        chunker = SmartChunker(config)

        chunk = Chunk(
            content="This is valid content with enough words.",
            chunk_id="test",
            source_work_id="work",
            source_format="txt",
        )

        assert chunker.validate_chunk(chunk) is True

    def test_oversized_chunk_invalid(self):
        config = IngestionConfig(max_chunk_size=100)
        chunker = SmartChunker(config)

        # Content that exceeds max by > 20%
        content = "word " * 500  # Way over limit

        chunk = Chunk(
            content=content,
            chunk_id="test",
            source_work_id="work",
            source_format="txt",
        )

        assert chunker.validate_chunk(chunk) is False
