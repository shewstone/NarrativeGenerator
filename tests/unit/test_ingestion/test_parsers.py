"""Tests for ingestion parsers."""

from pathlib import Path

import pytest

from narrative_engine.ingestion.models import DocumentType, SourceFormat
from narrative_engine.ingestion.parsers import (
    MarkdownParser,
    OcrParser,
    TxtParser,
    get_parser,
)


class TestTxtParser:
    """Tests for TXT parser."""

    @pytest.fixture
    def parser(self):
        return TxtParser()

    def test_can_parse_txt(self, parser, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")
        assert parser.can_parse(txt_file) is True

    def test_cannot_parse_pdf(self, parser, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("Not a real pdf")
        assert parser.can_parse(pdf_file) is False

    def test_parse_simple_text(self, parser, tmp_path):
        txt_file = tmp_path / "simple.txt"
        txt_file.write_text("This is a simple text file.")

        doc = parser.parse(txt_file)

        assert doc.metadata.source_format == SourceFormat.TXT
        assert doc.metadata.work_id == "simple"
        assert "This is a simple text file" in doc.content
        assert len(doc.structural_elements) == 1
        assert doc.structural_elements[0].element_type == "document"

    def test_parse_with_chapters(self, parser, tmp_path):
        txt_file = tmp_path / "chapters.txt"
        txt_file.write_text("Chapter 1\nFirst chapter content here.\n\n" "Chapter 2\nSecond chapter content here.")

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 2
        assert doc.structural_elements[0].element_type == "chapter"
        assert doc.structural_elements[1].element_type == "chapter"

    def test_parse_preserves_preamble_before_first_chapter(self, parser, tmp_path):
        txt_file = tmp_path / "preface.txt"
        txt_file.write_text(
            "Author's preface that must be ingested.\n\n"
            "Chapter 1\nFirst chapter content here.\n\n"
            "Chapter 2\nSecond chapter content here."
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 3
        assert doc.structural_elements[0].element_type == "document"
        assert "Author's preface" in doc.structural_elements[0].content

    def test_file_hash_computed(self, parser, tmp_path):
        txt_file = tmp_path / "hash_test.txt"
        txt_file.write_text("Test content for hashing")

        doc = parser.parse(txt_file)

        assert doc.metadata.file_hash is not None
        assert len(doc.metadata.file_hash) == 64  # SHA256 hex


class TestMarkdownParser:
    """Tests for Markdown parser."""

    @pytest.fixture
    def parser(self):
        return MarkdownParser()

    def test_can_parse_md(self, parser, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title")
        assert parser.can_parse(md_file) is True

    def test_parse_with_headers(self, parser, tmp_path):
        md_file = tmp_path / "headers.md"
        md_file.write_text(
            "# Chapter 1\nContent of chapter 1.\n\n"
            "## Section 1.1\nMore content.\n\n"
            "# Chapter 2\nContent of chapter 2."
        )

        doc = parser.parse(md_file)

        assert len(doc.structural_elements) == 3
        # Check hierarchy
        assert doc.structural_elements[0].level == 0  # #
        assert doc.structural_elements[1].level == 1  # ##
        assert doc.structural_elements[2].level == 0  # #

    def test_parse_with_frontmatter(self, parser, tmp_path):
        md_file = tmp_path / "with_frontmatter.md"
        md_file.write_text("---\ntitle: Test Document\nauthor: John Doe\n---\n\n# Content\nText here.")

        doc = parser.parse(md_file)

        assert doc.metadata.title == "Test Document"
        assert doc.metadata.author == "John Doe"

    def test_parse_preserves_content_before_first_header(self, parser, tmp_path):
        md_file = tmp_path / "preamble.md"
        md_file.write_text("Opening context.\n\n# Chapter 1\nBody text.")

        doc = parser.parse(md_file)

        assert [element.element_type for element in doc.structural_elements] == [
            "document",
            "section",
        ]
        assert doc.structural_elements[0].content == "Opening context."


class TestParserRegistry:
    """Tests for parser registry."""

    def test_get_parser_for_txt(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")
        parser = get_parser(txt_file)
        assert parser is not None
        assert isinstance(parser, TxtParser)

    def test_get_parser_for_md(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# test")
        parser = get_parser(md_file)
        assert parser is not None
        assert isinstance(parser, MarkdownParser)

    def test_no_parser_for_unknown(self, tmp_path):
        unknown_file = tmp_path / "test.xyz"
        unknown_file.write_text("test")
        parser = get_parser(unknown_file)
        assert parser is None

    def test_image_only_pdf_probe_precedes_generic_pdf_parser(self, tmp_path, monkeypatch):
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(b"not a text PDF")
        monkeypatch.setattr(OcrParser, "can_parse", lambda self, path: True)

        assert isinstance(get_parser(pdf_file), OcrParser)


class TestSourceMetadata:
    """Tests for SourceMetadata."""

    def test_metadata_creation(self):
        from narrative_engine.ingestion.models import SourceMetadata

        metadata = SourceMetadata(
            work_id="test-work",
            source_format=SourceFormat.TXT,
            title="Test Title",
            author="Test Author",
        )

        assert metadata.work_id == "test-work"
        assert metadata.source_format == SourceFormat.TXT
        assert metadata.title == "Test Title"
        assert metadata.author == "Test Author"
        assert metadata.ingested_at is not None

    def test_detect_document_type(self):

        parser = TxtParser()

        # Timeline detection
        timeline_path = Path("timeline_of_events.txt")
        doc_type = parser._detect_document_type(timeline_path, {})
        assert doc_type == DocumentType.TIMELINE

        # Article detection
        article_path = Path("research_article.txt")
        doc_type = parser._detect_document_type(article_path, {})
        assert doc_type == DocumentType.ARTICLE

        # Default to book
        book_path = Path("great_gatsby.txt")
        doc_type = parser._detect_document_type(book_path, {})
        assert doc_type == DocumentType.BOOK
