"""Tests for ingestion parsers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from narrative_engine.ingestion.models import DocumentType, SourceFormat
from narrative_engine.ingestion.parsers import (
    MarkdownParser,
    OcrParser,
    PdfParser,
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

    def test_parse_skips_gutenberg_front_matter_before_chapters(self, parser, tmp_path):
        txt_file = tmp_path / "gutenberg-front-matter.txt"
        txt_file.write_text(
            "Produced by a volunteer team for Project Gutenberg.\n\n"
            "PREFACE\nPublication notes.\n\nCONTENTS\n1. First.\n2. Second.\n\n"
            "Chapter 1\nFirst historical chapter.\n\n"
            "Chapter 2\nSecond historical chapter."
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 2
        assert all(element.element_type == "chapter" for element in doc.structural_elements)
        assert "Produced by" in doc.content

    def test_parse_skips_explicit_chapter_entries_repeated_in_contents(self, parser, tmp_path):
        txt_file = tmp_path / "chapter-contents.txt"
        txt_file.write_text(
            "TABLE OF CONTENTS\n"
            "Chapter I\nTHE BEGINNING\nChapter II\nTHE CHANGE\n\n"
            "Chapter I\nTHE BEGINNING\nHistorical body one.\n\n"
            "Chapter II\nTHE CHANGE\nHistorical body two."
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 2
        assert [element.title for element in doc.structural_elements] == ["Chapter I", "Chapter II"]
        assert doc.structural_elements[0].content.endswith("Historical body one.")

    def test_parse_matches_ocr_letter_t_body_heading_to_chapter_one(self, parser, tmp_path):
        txt_file = tmp_path / "ocr-chapter-one.txt"
        txt_file.write_text(
            "CONTENTS\n"
            "Chapter 1. Historical Setting 1\n"
            "Chapter 2. Society 80\n"
            "Chapter 3. Economy 170\n\n"
            "Chapter t. Historical Setting\nHistorical body one.\n\n"
            "Chapter 2. Society\nHistorical body two.\n\n"
            "Chapter 3. Economy\nHistorical body three."
        )

        doc = parser.parse(txt_file)

        assert [element.title for element in doc.structural_elements] == [
            "Chapter t. Historical Setting",
            "Chapter 2. Society",
            "Chapter 3. Economy",
        ]
        assert doc.structural_elements[0].content.endswith("Historical body one.")

    def test_parse_falls_back_when_only_bibliography_repeats_toc_chapters(self, parser, tmp_path):
        txt_file = tmp_path / "country-study.txt"
        narrative = " ".join(["historical narrative evidence"] * 700)
        txt_file.write_text(
            "CONTENTS\n"
            "Chapter 1. Historical Setting 1\n"
            "Chapter 2. Society 80\n"
            "Chapter 3. Politics 170\n\n"
            f"Historical Setting\n{narrative}\n\n"
            "BIBLIOGRAPHY\n"
            "Chapter 1\nReferences one.\n"
            "Chapter 2\nReferences two.\n"
            "Chapter 3\nReferences three.\n"
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 1
        assert doc.structural_elements[0].element_type == "document"
        assert narrative in doc.structural_elements[0].content
        assert "References three" in doc.structural_elements[0].content

    def test_parse_preserves_substantive_chapters_when_numbering_restarts(self, parser, tmp_path):
        txt_file = tmp_path / "multi-part.txt"
        substantive = " ".join(["historical evidence"] * 600)
        txt_file.write_text(
            "CONTENTS\nPART I\nI. Origins\nPART II\nI. Renewal\n\n"
            f"PART I\nChapter I\nORIGINS\n{substantive}\n\n"
            f"Chapter II\nCHANGE\n{substantive}\n\n"
            f"PART II\nChapter I\nRENEWAL\n{substantive}\n\n"
            f"Chapter II\nCONSOLIDATION\n{substantive}"
        )

        doc = parser.parse(txt_file)

        chapters = [element for element in doc.structural_elements if element.element_type == "chapter"]
        assert [element.title for element in chapters] == [
            "Chapter I",
            "Chapter II",
            "Chapter I",
            "Chapter II",
        ]

    def test_parse_does_not_treat_lowercase_prose_as_spelled_chapter_number(self, parser, tmp_path):
        txt_file = tmp_path / "chapter-prose.txt"
        txt_file.write_text(
            "Chapter I\nFirst body.\n\n"
            "Chapter II\nSecond body.\n\n"
            "chapter in the subject's life was difficult.\n\n"
            "End matter."
        )

        doc = parser.parse(txt_file)

        assert [element.title for element in doc.structural_elements] == ["Chapter I", "Chapter II"]
        assert "chapter in the subject's life" in doc.structural_elements[-1].content

    def test_parse_gutenberg_roman_numeral_headings_with_titles(self, parser, tmp_path):
        txt_file = tmp_path / "gutenberg.txt"
        txt_file.write_bytes(
            b"Preface.\r\n\r\n"
            b"CHAPTER I--THE FEUDAL AGE\r\nFirst chapter.\r\n\r\n"
            b"CHAPTER II.--LAW AND GOVERNMENT\r\nSecond chapter.\r\n"
        )

        doc = parser.parse(txt_file)

        assert [element.element_type for element in doc.structural_elements] == [
            "document",
            "chapter",
            "chapter",
        ]
        assert doc.structural_elements[1].title == "CHAPTER I--THE FEUDAL AGE"
        assert doc.structural_elements[2].title == "CHAPTER II.--LAW AND GOVERNMENT"

    def test_parse_bare_roman_numeral_chapter_headings(self, parser, tmp_path):
        txt_file = tmp_path / "roman-headings.txt"
        txt_file.write_text(
            "Preface.\n\n"
            "I\nAFRICA\nFirst chapter.\n\n"
            "II.\nTHE COMING OF THE PEOPLES\nSecond chapter.\n\n"
            "III\nEMPIRES OF THE NIGER\nThird chapter.\n"
        )

        doc = parser.parse(txt_file)

        assert [element.element_type for element in doc.structural_elements] == [
            "document",
            "chapter",
            "chapter",
            "chapter",
        ]
        assert "EMPIRES OF THE NIGER" in doc.structural_elements[-1].content

    def test_roman_heading_parser_skips_repeated_contents_entries(self, parser, tmp_path):
        txt_file = tmp_path / "roman-with-contents.txt"
        txt_file.write_text(
            "CONTENTS\nI AFRICA\nII PEOPLES\nIII EMPIRES\n\n"
            "PREFACE\nContext.\n\n"
            "I AFRICA\nBody one.\n\n"
            "II.\n\nPEOPLES\nBody two.\n\n"
            "III.\n\nEMPIRES\nBody three.\n"
        )

        doc = parser.parse(txt_file)

        chapters = [element for element in doc.structural_elements if element.element_type == "chapter"]
        assert len(chapters) == 3
        assert chapters[0].content.startswith("I AFRICA")
        assert doc.structural_elements[0] is chapters[0]

    def test_roman_heading_parser_matches_decorated_toc_to_body(self, parser, tmp_path):
        txt_file = tmp_path / "roman-decorated-contents.txt"
        txt_file.write_text(
            "CONTENTS\n"
            "I. OTHMAN (1288-1326)                           13\n"
            "II. ORCHAN (1326-59)                            20\n"
            "III. MURAD I (1359-89)                          31\n\n"
            "PART I\nTHE GROWTH OF EMPIRE\n\n"
            "OTHMAN\n\n1288-1326\nSubstantive body one.\n\n"
            "II\n\nORCHAN\n\n1326-59\nSubstantive body two.\n\n"
            "III\n\nMURAD I\n\n1359-89\nSubstantive body three.\n\n"
            "INDEX\nOTHMAN 1\nORCHAN 2\n"
        )

        doc = parser.parse(txt_file)

        chapters = [element for element in doc.structural_elements if element.element_type == "chapter"]
        assert len(chapters) == 3
        assert chapters[0].content.startswith("OTHMAN\n\n1288-1326")
        assert all("Substantive body" in chapter.content for chapter in chapters)
        assert "INDEX" not in chapters[-1].content

    def test_roman_parser_ignores_catalogue_initials_and_false_numerals(self, parser, tmp_path):
        txt_file = tmp_path / "roman-catalogue.txt"
        txt_file.write_text(
            "Transcriber's Note.\n\nC. EDMUND MAURICE\n\nCONTENTS.\n"
            "I.\nFIRST ERA                                      1-10\n"
            "II.\nSECOND ERA                                    11-20\n"
            "III.\nTHIRD ERA                                     21-30\n"
            "IV.\nFOURTH ERA                                    31-40\n"
            "I. TO DEATH OF A KING                         35\n"
            "V.\nFIFTH ERA                                     41-50\n\n"
            "I.\nFIRST ERA\nSubstantive one.\n\n"
            "II.\nSECOND ERA\nSubstantive two.\n\n"
            "III.\nTHIRD ERA\nSubstantive three.\n\n"
            "IV.\nFOURTH ERA\nThe king from III. was murdered.\n\n"
            "III. WAS MURDERED.\n"
            "V.\nFIFTH ERA\nSubstantive five.\n\n"
            "INDEX\nNames and pages."
        )

        doc = parser.parse(txt_file)

        chapters = [element for element in doc.structural_elements if element.element_type == "chapter"]
        assert len(chapters) == 5
        assert chapters[0].content.startswith("I.\nFIRST ERA\nSubstantive one.")
        assert chapters[-1].content.startswith("V.\nFIFTH ERA\nSubstantive five.")
        assert "Transcriber's Note" not in "\n".join(chapter.content for chapter in chapters)
        assert "INDEX" not in chapters[-1].content

    def test_roman_parser_rejects_unresolved_paginated_letter_contents(self, parser, tmp_path):
        txt_file = tmp_path / "letters-with-contents.txt"
        txt_file.write_text(
            "CONTENTS\n"
            "I. To a Friend                                      17\n"
            "II. To the Council                                  21\n"
            "III. To the Community                               28\n\n"
            "LETTERS\n\n"
            "LETTER I.\nSubstantive first letter.\n\n"
            "LETTER II.\nSubstantive second letter.\n\n"
            "LETTER III.\nSubstantive third letter.\n"
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 1
        assert doc.structural_elements[0].element_type == "document"
        assert "Substantive first letter" in doc.structural_elements[0].content
        assert "Substantive third letter" in doc.structural_elements[0].content

    def test_roman_parser_rejects_unresolved_unpaginated_contents(self, parser, tmp_path):
        txt_file = tmp_path / "diary-with-contents.txt"
        txt_file.write_text(
            "CONTENTS\n"
            "I. The First Day\n"
            "II. The Second Day\n"
            "III. The Third Day\n\n"
            "ILLUSTRATIONS\nSeveral plates.\n\n"
            "DIARY\n\n"
            + "Substantive unnumbered narrative. " * 300
        )

        doc = parser.parse(txt_file)

        assert len(doc.structural_elements) == 1
        assert doc.structural_elements[0].element_type == "document"
        assert "Substantive unnumbered narrative" in doc.structural_elements[0].content

    def test_multipart_roman_subsections_preserve_every_part(self, parser, tmp_path):
        txt_file = tmp_path / "multipart-treatise.txt"
        txt_file.write_text(
            "CONTENTS\nPART I\nI FIRST CLAIM\nII SECOND CLAIM\n"
            "PART II\nI FIRST OBJECTION\nII SECOND OBJECTION\n"
            "PART III\nI PROPOSED REMEDY\n\n"
            "PREFACE\nContext.\n\n"
            "PART I\nI\nFIRST CLAIM\nSubstantive first part.\n"
            "II\nSECOND CLAIM\nMore first part.\n\n"
            "PART II\nI\nFIRST OBJECTION\nSubstantive second part.\n"
            "II\nSECOND OBJECTION\nMore second part.\n\n"
            "PART III\nI\nPROPOSED REMEDY\nSubstantive final part.\n"
        )

        doc = parser.parse(txt_file)

        chapters = [element for element in doc.structural_elements if element.element_type == "chapter"]
        assert len(chapters) == 3
        assert chapters[0].content.startswith("PART I\nI\nFIRST CLAIM")
        assert "Substantive second part" in chapters[1].content
        assert "Substantive final part" in chapters[2].content

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

    def test_pdf_probe_checks_past_image_only_front_matter(self, tmp_path, monkeypatch):
        pdf_file = tmp_path / "text-after-cover.pdf"
        pdf_file.write_bytes(b"%PDF-1.7 test fixture")

        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        reader = SimpleNamespace(
            pages=[Page("") for _ in range(4)] + [Page("Historical narrative. " * 20)]
        )
        monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda _: reader))

        assert OcrParser().can_parse(pdf_file) is False
        assert isinstance(get_parser(pdf_file), PdfParser)


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
