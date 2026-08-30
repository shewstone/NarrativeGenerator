"""Parsers for different document formats.

Each parser converts a raw file into a ParsedDocument with
structural elements and metadata.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

from narrative_engine.ingestion.models import (
    DocumentType,
    ParsedDocument,
    SourceFormat,
    SourceMetadata,
    StructuralElement,
)
from narrative_engine.observability import get_logger

logger = get_logger(__name__)

_FRONT_MATTER_MARKER = re.compile(
    r"(?im)^[ \t]*(produced by|project gutenberg|preface|contents|table of contents|"
    r"list of (?:maps|illustrations|plates)|transcriber's note)\b"
)

_SPELLED_CHAPTER_NUMBER = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty"
)


def _looks_like_book_front_matter(text: str) -> bool:
    """Identify title/credit/contents blocks that are not narrative evidence.

    A generic introductory paragraph remains ingestible.  We only suppress a
    pre-chapter block when it has a Gutenberg credit marker or at least two
    distinct conventional front-matter headings.
    """
    markers = {match.casefold() for match in _FRONT_MATTER_MARKER.findall(text)}
    return bool(markers & {"produced by", "project gutenberg"}) or len(markers) >= 2


def _roman_heading_signature(match: re.Match) -> tuple[str, str]:
    """Canonicalize a Roman heading so decorated TOC entries match the body.

    Contents pages commonly append a reign range and a widely separated page
    number (``I. OTHMAN (1288-1326)    13``), while the body uses ``I`` then
    ``OTHMAN``.  Removing only those conventional suffixes lets duplicate-run
    detection choose the substantive body without fuzzy title matching.
    """
    title = (match.group("same_line") or match.group("next_line") or "").strip()
    title = re.sub(r"\s{2,}\d+(?:\s*[-–]\s*\d+)?\s*$", "", title)
    title = re.sub(r"\s*\((?:[A-Z.]+\s*)?\d{1,4}\s*[-–]\s*\d{1,4}\)\s*$", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).casefold().strip()
    return match.group("roman"), title


def _roman_value(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(numeral):
        value = values[char]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def _chapter_id_key(chapter_id: str) -> str:
    """Normalize OCR variants used when matching contents and body runs.

    Library scans occasionally recognize a decorative ``1`` as ``t`` in a
    body heading (``Chapter t. Historical Setting``), while the contents page
    retains ``Chapter 1``. Treating only that complete heading token as one
    lets duplicate-run detection select the body without broad fuzzy matching.
    """
    normalized = chapter_id.casefold()
    return "1" if normalized == "t" else normalized


def _looks_like_contents_chapter_run(content: str, headings: list[re.Match]) -> bool:
    """Distinguish a compact contents run from a real multi-part book.

    Repeating ``Chapter I`` is not sufficient: books commonly restart their
    numbering at each part.  A contents run is compact, whereas substantive
    chapters contain hundreds or thousands of words.  The median makes the
    check tolerant of one unusually detailed contents entry.
    """
    if not headings:
        return False
    word_counts = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else heading.end()
        word_counts.append(len(content[heading.start() : end].split()))
    ordered = sorted(word_counts)
    median = ordered[len(ordered) // 2]
    return median < 500


def _looks_like_paginated_contents_heading(heading: re.Match) -> bool:
    """Return true for a bare-Roman contents entry ending in a page number.

    Collections of letters often number their actual items as ``LETTER I``
    while listing them in the contents as ``I. Recipient             17``.
    The bare-Roman detector cannot map those two different heading forms, so
    treating the contents run as chapters produces dozens of tiny chunks and
    leaves the whole body under the final TOC entry.  A widely separated page
    number is specific enough to reject that unresolved run safely.
    """
    title = heading.group("same_line") or heading.group("next_line") or ""
    return bool(re.search(r"\s{2,}\d+(?:\s*[-–]\s*\d+)?\s*$", title))


class BaseParser(ABC):
    """Base class for document parsers."""

    def __init__(self, format_type: SourceFormat):
        self.format_type = format_type

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse the file and return a ParsedDocument."""
        pass

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_work_id(self, file_path: Path) -> str:
        """Generate a work ID from file path."""
        # Use filename without extension as work ID
        # Sanitize to be URL-safe
        name = file_path.stem.lower()
        name = re.sub(r"[^a-z0-9]+", "-", name)
        name = name.strip("-")
        return name

    def _detect_document_type(self, file_path: Path, metadata: dict) -> DocumentType:
        """Detect document type from filename and metadata."""
        name_lower = file_path.name.lower()

        if any(x in name_lower for x in ["timeline", "chronology"]):
            return DocumentType.TIMELINE
        if "chapter" in name_lower:
            return DocumentType.CHAPTER
        if any(x in name_lower for x in ["article", "paper"]):
            return DocumentType.ARTICLE

        # Default to book for longer documents
        return DocumentType.BOOK


class TxtParser(BaseParser):
    """Parser for plain text files."""

    def __init__(self):
        super().__init__(SourceFormat.TXT)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".txt", ".text"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a text file."""
        logger.info("parsing_text_file", file_path=str(file_path))

        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Try to detect chapters/sections by common patterns
        structural_elements = self._extract_structure(content)

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.TXT,
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            word_count=len(content.split()),
            document_type=self._detect_document_type(file_path, {}),
        )

        return ParsedDocument(
            metadata=metadata,
            content=content,
            structural_elements=structural_elements,
        )

    def _extract_structure(self, content: str) -> list:
        """Extract structural elements from text using heuristics."""
        elements = []
        synthetic_chapter_start: int | None = None
        rejected_late_reference_run = False

        # Match common plain-text book headings, including Gutenberg forms
        # such as "CHAPTER IV--A.D. 220-1200" and CRLF source files.  The
        # previous expression required the chapter number to be followed by
        # whitespace or a colon, so punctuation/title-bearing headings were
        # silently missed and an entire book became one structural element.
        chapter_pattern = (
            r"(?im)^[ \t]*chapter[ \t]+"
            rf"(?P<chapter_id>\d+|[ivxlcdm]+|t(?=[.\s])|{_SPELLED_CHAPTER_NUMBER})"
            r"\b[^\r\n]*(?:\r?\n|$)"
        )
        chapters = list(re.finditer(chapter_pattern, content))

        # A contents page can repeat the book's complete ``Chapter I...``
        # run before the real headings. If the pre-first-heading material is
        # recognisable front matter and Chapter I occurs again, start from
        # that second occurrence. This avoids spending LLM calls on one-line
        # TOC entries while preserving legitimate multi-part chapter resets
        # in books without a front-matter contents block.
        before_first_chapter = content[: chapters[0].start()] if chapters else ""
        has_contents_heading = bool(
            re.search(r"(?im)^[ \t]*(?:table of )?contents\.?[ \t]*$", before_first_chapter)
        )
        if chapters and (_looks_like_book_front_matter(before_first_chapter) or has_contents_heading):
            first_id = _chapter_id_key(chapters[0].group("chapter_id"))
            repeated_first = [
                index
                for index, heading in enumerate(chapters)
                if _chapter_id_key(heading.group("chapter_id")) == first_id
            ]
            if len(repeated_first) > 1:
                repeat_index = repeated_first[1]
                possible_contents = chapters[:repeat_index]
                if _looks_like_contents_chapter_run(content, possible_contents):
                    chapters = chapters[repeat_index:]

                    # Some OCR country studies omit ``Chapter N`` from the
                    # body title pages but repeat it in the bibliography.
                    # Treating that late reference list as the book's only
                    # structure silently discards nearly all narrative text.
                    # Falling back to whole-document chunking is safer and
                    # preserves every page for manifest-level selection.
                    if chapters and chapters[0].start() > len(content) * 0.75:
                        chapters = []
                        rejected_late_reference_run = True

        # Multipart treatises often restart bare Roman subsection numbering
        # inside each ``PART I``, ``PART II``, ... section. Treating those
        # subsection headings as one continuous chapter sequence stops at the
        # first reset and silently drops every later part. Prefer the larger
        # part boundaries themselves; SmartChunker will still divide each
        # part to the normal word budget without losing content.
        if not chapters and not rejected_late_reference_run:
            part_pattern = r"(?im)^[ \t]*PART[ \t]+(?P<part_id>\d+|[IVXLCDM]+)[ \t]*\r?$"
            part_headings = list(re.finditer(part_pattern, content))
            part_runs: list[tuple[int, list[re.Match]]] = []
            for start_index, heading in enumerate(part_headings):
                raw_id = heading.group("part_id")
                value = int(raw_id) if raw_id.isdigit() else _roman_value(raw_id.upper())
                if value != 1:
                    continue
                run = [heading]
                expected = 2
                for candidate in part_headings[start_index + 1 :]:
                    raw_candidate = candidate.group("part_id")
                    candidate_value = (
                        int(raw_candidate)
                        if raw_candidate.isdigit()
                        else _roman_value(raw_candidate.upper())
                    )
                    if candidate_value == 1:
                        break
                    if candidate_value == expected:
                        run.append(candidate)
                        expected += 1
                if len(run) >= 2:
                    part_runs.append((start_index, run))
            if part_runs:
                _, chapters = max(part_runs, key=lambda item: (len(item[1]), item[0]))

        # Some public-domain histories use a bare Roman numeral followed by
        # an uppercase title ("IV\nTHE NIGER AND ISLAM") instead of the word
        # "Chapter". Require at least three such headings to avoid treating an
        # isolated numeral in prose/front matter as document structure.
        if not chapters and not rejected_late_reference_run:
            resolved_roman_body = False
            roman_heading_pattern = (
                r"(?m)^[ \t]*(?P<roman>[IVXLCDM]{1,8})\.?"
                r"(?:[ \t]+(?P<same_line>[A-Z][^\r\n]{2,})[ \t]*\r?$|"
                r"[ \t]*\r?\n(?:[ \t]*\r?\n){0,3}"
                r"[ \t]*(?P<next_line>[A-Z][^\r\n]{2,})[ \t]*\r?$)"
            )
            roman_headings = list(re.finditer(roman_heading_pattern, content))
            contents_heading = re.search(
                r"(?im)^[ \t]*(?:table of )?contents\.?[ \t]*$",
                content,
            )
            if contents_heading:
                # Ignore isolated catalogue initials before the actual table
                # of contents (for example ``C. EDMUND MAURICE``), which can
                # otherwise prevent discovery of the real I, II, III run.
                roman_headings = [
                    heading for heading in roman_headings if heading.start() > contents_heading.end()
                ]
                # When both contents and body use bare numerals, select the
                # latest full sequential run. Scanning for each expected
                # value skips misleading numerals embedded in wrapped titles
                # or prose; preferring the later equally long run selects the
                # substantive body over the compact contents list.
                sequential_runs = []
                for start_index, start_heading in enumerate(roman_headings):
                    if _roman_value(start_heading.group("roman")) != 1:
                        continue
                    run = [start_heading]
                    expected_value = 2
                    for heading in roman_headings[start_index + 1 :]:
                        if _roman_value(heading.group("roman")) == expected_value:
                            run.append(heading)
                            expected_value += 1
                    if len(run) >= 3:
                        sequential_runs.append((start_index, run))
                if len(sequential_runs) >= 2:
                    selected_start, roman_headings = max(
                        sequential_runs,
                        key=lambda item: (len(item[1]), item[0]),
                    )
                    resolved_roman_body = selected_start > 0
            # Gutenberg contents pages often repeat the exact first heading.
            # Start from its final occurrence so the TOC does not become a
            # set of tiny duplicate chapters.
            if roman_headings:
                first = roman_headings[0]
                first_signature = _roman_heading_signature(first)
                repeated = [
                    index
                    for index, heading in enumerate(roman_headings)
                    if _roman_heading_signature(heading) == first_signature
                ]
                if len(repeated) > 1:
                    resolved_roman_body = True
                    body_start = repeated[-1]
                    toc_candidates = roman_headings[:body_start]
                    toc_sequence = []
                    expected_value = 1
                    for heading in toc_candidates:
                        if _roman_value(heading.group("roman")) == expected_value:
                            toc_sequence.append(heading)
                            expected_value += 1

                    body_candidates = roman_headings[body_start:]
                    body_headings = []
                    cursor = 0
                    for expected in toc_sequence:
                        signature = _roman_heading_signature(expected)
                        found = next(
                            (
                                index
                                for index in range(cursor, len(body_candidates))
                                if _roman_heading_signature(body_candidates[index]) == signature
                            ),
                            None,
                        )
                        if found is None:
                            break
                        body_headings.append(body_candidates[found])
                        cursor = found + 1
                    roman_headings = body_headings if len(body_headings) >= 3 else body_candidates
                elif re.search(r"(?im)^[ \t]*(?:table of )?contents[ \t]*$", content[: first.start()]):
                    # A few books omit the numeral from the first body
                    # chapter while retaining it in the contents page. Find
                    # where a later TOC heading (normally II) repeats, then
                    # recover the first title as an exact standalone line in
                    # the gap. This keeps Chapter I without accepting fuzzy
                    # or arbitrary all-caps headings.
                    for toc_index in range(1, len(roman_headings)):
                        signature = _roman_heading_signature(roman_headings[toc_index])
                        later = [
                            index
                            for index in range(toc_index + 1, len(roman_headings))
                            if _roman_heading_signature(roman_headings[index]) == signature
                        ]
                        if not later:
                            continue
                        body_index = later[0]
                        first_title = re.escape(first_signature[1])
                        title_matches = list(
                            re.finditer(
                                rf"(?im)^[ \t]*{first_title}[ \t]*\r?$",
                                content[roman_headings[body_index - 1].end() : roman_headings[body_index].start()],
                            )
                        )
                        if title_matches:
                            resolved_roman_body = True
                            search_start = roman_headings[body_index - 1].end()
                            synthetic_chapter_start = search_start + title_matches[-1].start()
                            # Match the rest of the body against the exact TOC
                            # title sequence. This skips prose such as
                            # ``I. Queen Elizabeth ...`` between Chapters XII
                            # and XIII without truncating the chapter run.
                            toc_headings = roman_headings[:body_index]
                            body_headings = []
                            cursor = body_index
                            for expected in toc_headings[toc_index:]:
                                expected_signature = _roman_heading_signature(expected)
                                found = next(
                                    (
                                        index
                                        for index in range(cursor, len(roman_headings))
                                        if _roman_heading_signature(roman_headings[index]) == expected_signature
                                    ),
                                    None,
                                )
                                if found is None:
                                    break
                                body_headings.append(roman_headings[found])
                                cursor = found + 1
                            roman_headings = body_headings
                        break
                # A chapter run is sequential (I, II, III, ...). Stop at the
                # first jump/reset so index initials, outline bullets, or
                # footnote numerals later in the file are not mistaken for
                # hundreds of tiny chapters.
                sequential = [roman_headings[0]]
                previous_value = _roman_value(roman_headings[0].group("roman"))
                for heading in roman_headings[1:]:
                    value = _roman_value(heading.group("roman"))
                    if value != previous_value + 1:
                        break
                    sequential.append(heading)
                    previous_value = value
                roman_headings = sequential

                # If the only resolved sequence is visibly a paginated table
                # of contents, fall back to whole-document chunking.  This is
                # preferable to emitting the TOC as tiny chapters and losing
                # the correspondence/body that uses a different heading form.
                if (
                    contents_heading
                    and len(roman_headings) >= 3
                    and sum(_looks_like_paginated_contents_heading(heading) for heading in roman_headings)
                    / len(roman_headings)
                    >= 0.6
                ):
                    roman_headings = []
                elif (
                    contents_heading
                    and not resolved_roman_body
                    and len(roman_headings) >= 3
                    and _looks_like_contents_chapter_run(content, roman_headings)
                ):
                    # Some illustrated histories number only their contents
                    # entries; the body is a continuous diary or collection
                    # of unnumbered sections.  In that case the last TOC item
                    # otherwise absorbs most of the book while every earlier
                    # item becomes a tiny chunk. Preserve the whole body and
                    # let the normal word-size chunker split it instead.
                    roman_headings = []
            if len(roman_headings) + int(synthetic_chapter_start is not None) >= 3:
                chapters = roman_headings

        if chapters:
            chapter_starts = [heading.start() for heading in chapters]
            if synthetic_chapter_start is not None:
                chapter_starts.insert(0, synthetic_chapter_start)
            # Preserve any foreword/preamble before the first detected
            # chapter; chunking consumes structural_elements rather than the
            # top-level content field, so dropping it here loses source text.
            preamble = content[: chapter_starts[0]].strip()
            preamble_is_front_matter = _looks_like_book_front_matter(preamble) or bool(
                re.search(r"(?im)^[ \t]*(?:table of )?contents\.?[ \t]*$", preamble)
            )
            if preamble and not preamble_is_front_matter:
                elements.append(
                    StructuralElement(
                        element_type="document",
                        level=0,
                        content=preamble,
                    )
                )

            # Split content by chapters
            for i, start in enumerate(chapter_starts):
                end = chapter_starts[i + 1] if i + 1 < len(chapter_starts) else len(content)
                if i + 1 == len(chapter_starts):
                    # A standalone index is reference apparatus, not
                    # narrative evidence. Keeping it inside the final chapter
                    # creates spurious person/place episodes and needless LLM
                    # calls, so truncate only this unambiguous marker while
                    # retaining appendices and endnotes.
                    index_heading = re.search(r"(?im)^[ \t]*INDEX[ \t]*\r?$", content[start:end])
                    if index_heading:
                        end = start + index_heading.start()

                chapter_content = content[start:end].strip()
                # Extract title from first line if present
                lines = chapter_content.split("\n", 2)
                title = lines[0].strip() if len(lines) > 0 else None

                elements.append(
                    StructuralElement(
                        element_type="chapter",
                        level=0,
                        title=title,
                        content=chapter_content,
                    )
                )
        else:
            # No chapters found, treat as single element
            elements.append(
                StructuralElement(
                    element_type="document",
                    level=0,
                    content=content,
                )
            )

        return elements


class MarkdownParser(BaseParser):
    """Parser for Markdown files."""

    def __init__(self):
        super().__init__(SourceFormat.MARKDOWN)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".md", ".markdown"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a Markdown file, extracting headers as structure."""
        logger.info("parsing_markdown_file", file_path=str(file_path))

        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Parse headers for structure
        structural_elements = self._extract_markdown_structure(content)

        # Try to extract YAML frontmatter
        frontmatter = self._extract_frontmatter(content)

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.MARKDOWN,
            title=frontmatter.get("title"),
            author=frontmatter.get("author"),
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            word_count=len(content.split()),
            document_type=self._detect_document_type(file_path, frontmatter),
        )

        return ParsedDocument(
            metadata=metadata,
            content=content,
            structural_elements=structural_elements,
            raw_data={"frontmatter": frontmatter},
        )

    def _extract_markdown_structure(self, content: str) -> list:
        """Extract headers as structural elements."""
        elements = []

        # Remove frontmatter for structure analysis
        content_no_fm = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

        # Match headers (# ## ###)
        header_pattern = r"^(#{1,6})\s+(.+)$"
        lines = content_no_fm.split("\n")

        current_element = None
        element_content = []
        preamble_content = []
        saw_header = False

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous element
                if current_element:
                    current_element.content = "\n".join(element_content)
                    elements.append(current_element)
                elif not saw_header:
                    preamble = "\n".join(preamble_content).strip()
                    if preamble:
                        elements.append(
                            StructuralElement(
                                element_type="document",
                                level=0,
                                content=preamble,
                            )
                        )

                saw_header = True
                level = len(match.group(1)) - 1  # 0-indexed
                title = match.group(2).strip()
                current_element = StructuralElement(
                    element_type="section",
                    level=level,
                    title=title,
                )
                element_content = []
            else:
                if current_element:
                    element_content.append(line)
                else:
                    preamble_content.append(line)

        # Save last element
        if current_element:
            current_element.content = "\n".join(element_content)
            elements.append(current_element)

        # If no headers found, treat as single element
        if not saw_header:
            elements.append(
                StructuralElement(
                    element_type="document",
                    level=0,
                    content=content_no_fm,
                )
            )

        return elements

    def _extract_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter if present."""
        match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not match:
            return {}

        import yaml

        try:
            return yaml.safe_load(match.group(1)) or {}
        except Exception:
            return {}


class PdfParser(BaseParser):
    """Parser for text-bearing PDF files (requires pypdf)."""

    def __init__(self):
        super().__init__(SourceFormat.PDF)
        self._has_pdf_lib = None

    def _check_dependencies(self) -> bool:
        """Check if PDF parsing libraries are available."""
        if self._has_pdf_lib is None:
            self._has_pdf_lib = find_spec("pypdf") is not None
        return self._has_pdf_lib

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF file."""
        logger.info("parsing_pdf_file", file_path=str(file_path))

        if not self._check_dependencies():
            raise ImportError("PDF parsing requires pypdf. Install with: pip install pypdf")

        from pypdf import PdfReader

        full_text = []
        structural_elements = []
        page_count = 0

        pdf = PdfReader(file_path)
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, 1):
            try:
                text = page.extract_text()
            except Exception as exc:
                # A damaged page should not discard an otherwise usable
                # government report or scanned-and-OCRed historical volume.
                logger.warning(
                    "pdf_page_text_extraction_failed",
                    file_path=str(file_path),
                    page=page_number,
                    error=str(exc),
                )
                continue
            if text:
                full_text.append(text)

        content = "\n\n".join(full_text)

        # Try to detect chapters
        structural_elements = self._extract_chapters_from_text(content)

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.PDF,
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            total_pages=page_count,
            word_count=len(content.split()),
            document_type=self._detect_document_type(file_path, {}),
        )

        return ParsedDocument(
            metadata=metadata,
            content=content,
            structural_elements=structural_elements,
        )

    def _extract_chapters_from_text(self, content: str) -> list:
        """Extract chapters using heuristics."""
        elements = []

        # Look for "Chapter X" or Roman numerals
        chapter_pattern = r"(?:^|\n\n)(?:Chapter|CHAPTER)\s+(\d+|I|II|III|IV|V|VI|VII|VIII|IX|X)[\s:]*\n"
        matches = list(re.finditer(chapter_pattern, content))

        if len(matches) > 2:  # Only if we find multiple chapters
            preamble = content[: matches[0].start()].strip()
            if preamble:
                elements.append(
                    StructuralElement(
                        element_type="document",
                        level=0,
                        content=preamble,
                    )
                )
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

                chapter_content = content[start:end].strip()
                elements.append(
                    StructuralElement(
                        element_type="chapter",
                        level=0,
                        title=match.group(0).strip(),
                        content=chapter_content,
                    )
                )
        else:
            # No chapters, treat as single element
            elements.append(
                StructuralElement(
                    element_type="document",
                    level=0,
                    content=content,
                )
            )

        return elements


class EpubParser(BaseParser):
    """Parser for EPUB files."""

    def __init__(self):
        super().__init__(SourceFormat.EPUB)
        self._has_epub_lib = None

    def _check_dependencies(self) -> bool:
        """Check if EPUB parsing libraries are available."""
        if self._has_epub_lib is None:
            self._has_epub_lib = find_spec("ebooklib") is not None
        return self._has_epub_lib

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".epub"

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse an EPUB file."""
        logger.info("parsing_epub_file", file_path=str(file_path))

        if not self._check_dependencies():
            raise ImportError("EPUB parsing requires ebooklib. Install with: " "pip install ebooklib beautifulsoup4")

        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(file_path)

        # Extract metadata
        title = self._get_metadata(book, "title")
        author = self._get_metadata(book, "creator")
        language = self._get_metadata(book, "language") or "en"

        # Extract chapters
        structural_elements = []
        full_text = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n")

                if text.strip():
                    full_text.append(text)
                    # Try to extract title from h1/h2
                    h1 = soup.find("h1")
                    h2 = soup.find("h2")
                    title_elem = h1 or h2
                    chapter_title = title_elem.get_text() if title_elem else None

                    structural_elements.append(
                        StructuralElement(
                            element_type="chapter",
                            level=0,
                            title=chapter_title,
                            content=text,
                        )
                    )

        content = "\n\n".join(full_text)

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.EPUB,
            title=title,
            author=author,
            language=language,
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            word_count=len(content.split()),
            document_type=DocumentType.BOOK,
        )

        return ParsedDocument(
            metadata=metadata,
            content=content,
            structural_elements=structural_elements,
        )

    def _get_metadata(self, book, field: str) -> Optional[str]:
        """Extract metadata from EPUB."""
        try:
            data = book.get_metadata("DC", field)
            return data[0][0] if data else None
        except Exception:
            return None


class HtmlParser(BaseParser):
    """Parser for HTML files (web archives, etc.)."""

    def __init__(self):
        super().__init__(SourceFormat.HTML)

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".html", ".htm"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse an HTML file."""
        logger.info("parsing_html_file", file_path=str(file_path))

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError("HTML parsing requires beautifulsoup4. Install with: pip install beautifulsoup4") from exc

        content = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text()

        # Extract headings for structure
        structural_elements = []
        text_content = soup.get_text(separator="\n")

        # Try to find article or main content
        article = soup.find("article") or soup.find("main") or soup.find("body")
        if article:
            text_content = article.get_text(separator="\n")

        structural_elements.append(
            StructuralElement(
                element_type="document",
                level=0,
                title=title,
                content=text_content,
            )
        )

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.HTML,
            title=title,
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            word_count=len(text_content.split()),
            document_type=DocumentType.ARTICLE,
        )

        return ParsedDocument(
            metadata=metadata,
            content=text_content,
            structural_elements=structural_elements,
        )


class OcrParser(BaseParser):
    """Parser for image-based PDFs requiring OCR.

    This is a wrapper that uses Tesseract OCR to extract text
    from images or scanned documents.
    """

    def __init__(self):
        super().__init__(SourceFormat.PDF)
        self._has_ocr = None
        self.page_batch_size = 4

    def _check_dependencies(self) -> bool:
        """Check if OCR libraries are available."""
        if self._has_ocr is None:
            if any(find_spec(package) is None for package in ("pytesseract", "PIL", "pdf2image")):
                self._has_ocr = False
                return False
            try:
                import pytesseract

                # Verify tesseract is installed
                pytesseract.get_tesseract_version()
                self._has_ocr = True
            except Exception:
                self._has_ocr = False
        return self._has_ocr

    def can_parse(self, file_path: Path) -> bool:
        """Check if file needs OCR."""
        if file_path.suffix.lower() != ".pdf":
            return False

        # Check if PDF has extractable text
        try:
            from pypdf import PdfReader

            pdf = PdfReader(file_path)
            # Covers and publishing pages are often image-only. Probe far
            # enough to reach the body without scanning the entire document.
            for page in pdf.pages[:8]:
                try:
                    text = page.extract_text()
                except Exception:
                    continue
                if text and len(text.strip()) > 50:
                    return False  # Has text, doesn't need OCR
            return True  # No text found, needs OCR
        except Exception:
            return True  # Error, assume needs OCR

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF using OCR."""
        logger.info("parsing_pdf_with_ocr", file_path=str(file_path))

        if not self._check_dependencies():
            raise ImportError(
                "OCR requires pytesseract and Pillow. Install with: "
                "pip install pytesseract pillow pdf2image. "
                "Also install Tesseract: brew install tesseract (macOS)"
            )

        import pytesseract
        from pdf2image import convert_from_path, pdfinfo_from_path

        page_count = int(pdfinfo_from_path(file_path).get("Pages", 0))
        if page_count <= 0:
            raise ValueError(f"Could not determine PDF page count for {file_path}")
        full_text = []

        # Rendering a whole scanned book at 300 DPI can consume gigabytes.
        # Bound live image memory to a small page batch and close every PIL
        # image promptly after OCR.
        for first_page in range(1, page_count + 1, self.page_batch_size):
            last_page = min(page_count, first_page + self.page_batch_size - 1)
            images = convert_from_path(
                file_path,
                dpi=300,
                first_page=first_page,
                last_page=last_page,
                thread_count=1,
            )
            try:
                for offset, image in enumerate(images):
                    page_number = first_page + offset
                    logger.debug("ocr_page", file_path=str(file_path), page=page_number)
                    full_text.append(pytesseract.image_to_string(image))
            finally:
                for image in images:
                    image.close()

        content = "\n\n".join(full_text)

        metadata = SourceMetadata(
            work_id=self._extract_work_id(file_path),
            source_format=SourceFormat.PDF,
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            total_pages=page_count,
            word_count=len(content.split()),
            document_type=self._detect_document_type(file_path, {}),
            requires_ocr=True,
        )

        return ParsedDocument(
            metadata=metadata,
            content=content,
            structural_elements=[
                StructuralElement(
                    element_type="document",
                    level=0,
                    content=content,
                )
            ],
        )


# Registry of available parsers
PARSERS: list = [
    TxtParser(),
    MarkdownParser(),
    # Probe for image-only PDFs before the generic PDF parser, whose
    # extension-only can_parse() would otherwise make OCR unreachable.
    OcrParser(),
    PdfParser(),
    EpubParser(),
    HtmlParser(),
]


def get_parser(file_path: Path) -> Optional[BaseParser]:
    """Get appropriate parser for file."""
    for parser in PARSERS:
        if parser.can_parse(file_path):
            return parser
    return None
