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

        # Pattern: Chapter X or CHAPTER X
        chapter_pattern = r"(?:^|\n)(?:Chapter|CHAPTER)\s+(\d+|\w+)[\s:]*\n"
        chapters = list(re.finditer(chapter_pattern, content))

        if chapters:
            # Preserve any foreword/preamble before the first detected
            # chapter; chunking consumes structural_elements rather than the
            # top-level content field, so dropping it here loses source text.
            preamble = content[: chapters[0].start()].strip()
            if preamble:
                elements.append(
                    StructuralElement(
                        element_type="document",
                        level=0,
                        content=preamble,
                    )
                )

            # Split content by chapters
            for i, match in enumerate(chapters):
                start = match.start()
                end = chapters[i + 1].start() if i + 1 < len(chapters) else len(content)

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
    """Parser for PDF files (requires PyPDF2 or pdfplumber)."""

    def __init__(self):
        super().__init__(SourceFormat.PDF)
        self._has_pdf_lib = None

    def _check_dependencies(self) -> bool:
        """Check if PDF parsing libraries are available."""
        if self._has_pdf_lib is None:
            self._has_pdf_lib = find_spec("pdfplumber") is not None
        return self._has_pdf_lib

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a PDF file."""
        logger.info("parsing_pdf_file", file_path=str(file_path))

        if not self._check_dependencies():
            raise ImportError("PDF parsing requires pdfplumber. Install with: " "pip install pdfplumber")

        import pdfplumber

        full_text = []
        structural_elements = []
        page_count = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)

                    # Simple heuristic: large font = chapter header
                    # (pdfplumber can extract font info but this is basic)

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
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:3]:  # Check first 3 pages
                    text = page.extract_text()
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
