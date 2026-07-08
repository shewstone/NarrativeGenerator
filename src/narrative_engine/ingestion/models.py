"""Data models for ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class SourceFormat(str, Enum):
    """Supported source formats."""

    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """Type of document for processing hints."""

    BOOK = "book"
    ARTICLE = "article"
    CHAPTER = "chapter"
    TIMELINE = "timeline"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


@dataclass
class SourceMetadata:
    """Metadata extracted from source document."""

    # Identifiers
    work_id: str  # Unique identifier for this work
    source_format: SourceFormat

    # Bibliographic info
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None

    # Content info
    language: str = "en"
    total_pages: Optional[int] = None
    word_count: Optional[int] = None

    # Historiographic school if known
    historiographic_school: Optional[str] = None
    # e.g., "Marxist", "Whig", "Annales", "Cliometric"

    # Source of the file
    source_url: Optional[str] = None
    archive_collection: Optional[str] = None

    # Processing hints
    document_type: DocumentType = DocumentType.UNKNOWN
    requires_ocr: bool = False

    # Provenance
    file_path: Optional[Path] = None
    file_hash: Optional[str] = None  # SHA256 of file for deduplication
    ingested_at: datetime = None

    def __post_init__(self):
        if self.ingested_at is None:
            self.ingested_at = datetime.utcnow()


@dataclass
class StructuralElement:
    """A structural element in the document (chapter, section, etc.)."""

    element_type: str  # "chapter", "section", "part", etc.
    level: int  # Hierarchy level (0 = top, 1 = subsection, etc.)
    title: Optional[str] = None
    content: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    children: List["StructuralElement"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class ParsedDocument:
    """Result of parsing a source document."""

    metadata: SourceMetadata
    content: str  # Full text content
    structural_elements: List[StructuralElement]

    # Raw format-specific data (for debugging/provenance)
    raw_data: Optional[Dict[str, Any]] = None

    # Processing info
    parsed_at: datetime = None
    parser_version: str = "1.0.0"

    def __post_init__(self):
        if self.parsed_at is None:
            self.parsed_at = datetime.utcnow()

    def get_text_by_structure(self) -> List[tuple[str, str, int]]:
        """Extract (element_type, title, content) tuples from structure."""
        results = []

        def traverse(element: StructuralElement, parent_title: str = ""):
            full_title = f"{parent_title} - {element.title}" if parent_title and element.title else (element.title or parent_title)
            if element.content.strip():
                results.append((element.element_type, full_title, element.content))
            for child in element.children:
                traverse(child, full_title)

        for elem in self.structural_elements:
            traverse(elem)
        return results


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""

    # Chunking parameters
    target_chunk_size: int = 4000  # Target tokens per chunk
    min_chunk_size: int = 1000  # Minimum tokens
    max_chunk_size: int = 8000  # Maximum tokens
    overlap_tokens: int = 200  # Overlap between chunks

    # Structural boundaries (don't split across these)
    respect_chapters: bool = True
    respect_sections: bool = True

    # OCR settings
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300

    # Metadata extraction
    extract_metadata_from_filename: bool = True
    extract_metadata_from_content: bool = True

    # Output settings
    output_format: str = "json"  # json, jsonl
    preserve_structure: bool = True  # Include structural metadata in chunks

    @classmethod
    def default(cls) -> "IngestionConfig":
        """Get default configuration."""
        return cls()


@dataclass
class IngestionResult:
    """Result of ingesting a document."""

    # Success / failure
    success: bool
    source_path: Path
    chunks_created: int = 0

    # Outputs
    output_files: List[Path] = None
    metadata: Optional[SourceMetadata] = None

    # Errors
    errors: List[str] = None
    warnings: List[str] = None

    # Processing info
    duration_seconds: float = 0.0
    processed_at: datetime = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.processed_at is None:
            self.processed_at = datetime.utcnow()
