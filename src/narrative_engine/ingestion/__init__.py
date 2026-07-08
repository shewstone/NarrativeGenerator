"""Ingestion module for Narrative Engine.

Converts raw sources (PDFs, EPUBs, text files) into structured chunks
ready for the extraction pipeline.

Pipeline:
    Raw file → Parser → Structural extraction → Smart chunking → Chunks

Supported formats:
    - PDF (with OCR for scanned documents)
    - EPUB
    - TXT / Markdown
    - HTML (for web archives)
"""

from narrative_engine.ingestion.chunker import Chunk, SmartChunker
from narrative_engine.ingestion.models import (
    IngestionConfig,
    IngestionResult,
    ParsedDocument,
    SourceMetadata,
)
from narrative_engine.ingestion.parsers import (
    EpubParser,
    HtmlParser,
    MarkdownParser,
    OcrParser,
    PdfParser,
    TxtParser,
)
from narrative_engine.ingestion.pipeline import IngestionPipeline

__all__ = [
    # Models
    "IngestionConfig",
    "IngestionResult",
    "ParsedDocument",
    "SourceMetadata",
    "Chunk",
    # Parsers
    "PdfParser",
    "EpubParser",
    "TxtParser",
    "MarkdownParser",
    "HtmlParser",
    "OcrParser",
    # Pipeline
    "SmartChunker",
    "IngestionPipeline",
]
