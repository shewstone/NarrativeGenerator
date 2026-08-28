"""Ingestion pipeline orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from narrative_engine.ingestion.chunker import Chunk, SmartChunker
from narrative_engine.ingestion.models import (
    IngestionConfig,
    IngestionResult,
    ParsedDocument,
)
from narrative_engine.ingestion.parsers import get_parser

# LogTimer comes from logging_config: the observability module exports a
# @contextmanager of the same name with an incompatible (operation-first)
# signature, and the calls below use the class form (logger, operation, **ctx).
from narrative_engine.logging_config import LogTimer
from narrative_engine.observability import (
    EpisodeNotFoundError,
    get_logger,
)

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline.

    Pipeline stages:
    1. Parse source file (PDF, EPUB, etc.)
    2. Extract structure (chapters, sections)
    3. Chunk with narrative boundary respect
    4. Output to JSON/JSONL
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig.default()
        self.chunker = SmartChunker(config)

    def ingest_file(self, file_path: Path, output_dir: Path) -> IngestionResult:
        """Ingest a single file through the pipeline."""
        start_time = time.time()
        errors = []
        warnings = []

        with LogTimer(logger, "ingest_file", file_path=str(file_path)):
            try:
                # Step 1: Parse
                parsed = self._parse(file_path)
                if not parsed:
                    errors.append(f"Failed to parse {file_path}")
                    return IngestionResult(
                        success=False,
                        source_path=file_path,
                        errors=errors,
                    )

                # Step 2: Chunk
                chunks = self._chunk(parsed)
                if not chunks:
                    warnings.append(f"No chunks produced from {file_path}")

                # Step 3: Output
                output_files = self._output(chunks, output_dir)

                duration = time.time() - start_time

                logger.info(
                    "ingestion_complete",
                    file_path=str(file_path),
                    chunks=len(chunks),
                    duration_seconds=duration,
                )

                return IngestionResult(
                    success=True,
                    source_path=file_path,
                    chunks_created=len(chunks),
                    output_files=output_files,
                    metadata=parsed.metadata,
                    errors=errors if errors else None,
                    warnings=warnings if warnings else None,
                    duration_seconds=duration,
                )

            except Exception as e:
                logger.error(
                    "ingestion_failed",
                    file_path=str(file_path),
                    error=str(e),
                    exc_info=True,
                )
                return IngestionResult(
                    success=False,
                    source_path=file_path,
                    errors=[str(e)],
                    duration_seconds=time.time() - start_time,
                )

    def ingest_directory(
        self,
        source_dir: Path,
        output_dir: Path,
        pattern: str = "*",
    ) -> List[IngestionResult]:
        """Ingest all matching files in a directory."""
        logger.info(
            "ingesting_directory",
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            pattern=pattern,
        )

        results = []
        for file_path in source_dir.rglob(pattern):
            if file_path.is_file():
                result = self.ingest_file(file_path, output_dir)
                results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)

        logger.info(
            "directory_ingestion_complete",
            files_processed=len(results),
            successful=success_count,
            failed=len(results) - success_count,
            total_chunks=total_chunks,
        )

        return results

    def _parse(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a source file."""
        parser = get_parser(file_path)
        if not parser:
            raise EpisodeNotFoundError(
                str(file_path),
                reason="No suitable parser found",
            )

        return parser.parse(file_path)

    def _chunk(self, parsed: ParsedDocument) -> List[Chunk]:
        """Chunk a parsed document."""
        return self.chunker.chunk_document(parsed)

    def _output(self, chunks: List[Chunk], output_dir: Path) -> List[Path]:
        """Write chunks to output files."""
        if not chunks:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []

        if self.config.output_format == "json":
            # Single JSON file with all chunks
            output_file = output_dir / f"{chunks[0].source_work_id}_chunks.json"
            data = {
                "work_id": chunks[0].source_work_id,
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "chapter_title": c.chapter_title,
                        "section_title": c.section_title,
                        "element_type": c.element_type,
                        "chunk_index": c.chunk_index,
                        "estimated_tokens": c.estimated_tokens,
                    }
                    for c in chunks
                ],
            }
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            output_files.append(output_file)

        elif self.config.output_format == "jsonl":
            # JSONL: one chunk per line
            output_file = output_dir / f"{chunks[0].source_work_id}_chunks.jsonl"
            with open(output_file, "w") as f:
                for chunk in chunks:
                    json.dump(
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "chapter_title": chunk.chapter_title,
                            "section_title": chunk.section_title,
                            "element_type": chunk.element_type,
                            "chunk_index": chunk.chunk_index,
                            "estimated_tokens": chunk.estimated_tokens,
                        },
                        f,
                    )
                    f.write("\n")
            output_files.append(output_file)

        return output_files
