"""Drop-directory watcher (T7, docs/tickets/T7-drop-directory-watcher.md).

Always-on polling loop: files dropped into the watch directory are hashed,
guarded against duplicates, parsed, chunked, and — when an LLM key is
configured — extracted into episodes, embedded, and composed into arc
instances. Every lifecycle transition lands on a SourceDocument row, which
is what the dashboard (T8) renders as the processing queue.

Polling, not inotify: dependency-free, and it works identically on Linux,
macOS bind mounts, and CI.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from narrative_engine.logging_config import get_logger
from narrative_engine.models import SourceDocument, SourceDocumentStatus
from narrative_engine.storage.repositories import (
    EpisodeRepository,
    SourceDocumentRepository,
)

logger = get_logger(__name__)


class ClaimLostError(RuntimeError):
    """Raised when another worker has taken over an expired claim."""


IGNORED_SUFFIXES = {".part", ".tmp", ".crdownload", ".swp"}
HASH_CHUNK_SIZE = 1024 * 1024


def _hash_file(path: Path) -> tuple[str, int]:
    """Hash a file without retaining its complete contents in memory."""
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def llm_configured() -> bool:
    """Extraction only runs when a model key is actually available; without
    one, files still ingest and the row says extraction_ran=False —
    visible degradation, never a silent skip."""
    return any(os.getenv(name) for name in ("NE_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"))


@dataclass
class WatcherConfig:
    watch_dir: Path
    interval_seconds: float = 3.0
    # Files must be untouched this long before pickup, so half-copied
    # files are never processed.
    settle_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        return cls(
            watch_dir=Path(os.getenv("NE_WATCH_DIR", "data/raw")),
            interval_seconds=float(os.getenv("NE_WATCH_INTERVAL", "3.0")),
            settle_seconds=float(os.getenv("NE_WATCH_SETTLE", "2.0")),
        )


class DocumentProcessor:
    """Processes one dropped file end-to-end. Collaborators are injectable
    so tests never need a model download or an API key."""

    def __init__(self, extractor=None, embedder=None, lease_seconds: float = 3600.0) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._lease_seconds = lease_seconds
        self._owns_extractor = extractor is None

    async def aclose(self) -> None:
        """Release network resources owned by a lazily-created extractor."""
        if not self._owns_extractor or self._extractor is None:
            return
        close = getattr(self._extractor, "aclose", None)
        if close is not None:
            await close()

    def _get_extractor(self):
        if self._extractor is None and llm_configured():
            from narrative_engine.extraction.pipeline import ExtractionOrchestrator

            self._extractor = ExtractionOrchestrator()
        return self._extractor

    def _get_embedder(self):
        if self._embedder is None:
            from narrative_engine.retrieval.embeddings import EmbeddingGenerator

            self._embedder = EmbeddingGenerator()
        return self._embedder

    def _lease_expiry(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)

    async def _heartbeat_claim(
        self,
        session_factory,
        document_id: UUID,
        claim_token: UUID,
        stopped: asyncio.Event,
        claim_lost: asyncio.Event,
    ) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                async with session_factory() as heartbeat_session:
                    repo = SourceDocumentRepository(heartbeat_session)
                    renewed = await repo.renew_claim(
                        document_id,
                        claim_token,
                        self._lease_expiry(),
                    )
                    await heartbeat_session.commit()
            except Exception as exc:
                logger.error(
                    "document_claim_heartbeat_failed",
                    document_id=str(document_id),
                    error=str(exc),
                )
                claim_lost.set()
                return
            if not renewed:
                claim_lost.set()
                return

    async def process_file(self, session: AsyncSession, path: Path) -> Optional[SourceDocument]:
        """Hash, dedupe-guard, and process one file. Returns the document
        row, or None when this (hash, filename) was already recorded."""
        repo = SourceDocumentRepository(session)

        content_hash, size_bytes = _hash_file(path)

        existing = await repo.get_by_hash_and_filename(content_hash, path.name)
        if existing is not None:
            if existing.status not in {
                SourceDocumentStatus.QUEUED,
                SourceDocumentStatus.PROCESSING,
            }:
                return None
            document = existing
        else:
            document = None

        original = await repo.get_original_by_hash(content_hash) if document is None else None
        if original is not None:
            document = SourceDocument(
                filename=path.name,
                content_hash=content_hash,
                size_bytes=size_bytes,
                status=SourceDocumentStatus.DUPLICATE,
                duplicate_of=original.id,
                error=f"Same content as {original.filename!r}; not reprocessed",
            )
            await repo.create(document)
            logger.info(
                "duplicate_source_rejected",
                filename=path.name,
                duplicate_of=original.filename,
            )
            return document

        if document is None:
            document = SourceDocument(
                filename=path.name,
                content_hash=content_hash,
                size_bytes=size_bytes,
            )
            try:
                async with session.begin_nested():
                    await repo.create(document)
            except IntegrityError:
                # Another watcher inserted this exact drop first.
                return None

        claim_token = uuid4()
        if not await repo.claim_available(
            document.id,
            claim_token,
            self._lease_expiry(),
        ):
            return None
        document.status = SourceDocumentStatus.PROCESSING
        document.error = None
        # Commit checkpoints so the dashboard's polling session sees status
        # transitions and per-chunk progress DURING long LLM runs, not just
        # at the end of the scan (READ COMMITTED hides uncommitted flushes).
        await session.commit()

        session_factory = async_sessionmaker(bind=session.bind, expire_on_commit=False)
        heartbeat_stopped = asyncio.Event()
        claim_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_claim(
                session_factory,
                document.id,
                claim_token,
                heartbeat_stopped,
                claim_lost,
            )
        )
        try:
            await self._run_pipeline(
                session,
                path,
                document,
                repo,
                claim_token,
                claim_lost,
            )
            if claim_lost.is_set():
                raise ClaimLostError("document claim was replaced")
            document.status = SourceDocumentStatus.COMPLETED
        except ClaimLostError:
            await session.rollback()
            return None
        except Exception as exc:  # one bad file must not stop the loop
            await session.rollback()
            logger.error("document_processing_failed", filename=path.name, error=str(exc))
            document.status = SourceDocumentStatus.FAILED
            document.error = str(exc)
        finally:
            heartbeat_stopped.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

        if not await repo.update_claimed(document, claim_token):
            await session.rollback()
            return None
        await session.commit()
        return document

    async def _run_pipeline(
        self,
        session: AsyncSession,
        path: Path,
        document: SourceDocument,
        repo: SourceDocumentRepository,
        claim_token: UUID,
        claim_lost: asyncio.Event,
    ) -> None:
        from narrative_engine.ingestion.chunker import SmartChunker
        from narrative_engine.ingestion.parsers import get_parser

        parser = get_parser(path)
        if parser is None:
            raise ValueError(f"No parser for file type: {path.suffix!r}")

        parsed = parser.parse(path)
        chunks = SmartChunker().chunk_document(parsed)
        # ParsedDocument retains the full text plus per-section copies. Once
        # chunking is complete, only the chunks are needed for the long LLM
        # phase, so release that duplicate document-sized object promptly.
        del parsed
        document.chunks_created = len(chunks)
        if not await repo.update_claimed(document, claim_token):
            raise ClaimLostError("document claim was replaced")
        await session.commit()  # total chunk count visible immediately

        extractor = self._get_extractor()
        if extractor is None:
            document.extraction_ran = False
            return

        episode_repo = EpisodeRepository(session)
        embedder = self._get_embedder()
        arc_types_seen = set()
        remaining_chunks = chunks[document.chunks_processed :]
        if not remaining_chunks:
            arc_types_seen.update(await episode_repo.get_arc_types_for_chunks([chunk.chunk_id for chunk in chunks]))

        for chunk in remaining_chunks:
            async with session.begin_nested():
                result = await extractor.process_text(
                    text=chunk.content,
                    source_chunk_id=chunk.chunk_id,
                    session=session,
                )
                if result.errors:
                    raise RuntimeError("; ".join(result.errors))
                for episode in result.episodes:
                    await episode_repo.update_embedding(
                        episode.id,
                        embedder.generate_surface_embedding(episode),
                        kind="surface",
                    )
                    await episode_repo.update_embedding(
                        episode.id,
                        embedder.generate_structural_embedding(episode),
                        kind="structural",
                    )
                    if episode.arc_type:
                        arc_types_seen.add(episode.arc_type)
                if claim_lost.is_set():
                    raise ClaimLostError("document claim was replaced")
            document.episodes_created += len(result.episodes)
            document.chunks_processed += 1
            # Per-chunk checkpoint: progress survives a crash mid-book and
            # the dashboard renders it live. Also makes extracted episodes
            # durable as they land instead of all-or-nothing at file end.
            if not await repo.update_claimed(document, claim_token):
                raise ClaimLostError("document claim was replaced")
            await session.commit()

        # Composition pass: stitch the new episodes (plus any existing
        # same-scope ones) into arc instances the dashboard can render.
        if arc_types_seen:
            from narrative_engine.composition.pipeline import CompositionPipeline

            composer = CompositionPipeline(session)
            for arc_type in arc_types_seen:
                instances = await composer.compose_arc_instances(arc_type)
                await composer.reconcile_instances(instances, arc_type)

        document.extraction_ran = True


def _settled_files(watch_dir: Path, settle_seconds: float) -> list[Path]:
    if not watch_dir.exists():
        return []
    now = time.time()
    files = []
    for path in sorted(watch_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if now - path.stat().st_mtime < settle_seconds:
            continue  # still being copied
        files.append(path)
    return files


async def scan_once(
    session: AsyncSession,
    config: WatcherConfig,
    processor: DocumentProcessor,
) -> list[SourceDocument]:
    """One pass over the watch directory. Returns rows touched this scan."""
    touched = []
    for path in _settled_files(config.watch_dir, config.settle_seconds):
        try:
            document = await processor.process_file(session, path)
        except Exception as exc:  # unreadable file etc. — keep scanning
            logger.error("watcher_scan_error", path=str(path), error=str(exc))
            continue
        if document is not None:
            touched.append(document)
    return touched


async def watch_loop(config: Optional[WatcherConfig] = None) -> None:
    """The always-on loop (run as an asyncio task by the API lifespan)."""
    from narrative_engine.storage.database import db_manager

    config = config or WatcherConfig.from_env()
    processor = DocumentProcessor()
    logger.info(
        "watcher_started",
        watch_dir=str(config.watch_dir),
        interval=config.interval_seconds,
        llm_configured=llm_configured(),
    )
    try:
        while True:
            try:
                async with db_manager.session() as session:
                    touched = await scan_once(session, config, processor)
                if touched:
                    logger.info("watcher_scan_complete", processed=len(touched))
            except asyncio.CancelledError:
                logger.info("watcher_stopped")
                raise
            except Exception as exc:  # DB hiccup etc. — the loop survives
                logger.error("watcher_loop_error", error=str(exc))
            await asyncio.sleep(config.interval_seconds)
    finally:
        await processor.aclose()
