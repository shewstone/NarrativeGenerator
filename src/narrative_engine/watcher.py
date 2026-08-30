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
import json
import os
import re
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
from narrative_engine.models import Episode, SourceDocument, SourceDocumentStatus
from narrative_engine.storage.repositories import (
    EpisodeRepository,
    SourceDocumentRepository,
)

logger = get_logger(__name__)


class ClaimLostError(RuntimeError):
    """Raised when another worker has taken over an expired claim."""


IGNORED_SUFFIXES = {".part", ".tmp", ".crdownload", ".swp"}
HASH_CHUNK_SIZE = 1024 * 1024
_DATED_SOURCE_RE = re.compile(r"(?:^|[-_])(\d{4})(?:-(\d{2})-(\d{2}))?$")


def _hash_file(path: Path) -> tuple[str, int]:
    """Hash a file without retaining its complete contents in memory."""
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _source_publication_date(
    filename: str,
    provenance_path: Optional[Path] = None,
) -> Optional[datetime]:
    """Read a conservative publication date from a dated corpus filename.

    Exact ISO dates remain exact. A year-only suffix is treated as available
    at year end so a backtest earlier in that year cannot see it.
    """
    if provenance_path is not None:
        try:
            registry = json.loads(provenance_path.read_text(encoding="utf-8"))
            entry = registry.get(filename)
            if isinstance(entry, dict) and "published_at" in entry:
                value = entry["published_at"]
                # An explicit null means the source date is unknown. It must
                # suppress filename inference because descriptive filenames
                # often contain an event year rather than a publication year.
                if value is None:
                    return None
                if isinstance(value, str):
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return (
                        parsed.replace(tzinfo=timezone.utc)
                        if parsed.tzinfo is None
                        else parsed.astimezone(timezone.utc)
                    )
                logger.warning("source_provenance_date_invalid", filename=filename)
                return None
        except (AttributeError, json.JSONDecodeError, OSError, ValueError):
            logger.warning("source_provenance_unreadable", filename=filename)

    match = _DATED_SOURCE_RE.search(Path(filename).stem)
    if match is None:
        return None
    year, month, day = match.groups()
    try:
        if month is None:
            return datetime(int(year), 12, 31, tzinfo=timezone.utc)
        return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
    except ValueError:
        return None


def _source_chunk_ranges(
    filename: str,
    provenance_path: Path,
) -> tuple[tuple[int, int], ...] | None:
    """Return intentionally selected inclusive chunk ranges, if configured."""
    try:
        registry = json.loads(provenance_path.read_text(encoding="utf-8"))
        raw_ranges = registry.get(filename, {}).get("include_chunk_ranges")
        if raw_ranges is None:
            return None
        ranges = tuple((int(item[0]), int(item[1])) for item in raw_ranges)
        if any(start < 0 or end < start for start, end in ranges):
            raise ValueError("invalid range")
        return ranges
    except (AttributeError, IndexError, json.JSONDecodeError, OSError, TypeError, ValueError):
        logger.warning("source_chunk_ranges_unreadable", filename=filename)
        return None


def _source_chunks_preselected(filename: str, provenance_path: Path) -> bool:
    """Whether range selection already supplied the narrative quality gate."""
    try:
        registry = json.loads(provenance_path.read_text(encoding="utf-8"))
        return registry.get(filename, {}).get("chunks_preselected") is True
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _constrain_episode_to_source_date(
    episode: Episode,
    source_published_at: Optional[datetime],
) -> bool:
    """Keep extracted history on the evidence side of its source date.

    Returns ``False`` when the episode begins after publication and therefore
    cannot be an observed historical event in that source.  Ongoing episodes
    are retained, but any end inferred beyond publication is capped at the
    last date the source could have observed.  This is deliberately
    conservative: forecasts and plans must not masquerade as realised events
    in walk-forward evaluation.
    """
    if source_published_at is None:
        return True

    published = (
        source_published_at.replace(tzinfo=timezone.utc)
        if source_published_at.tzinfo is None
        else source_published_at.astimezone(timezone.utc)
    )
    if episode.start_year is not None and episode.start_year > published.year:
        return False
    if episode.start_date is not None:
        start = (
            episode.start_date.replace(tzinfo=timezone.utc)
            if episode.start_date.tzinfo is None
            else episode.start_date.astimezone(timezone.utc)
        )
        if start > published:
            return False

    if episode.end_year is not None and episode.end_year > published.year:
        episode.end_year = published.year
    if episode.end_date is not None:
        end = (
            episode.end_date.replace(tzinfo=timezone.utc)
            if episode.end_date.tzinfo is None
            else episode.end_date.astimezone(timezone.utc)
        )
        if end > published:
            episode.end_date = published
    return True


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
    # Claims are renewed every third of this interval. Keeping the lease
    # short bounds restart recovery when a container is killed before its
    # cancellation handler can release the active document.
    lease_seconds: float = 180.0

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        return cls(
            watch_dir=Path(os.getenv("NE_WATCH_DIR", "data/raw")),
            interval_seconds=float(os.getenv("NE_WATCH_INTERVAL", "3.0")),
            settle_seconds=float(os.getenv("NE_WATCH_SETTLE", "2.0")),
            lease_seconds=float(os.getenv("NE_DOCUMENT_LEASE_SECONDS", "180.0")),
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
        except asyncio.CancelledError:
            # A normal server restart cancels the watcher task. Release its
            # lease immediately so the next process can resume from the last
            # committed chunk instead of leaving the book stranded until the
            # one-hour lease expires.
            await session.rollback()
            document.status = SourceDocumentStatus.QUEUED
            document.error = None
            if await repo.update_claimed(document, claim_token):
                await session.commit()
            else:
                await session.rollback()
            raise
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
        source_published_at = _source_publication_date(
            path.name,
            path.parent / ".source-provenance.json",
        )
        included_ranges = _source_chunk_ranges(
            path.name,
            path.parent / ".source-provenance.json",
        )
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

        for chunk_index, chunk in enumerate(
            remaining_chunks,
            start=document.chunks_processed,
        ):
            if included_ranges is not None and not any(
                start <= chunk_index <= end for start, end in included_ranges
            ):
                document.chunks_processed += 1
                if not await repo.update_claimed(document, claim_token):
                    raise ClaimLostError("document claim was replaced")
                await session.commit()
                continue
            async with session.begin_nested():
                process_options = {
                    "text": chunk.content,
                    "source_chunk_id": chunk.chunk_id,
                    "session": session,
                }
                result = await extractor.process_text(
                    **process_options,
                )
                if result.errors:
                    raise RuntimeError("; ".join(result.errors))
                retained_episodes = []
                for episode in result.episodes:
                    episode.source_published_at = source_published_at
                    if not _constrain_episode_to_source_date(episode, source_published_at):
                        logger.warning(
                            "future_episode_discarded",
                            episode_title=episode.title,
                            start_year=episode.start_year,
                            source_published_at=source_published_at,
                        )
                        await episode_repo.delete(episode.id)
                        continue
                    await episode_repo.update_source_temporal_bounds(episode)
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
                    retained_episodes.append(episode)
                result.episodes = retained_episodes
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

        reconcile_phases = getattr(extractor, "reconcile_document_phases", None)
        if callable(reconcile_phases):
            reconciled = await reconcile_phases(
                [chunk.chunk_id for chunk in chunks],
                session,
            )
            for episode in reconciled:
                if episode.arc_type:
                    arc_types_seen.add(episode.arc_type)
                # Phase and arc labels participate in the structural render;
                # refresh only that vector after document-level correction.
                await episode_repo.update_embedding(
                    episode.id,
                    embedder.generate_structural_embedding(episode),
                    kind="structural",
                )

        link_candidates = getattr(extractor, "link_document_candidates", None)
        if callable(link_candidates):
            await link_candidates(
                [chunk.chunk_id for chunk in chunks],
                session,
            )

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
        except FileNotFoundError:
            # A batch monitor may remove a completed source after this scan
            # listed it but before processing begins.  Cleanup is expected and
            # should not be reported as an ingestion failure.
            logger.debug("watcher_source_removed_during_scan", path=str(path))
            continue
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
    processor = DocumentProcessor(lease_seconds=config.lease_seconds)
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
