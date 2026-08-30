"""Drop-directory watcher tests (T7, docs/tickets/T7-drop-directory-watcher.md)."""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from narrative_engine.models import ArcType, Episode, SourceDocument, SourceDocumentStatus
from narrative_engine.storage.orm_models import SourceDocumentORM
from narrative_engine.storage.repositories import (
    SOURCE_DOCUMENT_FIELDS_EXCLUDED,
    EpisodeRepository,
    SourceDocumentRepository,
)
from narrative_engine.watcher import (
    DocumentProcessor,
    WatcherConfig,
    _constrain_episode_to_source_date,
    _hash_file,
    _source_chunk_ranges,
    _source_chunks_preselected,
    _source_publication_date,
    llm_configured,
    scan_once,
)
from tests.unit.test_roundtrip import assert_field_coverage

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _isolate_llm_credentials(monkeypatch):
    for name in ("NE_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_hash_file_streams_without_path_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    content = b"0123456789" * 1000
    path.write_bytes(content)

    def fail_read_bytes(self):
        raise AssertionError("whole-file read should not be used")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    digest, size_bytes = _hash_file(path)

    assert digest == hashlib.sha256(content).hexdigest()
    assert size_bytes == len(content)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("loc-china-country-study-1987.txt", datetime(1987, 12, 31, tzinfo=UTC)),
        ("crs-china-relations-2026-08-26.pdf", datetime(2026, 8, 26, tzinfo=UTC)),
        ("pg15359-the-negro.txt", None),
        ("bad-date-2026-02-30.pdf", None),
    ],
)
def test_source_publication_date_from_filename(filename, expected):
    assert _source_publication_date(filename) == expected


def test_source_publication_registry_overrides_filename_year(tmp_path):
    provenance = tmp_path / ".source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "loc-china-country-study-1987.txt": {
                    "published_at": "1988-12-31T00:00:00+00:00"
                }
            }
        )
    )

    assert _source_publication_date(
        "loc-china-country-study-1987.txt", provenance
    ) == datetime(1988, 12, 31, tzinfo=UTC)


def test_explicit_unknown_publication_date_suppresses_filename_inference(tmp_path):
    provenance = tmp_path / ".source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "china-revolution-1911.txt": {
                    "published_at": None,
                }
            }
        )
    )

    assert _source_publication_date("china-revolution-1911.txt", provenance) is None


def test_source_chunk_ranges_from_registry(tmp_path):
    provenance = tmp_path / ".source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "book.txt": {
                    "include_chunk_ranges": [[0, 2], [7, 9]],
                    "chunks_preselected": True,
                }
            }
        )
    )

    assert _source_chunk_ranges("book.txt", provenance) == ((0, 2), (7, 9))
    assert _source_chunks_preselected("book.txt", provenance) is True


def test_source_date_caps_an_ongoing_episode():
    episode = Episode(
        title="Ongoing reform",
        summary="A reform underway when the source was published.",
        start_year=1980,
        end_year=1989,
        start_date=datetime(1980, 1, 1, tzinfo=UTC),
        end_date=datetime(1989, 1, 1, tzinfo=UTC),
    )

    retained = _constrain_episode_to_source_date(
        episode,
        datetime(1988, 12, 31, tzinfo=UTC),
    )

    assert retained is True
    assert episode.end_year == 1988
    assert episode.end_date == datetime(1988, 12, 31, tzinfo=UTC)


@pytest.mark.parametrize(
    "episode",
    [
        Episode(title="Forecast", summary="Future plan", start_year=1990),
        Episode(
            title="Dated forecast",
            summary="Future plan",
            start_date=datetime(1989, 1, 1, tzinfo=UTC),
        ),
    ],
)
def test_source_date_rejects_episodes_that_begin_in_the_future(episode):
    assert not _constrain_episode_to_source_date(
        episode,
        datetime(1988, 12, 31, tzinfo=UTC),
    )


class TestSourceDocumentRoundTrip:
    @pytest.mark.asyncio
    async def test_maximal(self, db_session):
        repo = SourceDocumentRepository(db_session)
        original = SourceDocument(filename="orig.txt", content_hash="a" * 64)
        await repo.create(original)

        document = SourceDocument(
            id=uuid4(),
            filename="kindleberger-manias.txt",
            content_hash="b" * 64,
            size_bytes=123456,
            status=SourceDocumentStatus.FAILED,
            error="parser exploded",
            chunks_created=12,
            chunks_processed=5,
            episodes_created=7,
            extraction_ran=True,
            duplicate_of=original.id,
            created_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC),
            updated_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC),
        )
        assert_field_coverage(SourceDocument, document, SOURCE_DOCUMENT_FIELDS_EXCLUDED)

        await repo.create(document)
        fetched = await repo.get_by_id(document.id)

        assert fetched is not None
        assert fetched.model_dump() == document.model_dump()

    @pytest.mark.asyncio
    async def test_expired_claim_takeover_fences_old_worker(self, db_session):
        repo = SourceDocumentRepository(db_session)
        document = SourceDocument(filename="book.txt", content_hash="c" * 64)
        await repo.create(document)

        first_token = uuid4()
        second_token = uuid4()
        assert await repo.claim_available(
            document.id,
            first_token,
            datetime(2100, 1, 1, tzinfo=UTC),
        )
        assert not await repo.claim_available(
            document.id,
            second_token,
            datetime(2100, 1, 1, tzinfo=UTC),
        )

        row = await db_session.get(SourceDocumentORM, document.id)
        row.lease_expires_at = datetime(2020, 1, 1, tzinfo=UTC)
        await db_session.flush()

        assert await repo.claim_available(
            document.id,
            second_token,
            datetime(2100, 1, 1, tzinfo=UTC),
        )
        document.status = SourceDocumentStatus.PROCESSING
        document.chunks_processed = 1
        assert not await repo.update_claimed(document, first_token)
        assert await repo.update_claimed(document, second_token)


def _write(tmp_path: Path, name: str, content: bytes, old: bool = True) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    if old:
        # Make the file look settled (mtime in the past).
        stale = time.time() - 60
        os.utime(path, (stale, stale))
    return path


def _config(tmp_path: Path) -> WatcherConfig:
    return WatcherConfig(watch_dir=tmp_path, settle_seconds=2.0)


def test_watcher_config_uses_bounded_document_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("NE_WATCH_DIR", str(tmp_path))
    monkeypatch.setenv("NE_DOCUMENT_LEASE_SECONDS", "90")

    config = WatcherConfig.from_env()

    assert config.lease_seconds == 90


class TestDuplicateGuard:
    @pytest.mark.asyncio
    async def test_same_bytes_any_filename_becomes_duplicate_row(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "book.txt", b"the same book content")
        _write(tmp_path, "book-copy-from-other-folder.txt", b"the same book content")

        touched = await scan_once(db_session, _config(tmp_path), processor)

        by_status = {}
        for doc in touched:
            by_status.setdefault(doc.status, []).append(doc)
        assert len(by_status[SourceDocumentStatus.COMPLETED]) == 1
        assert len(by_status[SourceDocumentStatus.DUPLICATE]) == 1

        duplicate = by_status[SourceDocumentStatus.DUPLICATE][0]
        original = by_status[SourceDocumentStatus.COMPLETED][0]
        assert duplicate.duplicate_of == original.id
        assert duplicate.chunks_created == 0  # never processed

    @pytest.mark.asyncio
    async def test_rescan_does_not_rerecord(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "book.txt", b"content that stays in the folder")

        first = await scan_once(db_session, _config(tmp_path), processor)
        second = await scan_once(db_session, _config(tmp_path), processor)

        assert len(first) == 1
        assert second == []  # same (hash, filename): recorded once, forever

    @pytest.mark.asyncio
    async def test_different_bytes_same_name_is_not_duplicate(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "book.txt", b"first edition")
        first = await scan_once(db_session, _config(tmp_path), processor)

        _write(tmp_path, "book.txt", b"second edition, revised")
        second = await scan_once(db_session, _config(tmp_path), processor)

        assert first[0].status == SourceDocumentStatus.COMPLETED
        assert second[0].status == SourceDocumentStatus.COMPLETED  # new edition


class TestSettleGuard:
    @pytest.mark.asyncio
    async def test_fresh_files_are_not_picked_up(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "still-copying.txt", b"partial", old=False)

        touched = await scan_once(db_session, _config(tmp_path), processor)
        assert touched == []

    @pytest.mark.asyncio
    async def test_partial_suffixes_ignored(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "download.part", b"partial browser download")

        touched = await scan_once(db_session, _config(tmp_path), processor)
        assert touched == []


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_unparseable_file_fails_visibly_and_loop_continues(self, db_session, tmp_path):
        processor = DocumentProcessor(extractor=None)
        _write(tmp_path, "image.xyz", b"\x89PNG not a text format")
        _write(tmp_path, "readable.txt", b"a perfectly fine text file")

        touched = await scan_once(db_session, _config(tmp_path), processor)

        statuses = {d.filename: d.status for d in touched}
        assert statuses["image.xyz"] == SourceDocumentStatus.FAILED
        assert statuses["readable.txt"] == SourceDocumentStatus.COMPLETED
        failed = next(d for d in touched if d.filename == "image.xyz")
        assert failed.error


class FakeEmbedder:
    def generate_surface_embedding(self, episode):
        return [0.1] * 384

    def generate_structural_embedding(self, episode):
        return [0.2] * 384


class FakeExtractor:
    """Creates one real episode per chunk; optionally fails at chunk N."""

    def __init__(self, session, fail_on_call: int | None = None):
        self._session = session
        self._fail_on_call = fail_on_call
        self.calls = 0

    async def process_text(self, text, source_chunk_id, session, skip_segmentation=False):
        from types import SimpleNamespace

        from narrative_engine.models import Episode
        from narrative_engine.storage.repositories import EpisodeRepository

        self.calls += 1
        assert isinstance(skip_segmentation, bool)
        if self._fail_on_call is not None and self.calls == self._fail_on_call:
            raise RuntimeError("LLM exploded mid-book")
        episode = Episode(title=f"ep-{source_chunk_id}", summary="s")
        await EpisodeRepository(session).create(episode)
        return SimpleNamespace(episodes=[episode], errors=[])


def _multi_chunk_file(tmp_path: Path, name: str = "book.txt") -> Path:
    # Large enough to hard-split into several chunks under default config.
    return _write(tmp_path, name, ("word " * 30000).encode())


class TestChunkProgress:
    @pytest.mark.asyncio
    async def test_configured_chunk_ranges_skip_irrelevant_sections(
        self, db_session, tmp_path
    ):
        path = _multi_chunk_file(tmp_path, "selected-book.txt")
        (tmp_path / ".source-provenance.json").write_text(
            json.dumps(
                {
                    path.name: {
                        "include_chunk_ranges": [[1, 1]],
                        "chunks_preselected": True,
                    }
                }
            )
        )
        extractor = FakeExtractor(db_session)
        document = await DocumentProcessor(
            extractor=extractor,
            embedder=FakeEmbedder(),
        ).process_file(db_session, path)

        assert document is not None
        assert document.status == SourceDocumentStatus.COMPLETED
        assert document.chunks_created > 1
        assert document.chunks_processed == document.chunks_created
        assert document.episodes_created == 1
        assert extractor.calls == 1

    @pytest.mark.asyncio
    async def test_cancellation_requeues_document_for_immediate_resume(self, db_session, tmp_path):
        started = asyncio.Event()
        never_finishes = asyncio.Event()

        class BlockingExtractor(FakeExtractor):
            async def process_text(self, text, source_chunk_id, session):
                started.set()
                await never_finishes.wait()

        path = _write(tmp_path, "book.txt", b"one historical episode")
        processor = DocumentProcessor(
            extractor=BlockingExtractor(db_session),
            embedder=FakeEmbedder(),
        )
        task = asyncio.create_task(processor.process_file(db_session, path))
        await started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        document = await SourceDocumentRepository(db_session).get_by_hash_and_filename(content_hash, path.name)
        assert document is not None
        assert document.status == SourceDocumentStatus.QUEUED
        row = await db_session.get(SourceDocumentORM, document.id)
        assert row.claim_token is None
        assert row.lease_expires_at is None

    @pytest.mark.asyncio
    async def test_progress_counts_track_chunks(self, db_session, tmp_path):
        extractor = FakeExtractor(db_session)
        processor = DocumentProcessor(extractor=extractor, embedder=FakeEmbedder())
        _multi_chunk_file(tmp_path)

        touched = await scan_once(db_session, _config(tmp_path), processor)

        doc = touched[0]
        assert doc.status == SourceDocumentStatus.COMPLETED
        assert doc.chunks_created >= 2  # sanity: the fixture actually multi-chunks
        assert doc.chunks_processed == doc.chunks_created == extractor.calls
        assert doc.episodes_created == extractor.calls
        assert doc.extraction_ran is True

    @pytest.mark.asyncio
    async def test_source_removed_during_scan_is_ignored(self, db_session, tmp_path):
        path = _write(tmp_path, "cleaned.txt", b"already processed")

        class RemovedSourceProcessor:
            async def process_file(self, session, candidate):
                assert candidate == path
                raise FileNotFoundError(candidate)

        touched = await scan_once(
            db_session,
            _config(tmp_path),
            RemovedSourceProcessor(),
        )

        assert touched == []

    @pytest.mark.asyncio
    async def test_partial_progress_survives_midfile_failure(self, db_session, tmp_path):
        """A crash on chunk 3 must leave chunks 1-2 committed and visible --
        the dashboard shows how far the run got, and the extracted episodes
        are durable rather than all-or-nothing."""
        extractor = FakeExtractor(db_session, fail_on_call=3)
        processor = DocumentProcessor(extractor=extractor, embedder=FakeEmbedder())
        _multi_chunk_file(tmp_path)

        touched = await scan_once(db_session, _config(tmp_path), processor)

        doc = touched[0]
        assert doc.status == SourceDocumentStatus.FAILED
        assert "LLM exploded" in doc.error
        assert doc.chunks_processed == 2
        assert doc.episodes_created == 2
        assert doc.chunks_created > doc.chunks_processed  # gap is visible

    @pytest.mark.asyncio
    async def test_failed_document_resumes_from_next_chunk(self, db_session, tmp_path):
        failing = FakeExtractor(db_session, fail_on_call=3)
        _multi_chunk_file(tmp_path)

        first = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=failing, embedder=FakeEmbedder()),
        )
        failed = first[0]
        assert failed.status == SourceDocumentStatus.FAILED
        assert failed.chunks_processed == 2

        failed.status = SourceDocumentStatus.QUEUED
        await SourceDocumentRepository(db_session).update(failed)
        await db_session.commit()

        resumed_extractor = FakeExtractor(db_session)
        second = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=resumed_extractor, embedder=FakeEmbedder()),
        )

        assert len(second) == 1
        resumed = second[0]
        assert resumed.id == failed.id
        assert resumed.status == SourceDocumentStatus.COMPLETED
        assert resumed.chunks_processed == resumed.chunks_created
        assert resumed_extractor.calls == resumed.chunks_created - 2

    @pytest.mark.asyncio
    async def test_queued_failure_before_chunking_restarts_pipeline(self, db_session, tmp_path):
        path = _write(tmp_path, "book.txt", b"one historical episode")
        failed = SourceDocument(
            filename=path.name,
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            status=SourceDocumentStatus.QUEUED,
            chunks_created=0,
            chunks_processed=0,
        )
        await SourceDocumentRepository(db_session).create(failed)
        await db_session.commit()

        extractor = FakeExtractor(db_session)
        touched = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=extractor, embedder=FakeEmbedder()),
        )

        assert len(touched) == 1
        assert touched[0].id == failed.id
        assert touched[0].status == SourceDocumentStatus.COMPLETED
        assert touched[0].chunks_processed == touched[0].chunks_created == 1
        assert extractor.calls == 1

    @pytest.mark.asyncio
    async def test_active_processing_document_is_not_claimed_again(self, db_session, tmp_path):
        path = _multi_chunk_file(tmp_path)
        active = SourceDocument(
            filename=path.name,
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            status=SourceDocumentStatus.PROCESSING,
            chunks_created=3,
            chunks_processed=1,
        )
        await SourceDocumentRepository(db_session).create(active)
        active_row = await db_session.get(SourceDocumentORM, active.id)
        active_row.claim_token = uuid4()
        active_row.lease_expires_at = datetime(2100, 1, 1, tzinfo=UTC)
        await db_session.commit()

        extractor = FakeExtractor(db_session)
        touched = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=extractor, embedder=FakeEmbedder()),
        )

        assert touched == []
        assert extractor.calls == 0

    @pytest.mark.asyncio
    async def test_stale_processing_document_is_reclaimed(self, db_session, tmp_path):
        path = _write(tmp_path, "book.txt", b"one historical episode")
        stale = SourceDocument(
            filename=path.name,
            content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            status=SourceDocumentStatus.PROCESSING,
            chunks_created=1,
            chunks_processed=0,
            updated_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
        await SourceDocumentRepository(db_session).create(stale)
        await db_session.commit()

        extractor = FakeExtractor(db_session)
        touched = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(
                extractor=extractor,
                embedder=FakeEmbedder(),
                lease_seconds=60,
            ),
        )

        assert len(touched) == 1
        assert touched[0].id == stale.id
        assert touched[0].status == SourceDocumentStatus.COMPLETED
        assert extractor.calls == 1

    @pytest.mark.asyncio
    async def test_retry_after_composition_failure_does_not_reextract(self, db_session, tmp_path, monkeypatch):
        from narrative_engine.composition.pipeline import CompositionPipeline

        class ClassifiedExtractor(FakeExtractor):
            async def process_text(self, text, source_chunk_id, session):
                self.calls += 1
                episode = Episode(
                    title="classified",
                    summary="s",
                    arc_type=ArcType.CREDIT_BOOM_AND_BUST,
                    extracted_from=[source_chunk_id],
                )
                await EpisodeRepository(session).create(episode)
                return SimpleNamespace(episodes=[episode], errors=[])

        composition_calls = 0

        async def flaky_compose(self, arc_type):
            nonlocal composition_calls
            composition_calls += 1
            if composition_calls == 1:
                raise RuntimeError("composition failed")
            return []

        monkeypatch.setattr(CompositionPipeline, "compose_arc_instances", flaky_compose)
        _write(tmp_path, "book.txt", b"one historical episode")

        first_extractor = ClassifiedExtractor(db_session)
        first = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=first_extractor, embedder=FakeEmbedder()),
        )
        failed = first[0]
        assert failed.status == SourceDocumentStatus.FAILED
        assert failed.chunks_processed == failed.chunks_created == 1

        failed.status = SourceDocumentStatus.QUEUED
        await SourceDocumentRepository(db_session).update(failed)
        await db_session.commit()

        retry_extractor = ClassifiedExtractor(db_session)
        retried = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(extractor=retry_extractor, embedder=FakeEmbedder()),
        )

        assert retried[0].status == SourceDocumentStatus.COMPLETED
        assert retry_extractor.calls == 0
        assert composition_calls == 2

    @pytest.mark.asyncio
    async def test_pipeline_result_errors_do_not_checkpoint_chunk(self, db_session, tmp_path):
        class ErrorResultExtractor(FakeExtractor):
            async def process_text(self, text, source_chunk_id, session):
                result = await super().process_text(text, source_chunk_id, session)
                result.errors = ["Venice 402 insufficient balance"]
                return result

        _write(tmp_path, "book.txt", b"one short historical episode")
        touched = await scan_once(
            db_session,
            _config(tmp_path),
            DocumentProcessor(
                extractor=ErrorResultExtractor(db_session),
                embedder=FakeEmbedder(),
            ),
        )

        document = touched[0]
        assert document.status == SourceDocumentStatus.FAILED
        assert document.chunks_processed == 0
        assert document.episodes_created == 0
        assert await EpisodeRepository(db_session).count() == 0


class TestExtractionGating:
    @pytest.mark.asyncio
    async def test_no_llm_key_completes_with_extraction_pending(self, db_session, tmp_path, monkeypatch):
        for var in ("NE_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert not llm_configured()

        processor = DocumentProcessor()  # no injected extractor either
        _write(tmp_path, "book.txt", b"chapter one. things happened.")

        touched = await scan_once(db_session, _config(tmp_path), processor)

        assert touched[0].status == SourceDocumentStatus.COMPLETED
        assert touched[0].extraction_ran is False
        assert touched[0].chunks_created >= 1
