"""Unit tests for safe five-book batch helpers."""

from __future__ import annotations

import http.client
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "process_book_batch.py"
SPEC = importlib.util.spec_from_file_location("process_book_batch", SCRIPT)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def test_manifest_requires_exactly_five_unique_safe_source_files(tmp_path):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "url": f"https://example.test/{index}",
                    "filename": f"book-{index}.pdf" if index == 4 else f"book-{index}.txt",
                    "published_at": "2025-02-05",
                    "include_chunk_ranges": [[0, 2], [7, 9]],
                    "chunks_preselected": True,
                }
                for index in range(5)
            ]
        )
    )

    books = batch.load_manifest(manifest)

    assert len(books) == 5
    assert books[0].filename == "book-0.txt"
    assert books[0].published_at == "2025-02-05T00:00:00+00:00"
    assert books[0].include_chunk_ranges == ((0, 2), (7, 9))
    assert books[0].chunks_preselected is True


def test_manifest_rejects_invalid_publication_date(tmp_path):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "url": f"https://example.test/{index}",
                    "filename": f"book-{index}.txt",
                    "published_at": "2025-02-30" if index == 0 else None,
                }
                for index in range(5)
            ]
        )
    )

    with pytest.raises(ValueError, match="invalid published_at"):
        batch.load_manifest(manifest)


def test_source_provenance_registry_merges_atomically(tmp_path):
    path = tmp_path / ".source-provenance.json"
    path.write_text(json.dumps({"older.txt": {"published_at": "1900-01-01T00:00:00+00:00"}}))
    books = [
        batch.Book(
            url="https://example.test/new",
            filename="new.txt",
            published_at="2025-02-05T00:00:00+00:00",
            title="New source",
            source_page="https://example.test/new",
            include_chunk_ranges=((0, 2), (7, 9)),
            chunks_preselected=True,
        )
    ]

    batch._update_source_provenance(books, tmp_path)

    registry = json.loads(path.read_text())
    assert set(registry) == {"older.txt", "new.txt"}
    assert registry["new.txt"]["published_at"] == "2025-02-05T00:00:00+00:00"
    assert registry["new.txt"]["include_chunk_ranges"] == [[0, 2], [7, 9]]
    assert registry["new.txt"]["chunks_preselected"] is True
    assert not (tmp_path / ".source-provenance.tmp").exists()


@pytest.mark.parametrize("filename", ["../book.txt", "/tmp/book.txt", "book.epub"])
def test_manifest_rejects_unsafe_or_unsupported_filenames(tmp_path, filename):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            [
                {"url": f"https://example.test/{index}", "filename": filename if index == 0 else f"book-{index}.txt"}
                for index in range(5)
            ]
        )
    )

    with pytest.raises(ValueError):
        batch.load_manifest(manifest)


def test_gutenberg_boilerplate_is_removed():
    source = (
        "Distribution header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Produced by Example Person and the Online Distributed\n"
        "Proofreading Team.\n\n"
        "CHAPTER I\nHistory.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Distribution footer\n"
    )

    assert batch.strip_gutenberg_boilerplate(source) == "CHAPTER I\nHistory.\n"


def test_download_retries_an_incomplete_transfer(monkeypatch, tmp_path):
    payload = (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + ("Historical body. " * 100)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    ).encode()
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise http.client.IncompleteRead(b"partial", len(payload))
            return payload

    monkeypatch.setattr(batch.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    destination = tmp_path / "book.txt"

    batch.download_book(
        batch.Book(url="https://www.gutenberg.org/test.txt", filename="book.txt"),
        destination,
    )

    assert calls == 2
    assert destination.read_text().startswith("Historical body.")


def test_download_preserves_pdf_bytes(monkeypatch, tmp_path):
    payload = b"%PDF-1.7\n" + (b"document bytes" * 100)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(batch.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    destination = tmp_path / "report.pdf"

    batch.download_book(
        batch.Book(url="https://example.test/report.pdf", filename="report.pdf"),
        destination,
    )

    assert destination.read_bytes() == payload


def test_download_uses_source_page_as_referer(monkeypatch, tmp_path):
    payload = b"%PDF-1.7\n" + (b"document bytes" * 100)
    captured_request = None

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return payload

    def fake_urlopen(request, **_kwargs):
        nonlocal captured_request
        captured_request = request
        return Response()

    monkeypatch.setattr(batch.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "report.pdf"

    batch.download_book(
        batch.Book(
            url="https://files.example.test/report.pdf",
            filename="report.pdf",
            source_page="https://www.example.test/catalogue/report",
        ),
        destination,
    )

    assert captured_request is not None
    assert captured_request.get_header("Referer") == "https://www.example.test/catalogue/report"
    assert captured_request.get_header("Accept") == "application/pdf,text/plain;q=0.9,*/*;q=0.8"


def test_preflight_rejects_source_that_is_only_a_short_contents_fragment(tmp_path):
    source = tmp_path / "fragmented.txt"
    source.write_text(
        "CONTENTS\n"
        "I ALPHA\nA.\nII BETA\nB.\nIII GAMMA\nC.\nIV DELTA\nD.\nV EPSILON\nE.\n"
    )

    with pytest.raises(ValueError, match="source is too short"):
        batch.preflight_source(source)


def test_monitor_requeues_failed_document_then_cleans_it(monkeypatch, tmp_path):
    books = [batch.Book(url=f"https://example.test/{index}", filename=f"book-{index}.txt") for index in range(5)]
    for book in books:
        (tmp_path / book.filename).write_text("history")
    staged_at = datetime.now(timezone.utc)
    created_at = (staged_at + timedelta(seconds=1)).isoformat()

    def documents(failed: bool):
        return [
            {
                "id": f"doc-{index}",
                "filename": book.filename,
                "created_at": created_at,
                "status": "failed" if failed and index == 0 else "completed",
                "chunks_processed": 1,
                "chunks_created": 1,
                "episodes_created": 1,
            }
            for index, book in enumerate(books)
        ]

    responses = iter([documents(True), documents(False)])
    posts = []

    def fake_api_json(api_base, path, *, method="GET"):
        if method == "POST":
            posts.append(path)
            return {"retried": True}
        return next(responses)

    monkeypatch.setattr(batch, "api_json", fake_api_json)
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)

    assert batch.monitor_and_clean(books, tmp_path, "http://local", staged_at, 0, 1, 2)
    assert posts == ["/api/documents/doc-0/retry"]
    assert not any((tmp_path / book.filename).exists() for book in books)


def test_resume_selection_accepts_documents_created_before_restart():
    books = [batch.Book(url="https://example.test/book", filename="book.txt")]
    documents = [
        {
            "id": "doc-1",
            "filename": "book.txt",
            "created_at": "2020-01-01T00:00:00+00:00",
            "status": "processing",
        }
    ]

    assert batch._latest_batch_documents(documents, books, staged_at=None) == {
        "book.txt": documents[0]
    }
