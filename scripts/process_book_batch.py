#!/usr/bin/env python3
"""Download, stage, monitor, and clean one five-source ingestion batch.

The manifest is a JSON array with exactly five objects containing ``url`` and
``filename``. Downloads are completed in a temporary directory before any file
enters the watch directory, so the watcher never sees a partial book. A source
is deleted only after the API reports it completed (or as a duplicate of an
already processed source); failed sources remain available for retry.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BATCH_SIZE = 5
DOWNLOAD_ATTEMPTS = 4
TERMINAL_STATES = {"completed", "duplicate", "failed"}
SUCCESS_STATES = {"completed", "duplicate"}
ALLOWED_SUFFIXES = {".txt", ".text", ".pdf"}
USER_AGENT = "NarrativeEngine-CorpusBuilder/1.0 (+local research ingestion)"
MIN_STRUCTURAL_COVERAGE = 0.65
MAX_TINY_CHUNK_RATIO = 0.20
MIN_SOURCE_WORDS = 250


@dataclass(frozen=True)
class Book:
    url: str
    filename: str
    published_at: str | None = None
    title: str | None = None
    source_page: str | None = None
    include_chunk_ranges: tuple[tuple[int, int], ...] | None = None
    chunks_preselected: bool = False


def _published_at(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("published_at must be an ISO date")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid published_at date: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _include_chunk_ranges(value: Any) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("include_chunk_ranges must be a list of [start, end] pairs")
    ranges: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(index, int) or isinstance(index, bool) for index in item)
            or item[0] < 0
            or item[1] < item[0]
        ):
            raise ValueError("include_chunk_ranges must contain non-negative [start, end] pairs")
        ranges.append((item[0], item[1]))
    if ranges != sorted(ranges) or any(
        left[1] >= right[0] for left, right in zip(ranges, ranges[1:], strict=False)
    ):
        raise ValueError("include_chunk_ranges must be ordered and non-overlapping")
    return tuple(ranges)


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every manifest entry needs a non-empty filename")
    filename = value.strip()
    if filename != Path(filename).name or filename in {".", ".."}:
        raise ValueError(f"filename must not contain a path: {filename!r}")
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported batch source format: {filename!r}")
    return filename


def load_manifest(path: Path) -> list[Book]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != BATCH_SIZE:
        raise ValueError(f"manifest must contain exactly {BATCH_SIZE} books")

    books: list[Book] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("every manifest entry must be an object")
        url = entry.get("url")
        if not isinstance(url, str) or urlparse(url).scheme != "https":
            raise ValueError("every source URL must use HTTPS")
        books.append(
            Book(
                url=url,
                filename=_safe_filename(entry.get("filename")),
                published_at=_published_at(entry.get("published_at")),
                title=entry.get("title") if isinstance(entry.get("title"), str) else None,
                source_page=(
                    entry.get("source_page")
                    if isinstance(entry.get("source_page"), str)
                    else None
                ),
                include_chunk_ranges=_include_chunk_ranges(entry.get("include_chunk_ranges")),
                chunks_preselected=entry.get("chunks_preselected", False),
            )
        )
        if not isinstance(books[-1].chunks_preselected, bool):
            raise ValueError("chunks_preselected must be a boolean")

    filenames = [book.filename for book in books]
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest filenames must be unique")
    return books


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg's distribution wrapper and credit paragraph."""
    lines = text.splitlines()
    start = next(
        (
            index + 1
            for index, line in enumerate(lines)
            if line.strip().upper().startswith("*** START OF")
            and "PROJECT GUTENBERG" in line.upper()
        ),
        None,
    )
    end = next(
        (
            index
            for index, line in enumerate(lines[start or 0 :], start=start or 0)
            if line.strip().upper().startswith("*** END OF")
            and "PROJECT GUTENBERG" in line.upper()
        ),
        None,
    )
    if start is not None:
        lines = lines[start:end]
    body = "\n".join(lines).strip()
    paragraphs = re.split(r"\n\s*\n", body, maxsplit=1)
    if len(paragraphs) == 2 and paragraphs[0].lstrip().casefold().startswith(("produced by", "credits:")):
        body = paragraphs[1].lstrip()
    return body + "\n"


def download_book(book: Book, destination: Path) -> None:
    """Download a source with bounded retries before it reaches the watch dir."""
    headers = {
        "Accept": "application/pdf,text/plain;q=0.9,*/*;q=0.8",
        "User-Agent": USER_AGENT,
    }
    # Some public archives reject direct asset requests without the catalogue
    # page that led to them.  Supplying the manifest's provenance URL also
    # makes downloads behave like a normal documented source-link traversal.
    if book.source_page:
        headers["Referer"] = book.source_page
    request = urllib.request.Request(book.url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - HTTPS validated above
                payload = response.read()
            if len(payload) < 1_000:
                raise ValueError(f"download is unexpectedly small ({len(payload)} bytes)")
            if destination.suffix.lower() == ".pdf":
                if not payload.startswith(b"%PDF-"):
                    raise ValueError("download does not have a PDF signature")
                destination.write_bytes(payload)
                return
            text = payload.decode("utf-8-sig", errors="replace")
            cleaned = strip_gutenberg_boilerplate(text) if "gutenberg.org" in urlparse(book.url).netloc else text
            destination.write_text(cleaned, encoding="utf-8")
            return
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            last_error = exc
            if attempt == DOWNLOAD_ATTEMPTS:
                break
            delay = 2 ** (attempt - 1)
            print(
                f"download attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed for "
                f"{book.filename} ({exc}); retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise OSError(f"download failed for {book.filename} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def preflight_source(path: Path) -> dict[str, float | int]:
    """Reject structural parses likely to waste calls or discard the book."""
    from narrative_engine.ingestion.chunker import SmartChunker
    from narrative_engine.ingestion.parsers import get_parser

    parser = get_parser(path)
    if parser is None:
        raise ValueError(f"no parser available for {path.name}")
    parsed = parser.parse(path)
    chunks = SmartChunker().chunk_document(parsed)
    if not chunks:
        raise ValueError(f"preflight produced no chunks for {path.name}")

    source_words = max(1, parsed.metadata.word_count)
    structural_words = sum(len(element.content.split()) for element in parsed.structural_elements)
    coverage = structural_words / source_words
    tiny_chunks = sum(len(chunk.content.split()) < 100 for chunk in chunks)
    tiny_ratio = tiny_chunks / len(chunks)
    if source_words < MIN_SOURCE_WORDS:
        raise ValueError(
            f"preflight source is too short ({source_words} words) for {path.name}; "
            "inspect the download and contents-page detection"
        )
    if source_words >= 1_000 and coverage < MIN_STRUCTURAL_COVERAGE:
        raise ValueError(
            f"preflight retained only {coverage:.1%} of {path.name}; inspect chapter detection"
        )
    if len(chunks) >= 3 and tiny_ratio > MAX_TINY_CHUNK_RATIO:
        raise ValueError(
            f"preflight made {tiny_chunks}/{len(chunks)} tiny chunks for {path.name}; "
            "inspect contents-page detection"
        )
    return {
        "elements": len(parsed.structural_elements),
        "chunks": len(chunks),
        "coverage": coverage,
        "tiny_chunks": tiny_chunks,
    }


def api_json(api_base: str, path: str, *, method: str = "GET") -> Any:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}{path}",
        headers={"User-Agent": USER_AGENT},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - operator-supplied local API
        return json.load(response)


def stage_batch(books: list[Book], watch_dir: Path) -> datetime:
    watch_dir.mkdir(parents=True, exist_ok=True)
    collisions = [book.filename for book in books if (watch_dir / book.filename).exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite existing watch files: {', '.join(collisions)}")

    staged_at = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="narrative-book-batch-") as temp_name:
        temp_dir = Path(temp_name)
        for index, book in enumerate(books, 1):
            print(f"downloading {index}/{BATCH_SIZE}: {book.filename}", flush=True)
            download_book(book, temp_dir / book.filename)
        for book in books:
            report = preflight_source(temp_dir / book.filename)
            print(
                f"preflight: {book.filename}: {report['chunks']} chunks, "
                f"{report['coverage']:.1%} structural coverage, "
                f"{report['tiny_chunks']} tiny chunks",
                flush=True,
            )
        _update_source_provenance(books, watch_dir)
        for book in books:
            shutil.move(str(temp_dir / book.filename), watch_dir / book.filename)
            print(f"staged: {book.filename}", flush=True)
    return staged_at


def _update_source_provenance(books: list[Book], watch_dir: Path) -> None:
    """Persist small source metadata outside the disposable source files."""
    provenance_path = watch_dir / ".source-provenance.json"
    try:
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = {}
    if not isinstance(existing, dict):
        raise ValueError(f"invalid source provenance registry: {provenance_path}")
    for book in books:
        existing[book.filename] = {
            "published_at": book.published_at,
            "source_page": book.source_page,
            "title": book.title,
            "include_chunk_ranges": book.include_chunk_ranges,
            "chunks_preselected": book.chunks_preselected,
        }
    temporary = watch_dir / ".source-provenance.tmp"
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)


def _latest_batch_documents(
    documents: Any,
    books: list[Book],
    staged_at: datetime | None,
) -> dict[str, dict]:
    if not isinstance(documents, list):
        return {}
    wanted = {book.filename for book in books}
    selected: dict[str, dict] = {}
    for document in documents:
        if not isinstance(document, dict) or document.get("filename") not in wanted:
            continue
        try:
            created_at = datetime.fromisoformat(str(document.get("created_at", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if staged_at is not None and created_at < staged_at:
            continue
        filename = document["filename"]
        if filename not in selected or document.get("created_at", "") > selected[filename].get("created_at", ""):
            selected[filename] = document
    return selected


def monitor_and_clean(
    books: list[Book],
    watch_dir: Path,
    api_base: str,
    staged_at: datetime | None,
    poll_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    last_seen: dict[str, tuple] = {}
    cleaned: set[str] = set()
    retry_counts: dict[str, int] = {}

    while time.monotonic() < deadline:
        try:
            selected = _latest_batch_documents(api_json(api_base, "/api/documents"), books, staged_at)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"API unavailable ({exc}); retrying", file=sys.stderr, flush=True)
            time.sleep(poll_seconds)
            continue

        retry_queued = False
        for book in books:
            document = selected.get(book.filename)
            if not document:
                continue
            state = str(document.get("status", "unknown"))
            progress = (
                state,
                document.get("chunks_processed", 0),
                document.get("chunks_created", 0),
                document.get("episodes_created", 0),
            )
            if last_seen.get(book.filename) != progress:
                print(
                    f"{book.filename}: {state} "
                    f"{progress[1]}/{progress[2]} chunks, {progress[3]} episodes",
                    flush=True,
                )
                last_seen[book.filename] = progress

            if state in SUCCESS_STATES and book.filename not in cleaned:
                source = watch_dir / book.filename
                if source.is_file():
                    source.unlink()
                cleaned.add(book.filename)
                print(f"cleaned completed source: {book.filename}", flush=True)

            document_id = str(document.get("id", ""))
            retries = retry_counts.get(document_id, 0)
            if state == "failed" and document_id and retries < max_retries:
                try:
                    api_json(api_base, f"/api/documents/{document_id}/retry", method="POST")
                except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                    print(f"retry request failed for {book.filename} ({exc}); will try again", file=sys.stderr)
                    continue
                retry_counts[document_id] = retries + 1
                retry_queued = True
                print(
                    f"requeued failed source: {book.filename} "
                    f"(attempt {retry_counts[document_id]}/{max_retries})",
                    flush=True,
                )

        if not retry_queued and len(selected) == BATCH_SIZE and all(
            str(selected[book.filename].get("status")) in TERMINAL_STATES for book in books
        ):
            failed = [book.filename for book in books if selected[book.filename].get("status") == "failed"]
            if failed:
                print(f"batch finished with retained failures: {', '.join(failed)}", file=sys.stderr)
                return False
            return True
        time.sleep(poll_seconds)

    print("batch monitor timed out; unfinished sources were retained", file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest containing exactly five books")
    parser.add_argument("--watch-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="checkpoint-aware automatic retries per failed document (default: 2)",
    )
    parser.add_argument(
        "--resume-staged",
        action="store_true",
        help="monitor files/documents from an interrupted batch without downloading or staging again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        books = load_manifest(args.manifest)
        if args.resume_staged:
            staged_at = None
            print("resuming monitoring for an existing staged batch", flush=True)
        else:
            staged_at = stage_batch(books, args.watch_dir)
        succeeded = monitor_and_clean(
            books,
            args.watch_dir,
            args.api_base,
            staged_at,
            max(1.0, args.poll_seconds),
            max(0.1, args.timeout_hours) * 3600,
            max(0, args.max_retries),
        )
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"batch setup failed: {exc}", file=sys.stderr)
        return 2
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
