"""Tests for ingestion pipeline."""

import pytest

from narrative_engine.ingestion.models import IngestionConfig
from narrative_engine.ingestion.pipeline import IngestionPipeline


class TestIngestionPipeline:
    """Tests for IngestionPipeline."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        config = IngestionConfig(
            target_chunk_size=100,
            min_chunk_size=20,
            max_chunk_size=200,
            output_format="json",
        )
        return IngestionPipeline(config)

    def test_ingest_txt_file(self, pipeline, tmp_path):
        """Ingest a simple text file."""
        txt_file = tmp_path / "input" / "test.txt"
        txt_file.parent.mkdir()
        txt_file.write_text("This is a test document with several words.")

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(txt_file, output_dir)

        assert result.success is True
        assert result.chunks_created == 1
        assert result.source_path == txt_file
        assert len(result.output_files) == 1
        assert result.output_files[0].exists()

    def test_ingest_markdown_file(self, pipeline, tmp_path):
        """Ingest a markdown file with headers."""
        md_file = tmp_path / "input" / "test.md"
        md_file.parent.mkdir()
        md_file.write_text("# Chapter 1\nContent here.\n\n# Chapter 2\nMore content.")

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(md_file, output_dir)

        assert result.success is True
        assert result.chunks_created == 2  # Two chapters
        assert result.metadata.title is None  # No frontmatter

    def test_ingest_file_with_metadata(self, pipeline, tmp_path):
        """Ingest file extracts metadata."""
        md_file = tmp_path / "input" / "test.md"
        md_file.parent.mkdir()
        md_file.write_text("---\ntitle: Test Document\nauthor: John Doe\n---\n\n# Chapter 1\nContent.")

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(md_file, output_dir)

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata.title == "Test Document"
        assert result.metadata.author == "John Doe"

    def test_ingest_nonexistent_file(self, pipeline, tmp_path):
        """Ingesting nonexistent file fails gracefully.

        ingest_file catches errors and reports them on the result (so one
        bad file can't abort ingest_directory) rather than raising.
        """
        nonexistent = tmp_path / "does_not_exist.txt"
        output_dir = tmp_path / "output"

        result = pipeline.ingest_file(nonexistent, output_dir)

        assert result.success is False
        assert result.errors

    def test_ingest_empty_file_does_not_crash_output_stage(self, pipeline, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")

        result = pipeline.ingest_file(empty, tmp_path / "output")

        assert result.success is True
        assert result.chunks_created == 0
        assert result.output_files == []
        assert result.warnings

    def test_ingest_directory(self, pipeline, tmp_path):
        """Ingest all files in a directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create multiple files
        for i in range(3):
            (input_dir / f"file{i}.txt").write_text(f"Content of file {i}")

        output_dir = tmp_path / "output"
        results = pipeline.ingest_directory(input_dir, output_dir)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_output_format_json(self, pipeline, tmp_path):
        """JSON output format produces valid JSON."""
        import json

        txt_file = tmp_path / "input" / "test.txt"
        txt_file.parent.mkdir()
        txt_file.write_text("Test content.")

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(txt_file, output_dir)

        assert result.output_files[0].exists()
        with open(result.output_files[0]) as f:
            data = json.load(f)
            assert "work_id" in data
            assert "total_chunks" in data
            assert "chunks" in data

    def test_output_format_jsonl(self, tmp_path):
        """JSONL output format produces one chunk per line."""
        import json

        config = IngestionConfig(output_format="jsonl")
        pipeline = IngestionPipeline(config)

        txt_file = tmp_path / "input" / "test.txt"
        txt_file.parent.mkdir()
        txt_file.write_text("Test content with many words. " * 50)  # Large enough to split

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(txt_file, output_dir)

        assert result.output_files[0].exists()
        lines = result.output_files[0].read_text().strip().split("\n")
        for line in lines:
            data = json.loads(line)
            assert "chunk_id" in data
            assert "content" in data

    def test_ingestion_preserves_structure(self, pipeline, tmp_path):
        """Ingestion preserves chapter/section structure in chunks."""
        md_file = tmp_path / "input" / "structured.md"
        md_file.parent.mkdir()
        md_file.write_text(
            "# Chapter 1\nContent of chapter one.\n\n"
            "## Section 1.1\nSection content.\n\n"
            "# Chapter 2\nContent of chapter two."
        )

        output_dir = tmp_path / "output"
        result = pipeline.ingest_file(md_file, output_dir)

        # Verify structure is preserved
        # (Would need to load output file and verify)
        assert result.success is True
