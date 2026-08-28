"""CLI for ingestion pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from narrative_engine.ingestion.models import IngestionConfig
from narrative_engine.ingestion.pipeline import IngestionPipeline

console = Console()


@click.group()
def cli():
    """Narrative Engine Ingestion CLI.

    Convert raw sources (PDFs, EPUBs, text files) into structured chunks
    ready for the extraction pipeline.
    """
    pass


@cli.command()
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format (json or jsonl)",
)
@click.option(
    "--target-size",
    type=int,
    default=4000,
    help="Target chunk size in tokens",
)
@click.option(
    "--min-size",
    type=int,
    default=1000,
    help="Minimum chunk size in tokens",
)
@click.option(
    "--max-size",
    type=int,
    default=8000,
    help="Maximum chunk size in tokens",
)
@click.option(
    "--overlap",
    type=int,
    default=200,
    help="Overlap between chunks in tokens",
)
def ingest(
    source_path: Path,
    output_dir: Path,
    output_format: str,
    target_size: int,
    min_size: int,
    max_size: int,
    overlap: int,
):
    """Ingest a file or directory.

    SOURCE_PATH: Path to file or directory to ingest
    OUTPUT_DIR: Directory for output chunks
    """
    config = IngestionConfig(
        target_chunk_size=target_size,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
        overlap_tokens=overlap,
        output_format=output_format,
    )

    pipeline = IngestionPipeline(config)

    if source_path.is_file():
        console.print(f"[bold blue]Ingesting file:[/bold blue] {source_path}")
        result = pipeline.ingest_file(source_path, output_dir)
        _display_result(result)

        if not result.success:
            sys.exit(1)

    elif source_path.is_dir():
        console.print(f"[bold blue]Ingesting directory:[/bold blue] {source_path}")
        results = pipeline.ingest_directory(source_path, output_dir)

        # Display summary
        success_count = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)

        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Files Processed", str(len(results)))
        table.add_row("Successful", str(success_count))
        table.add_row("Failed", str(len(results) - success_count))
        table.add_row("Total Chunks", str(total_chunks))

        console.print(table)

        # Display individual results
        for result in results:
            _display_result(result, compact=True)

    else:
        console.print(f"[red]Error: {source_path} is not a file or directory[/red]")
        sys.exit(1)


@cli.command()
def list_parsers():
    """List available parsers."""
    from narrative_engine.ingestion.parsers import PARSERS

    table = Table(title="Available Parsers")
    table.add_column("Format", style="cyan")
    table.add_column("Extensions", style="magenta")
    table.add_column("Description", style="green")

    for parser in PARSERS:
        extensions = {
            "TxtParser": ".txt, .text",
            "MarkdownParser": ".md, .markdown",
            "PdfParser": ".pdf (text-based)",
            "EpubParser": ".epub",
            "HtmlParser": ".html, .htm",
            "OcrParser": ".pdf (scanned/OCR)",
        }.get(parser.__class__.__name__, "Unknown")

        table.add_row(
            parser.__class__.__name__.replace("Parser", ""),
            extensions,
            parser.__doc__.strip() if parser.__doc__ else "No description",
        )

    console.print(table)


def _display_result(result, compact: bool = False):
    """Display ingestion result."""
    if result.success:
        if compact:
            console.print(
                f"[green]✓[/green] {result.source_path.name}: "
                f"{result.chunks_created} chunks"
            )
        else:
            console.print("[bold green]Success![/bold green]")
            console.print(f"Chunks created: {result.chunks_created}")
            console.print(f"Duration: {result.duration_seconds:.2f}s")
            if result.output_files:
                console.print("Output files:")
                for f in result.output_files:
                    console.print(f"  - {f}")
    else:
        if compact:
            console.print(f"[red]✗[/red] {result.source_path.name}: FAILED")
        else:
            console.print("[bold red]Failed![/bold red]")
            if result.errors:
                for error in result.errors:
                    console.print(f"[red]Error: {error}[/red]")


if __name__ == "__main__":
    cli()
