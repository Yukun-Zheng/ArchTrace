"""ArchTrace command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from archtrace.ingest import build_static_ir, scan_repository
from archtrace.ir import load_atir_json
from archtrace.web_bundle import embed_source_files

app = typer.Typer(
    name="archtrace",
    help="Reconstruct and explore source-grounded ML architectures.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def inspect(
    repository: Annotated[Path, typer.Argument(help="Path to a research repository.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a table."),
    ] = False,
) -> None:
    """Scan a repository and identify Python structure and likely entrypoints."""
    summary = scan_repository(repository)

    if json_output:
        payload = {
            "root": str(summary.root),
            "python_files": [
                {
                    "path": item.path,
                    "imports": item.imports,
                    "functions": item.functions,
                    "classes": item.classes,
                    "model_classes": item.model_classes,
                    "entrypoint_score": item.entrypoint_score,
                }
                for item in summary.python_files
            ],
            "likely_entrypoints": [item.path for item in summary.likely_entrypoints],
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    console.print(f"[bold]Repository:[/bold] {summary.root}")
    console.print(f"[bold]Python files:[/bold] {len(summary.python_files)}")

    table = Table(title="Likely Entrypoints")
    table.add_column("Score", justify="right")
    table.add_column("Path")
    table.add_column("Model classes")
    for item in summary.likely_entrypoints[:20]:
        table.add_row(str(item.entrypoint_score), item.path, ", ".join(item.model_classes))
    console.print(table)


@app.command()
def analyze(
    repository: Annotated[Path, typer.Argument(help="Path to a research repository.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="ATIR JSON output path."),
    ] = Path(".archtrace/project.atir.json"),
) -> None:
    """Run the static analyzer and write a validated ATIR document."""
    summary = scan_repository(repository)
    graph = build_static_ir(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.model_dump_json(indent=2), encoding="utf-8")

    console.print(
        "[green]ATIR written[/green] "
        f"{output} ({len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.evidence)} evidence records)"
    )


@app.command("web-bundle")
def web_bundle(
    atir: Annotated[Path, typer.Argument(help="Path to a validated ATIR JSON document.")],
    source_root: Annotated[
        Path,
        typer.Option(
            "--source-root",
            help="Repository root used to safely embed source files referenced by ATIR spans.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Browser-ready ATIR bundle output path."),
    ] = Path(".archtrace/project.web.atir.json"),
    max_file_kb: Annotated[
        int,
        typer.Option(help="Maximum UTF-8 source file size to embed, in KiB."),
    ] = 256,
    max_total_mb: Annotated[
        int,
        typer.Option(help="Maximum total embedded source size, in MiB."),
    ] = 5,
) -> None:
    """Embed source text for the browser explorer without mutating ATIR facts."""
    graph = load_atir_json(atir.read_text(encoding="utf-8"))
    bundled = embed_source_files(
        graph,
        source_root,
        max_file_bytes=max_file_kb * 1024,
        max_total_bytes=max_total_mb * 1024 * 1024,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundled.model_dump_json(indent=2), encoding="utf-8")
    bundle_info = bundled.metadata.get("web", {}).get("source_bundle", {})
    console.print(
        "[green]Web bundle written[/green] "
        f"{output} ({bundle_info.get('embedded_files', 0)} source files embedded)"
    )


@app.command()
def validate(
    atir: Annotated[Path, typer.Argument(help="Path to an ATIR JSON document.")],
) -> None:
    """Migrate supported historical schemas and validate ATIR references."""
    graph = load_atir_json(atir.read_text(encoding="utf-8"))
    console.print(f"[green]Valid ATIR {graph.schema_version}:[/green] {atir}")


if __name__ == "__main__":
    app()
