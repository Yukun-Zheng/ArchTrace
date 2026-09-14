"""ArchTrace command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from archtrace.ingest import build_static_ir, scan_repository

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
    """Run the M0 static analyzer and write a validated ATIR document."""
    summary = scan_repository(repository)
    graph = build_static_ir(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.model_dump_json(indent=2), encoding="utf-8")

    console.print(
        "[green]ATIR written[/green] "
        f"{output} ({len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"{len(graph.evidence)} evidence records)"
    )


@app.command()
def validate(
    atir: Annotated[Path, typer.Argument(help="Path to an ATIR JSON document.")],
) -> None:
    """Validate ATIR schema and graph references."""
    from archtrace.ir import ArchTraceIR

    ArchTraceIR.model_validate_json(atir.read_text(encoding="utf-8"))
    console.print(f"[green]Valid ATIR:[/green] {atir}")


if __name__ == "__main__":
    app()
