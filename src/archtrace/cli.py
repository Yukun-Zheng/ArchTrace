"""ArchTrace command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from archtrace.benchmark import (
    BenchmarkResult,
    StaticBenchmarkMetrics,
    load_benchmark_manifest,
    load_runtime_environment,
    load_runtime_spec,
    load_system_spec,
    render_markdown_report,
    run_hybrid_benchmark,
    run_runtime_benchmark,
    run_static_benchmark,
    run_system_benchmark,
    run_system_hybrid_benchmark,
    write_benchmark_result,
)
from archtrace.ingest import build_static_ir, scan_repository
from archtrace.ir import load_atir_json
from archtrace.web_bundle import embed_source_files

app = typer.Typer(
    name="archtrace",
    help="Reconstruct and explore source-grounded ML architectures.",
    no_args_is_help=True,
)
console = Console()
benchmark_app = typer.Typer(help="Run reproducible real-world repository benchmarks.")
app.add_typer(benchmark_app, name="benchmark")


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


@benchmark_app.command("list")
def benchmark_list(
    manifest: Annotated[
        Path, typer.Argument(help="Benchmark manifest TOML path.")
    ] = Path("benchmarks/manifest.toml"),
) -> None:
    """List pinned benchmark cases and supported modes."""
    loaded = load_benchmark_manifest(manifest)
    table = Table(title=f"ArchTrace Benchmark {loaded.schema_version}")
    table.add_column("Case")
    table.add_column("Tier")
    table.add_column("Revision")
    table.add_column("Modes")
    table.add_column("Repository")
    for case in loaded.cases:
        table.add_row(
            case.id,
            case.tier.value,
            case.revision[:12],
            ", ".join(mode.value for mode in case.modes),
            case.repository,
        )
    console.print(table)


@benchmark_app.command("static")
def benchmark_static(
    manifest: Annotated[Path, typer.Argument(help="Benchmark manifest TOML path.")],
    case_id: Annotated[str, typer.Argument(help="Pinned benchmark case ID.")],
    repository: Annotated[Path, typer.Argument(help="Checked-out repository path.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Result JSON output path.")
    ] = None,
    allow_revision_mismatch: Annotated[
        bool,
        typer.Option(
            "--allow-revision-mismatch",
            help="Allow exploratory runs on a revision other than the pinned commit.",
        ),
    ] = False,
) -> None:
    """Run the static benchmark without importing target repository code."""
    case = load_benchmark_manifest(manifest).case(case_id)
    result = run_static_benchmark(
        case, repository, allow_revision_mismatch=allow_revision_mismatch
    )
    destination = output or Path("benchmarks/results") / f"{case.id}.static.json"
    write_benchmark_result(result, destination)
    console.print(
        f"[green]{result.status.value}[/green] {case.id} -> {destination} "
        f"({result.elapsed_seconds:.2f}s)"
    )
    if isinstance(result.metrics, StaticBenchmarkMetrics):
        console.print(
            f"parse={result.metrics.parse_success_rate:.1%} "
            f"calls={result.metrics.call_resolution_rate:.1%} "
            f"semantic={result.metrics.semantic_coverage:.1%}"
        )


@benchmark_app.command("runtime")
def benchmark_runtime(
    manifest: Annotated[Path, typer.Argument(help="Benchmark manifest TOML path.")],
    case_id: Annotated[str, typer.Argument(help="Pinned benchmark case ID.")],
    repository: Annotated[Path, typer.Argument(help="Checked-out repository path.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Result JSON output path.")
    ] = None,
    allow_revision_mismatch: Annotated[
        bool, typer.Option("--allow-revision-mismatch")
    ] = False,
) -> None:
    """Execute a declarative runtime target and capture runtime ATIR."""
    loaded = load_benchmark_manifest(manifest)
    case = loaded.case(case_id)
    if case.runtime_spec is None:
        raise typer.BadParameter(f"benchmark case {case_id!r} has no runtime_spec")
    spec = load_runtime_spec(manifest.parent / case.runtime_spec)
    environment = (
        load_runtime_environment(manifest.parent / case.runtime_environment)
        if case.runtime_environment is not None
        else None
    )
    result = run_runtime_benchmark(
        case,
        repository,
        spec,
        environment=environment,
        allow_revision_mismatch=allow_revision_mismatch,
    )
    destination = output or Path("benchmarks/results") / f"{case.id}.runtime.json"
    write_benchmark_result(result, destination)
    console.print(
        f"[green]{result.status.value}[/green] {case.id} runtime -> {destination} "
        f"({result.elapsed_seconds:.2f}s)"
    )


@benchmark_app.command("hybrid")
def benchmark_hybrid(
    manifest: Annotated[Path, typer.Argument(help="Benchmark manifest TOML path.")],
    case_id: Annotated[str, typer.Argument(help="Pinned benchmark case ID.")],
    repository: Annotated[Path, typer.Argument(help="Checked-out repository path.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Result JSON output path.")
    ] = None,
    allow_revision_mismatch: Annotated[
        bool, typer.Option("--allow-revision-mismatch")
    ] = False,
) -> None:
    """Run static + runtime analysis and measure source-grounded reconciliation."""
    loaded = load_benchmark_manifest(manifest)
    case = loaded.case(case_id)
    if case.runtime_spec is None:
        raise typer.BadParameter(f"benchmark case {case_id!r} has no runtime_spec")
    spec = load_runtime_spec(manifest.parent / case.runtime_spec)
    environment = (
        load_runtime_environment(manifest.parent / case.runtime_environment)
        if case.runtime_environment is not None
        else None
    )
    result = run_hybrid_benchmark(
        case,
        repository,
        spec,
        environment=environment,
        allow_revision_mismatch=allow_revision_mismatch,
    )
    destination = output or Path("benchmarks/results") / f"{case.id}.hybrid.json"
    write_benchmark_result(result, destination)
    console.print(
        f"[green]{result.status.value}[/green] {case.id} hybrid -> {destination} "
        f"({result.elapsed_seconds:.2f}s)"
    )


@benchmark_app.command("system-runtime")
def benchmark_system_runtime(
    manifest: Annotated[Path, typer.Argument(help="Benchmark manifest TOML path.")],
    case_id: Annotated[str, typer.Argument(help="Pinned benchmark case ID.")],
    repository: Annotated[Path, typer.Argument(help="Checked-out repository path.")],
    target_python: Annotated[
        Path, typer.Option("--target-python", help="Python executable for the target runtime.")
    ],
    runtime_cwd: Annotated[
        Path | None,
        typer.Option(
            "--runtime-cwd",
            help="Optional runtime working directory for external assets/resources.",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Result JSON output path.")
    ] = None,
    allow_revision_mismatch: Annotated[
        bool, typer.Option("--allow-revision-mismatch")
    ] = False,
) -> None:
    """Run a cross-runtime Python system scenario through the stdlib probe agent."""
    loaded = load_benchmark_manifest(manifest)
    case = loaded.case(case_id)
    if case.system_spec is None:
        raise typer.BadParameter(f"benchmark case {case_id!r} has no system_spec")
    spec_path = manifest.parent / case.system_spec
    spec = load_system_spec(spec_path)
    result = run_system_benchmark(
        case,
        repository,
        spec,
        spec_path=spec_path,
        target_python=target_python,
        runtime_cwd=runtime_cwd,
        allow_revision_mismatch=allow_revision_mismatch,
    )
    destination = output or Path("benchmarks/results") / f"{case.id}.system.runtime.json"
    write_benchmark_result(result, destination)
    console.print(
        f"[green]{result.status.value}[/green] {case.id} system runtime -> {destination} "
        f"({result.elapsed_seconds:.2f}s)"
    )


@benchmark_app.command("system-hybrid")
def benchmark_system_hybrid(
    manifest: Annotated[Path, typer.Argument(help="Benchmark manifest TOML path.")],
    case_id: Annotated[str, typer.Argument(help="Pinned benchmark case ID.")],
    repository: Annotated[Path, typer.Argument(help="Checked-out repository path.")],
    target_python: Annotated[
        Path, typer.Option("--target-python", help="Python executable for the target runtime.")
    ],
    runtime_cwd: Annotated[
        Path | None,
        typer.Option(
            "--runtime-cwd",
            help="Optional runtime working directory for external assets/resources.",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Result JSON output path.")
    ] = None,
    allow_revision_mismatch: Annotated[
        bool, typer.Option("--allow-revision-mismatch")
    ] = False,
) -> None:
    """Run system probes and reconcile source-grounded boundaries to static ATIR."""
    loaded = load_benchmark_manifest(manifest)
    case = loaded.case(case_id)
    if case.system_spec is None:
        raise typer.BadParameter(f"benchmark case {case_id!r} has no system_spec")
    spec_path = manifest.parent / case.system_spec
    spec = load_system_spec(spec_path)
    result = run_system_hybrid_benchmark(
        case,
        repository,
        spec,
        spec_path=spec_path,
        target_python=target_python,
        runtime_cwd=runtime_cwd,
        allow_revision_mismatch=allow_revision_mismatch,
    )
    destination = output or Path("benchmarks/results") / f"{case.id}.system.hybrid.json"
    write_benchmark_result(result, destination)
    console.print(
        f"[green]{result.status.value}[/green] {case.id} system hybrid -> {destination} "
        f"({result.elapsed_seconds:.2f}s)"
    )


@benchmark_app.command("report")
def benchmark_report(
    results: Annotated[list[Path], typer.Argument(help="Benchmark result JSON files.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Markdown report output path.")
    ] = None,
) -> None:
    """Render a deterministic Markdown scorecard from result JSON files."""
    loaded = [
        BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in results
    ]
    rendered = render_markdown_report(loaded)
    if output is None:
        typer.echo(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Benchmark report written[/green] {output}")


@app.command()
def validate(
    atir: Annotated[Path, typer.Argument(help="Path to an ATIR JSON document.")],
) -> None:
    """Migrate supported historical schemas and validate ATIR references."""
    graph = load_atir_json(atir.read_text(encoding="utf-8"))
    console.print(f"[green]Valid ATIR {graph.schema_version}:[/green] {atir}")


if __name__ == "__main__":
    app()
