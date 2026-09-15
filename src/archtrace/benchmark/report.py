"""Human-readable benchmark reporting."""

from __future__ import annotations

from archtrace.benchmark.models import BenchmarkResult


def render_markdown_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "# ArchTrace Benchmark Report",
        "",
        "| Case | Status | Parse | Call resolution | Semantic coverage | Nodes | Edges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result.metrics
        if metrics is None:
            lines.append(f"| {result.case_id} | {result.status.value} | — | — | — | — | — |")
            continue
        lines.append(
            "| "
            f"{result.case_id} | {result.status.value} | "
            f"{metrics.parse_success_rate:.1%} | {metrics.call_resolution_rate:.1%} | "
            f"{metrics.semantic_coverage:.1%} | {metrics.atir_nodes} | {metrics.atir_edges} |"
        )
    lines.extend(["", "## Failure taxonomy", ""])
    for result in results:
        if not result.failures:
            lines.append(f"- **{result.case_id}**: no recorded benchmark warnings/errors.")
            continue
        lines.append(f"- **{result.case_id}**")
        for failure in result.failures:
            lines.append(
                f"  - `{failure.category.value}` ({failure.severity.value}, n={failure.count}): "
                f"{failure.message}"
            )
    return "\n".join(lines) + "\n"
