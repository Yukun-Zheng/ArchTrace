"""Human-readable benchmark reporting."""

from __future__ import annotations

from archtrace.benchmark.models import (
    BenchmarkResult,
    HybridBenchmarkMetrics,
    RuntimeBenchmarkMetrics,
    StaticBenchmarkMetrics,
)


def render_markdown_report(results: list[BenchmarkResult]) -> str:
    lines = ["# ArchTrace Benchmark Report", ""]
    static_results = [
        result for result in results if isinstance(result.metrics, StaticBenchmarkMetrics)
    ]
    runtime_results = [
        result for result in results if isinstance(result.metrics, RuntimeBenchmarkMetrics)
    ]
    hybrid_results = [
        result for result in results if isinstance(result.metrics, HybridBenchmarkMetrics)
    ]
    if static_results:
        lines.extend(_static_table(static_results))
    if runtime_results:
        if len(lines) > 2:
            lines.append("")
        lines.extend(_runtime_table(runtime_results))
    if hybrid_results:
        if len(lines) > 2:
            lines.append("")
        lines.extend(_hybrid_table(hybrid_results))
    lines.extend(["", "## Failure taxonomy", ""])
    for result in results:
        if not result.failures:
            lines.append(
                f"- **{result.case_id} ({result.mode.value})**: "
                "no recorded warnings/errors."
            )
            continue
        lines.append(f"- **{result.case_id} ({result.mode.value})**")
        for failure in result.failures:
            lines.append(
                f"  - `{failure.category.value}` ({failure.severity.value}, n={failure.count}): "
                f"{failure.message}"
            )
    return "\n".join(lines) + "\n"


def _static_table(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "## Static",
        "",
        "| Case | Status | Parse | Call resolution | Semantic coverage | Nodes | Edges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result.metrics
        assert isinstance(metrics, StaticBenchmarkMetrics)
        lines.append(
            "| "
            f"{result.case_id} | {result.status.value} | {metrics.parse_success_rate:.1%} | "
            f"{metrics.call_resolution_rate:.1%} | {metrics.semantic_coverage:.1%} | "
            f"{metrics.atir_nodes} | {metrics.atir_edges} |"
        )
    return lines


def _runtime_table(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "## Runtime",
        "",
        "| Case | Status | Trace | Target module defs | Nodes | Op defs | Tensor spec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result.metrics
        assert isinstance(metrics, RuntimeBenchmarkMetrics)
        lines.append(
            "| "
            f"{result.case_id} | {result.status.value} | {metrics.trace_seconds:.2f}s | "
            f"{metrics.target_source_definitions} | {metrics.runtime_nodes} | "
            f"{metrics.operator_definitions} | {metrics.tensor_spec_coverage:.1%} |"
        )
    return lines


def _hybrid_table(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "## Hybrid",
        "",
        "| Case | Status | Reconcile | Target module alignment | Alignments | Merged nodes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result.metrics
        assert isinstance(metrics, HybridBenchmarkMetrics)
        lines.append(
            "| "
            f"{result.case_id} | {result.status.value} | "
            f"{metrics.reconciliation_seconds:.2f}s | {metrics.target_alignment_rate:.1%} | "
            f"{metrics.source_alignments} | {metrics.merged_nodes} |"
        )
    return lines
