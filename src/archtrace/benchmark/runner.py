"""Deterministic benchmark runner for repository-scale ArchTrace analysis."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

from archtrace.benchmark.models import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkStatus,
    FailureCategory,
    FailureSeverity,
    StaticBenchmarkMetrics,
)
from archtrace.ir import EdgeKind, NodeLevel
from archtrace.semantics import recover_semantics
from archtrace.static import CallResolution, index_repository, repository_index_to_atir


def run_static_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    *,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    """Analyze one pinned repository without importing or executing target code."""
    started = perf_counter()
    root = Path(repository).resolve()
    analyzer_revision, analyzer_dirty = _analyzer_source_state()
    failures: list[BenchmarkFailure] = []

    if not root.is_dir():
        return _failed_result(
            case,
            started,
            FailureCategory.MISSING_REPOSITORY,
            f"repository directory does not exist: {root}",
        )

    actual_revision = _git_revision(root)
    if actual_revision is not None and actual_revision != case.revision:
        severity = FailureSeverity.WARNING if allow_revision_mismatch else FailureSeverity.ERROR
        failures.append(
            BenchmarkFailure(
                category=FailureCategory.REVISION_MISMATCH,
                severity=severity,
                message=(
                    f"expected revision {case.revision}, found {actual_revision}; "
                    "benchmark results are not directly comparable"
                ),
            )
        )
        if not allow_revision_mismatch:
            return BenchmarkResult(
                case_id=case.id,
                repository=case.repository,
                expected_revision=case.revision,
                actual_revision=actual_revision,
                analyzer_revision=analyzer_revision,
                analyzer_dirty=analyzer_dirty,
                mode=BenchmarkMode.STATIC,
                status=BenchmarkStatus.FAILED,
                elapsed_seconds=perf_counter() - started,
                failures=failures,
            )

    try:
        stage_started = perf_counter()
        index = index_repository(root)
        index_seconds = perf_counter() - stage_started
        stage_started = perf_counter()
        graph = repository_index_to_atir(index)
        atir_seconds = perf_counter() - stage_started
        stage_started = perf_counter()
        semantic_graph = recover_semantics(graph)
        semantic_seconds = perf_counter() - stage_started
    except Exception as exc:
        return _failed_result(
            case,
            started,
            FailureCategory.ANALYSIS_EXCEPTION,
            f"{type(exc).__name__}: {exc}",
            actual_revision=actual_revision,
            existing_failures=failures,
        )

    resolution_counts = Counter(call.resolution for call in index.calls)
    parse_error_files = [item.path for item in index.files if item.parse_error is not None]
    dynamic_calls = [call for call in index.calls if call.resolution == CallResolution.DYNAMIC]
    unresolved_calls = [
        call for call in index.calls if call.resolution == CallResolution.UNRESOLVED
    ]
    dynamic_counts = Counter(call.callee_text for call in dynamic_calls)
    unresolved_counts = Counter(call.callee_text for call in unresolved_calls)
    dynamic_examples = [f"{name} ×{count}" for name, count in dynamic_counts.most_common(10)]
    unresolved_examples = [
        f"{name} ×{count}" for name, count in unresolved_counts.most_common(10)
    ]
    dynamic_shapes = Counter(_dynamic_call_shape(call.callee_text) for call in dynamic_calls)
    submodules = _git_submodule_status(root)
    missing_submodules = [path for path, materialized in submodules if not materialized]
    raw_config_references = graph.metadata.get("config_references", [])
    config_references = raw_config_references if isinstance(raw_config_references, list) else []
    dynamic_config_examples = [
        str(item.get("candidate_paths", []))
        for item in config_references
        if isinstance(item, dict) and item.get("status") == "dynamic"
    ]
    unresolved_config_examples = [
        str(item.get("candidate_paths", []))
        for item in config_references
        if isinstance(item, dict) and item.get("status") == "unresolved"
    ]

    _append_count_failure(
        failures,
        FailureCategory.PARSE_ERROR,
        "Python files could not be parsed by the current AST frontend.",
        parse_error_files,
    )
    _append_count_failure(
        failures,
        FailureCategory.DYNAMIC_CALL,
        "Call targets depend on runtime values and remain dynamic.",
        dynamic_examples,
        count=len(dynamic_calls),
    )
    _append_count_failure(
        failures,
        FailureCategory.UNRESOLVED_CALL,
        "Call targets could not be resolved by the static frontend.",
        unresolved_examples,
        count=len(unresolved_calls),
    )
    _append_count_failure(
        failures,
        FailureCategory.CONFIG_DYNAMIC,
        "Configuration path/name depends on runtime values.",
        dynamic_config_examples,
    )
    _append_count_failure(
        failures,
        FailureCategory.CONFIG_UNRESOLVED,
        "Hydra/OmegaConf configuration references could not be resolved statically.",
        unresolved_config_examples,
    )
    _append_count_failure(
        failures,
        FailureCategory.SUBMODULE_UNMATERIALIZED,
        "Git submodules are declared but not materialized in this checkout.",
        missing_submodules,
    )

    atir_nodes = len(semantic_graph.nodes)
    source_span_nodes = sum(bool(node.source) for node in semantic_graph.nodes)
    semantic_components = sum(node.level == NodeLevel.SEMANTIC for node in semantic_graph.nodes)
    semantic_metadata = semantic_graph.metadata.get("semantic", {})
    unclassified = (
        semantic_metadata.get("unclassified", {})
        if isinstance(semantic_metadata, dict)
        else {}
    )
    unclassified_count = len(unclassified) if isinstance(unclassified, dict) else 0
    mechanical_count = sum(
        node.level not in {NodeLevel.SEMANTIC, NodeLevel.PAPER} for node in semantic_graph.nodes
    )
    resolved_calls = (
        resolution_counts[CallResolution.LOCAL] + resolution_counts[CallResolution.EXTERNAL]
    )

    metrics = StaticBenchmarkMetrics(
        python_files=len(index.files),
        parse_errors=len(parse_error_files),
        parse_success_rate=_ratio(len(index.files) - len(parse_error_files), len(index.files)),
        symbols=len(index.symbols),
        calls=len(index.calls),
        local_calls=resolution_counts[CallResolution.LOCAL],
        external_calls=resolution_counts[CallResolution.EXTERNAL],
        dynamic_calls=resolution_counts[CallResolution.DYNAMIC],
        unresolved_calls=resolution_counts[CallResolution.UNRESOLVED],
        call_resolution_rate=_ratio(resolved_calls, len(index.calls)),
        config_entries=len(index.config_entries),
        config_references=len(config_references),
        dynamic_config_references=len(dynamic_config_examples),
        unresolved_config_references=len(unresolved_config_examples),
        entrypoints=len(index.entrypoints),
        dataflow_edges=sum(edge.kind == EdgeKind.DATA for edge in graph.edges),
        atir_nodes=atir_nodes,
        atir_edges=len(semantic_graph.edges),
        source_span_nodes=source_span_nodes,
        source_span_coverage=_ratio(source_span_nodes, atir_nodes),
        semantic_components=semantic_components,
        semantic_unclassified_nodes=unclassified_count,
        semantic_coverage=_ratio(mechanical_count - unclassified_count, mechanical_count),
        declared_submodules=len(submodules),
        unmaterialized_submodules=len(missing_submodules),
    )
    return BenchmarkResult(
        case_id=case.id,
        repository=case.repository,
        expected_revision=case.revision,
        actual_revision=actual_revision,
        analyzer_revision=analyzer_revision,
        analyzer_dirty=analyzer_dirty,
        mode=BenchmarkMode.STATIC,
        status=BenchmarkStatus.COMPLETED,
        elapsed_seconds=perf_counter() - started,
        metrics=metrics,
        failures=failures,
        metadata={
            "entrypoints": [candidate.path for candidate in index.entrypoints[:20]],
            "static_backend": graph.metadata.get("static_backend"),
            "semantic_backend": (
                semantic_metadata.get("backend") if isinstance(semantic_metadata, dict) else None
            ),
            "stage_seconds": {
                "index": index_seconds,
                "atir": atir_seconds,
                "semantic": semantic_seconds,
            },
            "dynamic_call_shapes": dict(sorted(dynamic_shapes.items())),
            "submodules": [
                {"path": path, "materialized": materialized}
                for path, materialized in submodules
            ],
        },
    )


def write_benchmark_result(result: BenchmarkResult, output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _append_count_failure(
    failures: list[BenchmarkFailure],
    category: FailureCategory,
    message: str,
    examples: list[str],
    *,
    count: int | None = None,
) -> None:
    if not examples:
        return
    failures.append(
        BenchmarkFailure(
            category=category,
            severity=FailureSeverity.WARNING,
            message=message,
            count=len(examples) if count is None else count,
            examples=examples[:10],
        )
    )


def _analyzer_source_state() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[3]
    revision = _git_revision(root)
    if revision is None:
        return None, None
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return revision, None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return revision, dirty


def _dynamic_call_shape(callee: str) -> str:
    if callee.startswith("super."):
        return "super_method"
    if callee.startswith("self."):
        return "self_attribute"
    if callee.startswith("<dynamic>"):
        return "call_result_attribute"
    if "." not in callee:
        return "bare_callable"
    return "receiver_attribute"


def _git_submodule_status(root: Path) -> list[tuple[str, bool]]:
    gitmodules = root / ".gitmodules"
    if not gitmodules.is_file():
        return []
    try:
        paths = subprocess.run(
            ["git", "-C", str(root), "config", "-f", ".gitmodules", "--get-regexp", "path"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    result: list[tuple[str, bool]] = []
    for line in paths.stdout.splitlines():
        _, _, path = line.partition(" ")
        path = path.strip()
        if not path:
            continue
        submodule_root = root / path
        git_marker = submodule_root / ".git"
        materialized = submodule_root.is_dir() and git_marker.exists()
        result.append((path, materialized))
    return sorted(result)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _failed_result(
    case: BenchmarkCase,
    started: float,
    category: FailureCategory,
    message: str,
    *,
    actual_revision: str | None = None,
    existing_failures: list[BenchmarkFailure] | None = None,
) -> BenchmarkResult:
    failures = list(existing_failures or [])
    failures.append(
        BenchmarkFailure(
            category=category,
            severity=FailureSeverity.ERROR,
            message=message,
        )
    )
    analyzer_revision, analyzer_dirty = _analyzer_source_state()
    return BenchmarkResult(
        case_id=case.id,
        repository=case.repository,
        expected_revision=case.revision,
        actual_revision=actual_revision,
        analyzer_revision=analyzer_revision,
        analyzer_dirty=analyzer_dirty,
        mode=BenchmarkMode.STATIC,
        status=BenchmarkStatus.FAILED,
        elapsed_seconds=perf_counter() - started,
        failures=failures,
    )
