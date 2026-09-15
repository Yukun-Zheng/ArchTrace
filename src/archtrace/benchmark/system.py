"""Cross-runtime Python system benchmarks using the stdlib probe agent."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter

from archtrace.align import reconcile_static_runtime
from archtrace.benchmark.models import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkStatus,
    FailureCategory,
    FailureSeverity,
    HybridBenchmarkMetrics,
    PythonSystemSpec,
    SystemRuntimeBenchmarkMetrics,
)
from archtrace.benchmark.runner import analyzer_source_state, git_revision
from archtrace.ir import ArchTraceIR, CoverageStatus, EdgeKind, IdentityKind
from archtrace.runtime import raw_python_trace_to_atir
from archtrace.static import index_repository, repository_index_to_atir


def load_system_spec(path: str | Path) -> PythonSystemSpec:
    return PythonSystemSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def run_system_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: PythonSystemSpec,
    *,
    spec_path: str | Path,
    target_python: str | Path,
    runtime_cwd: str | Path | None = None,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    started = perf_counter()
    root = Path(repository).resolve()
    preflight = _preflight(case, root, allow_revision_mismatch, BenchmarkMode.RUNTIME)
    if isinstance(preflight, BenchmarkResult):
        return preflight
    actual_revision, failures = preflight
    try:
        execution = _execute_system(
            case,
            root,
            spec,
            spec_path=Path(spec_path),
            target_python=Path(target_python),
            runtime_cwd=None if runtime_cwd is None else Path(runtime_cwd),
        )
    except _SystemBenchmarkError as exc:
        return _failed(case, started, actual_revision, failures, exc.category, exc.message)

    graph = execution.graph
    definitions = [node for node in graph.nodes if node.identity_kind == IdentityKind.DEFINITION]
    source_definitions = [node for node in definitions if node.source]
    external_definitions = [node for node in definitions if not node.source]
    occurrences = [node for node in graph.nodes if node.identity_kind == IdentityKind.OCCURRENCE]
    values = [node for node in graph.nodes if node.identity_kind == IdentityKind.VALUE]
    tensor_values = [node for node in values if node.tensor is not None]
    role_counts = Counter(str(node.attributes.get("probe_role")) for node in occurrences)
    metrics = SystemRuntimeBenchmarkMetrics(
        trace_seconds=execution.elapsed_seconds,
        runtime_nodes=len(graph.nodes),
        runtime_edges=len(graph.edges),
        definitions=len(definitions),
        source_definitions=len(source_definitions),
        external_definitions=len(external_definitions),
        occurrences=len(occurrences),
        values=len(values),
        tensor_values=len(tensor_values),
        tensor_spec_coverage=_ratio(
            sum(
                node.tensor is not None and node.tensor.shape is not None
                for node in tensor_values
            ),
            len(tensor_values),
        ),
        adapter_occurrences=role_counts["adapter"],
        transport_occurrences=role_counts["transport"],
        environment_occurrences=role_counts["environment"],
        consumes_edges=sum(edge.kind == EdgeKind.CONSUMES for edge in graph.edges),
        produces_edges=sum(edge.kind == EdgeKind.PRODUCES for edge in graph.edges),
        calls_edges=sum(edge.kind == EdgeKind.CALLS for edge in graph.edges),
        next_edges=sum(edge.kind == EdgeKind.NEXT for edge in graph.edges),
    )
    return _completed(
        case,
        BenchmarkMode.RUNTIME,
        started,
        actual_revision,
        failures,
        metrics,
        metadata=execution.metadata,
    )


def run_system_hybrid_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: PythonSystemSpec,
    *,
    spec_path: str | Path,
    target_python: str | Path,
    runtime_cwd: str | Path | None = None,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    started = perf_counter()
    root = Path(repository).resolve()
    preflight = _preflight(case, root, allow_revision_mismatch, BenchmarkMode.HYBRID)
    if isinstance(preflight, BenchmarkResult):
        return preflight
    actual_revision, failures = preflight
    try:
        static_started = perf_counter()
        static_graph = repository_index_to_atir(index_repository(root))
        static_seconds = perf_counter() - static_started
        execution = _execute_system(
            case,
            root,
            spec,
            spec_path=Path(spec_path),
            target_python=Path(target_python),
            runtime_cwd=None if runtime_cwd is None else Path(runtime_cwd),
        )
        reconcile_started = perf_counter()
        merged = reconcile_static_runtime(static_graph, execution.graph)
        reconciliation_seconds = perf_counter() - reconcile_started
    except _SystemBenchmarkError as exc:
        return _failed(
            case,
            started,
            actual_revision,
            failures,
            exc.category,
            exc.message,
            mode=BenchmarkMode.HYBRID,
        )
    except Exception as exc:
        return _failed(
            case,
            started,
            actual_revision,
            failures,
            FailureCategory.HYBRID_RECONCILIATION_FAILURE,
            f"{type(exc).__name__}: {exc}",
            mode=BenchmarkMode.HYBRID,
        )

    alignments_raw = merged.metadata.get("reconciliation", {}).get("source_alignments", [])
    alignments = alignments_raw if isinstance(alignments_raw, list) else []
    aligned_ids = {
        item.get("runtime_definition_id")
        for item in alignments
        if isinstance(item, dict) and isinstance(item.get("runtime_definition_id"), str)
    }
    target_definitions = [
        node
        for node in execution.graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION and node.source
    ]
    coverage_counts = Counter(record.status for record in merged.coverage)
    metrics = HybridBenchmarkMetrics(
        reconciliation_seconds=reconciliation_seconds,
        static_nodes=len(static_graph.nodes),
        static_edges=len(static_graph.edges),
        runtime_nodes=len(execution.graph.nodes),
        runtime_edges=len(execution.graph.edges),
        merged_nodes=len(merged.nodes),
        merged_edges=len(merged.edges),
        target_runtime_definitions=len(target_definitions),
        aligned_target_runtime_definitions=sum(
            node.id in aligned_ids for node in target_definitions
        ),
        target_alignment_rate=_ratio(
            sum(node.id in aligned_ids for node in target_definitions), len(target_definitions)
        ),
        source_alignments=len(alignments),
        alias_edges=sum(edge.kind == EdgeKind.ALIAS for edge in merged.edges),
        coverage_records=len(merged.coverage),
        always_observed=coverage_counts[CoverageStatus.ALWAYS_OBSERVED],
        sometimes_observed=coverage_counts[CoverageStatus.SOMETIMES_OBSERVED],
        static_reachable_unobserved=coverage_counts[CoverageStatus.STATIC_REACHABLE_UNOBSERVED],
    )
    unaligned = [node for node in target_definitions if node.id not in aligned_ids]
    if unaligned:
        failures.append(
            BenchmarkFailure(
                category=FailureCategory.HYBRID_ALIGNMENT_GAP,
                severity=FailureSeverity.WARNING,
                message=(
                    "Source-grounded system probe definitions were not aligned "
                    "to static source."
                ),
                count=len(unaligned),
                examples=[
                    f"{node.label}: {node.source[0].path}:{node.source[0].start_line}"
                    for node in unaligned[:10]
                ],
            )
        )
    return _completed(
        case,
        BenchmarkMode.HYBRID,
        started,
        actual_revision,
        failures,
        metrics,
        metadata={
            **execution.metadata,
            "stage_seconds": {
                "static": static_seconds,
                "system_runtime": execution.elapsed_seconds,
                "reconciliation": reconciliation_seconds,
            },
            "source_alignment_count": len(alignments),
            "unaligned_target_definitions": [node.id for node in unaligned],
        },
    )


class _SystemExecution:
    def __init__(
        self, graph: ArchTraceIR, elapsed_seconds: float, metadata: dict[str, object]
    ) -> None:
        self.graph = graph
        self.elapsed_seconds = elapsed_seconds
        self.metadata = metadata


class _SystemBenchmarkError(Exception):
    def __init__(self, category: FailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def _execute_system(
    case: BenchmarkCase,
    root: Path,
    spec: PythonSystemSpec,
    *,
    spec_path: Path,
    target_python: Path,
    runtime_cwd: Path | None,
) -> _SystemExecution:
    if not target_python.is_file():
        raise _SystemBenchmarkError(
            FailureCategory.SYSTEM_RUNTIME_LAUNCH_FAILURE,
            f"target Python does not exist: {target_python}",
        )
    scenario_path = (spec_path.parent / spec.scenario).resolve()
    if not scenario_path.is_file():
        raise _SystemBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            f"system scenario does not exist: {scenario_path}",
        )
    runtime_dir = root if runtime_cwd is None else runtime_cwd.resolve()
    if not runtime_dir.is_dir():
        raise _SystemBenchmarkError(
            FailureCategory.SYSTEM_RUNTIME_LAUNCH_FAILURE,
            f"runtime cwd does not exist: {runtime_dir}",
        )
    resource_metadata = _verify_runtime_resources(spec, runtime_dir)
    runtime_root = Path(__file__).resolve().parents[1] / "runtime"
    driver = runtime_root / "python_probe_driver.py"
    agent = runtime_root / "python_probe_agent.py"
    with tempfile.TemporaryDirectory(prefix="archtrace-system-") as temporary:
        raw_path = Path(temporary) / "raw.json"
        command = [
            str(target_python),
            str(driver),
            "--repository",
            str(root),
            "--spec",
            str(spec_path.resolve()),
            "--scenario",
            str(scenario_path),
            "--agent",
            str(agent),
            "--output",
            str(raw_path),
            "--runtime-cwd",
            str(runtime_dir),
        ]
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _SystemBenchmarkError(
                FailureCategory.SYSTEM_RUNTIME_LAUNCH_FAILURE,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise _SystemBenchmarkError(
                FailureCategory.SYSTEM_RUNTIME_EXECUTION_FAILURE,
                f"target process exited {completed.returncode}: {detail}",
            )
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            graph = raw_python_trace_to_atir(
                raw,
                project_name=case.id,
                repository_root=root,
                repository_url=case.repository,
                revision=case.revision,
            )
        except Exception as exc:
            raise _SystemBenchmarkError(
                FailureCategory.SYSTEM_RUNTIME_TRACE_INVALID,
                f"{type(exc).__name__}: {exc}",
            ) from exc
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    return _SystemExecution(
        graph,
        elapsed,
        {
            "target_python_version": raw.get("run", {}).get("python_version"),
            "target_python": metadata.get("target_python"),
            "runtime_cwd": metadata.get("runtime_cwd"),
            "runtime_resources": resource_metadata,
            "scenario_output": metadata.get("scenario_output"),
            "probe_count": metadata.get("probe_count"),
            "external_definition_count": metadata.get("external_definition_count"),
            "raw_trace": {
                "definitions": len(raw.get("definitions", [])),
                "occurrences": len(raw.get("occurrences", [])),
                "values": len(raw.get("values", [])),
                "edges": len(raw.get("edges", [])),
            },
        },
    )


def _verify_runtime_resources(
    spec: PythonSystemSpec, runtime_dir: Path
) -> list[dict[str, object]]:
    verified: list[dict[str, object]] = []
    for resource in spec.resources:
        candidate = (runtime_dir / resource.path).resolve()
        if not candidate.is_relative_to(runtime_dir):
            raise _SystemBenchmarkError(
                FailureCategory.SYSTEM_RUNTIME_RESOURCE_MISMATCH,
                f"runtime resource escapes runtime cwd: {resource.path}",
            )
        exists = candidate.is_file() if resource.kind == "file" else candidate.is_dir()
        if not exists:
            raise _SystemBenchmarkError(
                FailureCategory.SYSTEM_RUNTIME_RESOURCE_MISMATCH,
                f"missing runtime {resource.kind}: {resource.path}",
            )
        actual_sha256: str | None = None
        size_bytes: int | None = None
        if resource.kind == "file":
            size_bytes = candidate.stat().st_size
            if resource.sha256 is not None:
                digest = hashlib.sha256()
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                actual_sha256 = digest.hexdigest()
                if actual_sha256 != resource.sha256:
                    raise _SystemBenchmarkError(
                        FailureCategory.SYSTEM_RUNTIME_RESOURCE_MISMATCH,
                        f"runtime resource sha256 mismatch for {resource.path}: "
                        f"expected {resource.sha256}, found {actual_sha256}",
                    )
        verified.append(
            {
                "path": resource.path,
                "kind": resource.kind,
                "sha256": actual_sha256,
                "size_bytes": size_bytes,
            }
        )
    return verified


def _preflight(
    case: BenchmarkCase,
    root: Path,
    allow_revision_mismatch: bool,
    mode: BenchmarkMode,
) -> tuple[str | None, list[BenchmarkFailure]] | BenchmarkResult:
    analyzer_revision, analyzer_dirty = analyzer_source_state()
    if not root.is_dir():
        return BenchmarkResult(
            case_id=case.id,
            repository=case.repository,
            expected_revision=case.revision,
            analyzer_revision=analyzer_revision,
            analyzer_dirty=analyzer_dirty,
            mode=mode,
            status=BenchmarkStatus.FAILED,
            elapsed_seconds=0.0,
            failures=[
                BenchmarkFailure(
                    category=FailureCategory.MISSING_REPOSITORY,
                    severity=FailureSeverity.ERROR,
                    message=f"repository directory does not exist: {root}",
                )
            ],
        )
    actual_revision = git_revision(root)
    failures: list[BenchmarkFailure] = []
    if actual_revision is not None and actual_revision != case.revision:
        severity = FailureSeverity.WARNING if allow_revision_mismatch else FailureSeverity.ERROR
        failures.append(
            BenchmarkFailure(
                category=FailureCategory.REVISION_MISMATCH,
                severity=severity,
                message=f"expected revision {case.revision}, found {actual_revision}",
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
                mode=mode,
                status=BenchmarkStatus.FAILED,
                elapsed_seconds=0.0,
                failures=failures,
            )
    return actual_revision, failures


def _completed(
    case: BenchmarkCase,
    mode: BenchmarkMode,
    started: float,
    actual_revision: str | None,
    failures: list[BenchmarkFailure],
    metrics: SystemRuntimeBenchmarkMetrics | HybridBenchmarkMetrics,
    *,
    metadata: dict[str, object],
) -> BenchmarkResult:
    analyzer_revision, analyzer_dirty = analyzer_source_state()
    return BenchmarkResult(
        case_id=case.id,
        repository=case.repository,
        expected_revision=case.revision,
        actual_revision=actual_revision,
        analyzer_revision=analyzer_revision,
        analyzer_dirty=analyzer_dirty,
        mode=mode,
        status=BenchmarkStatus.COMPLETED,
        elapsed_seconds=perf_counter() - started,
        metrics=metrics,
        failures=failures,
        metadata=metadata,
    )


def _failed(
    case: BenchmarkCase,
    started: float,
    actual_revision: str | None,
    failures: list[BenchmarkFailure],
    category: FailureCategory,
    message: str,
    *,
    mode: BenchmarkMode = BenchmarkMode.RUNTIME,
) -> BenchmarkResult:
    analyzer_revision, analyzer_dirty = analyzer_source_state()
    return BenchmarkResult(
        case_id=case.id,
        repository=case.repository,
        expected_revision=case.revision,
        actual_revision=actual_revision,
        analyzer_revision=analyzer_revision,
        analyzer_dirty=analyzer_dirty,
        mode=mode,
        status=BenchmarkStatus.FAILED,
        elapsed_seconds=perf_counter() - started,
        failures=[
            *failures,
            BenchmarkFailure(
                category=category,
                severity=FailureSeverity.ERROR,
                message=message,
            ),
        ],
    )


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator
