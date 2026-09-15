"""Declarative PyTorch runtime and static/runtime hybrid benchmarks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

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
    RuntimeBenchmarkMetrics,
    RuntimeTargetSpec,
    RuntimeValueSpec,
)
from archtrace.benchmark.runner import analyzer_source_state, git_revision
from archtrace.ir import CoverageStatus, EdgeKind, IdentityKind
from archtrace.runtime import trace_model
from archtrace.static import index_repository, repository_index_to_atir


@dataclass(slots=True)
class _RuntimeExecution:
    graph: Any
    output: Any
    trace_seconds: float
    parameter_count: int
    output_shapes: list[list[int]]
    stage_seconds: dict[str, float]


def load_runtime_spec(path: str | Path) -> RuntimeTargetSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuntimeTargetSpec.model_validate(payload)


def run_runtime_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: RuntimeTargetSpec,
    *,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    started = perf_counter()
    root = Path(repository).resolve()
    preflight = _benchmark_preflight(
        case, root, allow_revision_mismatch, mode=BenchmarkMode.RUNTIME
    )
    if isinstance(preflight, BenchmarkResult):
        return preflight
    actual_revision, failures = preflight
    try:
        execution = _execute_pytorch_target(case, root, spec)
    except _RuntimeBenchmarkError as exc:
        return _failed_runtime_result(
            case,
            started,
            actual_revision,
            failures,
            exc.category,
            exc.message,
        )

    graph = execution.graph
    definitions = [node for node in graph.nodes if node.identity_kind == IdentityKind.DEFINITION]
    target_definitions = _target_source_definitions(graph, root)
    tensor_nodes = [node for node in graph.nodes if node.tensor is not None]
    tensor_spec_nodes = [
        node
        for node in tensor_nodes
        if node.tensor is not None
        and (node.tensor.shape is not None or node.tensor.dtype is not None)
    ]
    metrics = RuntimeBenchmarkMetrics(
        trace_seconds=execution.trace_seconds,
        runtime_nodes=len(graph.nodes),
        runtime_edges=len(graph.edges),
        definitions=len(definitions),
        module_definitions=sum(node.role == "pytorch_module_definition" for node in definitions),
        operator_definitions=sum(
            node.role == "pytorch_operator_definition" for node in definitions
        ),
        occurrences=sum(node.identity_kind == IdentityKind.OCCURRENCE for node in graph.nodes),
        values=sum(node.identity_kind == IdentityKind.VALUE for node in graph.nodes),
        states=sum(node.identity_kind == IdentityKind.STATE for node in graph.nodes),
        target_source_definitions=len(target_definitions),
        tensor_nodes=len(tensor_nodes),
        tensor_spec_nodes=len(tensor_spec_nodes),
        tensor_spec_coverage=_ratio(len(tensor_spec_nodes), len(tensor_nodes)),
        consumes_edges=sum(edge.kind == EdgeKind.CONSUMES for edge in graph.edges),
        produces_edges=sum(edge.kind == EdgeKind.PRODUCES for edge in graph.edges),
        derived_from_edges=sum(edge.kind == EdgeKind.DERIVED_FROM for edge in graph.edges),
        parameter_count=execution.parameter_count,
        output_tensor_count=len(execution.output_shapes),
    )
    return _completed_result(
        case,
        BenchmarkMode.RUNTIME,
        started,
        actual_revision,
        failures,
        metrics,
        metadata={
            "runtime_spec": spec.model_dump(mode="json"),
            "stage_seconds": execution.stage_seconds,
            "output_shapes": execution.output_shapes,
            "runtime_backend": graph.metadata.get("runtime_backend"),
            "operator_dispatch": graph.metadata.get("operator_dispatch"),
            "structured_captures": graph.metadata.get("structured_captures", []),
        },
    )


def run_hybrid_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: RuntimeTargetSpec,
    *,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    started = perf_counter()
    root = Path(repository).resolve()
    preflight = _benchmark_preflight(
        case, root, allow_revision_mismatch, mode=BenchmarkMode.HYBRID
    )
    if isinstance(preflight, BenchmarkResult):
        return preflight
    actual_revision, failures = preflight
    try:
        static_started = perf_counter()
        static_index = index_repository(root)
        static_graph = repository_index_to_atir(static_index)
        static_seconds = perf_counter() - static_started
        execution = _execute_pytorch_target(case, root, spec)
        reconcile_started = perf_counter()
        merged = reconcile_static_runtime(static_graph, execution.graph)
        reconciliation_seconds = perf_counter() - reconcile_started
    except _RuntimeBenchmarkError as exc:
        return _failed_runtime_result(
            case,
            started,
            actual_revision,
            failures,
            exc.category,
            exc.message,
            mode=BenchmarkMode.HYBRID,
        )
    except Exception as exc:
        return _failed_runtime_result(
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
    target_definitions = _target_source_definitions(execution.graph, root)
    coverage_counts = {status: 0 for status in CoverageStatus}
    for record in merged.coverage:
        coverage_counts[record.status] += 1
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
    return _completed_result(
        case,
        BenchmarkMode.HYBRID,
        started,
        actual_revision,
        failures,
        metrics,
        metadata={
            "runtime_spec": spec.model_dump(mode="json"),
            "stage_seconds": {
                "static": static_seconds,
                **execution.stage_seconds,
                "reconciliation": reconciliation_seconds,
            },
            "output_shapes": execution.output_shapes,
            "source_alignments": alignments,
        },
    )


class _RuntimeBenchmarkError(Exception):
    def __init__(self, category: FailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def _execute_pytorch_target(
    case: BenchmarkCase,
    repository: Path,
    spec: RuntimeTargetSpec,
) -> _RuntimeExecution:
    if spec.framework != "pytorch":
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_UNSUPPORTED_FRAMEWORK,
            f"unsupported runtime framework: {spec.framework}",
        )
    missing = [
        name
        for name in ["torch", *spec.required_imports]
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_DEPENDENCY_MISSING,
            "missing runtime import(s): " + ", ".join(missing),
        )
    import_root = (repository / spec.import_root).resolve()
    if not import_root.is_dir() or not import_root.is_relative_to(repository):
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            f"runtime import_root is outside/missing from repository: {spec.import_root}",
        )

    top_level = spec.module.split(".", 1)[0]
    with _isolated_module_namespace(top_level), _runtime_import_path(import_root):
        import_started = perf_counter()
        try:
            module = importlib.import_module(spec.module)
            target = getattr(module, spec.symbol)
        except ModuleNotFoundError as exc:
            category = (
                FailureCategory.RUNTIME_IMPORT_FAILURE
                if exc.name == top_level
                else FailureCategory.RUNTIME_DEPENDENCY_MISSING
            )
            raise _RuntimeBenchmarkError(category, f"{type(exc).__name__}: {exc}") from exc
        except (ImportError, AttributeError, OSError) as exc:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_IMPORT_FAILURE, f"{type(exc).__name__}: {exc}"
            ) from exc
        import_seconds = perf_counter() - import_started

        construct_started = perf_counter()
        try:
            torch = importlib.import_module("torch")
            torch.manual_seed(spec.seed)
            model = target(**spec.constructor_kwargs)
        except Exception as exc:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_CONSTRUCTION_FAILURE,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        construction_seconds = perf_counter() - construct_started

        try:
            if spec.eval_mode and hasattr(model, "eval"):
                model.eval()
            args = tuple(_materialize_value(item, torch) for item in spec.args)
            kwargs = {
                key: _materialize_value(item, torch) for key, item in spec.kwargs.items()
            }
            parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
            trace_started = perf_counter()
            grad_context = torch.no_grad() if spec.no_grad else nullcontext()
            with grad_context:
                traced = trace_model(
                    model,
                    args,
                    kwargs,
                    run_id=f"run.benchmark.{case.id}",
                    project_name=case.id,
                    capture_operators=spec.capture_operators,
                    capture_fx=spec.capture_fx,
                    capture_export=spec.capture_export,
                )
            trace_seconds = perf_counter() - trace_started
        except Exception as exc:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_EXECUTION_FAILURE,
                f"{type(exc).__name__}: {exc}",
            ) from exc

        output_shapes = _tensor_shapes(traced.output, torch)
        if spec.expected_output_shapes and output_shapes != spec.expected_output_shapes:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OUTPUT_MISMATCH,
                f"expected output shapes {spec.expected_output_shapes}, observed {output_shapes}",
            )
        return _RuntimeExecution(
            graph=traced.ir,
            output=traced.output,
            trace_seconds=trace_seconds,
            parameter_count=parameter_count,
            output_shapes=output_shapes,
            stage_seconds={
                "import": import_seconds,
                "construction": construction_seconds,
                "trace": trace_seconds,
            },
        )


def _materialize_value(spec: RuntimeValueSpec, torch: Any) -> Any:
    if spec.kind == "scalar":
        return spec.value
    dtype = getattr(torch, spec.dtype, None)
    if dtype is None:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID, f"unknown torch dtype: {spec.dtype}"
        )
    kwargs: dict[str, Any] = {"dtype": dtype, "device": "cpu"}
    if spec.generator == "randn":
        value = torch.randn(tuple(spec.shape), **kwargs)
    elif spec.generator == "zeros":
        value = torch.zeros(tuple(spec.shape), **kwargs)
    elif spec.generator == "ones":
        value = torch.ones(tuple(spec.shape), **kwargs)
    else:
        value = torch.tensor(spec.values, **kwargs)
        if spec.shape:
            value = value.reshape(tuple(spec.shape))
    if spec.requires_grad:
        value.requires_grad_(True)
    return value


def _tensor_shapes(value: Any, torch: Any) -> list[list[int]]:
    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, dict):
        result: list[list[int]] = []
        for key in sorted(value, key=str):
            result.extend(_tensor_shapes(value[key], torch))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_tensor_shapes(item, torch))
        return result
    return []


def _target_source_definitions(graph: Any, root: Path) -> list[Any]:
    result = []
    for node in graph.nodes:
        if node.identity_kind != IdentityKind.DEFINITION or not node.source:
            continue
        if any(_span_inside_root(span.path, root) for span in node.source):
            result.append(node)
    return result


def _span_inside_root(path: str, root: Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def _isolated_module_namespace(top_level: str) -> Iterator[None]:
    prefix = f"{top_level}."
    previous = {
        name: module
        for name, module in list(sys.modules.items())
        if name == top_level or name.startswith(prefix)
    }
    for name in previous:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == top_level or name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        importlib.invalidate_caches()


@contextmanager
def _runtime_import_path(import_root: Path) -> Iterator[None]:
    value = str(import_root)
    sys.path.insert(0, value)
    previous = os.getcwd()
    try:
        os.chdir(import_root)
        yield
    finally:
        os.chdir(previous)
        with suppress(ValueError):
            sys.path.remove(value)


def _benchmark_preflight(
    case: BenchmarkCase,
    root: Path,
    allow_revision_mismatch: bool,
    *,
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


def _completed_result(
    case: BenchmarkCase,
    mode: BenchmarkMode,
    started: float,
    actual_revision: str | None,
    failures: list[BenchmarkFailure],
    metrics: RuntimeBenchmarkMetrics | HybridBenchmarkMetrics,
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


def _failed_runtime_result(
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
