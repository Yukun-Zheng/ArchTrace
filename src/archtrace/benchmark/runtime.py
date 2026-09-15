"""Declarative PyTorch runtime and static/runtime hybrid benchmarks."""

from __future__ import annotations

import importlib
import importlib.metadata
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
    RuntimeEnvironmentSpec,
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


@dataclass(frozen=True, slots=True)
class _ResolvedOverlay:
    repository_root: Path
    installed_root: Path
    repository_path: str
    module: str
    matched_files: int
    private_copy_verified: bool


@dataclass(frozen=True, slots=True)
class _RuntimeEnvironmentState:
    packages: dict[str, str]
    overlays: list[_ResolvedOverlay]


def load_runtime_environment(path: str | Path) -> RuntimeEnvironmentSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuntimeEnvironmentSpec.model_validate(payload)


def load_runtime_spec(path: str | Path) -> RuntimeTargetSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuntimeTargetSpec.model_validate(payload)


def run_runtime_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: RuntimeTargetSpec,
    *,
    environment: RuntimeEnvironmentSpec | None = None,
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
        environment_state = _verify_runtime_environment(environment, root)
        execution = _execute_pytorch_target(
            case, root, spec, overlays=environment_state.overlays
        )
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
    target_definitions = _target_source_definitions(
        graph, root, role="pytorch_module_definition"
    )
    target_operator_definitions = _target_source_definitions(
        graph, root, role="pytorch_operator_definition"
    )
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
        target_operator_source_definitions=len(target_operator_definitions),
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
            "environment": _environment_metadata(environment_state),
            "stage_seconds": execution.stage_seconds,
            "output_shapes": execution.output_shapes,
            "runtime_backend": graph.metadata.get("runtime_backend"),
            "operator_dispatch": graph.metadata.get("operator_dispatch"),
            "structured_captures": graph.metadata.get("structured_captures", []),
            "runtime_source_overlay_remaps": graph.metadata.get(
                "runtime_source_overlay_remaps", 0
            ),
        },
    )


def run_hybrid_benchmark(
    case: BenchmarkCase,
    repository: str | Path,
    spec: RuntimeTargetSpec,
    *,
    environment: RuntimeEnvironmentSpec | None = None,
    allow_revision_mismatch: bool = False,
) -> BenchmarkResult:
    started = perf_counter()
    root = Path(repository).resolve()
    preflight = _benchmark_preflight(case, root, allow_revision_mismatch, mode=BenchmarkMode.HYBRID)
    if isinstance(preflight, BenchmarkResult):
        return preflight
    actual_revision, failures = preflight
    try:
        environment_state = _verify_runtime_environment(environment, root)
        static_started = perf_counter()
        static_index = index_repository(root)
        static_graph = repository_index_to_atir(static_index)
        static_seconds = perf_counter() - static_started
        execution = _execute_pytorch_target(
            case, root, spec, overlays=environment_state.overlays
        )
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
    target_definitions = _target_source_definitions(
        execution.graph, root, role="pytorch_module_definition"
    )
    target_operator_definitions = _target_source_definitions(
        execution.graph, root, role="pytorch_operator_definition"
    )
    unaligned_definitions = [
        node for node in target_definitions if node.id not in aligned_ids
    ]
    if unaligned_definitions:
        failures.append(
            BenchmarkFailure(
                category=FailureCategory.HYBRID_ALIGNMENT_GAP,
                severity=FailureSeverity.WARNING,
                message="Target-owned runtime definitions could not be source-aligned.",
                count=len(unaligned_definitions),
                examples=[
                    _definition_example(node) for node in unaligned_definitions[:10]
                ],
            )
        )
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
        target_operator_definitions=len(target_operator_definitions),
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
            "environment": _environment_metadata(environment_state),
            "stage_seconds": {
                "static": static_seconds,
                **execution.stage_seconds,
                "reconciliation": reconciliation_seconds,
            },
            "output_shapes": execution.output_shapes,
            "source_alignment_count": len(alignments),
            "source_alignments": alignments[:50],
            "unaligned_target_definitions": [
                _definition_record(node) for node in unaligned_definitions[:50]
            ],
            "runtime_source_overlay_remaps": execution.graph.metadata.get(
                "runtime_source_overlay_remaps", 0
            ),
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
    *,
    overlays: list[_ResolvedOverlay],
) -> _RuntimeExecution:
    if spec.framework != "pytorch":
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_UNSUPPORTED_FRAMEWORK,
            f"unsupported runtime framework: {spec.framework}",
        )
    missing = [
        name for name in ["torch", *spec.required_imports] if importlib.util.find_spec(name) is None
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
            if spec.torch_num_threads is not None:
                torch.set_num_threads(spec.torch_num_threads)
            if spec.torch_num_interop_threads is not None:
                torch.set_num_interop_threads(spec.torch_num_interop_threads)
            torch.manual_seed(spec.seed)
            constructor_kwargs = {
                key: _materialize_constructor_value(value)
                for key, value in spec.constructor_kwargs.items()
            }
            model = target(**constructor_kwargs)
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
            kwargs = {key: _materialize_value(item, torch) for key, item in spec.kwargs.items()}
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
            traced.ir = _remap_runtime_overlay_sources(
                traced.ir, repository, overlays
            )
        except _RuntimeBenchmarkError:
            raise
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


def _materialize_constructor_value(value: Any) -> Any:
    """Materialize declarative constructor values without repository-specific code.

    Two reserved object forms are supported recursively:
    ``{"$kind": "import", "module": "torch", "symbol": "float32"}``
    returns an imported object, while ``{"$kind": "construct", ...}`` invokes a
    declared factory/class with recursively materialized args and kwargs.
    """
    if isinstance(value, list):
        return [_materialize_constructor_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("$kind")
    if kind is None:
        return {key: _materialize_constructor_value(item) for key, item in value.items()}
    if kind not in {"import", "construct"}:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            f"unknown constructor value kind: {kind!r}",
        )

    module_name = value.get("module")
    symbol_name = value.get("symbol")
    if not isinstance(module_name, str) or not isinstance(symbol_name, str):
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            "constructor import/construct values require string module and symbol",
        )
    try:
        module = importlib.import_module(module_name)
        target = _resolve_symbol(module, symbol_name)
    except (ImportError, AttributeError) as exc:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            f"cannot materialize {module_name}.{symbol_name}: {type(exc).__name__}: {exc}",
        ) from exc
    if kind == "import":
        return target

    raw_args = value.get("args", [])
    raw_kwargs = value.get("kwargs", {})
    if not isinstance(raw_args, list) or not isinstance(raw_kwargs, dict):
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID,
            "construct values require list args and object kwargs",
        )
    args = [_materialize_constructor_value(item) for item in raw_args]
    kwargs = {key: _materialize_constructor_value(item) for key, item in raw_kwargs.items()}
    try:
        return target(*args, **kwargs)
    except Exception as exc:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_CONSTRUCTION_FAILURE,
            f"cannot construct {module_name}.{symbol_name}: {type(exc).__name__}: {exc}",
        ) from exc


def _resolve_symbol(module: Any, symbol: str) -> Any:
    value = module
    for part in symbol.split("."):
        value = getattr(value, part)
    return value


def _materialize_value(spec: RuntimeValueSpec, torch: Any) -> Any:
    if spec.kind == "scalar":
        return spec.value
    if spec.kind == "mapping":
        return {key: _materialize_value(item, torch) for key, item in spec.items.items()}
    if spec.kind == "sequence":
        values = [_materialize_value(item, torch) for item in spec.elements]
        return tuple(values) if spec.sequence_type == "tuple" else values
    if spec.kind == "object":
        if spec.module is None or spec.symbol is None:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_SPEC_INVALID,
                "object runtime values require module and symbol",
            )
        try:
            module = importlib.import_module(spec.module)
            target = _resolve_symbol(module, spec.symbol)
            args = [_materialize_value(item, torch) for item in spec.constructor_args]
            object_kwargs = {
                key: _materialize_value(item, torch)
                for key, item in spec.constructor_kwargs.items()
            }
            return target(*args, **object_kwargs)
        except _RuntimeBenchmarkError:
            raise
        except Exception as exc:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_SPEC_INVALID,
                f"cannot construct runtime value {spec.module}.{spec.symbol}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

    dtype = getattr(torch, spec.dtype, None)
    if dtype is None:
        raise _RuntimeBenchmarkError(
            FailureCategory.RUNTIME_SPEC_INVALID, f"unknown torch dtype: {spec.dtype}"
        )
    tensor_kwargs: dict[str, Any] = {"dtype": dtype, "device": "cpu"}
    if spec.generator == "randn":
        value = torch.randn(tuple(spec.shape), **tensor_kwargs)
    elif spec.generator == "zeros":
        value = torch.zeros(tuple(spec.shape), **tensor_kwargs)
    elif spec.generator == "ones":
        value = torch.ones(tuple(spec.shape), **tensor_kwargs)
    else:
        value = torch.tensor(spec.values, **tensor_kwargs)
        if spec.shape:
            value = value.reshape(tuple(spec.shape))
    if spec.requires_grad:
        value.requires_grad_(True)
    return value


def _definition_example(node: Any) -> str:
    if node.source:
        span = node.source[0]
        symbol = f"::{span.symbol}" if span.symbol else ""
        return f"{node.label} @ {span.path}:{span.start_line}{symbol}"
    return f"{node.label} ({node.id})"


def _definition_record(node: Any) -> dict[str, object]:
    return {
        "id": node.id,
        "label": node.label,
        "role": node.role,
        "source": [span.model_dump(mode="json") for span in node.source],
    }


def _verify_runtime_environment(
    spec: RuntimeEnvironmentSpec | None, repository: Path
) -> _RuntimeEnvironmentState:
    if spec is None:
        return _RuntimeEnvironmentState(packages={}, overlays=[])
    if spec.python is not None:
        actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual_python != spec.python:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_ENVIRONMENT_MISMATCH,
                f"expected Python {spec.python}, found {actual_python}",
            )

    versions: dict[str, str] = {}
    for requirement in spec.packages:
        try:
            actual = importlib.metadata.version(requirement.distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_ENVIRONMENT_MISMATCH,
                f"missing required distribution: {requirement.distribution}",
            ) from exc
        if not _version_matches(actual, requirement.version):
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_ENVIRONMENT_MISMATCH,
                f"expected {requirement.distribution} {requirement.version}, found {actual}",
            )
        versions[requirement.distribution] = actual

    overlays: list[_ResolvedOverlay] = []
    for overlay in spec.overlays:
        repository_root = (repository / overlay.repository_path).resolve()
        if not repository_root.is_dir() or not repository_root.is_relative_to(repository):
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OVERLAY_MISMATCH,
                f"overlay source is outside/missing from repository: {overlay.repository_path}",
            )
        module_spec = importlib.util.find_spec(overlay.module)
        if module_spec is None:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OVERLAY_MISMATCH,
                f"overlay destination module is not importable: {overlay.module}",
            )
        if module_spec.submodule_search_locations:
            installed_root = Path(next(iter(module_spec.submodule_search_locations))).resolve()
        elif module_spec.origin is not None:
            installed_root = Path(module_spec.origin).resolve().parent
        else:
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OVERLAY_MISMATCH,
                f"cannot resolve module filesystem root: {overlay.module}",
            )

        source_files = sorted(repository_root.rglob("*.py"))
        mismatches: list[str] = []
        shared_links: list[str] = []
        matched = 0
        for source_file in source_files:
            relative = source_file.relative_to(repository_root)
            installed_file = installed_root / relative
            if (
                not installed_file.is_file()
                or installed_file.read_bytes() != source_file.read_bytes()
            ):
                mismatches.append(relative.as_posix())
                continue
            if overlay.require_private_copy and installed_file.stat().st_nlink > 1:
                shared_links.append(relative.as_posix())
                continue
            matched += 1
        if mismatches:
            examples = ", ".join(mismatches[:5])
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OVERLAY_MISMATCH,
                f"overlay {overlay.module} differs from repository source in "
                f"{len(mismatches)} file(s): {examples}",
            )
        if shared_links:
            examples = ", ".join(shared_links[:5])
            raise _RuntimeBenchmarkError(
                FailureCategory.RUNTIME_OVERLAY_MISMATCH,
                f"overlay {overlay.module} is backed by shared hardlinks in "
                f"{len(shared_links)} file(s): {examples}",
            )
        overlays.append(
            _ResolvedOverlay(
                repository_root=repository_root,
                installed_root=installed_root,
                repository_path=overlay.repository_path,
                module=overlay.module,
                matched_files=matched,
                private_copy_verified=overlay.require_private_copy,
            )
        )
    return _RuntimeEnvironmentState(packages=versions, overlays=overlays)


def _version_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.split("+", 1)[0] == expected


def _environment_metadata(state: _RuntimeEnvironmentState) -> dict[str, object]:
    return {
        "packages": dict(sorted(state.packages.items())),
        "overlays": [
            {
                "module": overlay.module,
                "repository_path": overlay.repository_path,
                "matched_files": overlay.matched_files,
                "private_copy_verified": overlay.private_copy_verified,
            }
            for overlay in state.overlays
        ],
    }


def _remap_runtime_overlay_sources(
    graph: Any, repository: Path, overlays: list[_ResolvedOverlay]
) -> Any:
    if not overlays:
        return graph

    remapped_count = 0

    def remap_span(span: Any) -> Any:
        nonlocal remapped_count
        source_path = Path(span.path)
        if not source_path.is_absolute():
            return span
        resolved = source_path.resolve()
        for overlay in overlays:
            try:
                relative = resolved.relative_to(overlay.installed_root)
            except ValueError:
                continue
            repository_path = overlay.repository_root / relative
            if repository_path.is_file():
                remapped_count += 1
                return span.model_copy(
                    update={"path": repository_path.relative_to(repository).as_posix()}
                )
        return span

    nodes = [
        node.model_copy(update={"source": [remap_span(span) for span in node.source]})
        for node in graph.nodes
    ]
    evidence = [
        item.model_copy(update={"source": remap_span(item.source)})
        if item.source is not None
        else item
        for item in graph.evidence
    ]
    metadata = dict(graph.metadata)
    metadata["runtime_source_overlay_remaps"] = remapped_count
    return type(graph).model_validate(
        graph.model_copy(
            update={"nodes": nodes, "evidence": evidence, "metadata": metadata}
        ).model_dump()
    )


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


def _target_source_definitions(
    graph: Any, root: Path, *, role: str | None = None
) -> list[Any]:
    result = []
    for node in graph.nodes:
        if node.identity_kind != IdentityKind.DEFINITION or not node.source:
            continue
        if role is not None and node.role != role:
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
