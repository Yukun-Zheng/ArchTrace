"""Selective source-grounded tracing for arbitrary Python system boundaries."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
    TensorSpec,
    TraceRun,
)


@dataclass(frozen=True, slots=True)
class PythonProbeTarget:
    """One source-grounded Python function or method boundary to instrument."""

    module: str
    symbol: str
    role: str = "python"
    boundary: str = "local"

    @property
    def key(self) -> str:
        return f"{self.module}:{self.symbol}"


@dataclass(slots=True)
class PythonTraceResult:
    """One concrete Python execution and its normalized ATIR graph."""

    ir: ArchTraceIR
    output: Any


@dataclass(slots=True)
class _InstalledProbe:
    owner: Any
    name: str
    original: Any


class PythonProbeCapture:
    """Patch only declared Python boundaries and capture concrete call/data evidence."""

    def __init__(
        self,
        probes: Sequence[PythonProbeTarget],
        *,
        run_id: str = "run.python.0",
        project_name: str = "python-system",
        repository_root: str | Path | None = None,
        repository_url: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.probes = _deduplicate_probes(probes)
        self.run_id = run_id
        self.repository_root = (
            Path(repository_root).resolve() if repository_root is not None else None
        )
        self.graph = ArchTraceIR(
            project=ProjectInfo(
                name=project_name,
                root=str(self.repository_root) if self.repository_root is not None else None,
                repository_url=repository_url,
                revision=revision,
            ),
            runs=[
                TraceRun(
                    id=run_id,
                    framework="python",
                    framework_version=None,
                    python_version=sys.version.split()[0],
                    metadata={"backend": "selective_python_probes"},
                )
            ],
            metadata={"runtime_backend": "python_probe", "probe_count": len(self.probes)},
        )
        self._installed: list[_InstalledProbe] = []
        self._definitions: dict[str, str] = {}
        self._occurrence_counts: dict[str, int] = {}
        self._object_values: dict[int, str] = {}
        self._stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            f"archtrace_python_probe_stack_{id(self)}", default=()
        )
        self._last_occurrence: str | None = None
        self._edge_counter = 0
        self._evidence_counter = 0
        self._value_counter = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("PythonProbeCapture is already installed")
        try:
            for probe in self.probes:
                self._install_probe(probe)
        except Exception:
            self.remove()
            raise

    def remove(self) -> None:
        while self._installed:
            installed = self._installed.pop()
            setattr(installed.owner, installed.name, installed.original)

    def finish(self) -> ArchTraceIR:
        self.graph.metadata["captured_occurrences"] = sum(self._occurrence_counts.values())
        return ArchTraceIR.model_validate(self.graph.model_dump())

    def _install_probe(self, probe: PythonProbeTarget) -> None:
        module = importlib.import_module(probe.module)
        owner, name = _resolve_owner(module, probe.symbol)
        descriptor = inspect.getattr_static(owner, name)
        function, replacement_factory, skip_receiver = _unwrap_descriptor(descriptor)
        if not callable(function):
            raise TypeError(f"probe target is not callable: {probe.key}")

        source = _source_span(function, self.repository_root, probe)
        definition_id = f"python.def.{_slug(probe.key)}"
        if definition_id in self._definitions.values():
            raise ValueError(f"duplicate Python probe definition id: {definition_id}")
        evidence_id = self._add_evidence(
            description="Python callable selected as a runtime probe boundary.",
            source=source,
            status=FactStatus.DECLARED,
        )
        self.graph.nodes.append(
            ArchNode(
                id=definition_id,
                level=NodeLevel.SOURCE,
                kind=NodeKind.CONTROL,
                identity_kind=IdentityKind.DEFINITION,
                label=probe.symbol.rsplit(".", 1)[-1],
                role="python_probe_definition",
                source=[source] if source is not None else [],
                evidence_ids=[evidence_id],
                attributes={
                    "module": probe.module,
                    "qualname": probe.symbol,
                    "probe_role": probe.role,
                    "boundary": probe.boundary,
                },
            )
        )
        self._definitions[probe.key] = definition_id
        self._occurrence_counts[definition_id] = 0

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(
                probe,
                definition_id,
                function,
                args,
                kwargs,
                skip_receiver=skip_receiver,
            )

        replacement = replacement_factory(wrapper)
        self._installed.append(_InstalledProbe(owner=owner, name=name, original=descriptor))
        setattr(owner, name, replacement)

    def _invoke(
        self,
        probe: PythonProbeTarget,
        definition_id: str,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        skip_receiver: bool,
    ) -> Any:
        occurrence_index = self._occurrence_counts[definition_id]
        self._occurrence_counts[definition_id] += 1
        occurrence_id = f"python.call.{_slug(probe.key)}.{occurrence_index:04d}"
        definition = next(node for node in self.graph.nodes if node.id == definition_id)
        evidence_id = self._add_evidence(
            description="Python probe call observed at runtime.",
            source=definition.source[0] if definition.source else None,
            status=FactStatus.OBSERVED,
            metadata={"occurrence_index": occurrence_index},
        )
        stack = self._stack.get()
        parent_ids = [stack[-1]] if stack else []
        occurrence = ArchNode(
            id=occurrence_id,
            level=NodeLevel.SOURCE,
            kind=NodeKind.CONTROL,
            identity_kind=IdentityKind.OCCURRENCE,
            label=definition.label,
            role="python_probe_call",
            parent_ids=parent_ids,
            definition_id=definition_id,
            run_id=self.run_id,
            occurrence_index=occurrence_index,
            source=list(definition.source),
            evidence_ids=[evidence_id],
            attributes={
                "probe_role": probe.role,
                "boundary": probe.boundary,
                "status": "running",
            },
        )
        self.graph.nodes.append(occurrence)
        if stack:
            self._add_edge(stack[-1], occurrence_id, EdgeKind.CALLS, [evidence_id])
        if self._last_occurrence is not None:
            self._add_edge(self._last_occurrence, occurrence_id, EdgeKind.NEXT, [evidence_id])
        self._last_occurrence = occurrence_id

        for name, value in _bound_arguments(function, args, kwargs, skip_receiver=skip_receiver):
            value_id = self._value_node(
                value,
                label=f"{definition.label}.{name}",
                role="python_probe_input",
                evidence_id=evidence_id,
            )
            self._add_edge(value_id, occurrence_id, EdgeKind.CONSUMES, [evidence_id])

        token = self._stack.set((*stack, occurrence_id))
        try:
            output = function(*args, **kwargs)
        except BaseException as exc:
            occurrence.attributes.update(
                {
                    "status": "raised",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:512],
                }
            )
            raise
        finally:
            self._stack.reset(token)

        occurrence.attributes["status"] = "returned"
        output_id = self._value_node(
            output,
            label=f"{definition.label}.return",
            role="python_probe_output",
            evidence_id=evidence_id,
        )
        self._add_edge(occurrence_id, output_id, EdgeKind.PRODUCES, [evidence_id])
        return output

    def _value_node(
        self,
        value: Any,
        *,
        label: str,
        role: str,
        evidence_id: str,
    ) -> str:
        identity = _trackable_identity(value)
        if identity is not None and identity in self._object_values:
            return self._object_values[identity]

        value_id = f"python.value.{self._value_counter:07d}"
        self._value_counter += 1
        tensor = _tensor_spec(value)
        self.graph.nodes.append(
            ArchNode(
                id=value_id,
                level=NodeLevel.OPERATION,
                kind=NodeKind.TENSOR if tensor is not None else NodeKind.OTHER,
                identity_kind=IdentityKind.VALUE,
                label=label,
                role=role,
                run_id=self.run_id,
                evidence_ids=[evidence_id],
                tensor=tensor,
                attributes={"structure": _summarize_value(value)},
            )
        )
        if identity is not None:
            self._object_values[identity] = value_id
        return value_id

    def _add_evidence(
        self,
        *,
        description: str,
        source: SourceSpan | None,
        status: FactStatus,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        evidence_id = f"evidence.python.{self._evidence_counter:07d}"
        self._evidence_counter += 1
        self.graph.evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.RUNTIME,
                status=status,
                description=description,
                source=source,
                run_id=self.run_id,
                metadata={} if metadata is None else metadata,
            )
        )
        return evidence_id

    def _add_edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        evidence_ids: list[str],
    ) -> None:
        self.graph.edges.append(
            ArchEdge(
                id=f"edge.python.{self._edge_counter:07d}",
                source=source,
                target=target,
                kind=kind,
                evidence_ids=evidence_ids,
            )
        )
        self._edge_counter += 1


def trace_python_target(
    entrypoint: PythonProbeTarget,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    probes: Sequence[PythonProbeTarget] = (),
    run_id: str = "run.python.0",
    project_name: str = "python-system",
    repository_root: str | Path | None = None,
    repository_url: str | None = None,
    revision: str | None = None,
) -> PythonTraceResult:
    """Execute one Python entrypoint while selectively probing declared boundaries."""
    all_probes = [entrypoint, *probes]
    capture = PythonProbeCapture(
        all_probes,
        run_id=run_id,
        project_name=project_name,
        repository_root=repository_root,
        repository_url=repository_url,
        revision=revision,
    )
    capture.install()
    try:
        module = importlib.import_module(entrypoint.module)
        callable_target = _resolve_symbol(module, entrypoint.symbol)
        output = callable_target(*args, **({} if kwargs is None else kwargs))
    finally:
        capture.remove()
    graph = capture.finish()
    graph.runs[0].entrypoint = entrypoint.key
    return PythonTraceResult(ir=graph, output=output)


def _deduplicate_probes(probes: Sequence[PythonProbeTarget]) -> list[PythonProbeTarget]:
    by_key: dict[str, PythonProbeTarget] = {}
    for probe in probes:
        existing = by_key.get(probe.key)
        if existing is not None and existing != probe:
            raise ValueError(f"conflicting probe declarations for {probe.key}")
        by_key[probe.key] = probe
    return list(by_key.values())


def _resolve_owner(module: Any, symbol: str) -> tuple[Any, str]:
    parts = symbol.split(".")
    owner = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _resolve_symbol(module: Any, symbol: str) -> Any:
    value = module
    for part in symbol.split("."):
        value = getattr(value, part)
    return value


def _unwrap_descriptor(descriptor: Any) -> tuple[Callable[..., Any], Callable[[Any], Any], bool]:
    if isinstance(descriptor, staticmethod):
        return descriptor.__func__, staticmethod, False
    if isinstance(descriptor, classmethod):
        return descriptor.__func__, classmethod, True
    return descriptor, lambda value: value, inspect.isfunction(descriptor)


def _source_span(
    function: Callable[..., Any],
    repository_root: Path | None,
    probe: PythonProbeTarget,
) -> SourceSpan | None:
    try:
        path = inspect.getsourcefile(function) or inspect.getfile(function)
        lines, start_line = inspect.getsourcelines(function)
    except (OSError, TypeError):
        return None
    source_path = Path(path).resolve()
    rendered = str(source_path)
    if repository_root is not None:
        with suppress(ValueError):
            rendered = source_path.relative_to(repository_root).as_posix()
    return SourceSpan(
        path=rendered,
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        symbol=f"{probe.module}.{probe.symbol}",
    )


def _bound_arguments(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    skip_receiver: bool,
) -> list[tuple[str, Any]]:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        items = list(bound.arguments.items())
    except (TypeError, ValueError):
        items = [(f"arg{index}", value) for index, value in enumerate(args)]
        items.extend((key, value) for key, value in kwargs.items())
    if skip_receiver and items and items[0][0] in {"self", "cls"}:
        items = items[1:]
    return items


def _trackable_identity(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return None
    return id(value)


def _tensor_spec(value: Any) -> TensorSpec | None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    rendered_shape: list[int | str | None]
    try:
        rendered_shape = [int(item) for item in tuple(shape)]
    except (TypeError, ValueError):
        rendered_shape = [str(item) for item in tuple(shape)]
    device = getattr(value, "device", None)
    requires_grad = getattr(value, "requires_grad", None)
    return TensorSpec(
        shape=rendered_shape,
        dtype=str(dtype).removeprefix("torch."),
        device=None if device is None else str(device),
        requires_grad=requires_grad if isinstance(requires_grad, bool) else None,
    )


def _summarize_value(value: Any, *, depth: int = 0) -> dict[str, Any]:
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    tensor = _tensor_spec(value)
    if tensor is not None:
        return {"type": value_type, "tensor": tensor.model_dump(mode="json")}
    if value is None or isinstance(value, (bool, int, float)):
        return {"type": value_type, "value": value}
    if isinstance(value, str):
        return {"type": value_type, "value": value[:128], "length": len(value)}
    if isinstance(value, bytes):
        return {"type": value_type, "length": len(value)}
    if depth >= 2:
        return {"type": value_type}
    if isinstance(value, Mapping):
        keys = list(value.keys())
        items: dict[str, Any] = {}
        for key in keys[:20]:
            try:
                items[str(key)] = _summarize_value(value[key], depth=depth + 1)
            except Exception:
                items[str(key)] = {"type": "unavailable"}
        return {
            "type": value_type,
            "length": len(value),
            "keys": [str(key) for key in keys[:50]],
            "items": items,
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "type": value_type,
            "length": len(value),
            "items": [_summarize_value(item, depth=depth + 1) for item in value[:10]],
        }
    if hasattr(value, "__dict__") and type(value).__module__ == "types":
        public = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return {
            "type": value_type,
            "attributes": {
                key: _summarize_value(item, depth=depth + 1)
                for key, item in list(public.items())[:20]
            },
        }
    return {"type": value_type}



def raw_python_trace_to_atir(
    raw: dict[str, Any],
    *,
    project_name: str,
    repository_root: str | Path | None = None,
    repository_url: str | None = None,
    revision: str | None = None,
) -> ArchTraceIR:
    """Normalize a stdlib-agent raw trace into canonical ATIR."""
    run = raw.get("run", {})
    run_id = str(run.get("id", "run.python.raw.0"))
    root = repository_root if repository_root is not None else raw.get("repository_root")
    graph = ArchTraceIR(
        project=ProjectInfo(
            name=project_name,
            root=None if root is None else str(root),
            repository_url=repository_url,
            revision=revision,
        ),
        runs=[
            TraceRun(
                id=run_id,
                framework=str(run.get("framework", "python")),
                python_version=(
                    str(run["python_version"]) if run.get("python_version") is not None else None
                ),
                metadata={"backend": run.get("backend")},
            )
        ],
        metadata={
            "runtime_backend": "python_probe_agent",
            "raw_schema_version": raw.get("schema_version"),
            **(raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}),
        },
    )
    for item in raw.get("evidence", []):
        source = _raw_source_span(item.get("source"))
        graph.evidence.append(
            Evidence(
                id=str(item["id"]),
                kind=EvidenceKind.RUNTIME,
                status=FactStatus(str(item.get("status", "observed"))),
                description=item.get("description"),
                source=source,
                run_id=item.get("run_id") or run_id,
                metadata=item.get("metadata", {}),
            )
        )
    for item in raw.get("definitions", []):
        source = _raw_source_span(item.get("source"))
        graph.nodes.append(
            ArchNode(
                id=str(item["id"]),
                level=NodeLevel.SOURCE,
                kind=NodeKind.CONTROL,
                identity_kind=IdentityKind.DEFINITION,
                label=str(item.get("label", item["id"])),
                role=(
                    "python_external_boundary_definition"
                    if item.get("external")
                    else "python_probe_definition"
                ),
                source=[source] if source is not None else [],
                evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
                attributes={
                    "probe_role": item.get("role"),
                    "boundary": item.get("boundary"),
                    "external": bool(item.get("external", False)),
                    "stubbed": bool(item.get("stubbed", False)),
                    "probe_key": item.get("key"),
                },
            )
        )
    for item in raw.get("occurrences", []):
        source = _raw_source_span(item.get("source"))
        graph.nodes.append(
            ArchNode(
                id=str(item["id"]),
                level=NodeLevel.SOURCE,
                kind=NodeKind.CONTROL,
                identity_kind=IdentityKind.OCCURRENCE,
                label=str(item.get("label", item["id"])),
                role="python_probe_call",
                parent_ids=[str(value) for value in item.get("parent_ids", [])],
                definition_id=str(item["definition_id"]),
                run_id=str(item.get("run_id", run_id)),
                occurrence_index=int(item.get("occurrence_index", 0)),
                source=[source] if source is not None else [],
                evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
                attributes={
                    "probe_role": item.get("role"),
                    "boundary": item.get("boundary"),
                    "status": item.get("status"),
                    **(
                        item.get("attributes", {})
                        if isinstance(item.get("attributes"), dict)
                        else {}
                    ),
                },
            )
        )
    for item in raw.get("values", []):
        tensor_payload = item.get("tensor")
        graph.nodes.append(
            ArchNode(
                id=str(item["id"]),
                level=NodeLevel.OPERATION,
                kind=NodeKind.TENSOR if isinstance(tensor_payload, dict) else NodeKind.OTHER,
                identity_kind=IdentityKind.VALUE,
                label=str(item.get("label", item["id"])),
                role=item.get("role"),
                run_id=str(item.get("run_id", run_id)),
                evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
                tensor=(
                    TensorSpec.model_validate(tensor_payload)
                    if isinstance(tensor_payload, dict)
                    else None
                ),
                attributes={"structure": item.get("structure", {})},
            )
        )
    for item in raw.get("edges", []):
        graph.edges.append(
            ArchEdge(
                id=str(item["id"]),
                source=str(item["source"]),
                target=str(item["target"]),
                kind=EdgeKind(str(item["kind"])),
                evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
            )
        )
    return ArchTraceIR.model_validate(graph.model_dump())


def _raw_source_span(value: Any) -> SourceSpan | None:
    if not isinstance(value, dict):
        return None
    return SourceSpan.model_validate(value)

def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return normalized.strip("_.-") or "probe"
