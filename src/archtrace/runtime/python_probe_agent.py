"""Stdlib-only selective Python probe agent for target runtimes.

This module must remain importable as a standalone file on Python 3.10. It does
not import ArchTrace, Pydantic, NumPy, PyTorch, or any other third-party package.
The target process emits a neutral raw trace; ArchTrace normalizes that trace to
ATIR in a separate, supported Python >=3.11 process.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import inspect
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProbeTarget:
    module: str
    symbol: str
    role: str = "python"
    boundary: str = "local"

    @property
    def key(self) -> str:
        return f"{self.module}:{self.symbol}"


@dataclass
class _InstalledProbe:
    owner: Any
    name: str
    original: Any


class RawProbeCapture:
    """Capture selected Python boundaries into a JSON-serializable raw trace."""

    def __init__(
        self,
        probes: Sequence[ProbeTarget],
        *,
        run_id: str = "run.python.raw.0",
        repository_root: str | Path | None = None,
    ) -> None:
        self.probes = _deduplicate_probes(probes)
        self.run_id = run_id
        self.repository_root = (
            Path(repository_root).resolve() if repository_root is not None else None
        )
        self.definitions: list[dict[str, Any]] = []
        self.occurrences: list[dict[str, Any]] = []
        self.values: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self._installed: list[_InstalledProbe] = []
        self._definition_ids: dict[str, str] = {}
        self._occurrence_counts: dict[str, int] = {}
        self._external_definitions: dict[tuple[str, str, str], str] = {}
        self._object_values: dict[int, str] = {}
        self._stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            f"archtrace_raw_probe_stack_{id(self)}", default=()
        )
        self._last_occurrence: str | None = None
        self._edge_counter = 0
        self._evidence_counter = 0
        self._value_counter = 0
        self._external_counter = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("RawProbeCapture is already installed")
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

    def record_boundary(
        self,
        label: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        output: Any = None,
        role: str = "external",
        boundary: str = "external",
        stubbed: bool = True,
    ) -> Any:
        """Record an explicit external/transport boundary without inventing internals."""
        definition_key = (label, role, boundary)
        definition_id = self._external_definitions.get(definition_key)
        if definition_id is None:
            definition_id = f"python.external.def.{self._external_counter:07d}"
            self._external_counter += 1
            evidence_id = self._add_evidence(
                description="External runtime boundary declared by the capture scenario.",
                source=None,
                status="declared",
                metadata={"stubbed": stubbed},
            )
            self.definitions.append(
                {
                    "id": definition_id,
                    "key": f"external:{label}",
                    "label": label,
                    "role": role,
                    "boundary": boundary,
                    "source": None,
                    "evidence_ids": [evidence_id],
                    "external": True,
                    "stubbed": stubbed,
                }
            )
            self._external_definitions[definition_key] = definition_id
            self._occurrence_counts[definition_id] = 0

        occurrence_id, evidence_id = self._start_occurrence(
            definition_id=definition_id,
            label=label,
            role=role,
            boundary=boundary,
            source=None,
            attributes={"external": True, "stubbed": stubbed},
        )
        for name, value in (inputs or {}).items():
            value_id = self._value_node(
                value,
                label=f"{label}.{name}",
                role="python_probe_input",
                evidence_id=evidence_id,
            )
            self._add_edge(value_id, occurrence_id, "consumes", [evidence_id])
        output_id = self._value_node(
            output,
            label=f"{label}.return",
            role="python_probe_output",
            evidence_id=evidence_id,
        )
        self._add_edge(occurrence_id, output_id, "produces", [evidence_id])
        self._finish_occurrence(occurrence_id, "returned")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "run": {
                "id": self.run_id,
                "framework": "python",
                "python_version": sys.version.split()[0],
                "backend": "selective_python_raw_probes",
            },
            "repository_root": (
                str(self.repository_root) if self.repository_root is not None else None
            ),
            "definitions": self.definitions,
            "occurrences": self.occurrences,
            "values": self.values,
            "edges": self.edges,
            "evidence": self.evidence,
            "metadata": {
                "probe_count": len(self.probes),
                "captured_occurrences": len(self.occurrences),
                "external_definition_count": len(self._external_definitions),
            },
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def _install_probe(self, probe: ProbeTarget) -> None:
        module = importlib.import_module(probe.module)
        owner, name = _resolve_owner(module, probe.symbol)
        descriptor = inspect.getattr_static(owner, name)
        function, replacement_factory, skip_receiver = _unwrap_descriptor(descriptor)
        if not callable(function):
            raise TypeError(f"probe target is not callable: {probe.key}")
        source = _source_span(function, self.repository_root, probe)
        definition_id = f"python.def.{_slug(probe.key)}"
        if definition_id in self._definition_ids.values():
            raise ValueError(f"duplicate raw probe definition id: {definition_id}")
        evidence_id = self._add_evidence(
            description="Python callable selected as a runtime probe boundary.",
            source=source,
            status="declared",
        )
        self.definitions.append(
            {
                "id": definition_id,
                "key": probe.key,
                "label": probe.symbol.rsplit(".", 1)[-1],
                "role": probe.role,
                "boundary": probe.boundary,
                "source": source,
                "evidence_ids": [evidence_id],
                "external": False,
                "stubbed": False,
            }
        )
        self._definition_ids[probe.key] = definition_id
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
        probe: ProbeTarget,
        definition_id: str,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        skip_receiver: bool,
    ) -> Any:
        definition = next(item for item in self.definitions if item["id"] == definition_id)
        occurrence_id, evidence_id = self._start_occurrence(
            definition_id=definition_id,
            label=definition["label"],
            role=probe.role,
            boundary=probe.boundary,
            source=definition["source"],
            attributes={"external": False, "stubbed": False},
        )
        for name, value in _bound_arguments(function, args, kwargs, skip_receiver=skip_receiver):
            value_id = self._value_node(
                value,
                label="{}.{}".format(definition["label"], name),
                role="python_probe_input",
                evidence_id=evidence_id,
            )
            self._add_edge(value_id, occurrence_id, "consumes", [evidence_id])

        stack = self._stack.get()
        token = self._stack.set((*stack, occurrence_id))
        try:
            output = function(*args, **kwargs)
        except BaseException as exc:
            self._finish_occurrence(
                occurrence_id,
                "raised",
                exception_type=type(exc).__name__,
                exception_message=str(exc)[:512],
            )
            raise
        finally:
            self._stack.reset(token)

        output_id = self._value_node(
            output,
            label="{}.return".format(definition["label"]),
            role="python_probe_output",
            evidence_id=evidence_id,
        )
        self._add_edge(occurrence_id, output_id, "produces", [evidence_id])
        self._finish_occurrence(occurrence_id, "returned")
        return output

    def _start_occurrence(
        self,
        *,
        definition_id: str,
        label: str,
        role: str,
        boundary: str,
        source: dict[str, Any] | None,
        attributes: dict[str, Any],
    ) -> tuple[str, str]:
        occurrence_index = self._occurrence_counts[definition_id]
        self._occurrence_counts[definition_id] += 1
        occurrence_id = f"python.call.{len(self.occurrences):07d}"
        evidence_id = self._add_evidence(
            description="Python probe call observed at runtime.",
            source=source,
            status="observed",
            metadata={"occurrence_index": occurrence_index},
        )
        stack = self._stack.get()
        occurrence = {
            "id": occurrence_id,
            "definition_id": definition_id,
            "label": label,
            "role": role,
            "boundary": boundary,
            "run_id": self.run_id,
            "occurrence_index": occurrence_index,
            "parent_ids": [stack[-1]] if stack else [],
            "source": source,
            "evidence_ids": [evidence_id],
            "status": "running",
            "attributes": dict(attributes),
        }
        self.occurrences.append(occurrence)
        if stack:
            self._add_edge(stack[-1], occurrence_id, "calls", [evidence_id])
        if self._last_occurrence is not None:
            self._add_edge(self._last_occurrence, occurrence_id, "next", [evidence_id])
        self._last_occurrence = occurrence_id
        return occurrence_id, evidence_id

    def _finish_occurrence(
        self,
        occurrence_id: str,
        status: str,
        *,
        exception_type: str | None = None,
        exception_message: str | None = None,
    ) -> None:
        occurrence = next(item for item in self.occurrences if item["id"] == occurrence_id)
        occurrence["status"] = status
        if exception_type is not None:
            occurrence["attributes"]["exception_type"] = exception_type
        if exception_message is not None:
            occurrence["attributes"]["exception_message"] = exception_message

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
        self.values.append(
            {
                "id": value_id,
                "label": label,
                "role": role,
                "run_id": self.run_id,
                "evidence_ids": [evidence_id],
                "tensor": _tensor_summary(value),
                "structure": _summarize_value(value),
            }
        )
        if identity is not None:
            self._object_values[identity] = value_id
        return value_id

    def _add_evidence(
        self,
        *,
        description: str,
        source: dict[str, Any] | None,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        evidence_id = f"evidence.python.{self._evidence_counter:07d}"
        self._evidence_counter += 1
        self.evidence.append(
            {
                "id": evidence_id,
                "status": status,
                "description": description,
                "source": source,
                "run_id": self.run_id,
                "metadata": {} if metadata is None else metadata,
            }
        )
        return evidence_id

    def _add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        evidence_ids: list[str],
    ) -> None:
        self.edges.append(
            {
                "id": f"edge.python.{self._edge_counter:07d}",
                "source": source,
                "target": target,
                "kind": kind,
                "evidence_ids": evidence_ids,
            }
        )
        self._edge_counter += 1


def _deduplicate_probes(probes: Sequence[ProbeTarget]) -> list[ProbeTarget]:
    by_key: dict[str, ProbeTarget] = {}
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


def _unwrap_descriptor(descriptor: Any) -> tuple[Callable[..., Any], Callable[[Any], Any], bool]:
    if isinstance(descriptor, staticmethod):
        return descriptor.__func__, staticmethod, False
    if isinstance(descriptor, classmethod):
        return descriptor.__func__, classmethod, True
    return descriptor, lambda value: value, inspect.isfunction(descriptor)


def _source_span(
    function: Callable[..., Any],
    repository_root: Path | None,
    probe: ProbeTarget,
) -> dict[str, Any] | None:
    try:
        path = inspect.getsourcefile(function) or inspect.getfile(function)
        lines, start_line = inspect.getsourcelines(function)
    except (OSError, TypeError):
        return None
    source_path = Path(path).resolve()
    rendered = str(source_path)
    if repository_root is not None:
        with contextlib.suppress(ValueError):
            rendered = source_path.relative_to(repository_root).as_posix()
    return {
        "path": rendered,
        "start_line": start_line,
        "end_line": start_line + len(lines) - 1,
        "symbol": f"{probe.module}.{probe.symbol}",
    }


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


def _tensor_summary(value: Any) -> dict[str, Any] | None:
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
    return {
        "shape": rendered_shape,
        "dtype": str(dtype).removeprefix("torch."),
        "device": None if device is None else str(device),
        "requires_grad": requires_grad if isinstance(requires_grad, bool) else None,
    }


def _summarize_value(value: Any, *, depth: int = 0) -> dict[str, Any]:
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    tensor = _tensor_summary(value)
    if tensor is not None:
        return {"type": value_type, "tensor": tensor}
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


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return normalized.strip("_.-") or "probe"
