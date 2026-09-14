"""PyTorch operator-dispatch capture into the shared ATIR runtime state."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from archtrace.ir import (
    ArchNode,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    SourceSpan,
)
from archtrace.runtime._pytorch_capture import PyTorchModuleCapture
from archtrace.runtime._pytorch_utils import caller_source_span, iter_tensors, safe_fragment


@dataclass(slots=True)
class _OperatorContext:
    occurrence_id: str
    evidence_id: str
    input_object_ids: set[int]
    mutable: bool


class PyTorchDispatchRecorder:
    """Normalize concrete ``__torch_dispatch__`` calls into ATIR."""

    def __init__(self, capture: PyTorchModuleCapture, torch: Any) -> None:
        self.capture = capture
        self.torch = torch
        self._definitions: dict[tuple[str, str | None, int | None, str | None], str] = {}
        self._definition_counts: dict[str, int] = {}
        self._definition_counter = 0
        self._occurrence_counter = 0

    def begin(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _OperatorContext:
        operator_name = str(func)
        span = caller_source_span()
        module_definition_id = self.capture.current_definition_id
        definition_id = self._ensure_definition(
            operator_name,
            span,
            module_definition_id,
        )

        local_index = self._definition_counts.get(definition_id, 0)
        self._definition_counts[definition_id] = local_index + 1
        occurrence_id = (
            f"op.{safe_fragment(self.capture.run.id)}.{self._occurrence_counter:07d}"
        )
        self._occurrence_counter += 1

        evidence_id = self.capture.next_evidence_id("runtime.operator")
        self.capture.add_evidence(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.RUNTIME,
                status=FactStatus.OBSERVED,
                confidence=1.0,
                description="Concrete PyTorch operator dispatch observed.",
                source=span,
                run_id=self.capture.run.id,
                metadata={
                    "operator": operator_name,
                    "module_definition_id": module_definition_id,
                },
            )
        )
        self.capture.add_node(
            ArchNode(
                id=occurrence_id,
                level=NodeLevel.OPERATION,
                kind=NodeKind.OPERATION,
                identity_kind=IdentityKind.OCCURRENCE,
                label=operator_name,
                role="pytorch_operator_call",
                definition_id=definition_id,
                run_id=self.capture.run.id,
                occurrence_index=local_index,
                source=[] if span is None else [span],
                evidence_ids=[evidence_id],
            )
        )

        current_call = self.capture.current_call_id
        if current_call is not None:
            self.capture.add_relation(
                current_call,
                occurrence_id,
                EdgeKind.CONTAINS,
                [evidence_id],
                namespace="operator",
            )

        input_object_ids: set[int] = set()
        seen_entities: set[str] = set()
        for tensor in iter_tensors((args, kwargs), self.torch):
            input_object_ids.add(id(tensor))
            entity_id = self.capture.ensure_tensor_entity(
                tensor,
                evidence_id=evidence_id,
            )
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            self.capture.add_relation(
                entity_id,
                occurrence_id,
                EdgeKind.CONSUMES,
                [evidence_id],
                namespace="operator",
            )

        schema = getattr(func, "_schema", None)
        mutable = bool(getattr(schema, "is_mutable", False))
        return _OperatorContext(
            occurrence_id=occurrence_id,
            evidence_id=evidence_id,
            input_object_ids=input_object_ids,
            mutable=mutable,
        )

    def end(self, context: _OperatorContext, output: Any) -> None:
        seen_entities: set[str] = set()
        for tensor in iter_tensors(output, self.torch):
            if context.mutable and id(tensor) in context.input_object_ids:
                entity_id, previous_id = self.capture.version_tensor_value(
                    tensor,
                    evidence_id=context.evidence_id,
                )
                if previous_id is not None:
                    self.capture.add_relation(
                        previous_id,
                        entity_id,
                        EdgeKind.DERIVED_FROM,
                        [context.evidence_id],
                        namespace="mutation",
                    )
            else:
                entity_id = self.capture.ensure_tensor_entity(
                    tensor,
                    evidence_id=context.evidence_id,
                )

            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            self.capture.add_relation(
                context.occurrence_id,
                entity_id,
                EdgeKind.PRODUCES,
                [context.evidence_id],
                namespace="operator",
            )

    def _ensure_definition(
        self,
        operator_name: str,
        span: SourceSpan | None,
        module_definition_id: str | None,
    ) -> str:
        key = (
            operator_name,
            None if span is None else span.path,
            None if span is None else span.start_line,
            module_definition_id,
        )
        existing = self._definitions.get(key)
        if existing is not None:
            return existing

        definition_id = (
            f"op.def.{self._definition_counter:06d}.{safe_fragment(operator_name)}"
        )
        self._definition_counter += 1
        evidence_ids: list[str] = []
        if span is not None:
            source_evidence_id = self.capture.next_evidence_id("source.operator")
            self.capture.add_evidence(
                Evidence(
                    id=source_evidence_id,
                    kind=EvidenceKind.SOURCE,
                    status=FactStatus.OBSERVED,
                    description="Python call site associated with dispatched operator.",
                    source=span,
                )
            )
            evidence_ids.append(source_evidence_id)

        self.capture.add_node(
            ArchNode(
                id=definition_id,
                level=NodeLevel.OPERATION,
                kind=NodeKind.OPERATION,
                identity_kind=IdentityKind.DEFINITION,
                label=operator_name,
                role="pytorch_operator_definition",
                parent_ids=(
                    [] if module_definition_id is None else [module_definition_id]
                ),
                source=[] if span is None else [span],
                evidence_ids=evidence_ids,
                attributes={"operator": operator_name},
            )
        )
        self._definitions[key] = definition_id
        return definition_id


def make_dispatch_mode(capture: PyTorchModuleCapture, torch: Any) -> Any:
    """Create a TorchDispatchMode without importing torch at module import time."""

    dispatch_module = importlib.import_module("torch.utils._python_dispatch")
    base = dispatch_module.TorchDispatchMode
    recorder = PyTorchDispatchRecorder(capture, torch)

    class ArchTraceDispatchMode(base):  # type: ignore[misc, valid-type]
        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            actual_kwargs = {} if kwargs is None else kwargs
            context = recorder.begin(func, args, actual_kwargs)
            result = func(*args, **actual_kwargs)
            recorder.end(context, result)
            return result

    return ArchTraceDispatchMode()
