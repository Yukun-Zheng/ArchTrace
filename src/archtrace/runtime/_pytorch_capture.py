"""Module/tensor runtime state shared by PyTorch capture backends."""

from __future__ import annotations

import platform
from dataclasses import dataclass
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
    TraceRun,
)
from archtrace.runtime._pytorch_utils import (
    iter_tensors,
    named_buffers,
    named_modules,
    named_parameters,
    safe_fragment,
    source_span,
    tensor_spec,
)


@dataclass(slots=True)
class _Definition:
    module: Any
    node_id: str
    aliases: list[str]
    source: SourceSpan | None


class PyTorchModuleCapture:
    """Shared mutable capture state for one concrete PyTorch execution."""

    def __init__(
        self,
        model: Any,
        *,
        torch: Any,
        run_id: str,
        project_name: str,
    ) -> None:
        self.model = model
        self.torch = torch
        self.project = ProjectInfo(name=project_name)
        self.run = TraceRun(
            id=run_id,
            framework="pytorch",
            framework_version=str(torch.__version__),
            python_version=platform.python_version(),
        )
        self.nodes: list[ArchNode] = []
        self.edges: list[ArchEdge] = []
        self.evidence: list[Evidence] = []
        self._nodes: dict[str, ArchNode] = {}
        self._definitions: dict[int, _Definition] = {}
        self._alias_to_definition: dict[str, str] = {}
        self._tensor_nodes: dict[int, str] = {}
        self._state_nodes: dict[int, str] = {}
        self._stack: list[str] = []
        self._occurrence_counts: dict[str, int] = {}
        self._handles: list[Any] = []
        self._value_counter = 0
        self._call_counter = 0
        self._edge_counter = 0
        self._evidence_counter = 0
        self._build_definitions()
        self._build_state_nodes()

    @property
    def current_call_id(self) -> str | None:
        return self._stack[-1] if self._stack else None

    @property
    def current_definition_id(self) -> str | None:
        call_id = self.current_call_id
        if call_id is None:
            return None
        return self._nodes[call_id].definition_id

    def install(self) -> None:
        for definition in self._definitions.values():
            module = definition.module
            self._handles.append(
                module.register_forward_pre_hook(
                    self._pre_hook(definition),
                    with_kwargs=True,
                )
            )
            hook = self._post_hook(definition)
            try:
                handle = module.register_forward_hook(
                    hook,
                    with_kwargs=True,
                    always_call=True,
                )
            except TypeError:
                handle = module.register_forward_hook(hook, with_kwargs=True)
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def finish(self) -> ArchTraceIR:
        if self._stack:
            raise RuntimeError("ArchTrace module stack was not empty after execution")
        return ArchTraceIR(
            project=self.project,
            runs=[self.run],
            nodes=self.nodes,
            edges=self.edges,
            evidence=self.evidence,
            metadata={
                "runtime_backend": "pytorch_module_hooks",
                "runtime_backend_version": "m1-v2",
            },
        )

    def add_node(self, node: ArchNode) -> None:
        if node.id in self._nodes:
            raise RuntimeError(f"duplicate runtime node id: {node.id}")
        self.nodes.append(node)
        self._nodes[node.id] = node

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def add_relation(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        evidence_ids: list[str],
        *,
        namespace: str = "runtime",
    ) -> None:
        edge_id = f"edge.{namespace}.{self._edge_counter:07d}"
        self._edge_counter += 1
        self.edges.append(
            ArchEdge(
                id=edge_id,
                source=source,
                target=target,
                kind=kind,
                evidence_ids=evidence_ids,
            )
        )

    def next_evidence_id(self, namespace: str) -> str:
        evidence_id = f"evidence.{namespace}.{self._evidence_counter:07d}"
        self._evidence_counter += 1
        return evidence_id

    def ensure_tensor_entity(
        self,
        tensor: Any,
        *,
        evidence_id: str,
        kind: NodeKind = NodeKind.TENSOR,
        role: str = "tensor",
    ) -> str:
        key = id(tensor)
        state_id = self._state_nodes.get(key)
        if state_id is not None:
            node = self._nodes[state_id]
            if evidence_id not in node.evidence_ids:
                node.evidence_ids.append(evidence_id)
            return state_id

        existing = self._tensor_nodes.get(key)
        if existing is not None:
            node = self._nodes[existing]
            if kind in {NodeKind.INPUT, NodeKind.OUTPUT}:
                node.kind = kind
                node.role = role
            if evidence_id not in node.evidence_ids:
                node.evidence_ids.append(evidence_id)
            return existing

        value_id = f"value.{safe_fragment(self.run.id)}.{self._value_counter:06d}"
        self._value_counter += 1
        node = ArchNode(
            id=value_id,
            level=NodeLevel.OPERATION,
            kind=kind,
            identity_kind=IdentityKind.VALUE,
            label=value_id,
            role=role,
            run_id=self.run.id,
            evidence_ids=[evidence_id],
            tensor=tensor_spec(tensor),
        )
        self._tensor_nodes[key] = value_id
        self.add_node(node)
        return value_id

    def version_tensor_value(self, tensor: Any, *, evidence_id: str) -> tuple[str, str | None]:
        """Create a new runtime value version for an in-place tensor mutation."""

        key = id(tensor)
        if key in self._state_nodes:
            state_id = self.ensure_tensor_entity(tensor, evidence_id=evidence_id)
            return state_id, None
        previous = self._tensor_nodes.get(key)
        if previous is None:
            return self.ensure_tensor_entity(tensor, evidence_id=evidence_id), None

        value_id = f"value.{safe_fragment(self.run.id)}.{self._value_counter:06d}"
        self._value_counter += 1
        node = ArchNode(
            id=value_id,
            level=NodeLevel.OPERATION,
            kind=NodeKind.TENSOR,
            identity_kind=IdentityKind.VALUE,
            label=value_id,
            role="tensor_version",
            run_id=self.run.id,
            evidence_ids=[evidence_id],
            tensor=tensor_spec(tensor),
        )
        self._tensor_nodes[key] = value_id
        self.add_node(node)
        return value_id, previous

    def _build_definitions(self) -> None:
        grouped: dict[int, tuple[Any, list[str]]] = {}
        for path, module in named_modules(self.model):
            alias = "model" if not path else f"model.{path}"
            key = id(module)
            if key not in grouped:
                grouped[key] = (module, [])
            grouped[key][1].append(alias)

        ordered = sorted(grouped.values(), key=lambda item: min(item[1]))
        for index, (module, aliases) in enumerate(ordered):
            aliases = sorted(set(aliases))
            canonical = aliases[0]
            node_id = f"module.def.{index:04d}.{safe_fragment(canonical)}"
            span = source_span(module.forward)
            evidence_ids: list[str] = []
            if span is not None:
                evidence_id = self.next_evidence_id("source.module")
                self.add_evidence(
                    Evidence(
                        id=evidence_id,
                        kind=EvidenceKind.SOURCE,
                        status=FactStatus.OBSERVED,
                        source=span,
                        description="Python forward implementation located.",
                    )
                )
                evidence_ids.append(evidence_id)

            self.add_node(
                ArchNode(
                    id=node_id,
                    level=NodeLevel.MODULE,
                    kind=NodeKind.MODULE,
                    identity_kind=IdentityKind.DEFINITION,
                    label=type(module).__name__,
                    role="pytorch_module_definition",
                    source=[] if span is None else [span],
                    evidence_ids=evidence_ids,
                    attributes={
                        "canonical_path": canonical,
                        "aliases": aliases,
                        "python_type": f"{type(module).__module__}.{type(module).__qualname__}",
                    },
                )
            )
            self._definitions[id(module)] = _Definition(module, node_id, aliases, span)
            for alias in aliases:
                self._alias_to_definition[alias] = node_id

        for definition in self._definitions.values():
            parents: set[str] = set()
            for alias in definition.aliases:
                if alias == "model" or "." not in alias:
                    continue
                parent_id = self._alias_to_definition.get(alias.rsplit(".", 1)[0])
                if parent_id is not None and parent_id != definition.node_id:
                    parents.add(parent_id)
            self._nodes[definition.node_id].parent_ids = sorted(parents)

    def _build_state_nodes(self) -> None:
        self._add_named_state(list(named_parameters(self.model)), role="pytorch_parameter")
        self._add_named_state(list(named_buffers(self.model)), role="pytorch_buffer")

    def _add_named_state(self, items: list[tuple[str, Any]], *, role: str) -> None:
        grouped: dict[int, tuple[Any, list[str]]] = {}
        for path, tensor in items:
            if id(tensor) in self._state_nodes:
                continue
            alias = f"model.{path}"
            key = id(tensor)
            if key not in grouped:
                grouped[key] = (tensor, [])
            grouped[key][1].append(alias)

        offset = len(self._state_nodes)
        for index, (tensor, aliases) in enumerate(
            sorted(grouped.values(), key=lambda item: min(item[1]))
        ):
            aliases = sorted(set(aliases))
            evidence_id = self.next_evidence_id("runtime.state")
            self.add_evidence(
                Evidence(
                    id=evidence_id,
                    kind=EvidenceKind.RUNTIME,
                    status=FactStatus.OBSERVED,
                    run_id=self.run.id,
                    description=f"PyTorch {role.removeprefix('pytorch_')} discovered at runtime.",
                )
            )
            parents: set[str] = set()
            for alias in aliases:
                owner = alias.rsplit(".", 1)[0] if "." in alias else "model"
                parent_id = self._alias_to_definition.get(owner)
                if parent_id is not None:
                    parents.add(parent_id)
            node_id = f"state.{offset + index:05d}.{safe_fragment(aliases[0])}"
            self.add_node(
                ArchNode(
                    id=node_id,
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.PARAMETER,
                    identity_kind=IdentityKind.STATE,
                    label=aliases[0],
                    role=role,
                    parent_ids=sorted(parents),
                    evidence_ids=[evidence_id],
                    tensor=tensor_spec(tensor),
                    attributes={"aliases": aliases},
                )
            )
            self._state_nodes[id(tensor)] = node_id

    def _pre_hook(self, definition: _Definition) -> Any:
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module
            local_index = self._occurrence_counts.get(definition.node_id, 0)
            self._occurrence_counts[definition.node_id] = local_index + 1
            call_id = f"call.{safe_fragment(self.run.id)}.{self._call_counter:06d}"
            self._call_counter += 1
            evidence_id = self._runtime_call_evidence(definition)
            self.add_node(
                ArchNode(
                    id=call_id,
                    level=NodeLevel.MODULE,
                    kind=NodeKind.MODULE,
                    identity_kind=IdentityKind.OCCURRENCE,
                    label=definition.aliases[0],
                    role="pytorch_module_call",
                    definition_id=definition.node_id,
                    run_id=self.run.id,
                    occurrence_index=local_index,
                    source=[] if definition.source is None else [definition.source],
                    evidence_ids=[evidence_id],
                    attributes={"aliases": definition.aliases},
                )
            )
            if self._stack:
                self.add_relation(
                    self._stack[-1], call_id, EdgeKind.CONTAINS, [evidence_id]
                )

            is_root = "model" in definition.aliases
            for tensor in iter_tensors((args, kwargs), self.torch):
                value_id = self.ensure_tensor_entity(
                    tensor,
                    evidence_id=evidence_id,
                    kind=NodeKind.INPUT if is_root else NodeKind.TENSOR,
                    role="model_input" if is_root else "tensor",
                )
                self.add_relation(value_id, call_id, EdgeKind.CONSUMES, [evidence_id])
            self._stack.append(call_id)

        return hook

    def _post_hook(self, definition: _Definition) -> Any:
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            if not self._stack:
                raise RuntimeError("ArchTrace module stack underflow")
            call_id = self._stack[-1]
            call = self._nodes[call_id]
            if call.definition_id != definition.node_id:
                raise RuntimeError("ArchTrace module stack mismatch")
            evidence_id = call.evidence_ids[0]
            is_root = "model" in definition.aliases
            for tensor in iter_tensors(output, self.torch):
                value_id = self.ensure_tensor_entity(
                    tensor,
                    evidence_id=evidence_id,
                    kind=NodeKind.OUTPUT if is_root else NodeKind.TENSOR,
                    role="model_output" if is_root else "tensor",
                )
                self.add_relation(call_id, value_id, EdgeKind.PRODUCES, [evidence_id])
            self._stack.pop()

        return hook

    def _runtime_call_evidence(self, definition: _Definition) -> str:
        evidence_id = self.next_evidence_id("runtime.call")
        self.add_evidence(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.RUNTIME,
                status=FactStatus.OBSERVED,
                confidence=1.0,
                description="Concrete PyTorch module forward call observed.",
                source=definition.source,
                run_id=self.run.id,
                metadata={"module_definition_id": definition.node_id},
            )
        )
        return evidence_id
