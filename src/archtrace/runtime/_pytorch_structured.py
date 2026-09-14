"""FX and torch.export structured evidence capture and runtime alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
)
from archtrace.runtime._pytorch_capture import PyTorchModuleCapture
from archtrace.runtime._pytorch_utils import iter_tensors, safe_fragment, tensor_spec


@dataclass(slots=True)
class StructuredCaptureReport:
    backend: str
    status: str
    nodes: int = 0
    data_edges: int = 0
    aligned_nodes: int = 0
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_structured_graphs(
    capture: PyTorchModuleCapture,
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    torch: Any,
    capture_fx: bool,
    capture_export: bool,
) -> list[StructuredCaptureReport]:
    """Capture optional structural backends without making them fatal."""

    reports: list[StructuredCaptureReport] = []
    if capture_fx:
        reports.append(_capture_backend(capture, "fx", lambda: torch.fx.symbolic_trace(model)))
    if capture_export:
        export_namespace = getattr(torch, "export", None)
        export_fn = None if export_namespace is None else getattr(export_namespace, "export", None)
        if export_fn is None:
            reports.append(
                StructuredCaptureReport(
                    backend="export",
                    status="unavailable",
                    error="torch.export.export is unavailable in this PyTorch build",
                )
            )
        else:
            reports.append(
                _capture_backend(
                    capture,
                    "export",
                    lambda: export_fn(model, args, kwargs),
                )
            )
    return reports


def _capture_backend(
    capture: PyTorchModuleCapture,
    backend: str,
    factory: Any,
) -> StructuredCaptureReport:
    try:
        captured = factory()
        graph = captured.graph if backend == "fx" else captured.graph_module.graph
        return _normalize_graph(capture, graph, backend)
    except Exception as exc:  # structural tracing is intentionally best-effort
        return StructuredCaptureReport(
            backend=backend,
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )


def _normalize_graph(
    capture: PyTorchModuleCapture,
    graph: Any,
    backend: str,
) -> StructuredCaptureReport:
    node_ids: dict[Any, str] = {}
    evidence_ids: dict[Any, str] = {}
    aligned_nodes = 0
    data_edges = 0

    graph_nodes = list(graph.nodes)
    for index, node in enumerate(graph_nodes):
        node_id = f"structured.{backend}.{index:06d}.{safe_fragment(str(node.name))}"
        evidence_id = capture.next_evidence_id(f"imported.{backend}")
        evidence_ids[node] = evidence_id
        capture.add_evidence(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.IMPORTED,
                status=FactStatus.OBSERVED,
                confidence=1.0,
                description=f"Node imported from PyTorch {backend} graph capture.",
                metadata={
                    "backend": backend,
                    "graph_op": str(node.op),
                    "target": _target_text(node.target),
                    "name": str(node.name),
                },
            )
        )

        target_text = _target_text(node.target)
        kind = _structured_kind(str(node.op))
        tensor = _structured_tensor_spec(node, capture.torch)
        capture.add_node(
            ArchNode(
                id=node_id,
                level=NodeLevel.MODULE if kind == NodeKind.MODULE else NodeLevel.OPERATION,
                kind=kind,
                identity_kind=IdentityKind.DEFINITION,
                label=target_text or str(node.name),
                role=f"pytorch_{backend}_node",
                evidence_ids=[evidence_id],
                tensor=tensor,
                attributes={
                    "backend": backend,
                    "graph_op": str(node.op),
                    "target": target_text,
                    "name": str(node.name),
                },
            )
        )
        node_ids[node] = node_id

        aligned_id = _find_runtime_alignment(capture, str(node.op), target_text)
        if aligned_id is not None:
            capture.add_relation(
                node_id,
                aligned_id,
                EdgeKind.DERIVED_FROM,
                [evidence_id],
                namespace=f"align.{backend}",
            )
            aligned_nodes += 1

    for node in graph_nodes:
        target_id = node_ids[node]
        evidence_id = evidence_ids[node]
        for source_node in getattr(node, "all_input_nodes", ()):
            source_id = node_ids.get(source_node)
            if source_id is None:
                continue
            capture.add_relation(
                source_id,
                target_id,
                EdgeKind.DATA,
                [evidence_id],
                namespace=f"structured.{backend}",
            )
            data_edges += 1

    return StructuredCaptureReport(
        backend=backend,
        status="success",
        nodes=len(graph_nodes),
        data_edges=data_edges,
        aligned_nodes=aligned_nodes,
    )


def _find_runtime_alignment(
    capture: PyTorchModuleCapture,
    graph_op: str,
    target_text: str,
) -> str | None:
    if graph_op == "call_module":
        alias = target_text if target_text.startswith("model.") else f"model.{target_text}"
        for node in capture.nodes:
            if node.role != "pytorch_module_definition":
                continue
            aliases = node.attributes.get("aliases", [])
            if alias in aliases:
                return node.id
        return None

    if graph_op not in {"call_function", "call_method"}:
        return None

    candidates = [
        node
        for node in capture.nodes
        if node.role == "pytorch_operator_definition"
    ]
    for node in candidates:
        if node.label == target_text:
            return node.id

    simple_name = target_text.rsplit(".", 1)[-1]
    if simple_name in {"default", "Tensor"} and "." in target_text:
        simple_name = target_text.rsplit(".", 2)[-2]
    for node in candidates:
        pieces = node.label.split(".")
        if simple_name in pieces:
            return node.id
    return None


def _structured_kind(graph_op: str) -> NodeKind:
    if graph_op == "placeholder":
        return NodeKind.INPUT
    if graph_op == "output":
        return NodeKind.OUTPUT
    if graph_op == "get_attr":
        return NodeKind.PARAMETER
    if graph_op == "call_module":
        return NodeKind.MODULE
    return NodeKind.OPERATION


def _target_text(target: Any) -> str:
    if isinstance(target, str):
        return target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return str(target)
    module = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return str(target)


def _structured_tensor_spec(node: Any, torch: Any) -> Any:
    value = getattr(node, "meta", {}).get("val")
    tensors = list(iter_tensors(value, torch))
    if len(tensors) == 1:
        return tensor_spec(tensors[0])
    return None
