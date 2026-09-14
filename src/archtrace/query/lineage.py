"""Data-lineage queries over ATIR."""

from __future__ import annotations

from collections import deque

from archtrace.ir import ArchTraceIR, EdgeKind, IdentityKind, NodeKind

_RUNTIME_FLOW_EDGES = {
    EdgeKind.CONSUMES,
    EdgeKind.PRODUCES,
    EdgeKind.DERIVED_FROM,
    EdgeKind.ALIAS,
}


def runtime_input_ids(graph: ArchTraceIR) -> list[str]:
    """Return concrete runtime input value IDs."""

    return [
        node.id
        for node in graph.nodes
        if node.identity_kind == IdentityKind.VALUE and node.kind == NodeKind.INPUT
    ]


def runtime_output_ids(graph: ArchTraceIR) -> list[str]:
    """Return concrete runtime output value IDs."""

    return [
        node.id
        for node in graph.nodes
        if node.identity_kind == IdentityKind.VALUE and node.kind == NodeKind.OUTPUT
    ]


def find_runtime_operator_path(
    graph: ArchTraceIR,
    source_id: str,
    target_id: str,
) -> list[str] | None:
    """Find a fine-grained runtime path using only values/state and operator calls.

    Module-call nodes are deliberately excluded. A successful path therefore
    proves that ArchTrace's operator/tensor layer itself connects the requested
    source and target rather than relying on a coarse module input/output edge.
    """

    nodes = {node.id: node for node in graph.nodes}
    if source_id not in nodes or target_id not in nodes:
        return None

    allowed = {
        node.id
        for node in graph.nodes
        if node.identity_kind in {IdentityKind.VALUE, IdentityKind.STATE}
        or (node.identity_kind == IdentityKind.OCCURRENCE and node.role == "pytorch_operator_call")
    }
    if source_id not in allowed or target_id not in allowed:
        return None

    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind not in _RUNTIME_FLOW_EDGES:
            continue
        if edge.source not in allowed or edge.target not in allowed:
            continue
        adjacency.setdefault(edge.source, []).append(edge.target)

    queue: deque[str] = deque([source_id])
    previous: dict[str, str | None] = {source_id: None}

    while queue:
        current = queue.popleft()
        if current == target_id:
            return _reconstruct_path(previous, target_id)
        for neighbor in adjacency.get(current, []):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)
    return None


def _reconstruct_path(previous: dict[str, str | None], target_id: str) -> list[str]:
    path = [target_id]
    current = target_id
    while previous[current] is not None:
        parent = previous[current]
        assert parent is not None
        path.append(parent)
        current = parent
    path.reverse()
    return path
