"""Deterministic L0 paper projection from L1 semantic ATIR groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from archtrace.ir import ArchNode, ArchTraceIR, EdgeKind, NodeLevel
from archtrace.semantics.ontology import SemanticRole, role_spec


@dataclass(frozen=True, slots=True)
class PaperViewPolicy:
    min_confidence: float = 0.55
    max_components: int = 12
    include_roles: tuple[SemanticRole, ...] = ()
    exclude_roles: tuple[SemanticRole, ...] = ()


@dataclass(frozen=True, slots=True)
class PaperNode:
    id: str
    label: str
    role: str
    member_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    confidence: float


@dataclass(slots=True)
class PaperEdge:
    source: str
    target: str
    evidence_edge_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperView:
    nodes: list[PaperNode]
    edges: list[PaperEdge]


def project_paper_view(
    graph: ArchTraceIR,
    *,
    policy: PaperViewPolicy | None = None,
) -> PaperView:
    """Project semantic components without inventing architecture edges."""

    selected_policy = policy or PaperViewPolicy()
    if not 0.0 <= selected_policy.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if selected_policy.max_components < 0:
        raise ValueError("max_components must be non-negative")

    include = {role.value for role in selected_policy.include_roles}
    exclude = {role.value for role in selected_policy.exclude_roles}
    semantic_nodes: list[ArchNode] = []
    for node in graph.nodes:
        if node.level != NodeLevel.SEMANTIC or node.role is None:
            continue
        confidence = _confidence(node.attributes.get("confidence"))
        if confidence < selected_policy.min_confidence:
            continue
        if include and node.role not in include:
            continue
        if node.role in exclude:
            continue
        semantic_nodes.append(node)

    semantic_nodes.sort(
        key=lambda node: (
            _role_priority(node.role or "unknown"),
            _source_sort_key(node),
            node.id,
        )
    )
    semantic_nodes = semantic_nodes[: selected_policy.max_components]

    paper_nodes = [
        PaperNode(
            id=node.id,
            label=node.label,
            role=node.role or "unknown",
            member_ids=tuple(sorted(_member_ids(node))),
            modalities=tuple(sorted(_modalities(node))),
            confidence=_confidence(node.attributes.get("confidence")),
        )
        for node in semantic_nodes
    ]
    selected_ids = {node.id for node in paper_nodes}
    member_to_semantic: dict[str, str] = {}
    for node in paper_nodes:
        for member_id in node.member_ids:
            member_to_semantic.setdefault(member_id, node.id)

    pair_to_edges: dict[tuple[str, str], list[str]] = {}
    for edge in graph.edges:
        if edge.kind == EdgeKind.CONTAINS:
            continue
        source_group = member_to_semantic.get(edge.source)
        target_group = member_to_semantic.get(edge.target)
        if source_group is None or target_group is None or source_group == target_group:
            continue
        if source_group not in selected_ids or target_group not in selected_ids:
            continue
        pair_to_edges.setdefault((source_group, target_group), []).append(edge.id)

    paper_edges = [
        PaperEdge(
            source=source,
            target=target,
            evidence_edge_ids=sorted(set(edge_ids)),
        )
        for (source, target), edge_ids in sorted(pair_to_edges.items())
    ]
    return PaperView(nodes=paper_nodes, edges=paper_edges)


def _member_ids(node: ArchNode) -> list[str]:
    raw = node.attributes.get("member_ids", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _modalities(node: ArchNode) -> list[str]:
    raw = node.attributes.get("modalities", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _role_priority(role_value: str) -> int:
    try:
        role = SemanticRole(role_value)
    except ValueError:
        return 999
    spec = role_spec(role)
    return 999 if spec is None else spec.priority


def _source_sort_key(node: ArchNode) -> tuple[str, int]:
    if not node.source:
        return ("~", 1_000_000_000)
    span = node.source[0]
    return (span.path, span.start_line)
