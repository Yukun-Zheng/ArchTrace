"""Deterministic L0 paper projection from L1 semantic ATIR groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from archtrace.ir import ArchNode, ArchTraceIR, EdgeKind, NodeLevel
from archtrace.semantics.ontology import SemanticPhase, SemanticRole, role_spec


@dataclass(frozen=True, slots=True)
class PaperViewPolicy:
    min_confidence: float = 0.55
    max_components: int = 12
    include_roles: tuple[SemanticRole, ...] = ()
    exclude_roles: tuple[SemanticRole, ...] = ()
    phases: tuple[SemanticPhase, ...] = ()
    include_author_declarations: bool = False


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


def select_semantic_components(
    graph: ArchTraceIR,
    *,
    phase: SemanticPhase | None = None,
    min_confidence: float = 0.0,
    include_author_declarations: bool = False,
) -> list[ArchNode]:
    """Return deterministic L1 components for a training or inference view."""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    selected: list[ArchNode] = []
    for node in graph.nodes:
        if node.level != NodeLevel.SEMANTIC or node.role is None:
            continue
        if (
            not include_author_declarations
            and node.attributes.get("declaration_only") is True
        ):
            continue
        if _confidence(node.attributes.get("confidence")) < min_confidence:
            continue
        if phase is not None and not _phase_matches(node, (phase,)):
            continue
        selected.append(node)

    selected.sort(
        key=lambda node: (
            _role_priority(node.role or "unknown"),
            _source_sort_key(node),
            node.id,
        )
    )
    return selected


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
    semantic_nodes = select_semantic_components(
        graph,
        min_confidence=selected_policy.min_confidence,
        include_author_declarations=selected_policy.include_author_declarations,
    )
    semantic_nodes = [
        node
        for node in semantic_nodes
        if (not include or node.role in include)
        and node.role not in exclude
        and _phase_matches(node, selected_policy.phases)
    ][: selected_policy.max_components]

    paper_nodes = [
        PaperNode(
            id=semantic_node.id,
            label=semantic_node.label,
            role=semantic_node.role or "unknown",
            member_ids=tuple(sorted(_member_ids(semantic_node))),
            modalities=tuple(sorted(_modalities(semantic_node))),
            confidence=_confidence(semantic_node.attributes.get("confidence")),
        )
        for semantic_node in semantic_nodes
    ]
    selected_ids = {paper_node.id for paper_node in paper_nodes}
    member_to_semantic: dict[str, str] = {}
    for paper_node in paper_nodes:
        for member_id in paper_node.member_ids:
            member_to_semantic.setdefault(member_id, paper_node.id)

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


def _phase_matches(node: ArchNode, requested: tuple[SemanticPhase, ...]) -> bool:
    if not requested:
        return True
    raw_phase = node.attributes.get("phase", SemanticPhase.BOTH.value)
    try:
        component_phase = SemanticPhase(str(raw_phase))
    except ValueError:
        return False
    if component_phase == SemanticPhase.BOTH:
        return True
    return component_phase in requested


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
    return role_spec(role).priority


def _source_sort_key(node: ArchNode) -> tuple[str, int]:
    if not node.source:
        return ("~", 1_000_000_000)
    span = node.source[0]
    return (span.path, span.start_line)
