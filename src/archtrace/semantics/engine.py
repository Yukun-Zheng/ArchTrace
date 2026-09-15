"""Deterministic, evidence-grounded semantic recovery for ATIR."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
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
)
from archtrace.semantics.ontology import (
    MODALITY_KEYWORDS,
    ROLE_SPECS,
    Modality,
    SemanticPhase,
    SemanticRole,
    role_spec,
)

_FLOW_EDGE_KINDS = {
    EdgeKind.DATA,
    EdgeKind.PRODUCES,
    EdgeKind.CONSUMES,
    EdgeKind.DERIVED_FROM,
    EdgeKind.NEXT,
}
_PRIMARY_MODALITIES = {
    Modality.VISION,
    Modality.LANGUAGE,
    Modality.PROPRIOCEPTION,
    Modality.DEPTH,
    Modality.POINT_CLOUD,
    Modality.AUDIO,
    Modality.STATE,
}


@dataclass(frozen=True, slots=True)
class SemanticHypothesis:
    role: SemanticRole
    confidence: float
    reasons: tuple[str, ...]
    modalities: tuple[Modality, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticOverride:
    member_ids: tuple[str, ...]
    role: SemanticRole
    label: str | None = None
    modalities: tuple[Modality, ...] = ()
    reason: str = "user correction"


@dataclass(slots=True)
class _SemanticGraphIndex:
    node_by_id: dict[str, ArchNode]
    flow_in: dict[str, list[str]]
    flow_out: dict[str, list[str]]
    containment_children: dict[str, list[str]]


@dataclass(slots=True)
class _IdAllocator:
    used_ids: set[str]
    next_indices: dict[str, int] = field(default_factory=dict)

    def next(self, prefix: str) -> str:
        index = self.next_indices.get(prefix, 0)
        while f"{prefix}.{index:07d}" in self.used_ids:
            index += 1
        value = f"{prefix}.{index:07d}"
        self.used_ids.add(value)
        self.next_indices[prefix] = index + 1
        return value


def recover_semantics(
    graph: ArchTraceIR,
    *,
    overrides: list[SemanticOverride] | None = None,
    min_confidence: float = 0.55,
) -> ArchTraceIR:
    """Return a new ATIR with reversible L1 semantic groups added.

    This pass never changes or fabricates mechanical DATA/control/tensor facts.
    It only adds semantic Evidence, semantic group nodes, and CONTAINS edges.
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    result = graph.model_copy(deep=True)
    mechanical_nodes = [
        node for node in result.nodes if node.level not in {NodeLevel.SEMANTIC, NodeLevel.PAPER}
    ]
    graph_index = _build_semantic_graph_index(result, mechanical_nodes)
    node_by_id = graph_index.node_by_id
    modalities = _propagate_modalities(mechanical_nodes, graph_index.flow_out)
    hypotheses = {
        node.id: _infer_node_hypotheses(node, modalities, graph_index.flow_in)
        for node in mechanical_nodes
    }

    id_allocator = _IdAllocator(_all_record_ids(result))
    overridden_ids = {
        member_id for override in overrides or [] for member_id in override.member_ids
    }
    grouped = _group_candidates(
        result,
        mechanical_nodes,
        hypotheses,
        min_confidence,
        overridden_ids,
        node_by_id=graph_index.node_by_id,
        containment_children=graph_index.containment_children,
    )

    for anchor_id, hypothesis, member_ids in grouped:
        _append_semantic_group(
            result,
            id_allocator,
            graph_index.node_by_id,
            anchor_id=anchor_id,
            hypothesis=hypothesis,
            member_ids=member_ids,
            user_override=None,
            machine_hypotheses=hypotheses,
        )

    for override in overrides or []:
        missing = sorted(set(override.member_ids) - node_by_id.keys())
        if missing:
            raise ValueError("semantic override references unknown node(s): " + ", ".join(missing))
        if not override.member_ids:
            raise ValueError("semantic override requires at least one member_id")
        spec = role_spec(override.role)
        hypothesis = SemanticHypothesis(
            role=override.role,
            confidence=1.0,
            reasons=(override.reason,),
            modalities=(
                override.modalities
                if override.modalities
                else (() if spec is None else spec.default_modalities)
            ),
        )
        _append_semantic_group(
            result,
            id_allocator,
            graph_index.node_by_id,
            anchor_id=override.member_ids[0],
            hypothesis=hypothesis,
            member_ids=list(override.member_ids),
            user_override=override,
            machine_hypotheses=hypotheses,
        )

    classified_members = {
        member_id
        for node in result.nodes
        if node.level == NodeLevel.SEMANTIC
        for member_id in _member_ids(node)
    }
    unclassified: dict[str, Any] = {}
    for node in mechanical_nodes:
        if node.id in classified_members:
            continue
        node_hypotheses = hypotheses[node.id]
        unclassified[node.id] = {
            "status": "unknown" if not node_hypotheses else "low_confidence",
            "candidates": [_hypothesis_payload(item) for item in node_hypotheses[:3]],
        }

    semantic_metadata = dict(result.metadata.get("semantic", {}))
    semantic_metadata.update(
        {
            "backend": "deterministic_rules_v0",
            "min_confidence": min_confidence,
            "node_modalities": {
                node_id: sorted(modality.value for modality in values)
                for node_id, values in modalities.items()
                if values
            },
            "unclassified": unclassified,
        }
    )
    result.metadata["semantic"] = semantic_metadata
    return ArchTraceIR.model_validate(result.model_dump(mode="python"))


def _group_candidates(
    graph: ArchTraceIR,
    mechanical_nodes: list[ArchNode],
    hypotheses: dict[str, list[SemanticHypothesis]],
    min_confidence: float,
    overridden_ids: set[str],
    *,
    node_by_id: dict[str, ArchNode],
    containment_children: dict[str, list[str]],
) -> list[tuple[str, SemanticHypothesis, list[str]]]:
    candidates_by_anchor: dict[str, list[tuple[str, SemanticHypothesis]]] = {}
    for node in mechanical_nodes:
        if node.id in overridden_ids or not hypotheses[node.id]:
            continue
        anchor_id = _nearest_module_anchor(node, node_by_id)
        candidates_by_anchor.setdefault(anchor_id, []).append((node.id, hypotheses[node.id][0]))

    groups: list[tuple[str, SemanticHypothesis, list[str]]] = []
    for anchor_id in sorted(candidates_by_anchor):
        choices = candidates_by_anchor[anchor_id]
        choices.sort(
            key=lambda item: (
                -item[1].confidence,
                _role_priority(item[1].role),
                item[0],
            )
        )
        winner = choices[0][1]
        if winner.confidence < min_confidence or winner.role == SemanticRole.UNKNOWN:
            continue
        member_ids = _anchor_members(
            anchor_id, node_by_id, containment_children, overridden_ids
        )
        if not member_ids:
            continue
        groups.append((anchor_id, winner, member_ids))
    return groups


def _infer_node_hypotheses(
    node: ArchNode,
    modalities: dict[str, set[Modality]],
    flow_in: dict[str, list[str]],
) -> list[SemanticHypothesis]:
    text = _node_text(node)
    scored: dict[SemanticRole, tuple[float, list[str]]] = {}

    structural = _structural_role(node)
    if structural is not None:
        role, confidence, reason = structural
        scored[role] = (confidence, [reason])

    for spec in ROLE_SPECS:
        matched = [keyword for keyword in spec.keywords if _keyword_matches(text, keyword)]
        if not matched:
            continue
        confidence = (
            0.58
            if spec.role
            in {
                SemanticRole.GENERIC_ENCODER,
                SemanticRole.GENERIC_DECODER,
            }
            else 0.88
        )
        if spec.role == SemanticRole.MULTIMODAL_FUSION:
            confidence = 0.91
        reason = "name/source keyword: " + ", ".join(sorted(set(matched))[:3])
        current = scored.get(spec.role)
        if current is None or confidence > current[0]:
            scored[spec.role] = (confidence, [reason])
        elif confidence == current[0]:
            current[1].append(reason)

    incoming_modalities = _incoming_modalities(node.id, flow_in, modalities)
    primary = incoming_modalities & _PRIMARY_MODALITIES
    if len(primary) >= 2:
        confidence = 0.82
        reason = "multiple upstream modalities: " + ", ".join(
            sorted(modality.value for modality in primary)
        )
        current = scored.get(SemanticRole.MULTIMODAL_FUSION)
        if current is None or confidence > current[0]:
            scored[SemanticRole.MULTIMODAL_FUSION] = (confidence, [reason])
        elif confidence == current[0]:
            current[1].append(reason)

    results: list[SemanticHypothesis] = []
    for role, (confidence, reasons) in scored.items():
        spec = role_spec(role)
        role_modalities = set(modalities.get(node.id, set()))
        if spec is not None:
            role_modalities.update(spec.default_modalities)
        results.append(
            SemanticHypothesis(
                role=role,
                confidence=confidence,
                reasons=tuple(sorted(set(reasons))),
                modalities=tuple(sorted(role_modalities, key=lambda item: item.value)),
            )
        )
    results.sort(
        key=lambda item: (
            -item.confidence,
            _role_priority(item.role),
            item.role.value,
        )
    )
    return results


def _structural_role(
    node: ArchNode,
) -> tuple[SemanticRole, float, str] | None:
    if node.kind == NodeKind.LOSS:
        return SemanticRole.LOSS, 0.99, "ATIR node kind: loss"
    if node.kind == NodeKind.DATASET:
        return SemanticRole.DATASET, 0.99, "ATIR node kind: dataset"
    if node.kind == NodeKind.ENVIRONMENT:
        return SemanticRole.ENVIRONMENT, 0.99, "ATIR node kind: environment"
    return None


def _build_semantic_graph_index(
    graph: ArchTraceIR,
    mechanical_nodes: list[ArchNode],
) -> _SemanticGraphIndex:
    node_by_id = {node.id: node for node in graph.nodes}
    mechanical_ids = {node.id for node in mechanical_nodes}
    flow_in: dict[str, list[str]] = {}
    flow_out: dict[str, list[str]] = {}
    containment_children: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == EdgeKind.CONTAINS:
            containment_children.setdefault(edge.source, []).append(edge.target)
        if (
            edge.kind in _FLOW_EDGE_KINDS
            and edge.source in mechanical_ids
            and edge.target in mechanical_ids
        ):
            flow_out.setdefault(edge.source, []).append(edge.target)
            flow_in.setdefault(edge.target, []).append(edge.source)
    for mapping in (flow_in, flow_out, containment_children):
        for values in mapping.values():
            values.sort()
    return _SemanticGraphIndex(
        node_by_id=node_by_id,
        flow_in=flow_in,
        flow_out=flow_out,
        containment_children=containment_children,
    )


def _propagate_modalities(
    nodes: list[ArchNode],
    flow_out: dict[str, list[str]],
) -> dict[str, set[Modality]]:
    modalities = {node.id: _seed_modalities(node) for node in nodes}
    queue = deque(node_id for node_id, values in modalities.items() if values)
    queued = set(queue)
    while queue:
        source_id = queue.popleft()
        queued.discard(source_id)
        source_modalities = modalities[source_id]
        for target_id in flow_out.get(source_id, []):
            before = len(modalities[target_id])
            modalities[target_id].update(source_modalities)
            if len(modalities[target_id]) != before and target_id not in queued:
                queue.append(target_id)
                queued.add(target_id)
    return modalities


def _seed_modalities(node: ArchNode) -> set[Modality]:
    text = _node_text(node)
    result = {
        modality
        for modality, keywords in MODALITY_KEYWORDS.items()
        if any(_keyword_matches(text, keyword) for keyword in keywords)
    }
    if node.tensor is not None:
        semantic_text = _normalize_text(" ".join(node.tensor.semantics))
        for modality, keywords in MODALITY_KEYWORDS.items():
            if any(_keyword_matches(semantic_text, keyword) for keyword in keywords):
                result.add(modality)
    return result


def _incoming_modalities(
    node_id: str,
    flow_in: dict[str, list[str]],
    modalities: dict[str, set[Modality]],
) -> set[Modality]:
    result: set[Modality] = set()
    for source_id in flow_in.get(node_id, []):
        result.update(modalities.get(source_id, set()))
    return result


def _append_semantic_group(
    graph: ArchTraceIR,
    id_allocator: _IdAllocator,
    node_by_id: dict[str, ArchNode],
    *,
    anchor_id: str,
    hypothesis: SemanticHypothesis,
    member_ids: list[str],
    user_override: SemanticOverride | None,
    machine_hypotheses: dict[str, list[SemanticHypothesis]],
) -> None:
    members = [node_by_id[member_id] for member_id in member_ids]
    support_ids = sorted({evidence_id for member in members for evidence_id in member.evidence_ids})
    source = next((member.source[0] for member in members if member.source), None)
    spec = role_spec(hypothesis.role)
    label = (
        user_override.label
        if user_override is not None and user_override.label
        else (hypothesis.role.value if spec is None else spec.display_label)
    )
    phase = SemanticPhase.BOTH if spec is None else spec.phase
    evidence_id = id_allocator.next("evidence.semantic")
    semantic_id = id_allocator.next("semantic.component")
    evidence_kind = EvidenceKind.USER if user_override is not None else EvidenceKind.SEMANTIC
    status = FactStatus.CORRECTED if user_override is not None else FactStatus.INFERRED

    graph.evidence.append(
        Evidence(
            id=evidence_id,
            kind=evidence_kind,
            status=status,
            confidence=hypothesis.confidence,
            description=(
                "Human semantic correction."
                if user_override is not None
                else "Deterministic semantic role inference."
            ),
            source=source,
            metadata={
                "anchor_id": anchor_id,
                "member_ids": member_ids,
                "support_evidence_ids": support_ids,
                "reasons": list(hypothesis.reasons),
            },
        )
    )

    alternatives = _alternatives_for_members(member_ids, machine_hypotheses)
    graph.nodes.append(
        ArchNode(
            id=semantic_id,
            level=NodeLevel.SEMANTIC,
            kind=NodeKind.SEMANTIC_COMPONENT,
            identity_kind=IdentityKind.GROUP,
            label=label,
            role=hypothesis.role.value,
            source=[] if source is None else [source],
            evidence_ids=[evidence_id],
            attributes={
                "confidence": hypothesis.confidence,
                "member_ids": member_ids,
                "modalities": [item.value for item in hypothesis.modalities],
                "reasons": list(hypothesis.reasons),
                "alternatives": alternatives,
                "phase": phase.value,
                "inference_backend": (
                    "user_override" if user_override is not None else "deterministic_rules_v0"
                ),
                "support_evidence_ids": support_ids,
                "machine_hypotheses": alternatives if user_override is not None else [],
            },
        )
    )
    for member_id in member_ids:
        graph.edges.append(
            ArchEdge(
                id=id_allocator.next("edge.semantic.contains"),
                source=semantic_id,
                target=member_id,
                kind=EdgeKind.CONTAINS,
                evidence_ids=[evidence_id],
                attributes={"semantic_group": True},
            )
        )


def _nearest_module_anchor(node: ArchNode, node_by_id: dict[str, ArchNode]) -> str:
    if node.level == NodeLevel.MODULE:
        return node.id
    queue = list(node.parent_ids)
    visited: set[str] = set()
    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)
        current = node_by_id.get(current_id)
        if current is None:
            continue
        if current.level == NodeLevel.MODULE:
            return current.id
        queue.extend(current.parent_ids)
    return node.id


def _anchor_members(
    anchor_id: str,
    node_by_id: dict[str, ArchNode],
    containment_children: dict[str, list[str]],
    overridden_ids: set[str],
) -> list[str]:
    result: list[str] = []
    stack = [anchor_id]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        node = node_by_id.get(current)
        if (
            node is not None
            and node.level not in {NodeLevel.SEMANTIC, NodeLevel.PAPER}
            and current not in overridden_ids
        ):
            result.append(current)
        stack.extend(reversed(containment_children.get(current, [])))
    return sorted(set(result))


def _alternatives_for_members(
    member_ids: list[str],
    hypotheses: dict[str, list[SemanticHypothesis]],
) -> list[dict[str, Any]]:
    best_by_role: dict[SemanticRole, SemanticHypothesis] = {}
    for member_id in member_ids:
        for hypothesis in hypotheses.get(member_id, []):
            current = best_by_role.get(hypothesis.role)
            if current is None or hypothesis.confidence > current.confidence:
                best_by_role[hypothesis.role] = hypothesis
    ordered = sorted(
        best_by_role.values(),
        key=lambda item: (-item.confidence, _role_priority(item.role), item.role.value),
    )
    return [_hypothesis_payload(item) for item in ordered[:5]]


def _hypothesis_payload(hypothesis: SemanticHypothesis) -> dict[str, Any]:
    return {
        "role": hypothesis.role.value,
        "confidence": hypothesis.confidence,
        "reasons": list(hypothesis.reasons),
        "modalities": [item.value for item in hypothesis.modalities],
    }


def _member_ids(node: ArchNode) -> list[str]:
    raw = node.attributes.get("member_ids", [])
    return [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []


def _role_priority(role: SemanticRole) -> int:
    spec = role_spec(role)
    return 999 if spec is None else spec.priority


def _node_text(node: ArchNode) -> str:
    parts = [node.label, node.role or ""]
    parts.extend(span.path for span in node.source)
    parts.extend(span.symbol or "" for span in node.source)
    for key, value in sorted(node.attributes.items()):
        if isinstance(value, (str, int, float, bool)):
            parts.extend((str(key), str(value)))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return _normalize_text(" ".join(parts))


def _normalize_text(value: str) -> str:
    lowered = value.lower().replace("-", "_").replace("/", "_").replace(".", "_")
    return re.sub(r"[^a-z0-9_]+", "_", lowered)


def _keyword_matches(text: str, keyword: str) -> bool:
    normalized = _normalize_text(keyword)
    if normalized in text:
        return True
    compact_text = text.replace("_", "")
    return normalized.replace("_", "") in compact_text


def _all_record_ids(graph: ArchTraceIR) -> set[str]:
    return {
        record.id
        for group in (
            graph.nodes,
            graph.edges,
            graph.evidence,
            graph.runs,
            graph.claims,
            graph.conflicts,
            graph.coverage,
        )
        for record in group
    }
