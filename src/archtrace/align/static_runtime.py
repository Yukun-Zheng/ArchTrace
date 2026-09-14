"""Lossless reconciliation of repository-static and runtime ATIR evidence."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel

from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    CoverageRecord,
    CoverageStatus,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    ProjectInfo,
    SourceSpan,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def reconcile_static_runtime(
    static_graph: ArchTraceIR,
    runtime_graph: ArchTraceIR,
) -> ArchTraceIR:
    """Combine evidence graphs and add conservative source-based alignment."""

    _validate_projects(static_graph.project, runtime_graph.project)
    _validate_collisions(static_graph, runtime_graph)
    nodes = _merge(static_graph.nodes, runtime_graph.nodes)
    edges = _merge(static_graph.edges, runtime_graph.edges)
    evidence = _merge(static_graph.evidence, runtime_graph.evidence)
    runs = _merge(static_graph.runs, runtime_graph.runs)
    claims = _merge(static_graph.claims, runtime_graph.claims)
    conflicts = _merge(static_graph.conflicts, runtime_graph.conflicts)
    coverage = _merge(static_graph.coverage, runtime_graph.coverage)
    used = _all_ids(nodes, edges, evidence, runs, claims, conflicts, coverage)

    static_defs = [
        node
        for node in static_graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION and node.source
    ]
    runtime_defs = [
        node
        for node in runtime_graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION and node.source
    ]
    alignments: list[dict[str, Any]] = []
    for runtime_node in runtime_defs:
        matched = _best_match(runtime_node, static_defs)
        if matched is None:
            continue
        static_node, score = matched
        confidence = min(0.99, score / 12.0)
        evidence_id = _next_id("evidence.align", used)
        evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.STATIC,
                status=FactStatus.INFERRED,
                confidence=confidence,
                description="Runtime definition aligned to repository source definition.",
                source=runtime_node.source[0],
                metadata={
                    "runtime_definition_id": runtime_node.id,
                    "static_definition_id": static_node.id,
                    "alignment_score": score,
                },
            )
        )
        edges.append(
            ArchEdge(
                id=_next_id("edge.align", used),
                source=runtime_node.id,
                target=static_node.id,
                kind=EdgeKind.ALIAS,
                evidence_ids=[evidence_id],
                attributes={"alignment": "source_span", "score": score},
            )
        )
        record = _coverage_record(
            static_node,
            runtime_node,
            runtime_graph,
            evidence_id,
            used,
        )
        if record is not None:
            coverage.append(record)
        alignments.append(
            {
                "runtime_definition_id": runtime_node.id,
                "static_definition_id": static_node.id,
                "score": score,
                "confidence": confidence,
            }
        )

    return ArchTraceIR(
        project=_merge_project(static_graph.project, runtime_graph.project),
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        runs=runs,
        claims=claims,
        conflicts=conflicts,
        coverage=coverage,
        metadata={
            "reconciliation": {
                "static_metadata": static_graph.metadata,
                "runtime_metadata": runtime_graph.metadata,
                "source_alignments": alignments,
            }
        },
    )


def _coverage_record(
    static_node: ArchNode,
    runtime_node: ArchNode,
    runtime_graph: ArchTraceIR,
    evidence_id: str,
    used: set[str],
) -> CoverageRecord | None:
    considered = sorted(run.id for run in runtime_graph.runs)
    if not considered:
        return None
    observed = sorted(
        {
            node.run_id
            for node in runtime_graph.nodes
            if node.identity_kind == IdentityKind.OCCURRENCE
            and node.definition_id == runtime_node.id
            and node.run_id is not None
            and node.run_id in considered
        }
    )
    if observed == considered:
        status = CoverageStatus.ALWAYS_OBSERVED
    elif observed:
        status = CoverageStatus.SOMETIMES_OBSERVED
    else:
        status = CoverageStatus.STATIC_REACHABLE_UNOBSERVED
    return CoverageRecord(
        id=_next_id("coverage.align", used),
        subject_id=static_node.id,
        status=status,
        considered_run_ids=considered,
        observed_run_ids=observed,
        evidence_ids=[evidence_id],
        metadata={"runtime_definition_id": runtime_node.id},
    )


def _best_match(
    runtime_node: ArchNode,
    static_nodes: list[ArchNode],
) -> tuple[ArchNode, int] | None:
    candidates = [
        (score, node)
        for node in static_nodes
        if (score := _match_score(runtime_node, node)) >= 8
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].id))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    score, node = candidates[0]
    return node, score


def _match_score(runtime_node: ArchNode, static_node: ArchNode) -> int:
    best = 0
    for left in runtime_node.source:
        for right in static_node.source:
            if not _path_matches(left.path, right.path):
                continue
            score = 4
            if left.start_line == right.start_line:
                score += 5
            elif _overlaps(left, right):
                score += 2
            if static_node.label == "forward":
                score += 2
            if static_node.role == "python_method_definition":
                score += 1
            if _tail(left.symbol) and _tail(left.symbol) == _tail(right.symbol):
                score += 2
            best = max(best, score)
    return best


def _overlaps(left: SourceSpan, right: SourceSpan) -> bool:
    left_end = left.end_line or left.start_line
    right_end = right.end_line or right.start_line
    return left.start_line <= right_end and right.start_line <= left_end


def _path_matches(left: str, right: str) -> bool:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    return (
        left_norm == right_norm
        or left_norm.endswith(f"/{right_norm}")
        or right_norm.endswith(f"/{left_norm}")
    )


def _normalize(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")


def _tail(value: str | None) -> str | None:
    return None if not value else value.rsplit(".", 1)[-1]


def _validate_projects(static: ProjectInfo, runtime: ProjectInfo) -> None:
    if (
        static.repository_url
        and runtime.repository_url
        and static.repository_url != runtime.repository_url
    ):
        raise ValueError("cannot reconcile different repository URLs")
    if static.revision and runtime.revision and static.revision != runtime.revision:
        raise ValueError("cannot reconcile different repository revisions")


def _merge_project(static: ProjectInfo, runtime: ProjectInfo) -> ProjectInfo:
    return ProjectInfo(
        name=static.name or runtime.name,
        root=static.root or runtime.root,
        repository_url=static.repository_url or runtime.repository_url,
        revision=static.revision or runtime.revision,
        metadata={"static": static.metadata, "runtime": runtime.metadata},
    )


def _merge(left: list[_ModelT], right: list[_ModelT]) -> list[_ModelT]:
    result = list(left)
    by_id = {str(getattr(item, "id")): item for item in left}
    for item in right:
        item_id = str(getattr(item, "id"))
        existing = by_id.get(item_id)
        if existing is None:
            result.append(item)
            by_id[item_id] = item
        elif existing.model_dump(mode="json") != item.model_dump(mode="json"):
            raise ValueError(f"record id collision with different payload: {item_id}")
    return result


def _validate_collisions(static: ArchTraceIR, runtime: ArchTraceIR) -> None:
    left = _records(static)
    right = _records(runtime)
    for record_id in sorted(left.keys() & right.keys()):
        if left[record_id] != right[record_id]:
            raise ValueError(f"cross-graph record id collision: {record_id}")


def _records(graph: ArchTraceIR) -> dict[str, tuple[str, dict[str, Any]]]:
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    groups: list[tuple[str, list[Any]]] = [
        ("node", list(graph.nodes)),
        ("edge", list(graph.edges)),
        ("evidence", list(graph.evidence)),
        ("run", list(graph.runs)),
        ("claim", list(graph.claims)),
        ("conflict", list(graph.conflicts)),
        ("coverage", list(graph.coverage)),
    ]
    for category, records in groups:
        for record in records:
            result[str(record.id)] = (category, record.model_dump(mode="json"))
    return result


def _all_ids(*groups: list[Any]) -> set[str]:
    return {str(record.id) for group in groups for record in group}


def _next_id(prefix: str, used: set[str]) -> str:
    index = 0
    while f"{prefix}.{index:07d}" in used:
        index += 1
    value = f"{prefix}.{index:07d}"
    used.add(value)
    return value
