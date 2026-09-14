"""Normalize repository static index facts into ATIR v0.2."""

from __future__ import annotations

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
)
from archtrace.static.config_resolution import resolve_config_references
from archtrace.static.interprocedural import analyze_interprocedural_flow
from archtrace.static.python_index import RepositoryIndex, SymbolKind


def repository_index_to_atir(index: RepositoryIndex) -> ArchTraceIR:
    nodes: list[ArchNode] = []
    edges: list[ArchEdge] = []
    evidence: list[Evidence] = []
    evidence_counter = 0
    edge_counter = 0

    def add_evidence(
        description: str,
        source: SourceSpan,
        *,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        nonlocal evidence_counter
        evidence_id = f"evidence.static.{evidence_counter:07d}"
        evidence_counter += 1
        evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.STATIC,
                status=FactStatus.INFERRED,
                confidence=confidence,
                description=description,
                source=source,
                metadata={} if metadata is None else metadata,
            )
        )
        return evidence_id

    def add_edge(
        source: str,
        target: str,
        kind: EdgeKind,
        evidence_ids: list[str],
        *,
        label: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        nonlocal edge_counter
        edges.append(
            ArchEdge(
                id=f"edge.static.{edge_counter:07d}",
                source=source,
                target=target,
                kind=kind,
                label=label,
                evidence_ids=evidence_ids,
                attributes={} if attributes is None else attributes,
            )
        )
        edge_counter += 1

    symbol_ids = {symbol.id for symbol in index.symbols}
    call_ids = {call.id for call in index.calls}
    calls_by_id = {call.id: call for call in index.calls}

    for symbol in index.symbols:
        evidence_id = add_evidence(
            "Python symbol discovered by AST analysis.",
            symbol.span,
        )
        kind = NodeKind.MODULE if symbol.kind == SymbolKind.CLASS else NodeKind.OTHER
        level = NodeLevel.MODULE if symbol.kind == SymbolKind.CLASS else NodeLevel.SOURCE
        nodes.append(
            ArchNode(
                id=symbol.id,
                level=level,
                kind=kind,
                identity_kind=IdentityKind.DEFINITION,
                label=symbol.name,
                role=f"python_{symbol.kind.value}_definition",
                parent_ids=(
                    [symbol.parent_id]
                    if symbol.parent_id is not None and symbol.parent_id in symbol_ids
                    else []
                ),
                source=[symbol.span],
                evidence_ids=[evidence_id],
                attributes={
                    "module": symbol.module,
                    "qualname": symbol.qualname,
                },
            )
        )
        if symbol.parent_id is not None and symbol.parent_id in symbol_ids:
            add_edge(symbol.parent_id, symbol.id, EdgeKind.CONTAINS, [evidence_id])

    call_evidence: dict[str, str] = {}
    for call in index.calls:
        confidence = 1.0 if call.resolved_symbol_id is not None else 0.7
        evidence_id = add_evidence(
            f"Python call site statically indexed ({call.resolution.value}).",
            call.span,
            confidence=confidence,
        )
        call_evidence[call.id] = evidence_id
        nodes.append(
            ArchNode(
                id=call.id,
                level=NodeLevel.OPERATION,
                kind=NodeKind.OPERATION,
                identity_kind=IdentityKind.DEFINITION,
                label=call.callee_text,
                role="python_call_site",
                parent_ids=(
                    [call.caller_symbol_id]
                    if call.caller_symbol_id in symbol_ids
                    else []
                ),
                source=[call.span],
                evidence_ids=[evidence_id],
                attributes={
                    "module": call.module,
                    "resolution": call.resolution.value,
                    "resolved_target": call.resolved_target,
                    "result_targets": call.result_targets,
                    "argument_names": call.argument_names,
                },
            )
        )
        if call.caller_symbol_id in symbol_ids:
            add_edge(
                call.caller_symbol_id,
                call.id,
                EdgeKind.CONTAINS,
                [evidence_id],
            )
        if call.resolved_symbol_id is not None and call.resolved_symbol_id in symbol_ids:
            add_edge(
                call.id,
                call.resolved_symbol_id,
                EdgeKind.CALLS,
                [evidence_id],
            )

    for intra_link in index.dataflow:
        if (
            intra_link.producer_call_id not in call_ids
            or intra_link.consumer_call_id not in call_ids
        ):
            continue
        evidence_ids = [
            call_evidence[intra_link.producer_call_id],
            call_evidence[intra_link.consumer_call_id],
        ]
        add_edge(
            intra_link.producer_call_id,
            intra_link.consumer_call_id,
            EdgeKind.DATA,
            evidence_ids,
            label=intra_link.variable,
            attributes={"flow_kind": "intraprocedural"},
        )

    for boundary_link in analyze_interprocedural_flow(index):
        producer = calls_by_id.get(boundary_link.producer_call_id)
        consumer = calls_by_id.get(boundary_link.consumer_call_id)
        if producer is None or consumer is None:
            continue
        evidence_id = add_evidence(
            "Direct interprocedural value flow inferred across a resolved local call.",
            consumer.span,
            confidence=0.9,
            metadata={
                "producer_call_id": producer.id,
                "consumer_call_id": consumer.id,
                "flow_kind": boundary_link.kind.value,
            },
        )
        add_edge(
            producer.id,
            consumer.id,
            EdgeKind.DATA,
            [evidence_id],
            label=boundary_link.variable,
            attributes={
                "flow_kind": boundary_link.kind.value,
                **boundary_link.metadata,
            },
        )

    config_nodes: dict[str, ArchNode] = {}
    config_evidence: dict[str, str] = {}
    for entry in index.config_entries:
        evidence_id = add_evidence(
            f"Configuration value discovered from {entry.kind}.",
            entry.span,
        )
        node = ArchNode(
            id=entry.id,
            level=NodeLevel.SOURCE,
            kind=NodeKind.CONFIG,
            identity_kind=IdentityKind.STATE,
            label=entry.key,
            role=f"config_{entry.kind}",
            source=[entry.span],
            evidence_ids=[evidence_id],
            attributes={
                "path": entry.path,
                "key": entry.key,
                "value": entry.value,
                **entry.metadata,
            },
        )
        nodes.append(node)
        config_nodes[entry.id] = node
        config_evidence[entry.id] = evidence_id

    config_references: list[dict[str, Any]] = []
    for reference in resolve_config_references(index):
        source_node = config_nodes.get(reference.source_entry_id)
        if source_node is None:
            continue

        source_node.attributes["reference_status"] = reference.status.value
        source_node.attributes["candidate_paths"] = reference.candidate_paths
        source_node.attributes["target_entry_ids"] = reference.target_entry_ids
        config_references.append(
            {
                "source_entry_id": reference.source_entry_id,
                "status": reference.status.value,
                "candidate_paths": reference.candidate_paths,
                "target_entry_ids": reference.target_entry_ids,
            }
        )

        for target_id in reference.target_entry_ids:
            target_node = config_nodes.get(target_id)
            if target_node is None:
                continue
            add_edge(
                source_node.id,
                target_node.id,
                EdgeKind.READS,
                [
                    config_evidence[source_node.id],
                    config_evidence[target_node.id],
                ],
                label="resolves_config",
                attributes={"resolution": "static_config_reference"},
            )

    return ArchTraceIR(
        project=ProjectInfo(name=index.root.name, root=str(index.root)),
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        metadata={
            "static_backend": "python_ast_repository_index",
            "entrypoints": [
                {
                    "path": candidate.path,
                    "score": candidate.score,
                    "reasons": candidate.reasons,
                }
                for candidate in index.entrypoints
            ],
            "config_references": config_references,
            "parse_errors": {
                file.path: file.parse_error
                for file in index.files
                if file.parse_error is not None
            },
        },
    )
