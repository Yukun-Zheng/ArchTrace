"""Normalize repository static index facts into ATIR v0.2."""

from __future__ import annotations

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
            )
        )
        edge_counter += 1

    symbol_ids = {symbol.id for symbol in index.symbols}
    call_ids = {call.id for call in index.calls}

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

    for link in index.dataflow:
        if link.producer_call_id not in call_ids or link.consumer_call_id not in call_ids:
            continue
        evidence_ids = [
            call_evidence[link.producer_call_id],
            call_evidence[link.consumer_call_id],
        ]
        add_edge(
            link.producer_call_id,
            link.consumer_call_id,
            EdgeKind.DATA,
            evidence_ids,
            label=link.variable,
        )

    for entry in index.config_entries:
        evidence_id = add_evidence(
            f"Configuration value discovered from {entry.kind}.",
            entry.span,
        )
        nodes.append(
            ArchNode(
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
            "parse_errors": {
                file.path: file.parse_error
                for file in index.files
                if file.parse_error is not None
            },
        },
    )
