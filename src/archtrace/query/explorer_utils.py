"""Shared key, metadata, and search helpers for the explorer query layer."""

from __future__ import annotations

from urllib.parse import quote, unquote

from archtrace.ir import ArchNode, NodeLevel
from archtrace.query.models import (
    Breadcrumb,
    ExplorerLevel,
    ExplorerNode,
    ExplorerSourceSpan,
    SearchHit,
)


def make_key(level: ExplorerLevel, entity_id: str) -> str:
    return f"{level.value}:{quote(entity_id, safe='')}"


def parse_key(key: str) -> tuple[ExplorerLevel, str]:
    raw_level, separator, raw_id = key.partition(":")
    if not separator or not raw_id:
        raise ValueError(f"invalid explorer key: {key!r}")
    try:
        level = ExplorerLevel(raw_level)
    except ValueError as exc:
        raise ValueError(f"invalid explorer level in key: {key!r}") from exc
    return level, unquote(raw_id)


def explorer_level(node: ArchNode) -> ExplorerLevel:
    if node.level == NodeLevel.PAPER:
        return ExplorerLevel.PAPER
    if node.level == NodeLevel.SEMANTIC:
        return ExplorerLevel.SEMANTIC
    if node.level == NodeLevel.MODULE:
        return ExplorerLevel.MODULE
    if node.level == NodeLevel.SOURCE:
        return ExplorerLevel.SOURCE
    return ExplorerLevel.OPERATION


def explorer_source(span: object) -> ExplorerSourceSpan:
    return ExplorerSourceSpan(
        path=str(span.path),
        start_line=int(span.start_line),
        end_line=span.end_line,
        start_column=span.start_column,
        end_column=span.end_column,
        symbol=span.symbol,
    )


def explorer_node(
    node: ArchNode,
    *,
    child_count: int,
) -> ExplorerNode:
    metadata: dict[str, object] = {
        "attributes": node.attributes,
        "evidence_ids": node.evidence_ids,
    }
    if node.tensor is not None:
        metadata["tensor"] = node.tensor.model_dump(mode="json")
    for name in ("confidence", "modalities", "phase"):
        value = node.attributes.get(name)
        if value is not None:
            metadata[name] = value
    return ExplorerNode(
        key=make_key(explorer_level(node), node.id),
        entity_id=node.id,
        level=explorer_level(node),
        label=node.label,
        kind=node.kind.value,
        role=node.role,
        identity_kind=node.identity_kind.value,
        child_count=child_count,
        run_id=node.run_id,
        occurrence_index=node.occurrence_index,
        definition_id=node.definition_id,
        source=[explorer_source(span) for span in node.source],
        metadata=metadata,
    )


def breadcrumb(node: ArchNode) -> Breadcrumb:
    level = explorer_level(node)
    return Breadcrumb(
        key=make_key(level, node.id),
        label=node.label,
        level=level,
    )


def search_text(node: ArchNode) -> str:
    parts = [node.id, node.label, node.kind.value, node.role or ""]
    parts.extend(span.path for span in node.source)
    parts.extend(span.symbol or "" for span in node.source)
    if node.tensor is not None:
        parts.extend(
            [
                node.tensor.dtype or "",
                node.tensor.device or "",
                *node.tensor.semantics,
                str(node.tensor.shape or ""),
            ]
        )
    for key, value in node.attributes.items():
        if isinstance(value, (str, int, float, bool)):
            parts.extend((str(key), str(value)))
    return " ".join(parts).lower()


def search_score(node: ArchNode, query: str, tokens: list[str]) -> int:
    label = node.label.lower()
    role = (node.role or "").lower()
    paths = " ".join(span.path.lower() for span in node.source)
    score = 10 * len(tokens)
    if query == node.id.lower() or query == label:
        score += 100
    elif label.startswith(query):
        score += 70
    elif query in label:
        score += 50
    if query and query in role:
        score += 40
    if query and query in paths:
        score += 30
    if node.tensor is not None:
        tensor_text = " ".join(
            filter(
                None,
                [node.tensor.dtype, node.tensor.device, *node.tensor.semantics],
            )
        ).lower()
        if query in tensor_text:
            score += 20
    return score


def search_hit(node: ArchNode, score: int) -> SearchHit:
    level = explorer_level(node)
    return SearchHit(
        key=make_key(level, node.id),
        entity_id=node.id,
        label=node.label,
        level=level,
        kind=node.kind.value,
        role=node.role,
        source_path=node.source[0].path if node.source else None,
        score=score,
    )
