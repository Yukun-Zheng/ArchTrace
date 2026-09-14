"""Exact source-span projection for synchronized explorer code panes."""

from __future__ import annotations

from pathlib import Path

from archtrace.ir import ArchNode, ArchTraceIR, NodeLevel
from archtrace.query.explorer_utils import parse_key
from archtrace.query.models import SourceExcerpt, SourcePane


def source_pane(
    graph: ArchTraceIR,
    key: str,
    *,
    root: str | Path | None = None,
    context_lines: int = 3,
    max_excerpts: int = 8,
) -> SourcePane:
    """Return exact source spans and optional local text for one explorer entity."""

    if context_lines < 0 or max_excerpts < 1:
        raise ValueError("invalid source pane bounds")
    _, entity_id = parse_key(key)
    nodes = {node.id: node for node in graph.nodes}
    node = nodes.get(entity_id)
    if node is None:
        return SourcePane(key=key)

    spans = list(node.source)
    if not spans and node.level == NodeLevel.SEMANTIC:
        member_ids = _semantic_member_ids(graph, node)
        spans = [
            span
            for member_id in member_ids
            if member_id in nodes
            for span in nodes[member_id].source
        ]

    excerpts = [
        _excerpt(
            span.path,
            span.start_line,
            span.end_line or span.start_line,
            span.symbol,
            root,
            context_lines,
        )
        for span in spans[:max_excerpts]
    ]
    return SourcePane(key=key, excerpts=excerpts)


def _semantic_member_ids(graph: ArchTraceIR, node: ArchNode) -> list[str]:
    raw = node.attributes.get("member_ids", [])
    members = (
        {item for item in raw if isinstance(item, str)}
        if isinstance(raw, list)
        else set()
    )
    for edge in graph.edges:
        if edge.source == node.id and edge.kind.value == "contains":
            members.add(edge.target)
    return sorted(members)


def _excerpt(
    source_path: str,
    start_line: int,
    end_line: int,
    symbol: str | None,
    root: str | Path | None,
    context_lines: int,
) -> SourceExcerpt:
    if root is None:
        return SourceExcerpt(
            path=source_path,
            start_line=start_line,
            end_line=end_line,
            symbol=symbol,
        )

    text, excerpt_start, excerpt_end = _read_source(
        Path(root),
        source_path,
        start_line,
        end_line,
        context_lines,
    )
    return SourceExcerpt(
        path=source_path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        text=text,
        excerpt_start_line=excerpt_start,
        excerpt_end_line=excerpt_end,
    )


def _read_source(
    root: Path,
    source_path: str,
    start_line: int,
    end_line: int,
    context_lines: int,
) -> tuple[str | None, int | None, int | None]:
    root = root.resolve()
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None, None, None
    try:
        lines = candidate.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return None, None, None

    excerpt_start = max(1, start_line - context_lines)
    excerpt_end = min(len(lines), end_line + context_lines)
    text = "\n".join(lines[excerpt_start - 1 : excerpt_end])
    return text, excerpt_start, excerpt_end
