"""Create browser-friendly ATIR bundles without changing mechanical facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from archtrace.ir import ArchTraceIR


def embed_source_files(
    graph: ArchTraceIR,
    source_root: Path,
    *,
    max_file_bytes: int = 256_000,
    max_total_bytes: int = 5_000_000,
) -> ArchTraceIR:
    """Embed source text referenced by ATIR spans into presentation metadata.

    Files are only read when they resolve inside ``source_root``. The operation
    produces a deep copy and never changes nodes, edges, claims, or evidence.
    """

    if max_file_bytes < 1 or max_total_bytes < 1:
        raise ValueError("source bundle byte limits must be positive")

    root = source_root.expanduser().resolve()
    result = graph.model_copy(deep=True)
    candidate_paths = _referenced_paths(result)
    source_files: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    total = 0

    for display_path in candidate_paths:
        resolved = _resolve_source_path(root, display_path)
        if resolved is None:
            skipped.append({"path": display_path, "reason": "outside_source_root"})
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            skipped.append({"path": display_path, "reason": "missing"})
            continue
        if size > max_file_bytes:
            skipped.append({"path": display_path, "reason": "file_too_large", "bytes": size})
            continue
        if total + size > max_total_bytes:
            skipped.append({"path": display_path, "reason": "total_limit"})
            continue
        try:
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped.append({"path": display_path, "reason": "not_utf8_text"})
            continue
        source_files[display_path] = text
        total += len(text.encode("utf-8"))

    metadata = dict(result.metadata)
    web_metadata = metadata.get("web")
    web = dict(web_metadata) if isinstance(web_metadata, dict) else {}
    web.update(
        {
            "source_files": source_files,
            "source_bundle": {
                "root": str(root),
                "embedded_files": len(source_files),
                "embedded_bytes": total,
                "skipped": skipped,
            },
        }
    )
    metadata["web"] = web
    result.metadata = metadata
    return ArchTraceIR.model_validate(result.model_dump(mode="python"))


def _referenced_paths(graph: ArchTraceIR) -> list[str]:
    paths: set[str] = set()
    for node in graph.nodes:
        paths.update(span.path for span in node.source)
    for evidence in graph.evidence:
        if evidence.source is not None:
            paths.add(evidence.source.path)
    return sorted(paths)


def _resolve_source_path(root: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved
