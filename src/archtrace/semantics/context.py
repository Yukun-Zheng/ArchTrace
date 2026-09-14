"""Repository text context collection for evidence-grounded semantic recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_CONTEXT_SUFFIXES = {".md", ".rst", ".txt"}
_IGNORED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
}


@dataclass(frozen=True, slots=True)
class ContextSnippet:
    path: str
    text: str


def collect_repository_context(
    root: str | Path,
    *,
    max_files: int = 24,
    max_chars: int = 40_000,
    max_file_chars: int = 12_000,
) -> list[ContextSnippet]:
    """Collect deterministic README/docs text without executing repository code."""

    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    if max_files < 0 or max_chars < 0 or max_file_chars < 0:
        raise ValueError("context limits must be non-negative")

    candidates = sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _CONTEXT_SUFFIXES
            and not _ignored(path)
            and _is_context_candidate(root_path, path)
        ),
        key=lambda path: _context_sort_key(root_path, path),
    )

    snippets: list[ContextSnippet] = []
    remaining = max_chars
    for path in candidates[:max_files]:
        if remaining <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        clipped = text[: min(max_file_chars, remaining)]
        if not clipped:
            continue
        snippets.append(
            ContextSnippet(
                path=path.relative_to(root_path).as_posix(),
                text=clipped,
            )
        )
        remaining -= len(clipped)
    return snippets


def _is_context_candidate(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    name = path.name.lower()
    if name.startswith("readme"):
        return True
    parts = {part.lower() for part in relative.parts[:-1]}
    return bool(parts & {"doc", "docs", "documentation"})


def _context_sort_key(root: Path, path: Path) -> tuple[int, str]:
    relative = path.relative_to(root).as_posix()
    is_readme = path.name.lower().startswith("readme")
    return (0 if is_readme else 1, relative.lower())


def _ignored(path: Path) -> bool:
    return any(part in _IGNORED_PARTS for part in path.parts)
