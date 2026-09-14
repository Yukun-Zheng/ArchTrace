"""Repository ingestion for ArchTrace M0.

This module deliberately starts with framework-neutral Python structure. Runtime
framework adapters will enrich the same ATIR graph later rather than creating a
second graph model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from archtrace import ir as atir

IGNORED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
    "node_modules",
}

ENTRYPOINT_NAMES = {
    "main.py",
    "train.py",
    "eval.py",
    "evaluate.py",
    "inference.py",
    "demo.py",
    "run.py",
    "app.py",
}

MODEL_BASE_HINTS = {
    "Module",
    "nn.Module",
    "torch.nn.Module",
    "LightningModule",
    "PreTrainedModel",
    "Model",
}


@dataclass(slots=True)
class FileSummary:
    path: str
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    model_classes: list[str] = field(default_factory=list)
    entrypoint_score: int = 0


@dataclass(slots=True)
class RepositorySummary:
    root: Path
    python_files: list[FileSummary]

    @property
    def likely_entrypoints(self) -> list[FileSummary]:
        return sorted(
            (item for item in self.python_files if item.entrypoint_score > 0),
            key=lambda item: (-item.entrypoint_score, item.path),
        )


def scan_repository(root: str | Path) -> RepositorySummary:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    summaries: list[FileSummary] = []
    for path in sorted(root_path.rglob("*.py")):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        summaries.append(_summarize_python_file(root_path, path))

    return RepositorySummary(root=root_path, python_files=summaries)


def build_static_ir(summary: RepositorySummary) -> atir.ArchTraceIR:
    project = atir.ProjectInfo(name=summary.root.name, root=str(summary.root))
    nodes: list[atir.ArchNode] = []
    edges: list[atir.ArchEdge] = []
    evidence: list[atir.Evidence] = []

    repository_node = atir.ArchNode(
        id="repo.root",
        level=atir.NodeLevel.SEMANTIC,
        kind=atir.NodeKind.OTHER,
        identity_kind=atir.IdentityKind.GROUP,
        label=summary.root.name,
        role="repository",
    )
    nodes.append(repository_node)

    for file_index, file_summary in enumerate(summary.python_files):
        file_id = f"source.file.{file_index}"
        file_span = atir.SourceSpan(path=file_summary.path, start_line=1)
        evidence_id = f"evidence.static.file.{file_index}"
        evidence.append(
            atir.Evidence(
                id=evidence_id,
                kind=atir.EvidenceKind.STATIC,
                status=atir.FactStatus.OBSERVED,
                description="Python source discovered during repository scan.",
                source=file_span,
            )
        )
        nodes.append(
            atir.ArchNode(
                id=file_id,
                level=atir.NodeLevel.SOURCE,
                kind=atir.NodeKind.SOURCE,
                identity_kind=atir.IdentityKind.SOURCE,
                label=file_summary.path,
                parent_ids=[repository_node.id],
                source=[file_span],
                evidence_ids=[evidence_id],
                attributes={
                    "imports": file_summary.imports,
                    "functions": file_summary.functions,
                    "classes": file_summary.classes,
                    "model_classes": file_summary.model_classes,
                    "entrypoint_score": file_summary.entrypoint_score,
                },
            )
        )
        edges.append(
            atir.ArchEdge(
                id=f"edge.repo.file.{file_index}",
                source=repository_node.id,
                target=file_id,
                kind=atir.EdgeKind.CONTAINS,
                evidence_ids=[evidence_id],
            )
        )

        for class_index, class_name in enumerate(file_summary.model_classes):
            class_id = f"module.{file_index}.{class_index}"
            class_evidence_id = f"evidence.static.module.{file_index}.{class_index}"
            evidence.append(
                atir.Evidence(
                    id=class_evidence_id,
                    kind=atir.EvidenceKind.STATIC,
                    status=atir.FactStatus.INFERRED,
                    confidence=0.8,
                    description="Class name/base suggests a model/module definition.",
                    source=file_span,
                )
            )
            nodes.append(
                atir.ArchNode(
                    id=class_id,
                    level=atir.NodeLevel.MODULE,
                    kind=atir.NodeKind.MODULE,
                    identity_kind=atir.IdentityKind.DEFINITION,
                    label=class_name,
                    parent_ids=[file_id],
                    source=[file_span],
                    evidence_ids=[class_evidence_id],
                )
            )
            edges.append(
                atir.ArchEdge(
                    id=f"edge.file.module.{file_index}.{class_index}",
                    source=file_id,
                    target=class_id,
                    kind=atir.EdgeKind.CONTAINS,
                    evidence_ids=[class_evidence_id],
                )
            )

    return atir.ArchTraceIR(project=project, nodes=nodes, edges=edges, evidence=evidence)


def _summarize_python_file(root: Path, path: Path) -> FileSummary:
    relative = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")

    summary = FileSummary(path=relative)
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError:
        return summary

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            summary.imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            summary.imports.append(module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary.functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            summary.classes.append(node.name)
            bases = {_expr_name(base) for base in node.bases}
            if bases & MODEL_BASE_HINTS or any(
                hint in base for base in bases for hint in ("Module", "Model")
            ):
                summary.model_classes.append(node.name)

    name = path.name.lower()
    if name in ENTRYPOINT_NAMES:
        summary.entrypoint_score += 4
    if "__main__" in source:
        summary.entrypoint_score += 3
    if any(name.startswith(prefix) for prefix in ("train", "eval", "infer", "demo", "run")):
        summary.entrypoint_score += 1
    if any(item in summary.imports for item in ("torch", "jax", "tensorflow")):
        summary.entrypoint_score += 1

    summary.imports = sorted(set(item for item in summary.imports if item))
    summary.functions = sorted(set(summary.functions))
    summary.classes = sorted(set(summary.classes))
    summary.model_classes = sorted(set(summary.model_classes))
    return summary


def _expr_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return ""
