from __future__ import annotations

from pathlib import Path

import pytest

from archtrace.align import reconcile_static_runtime
from archtrace.ir import (
    ArchNode,
    ArchTraceIR,
    CoverageStatus,
    EdgeKind,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
    TraceRun,
)
from archtrace.static import index_repository, repository_index_to_atir


def _static_graph(root: Path) -> ArchTraceIR:
    (root / "model.py").write_text(
        """
class Tiny:
    def forward(self, x):
        return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return repository_index_to_atir(index_repository(root))


def _runtime_graph(
    root: Path,
    *,
    observed: bool,
    definition_id: str = "runtime.module.tiny",
) -> ArchTraceIR:
    span = SourceSpan(
        path=f"/workspace/{root.name}/model.py",
        start_line=2,
        end_line=3,
        symbol="Tiny.forward",
    )
    nodes = [
        ArchNode(
            id=definition_id,
            level=NodeLevel.MODULE,
            kind=NodeKind.MODULE,
            identity_kind=IdentityKind.DEFINITION,
            label="Tiny",
            role="pytorch_module_definition",
            source=[span],
        )
    ]
    if observed:
        nodes.append(
            ArchNode(
                id="runtime.call.tiny.0",
                level=NodeLevel.MODULE,
                kind=NodeKind.MODULE,
                identity_kind=IdentityKind.OCCURRENCE,
                label="model",
                role="pytorch_module_call",
                definition_id=definition_id,
                run_id="run.0",
                occurrence_index=0,
                source=[span],
            )
        )
    return ArchTraceIR(
        project=ProjectInfo(name=root.name),
        nodes=nodes,
        runs=[TraceRun(id="run.0", framework="pytorch")],
        metadata={"runtime_backend": "test"},
    )


def test_reconciliation_keeps_both_graphs_and_adds_source_alignment(
    tmp_path: Path,
) -> None:
    static = _static_graph(tmp_path)
    runtime = _runtime_graph(tmp_path, observed=True)
    merged = reconcile_static_runtime(static, runtime)

    static_forward = next(
        node
        for node in static.nodes
        if node.label == "forward" and node.role == "python_method_definition"
    )
    alias = next(
        edge
        for edge in merged.edges
        if edge.kind == EdgeKind.ALIAS and edge.source == "runtime.module.tiny"
    )
    assert alias.target == static_forward.id
    assert any(node.id == "runtime.call.tiny.0" for node in merged.nodes)
    assert any(node.id == static_forward.id for node in merged.nodes)

    coverage = next(record for record in merged.coverage if record.subject_id == static_forward.id)
    assert coverage.status == CoverageStatus.ALWAYS_OBSERVED
    assert coverage.observed_run_ids == ["run.0"]


def test_reconciliation_marks_aligned_but_unobserved_definition(
    tmp_path: Path,
) -> None:
    static = _static_graph(tmp_path)
    runtime = _runtime_graph(tmp_path, observed=False)
    merged = reconcile_static_runtime(static, runtime)

    static_forward = next(node for node in static.nodes if node.label == "forward")
    coverage = next(record for record in merged.coverage if record.subject_id == static_forward.id)
    assert coverage.status == CoverageStatus.STATIC_REACHABLE_UNOBSERVED
    assert coverage.observed_run_ids == []


def test_reconciliation_rejects_conflicting_global_ids(tmp_path: Path) -> None:
    static = _static_graph(tmp_path)
    static_forward = next(node for node in static.nodes if node.label == "forward")
    runtime = _runtime_graph(
        tmp_path,
        observed=False,
        definition_id=static_forward.id,
    )

    with pytest.raises(ValueError, match="cross-graph record id collision"):
        reconcile_static_runtime(static, runtime)
