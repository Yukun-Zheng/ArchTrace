from pathlib import Path

from archtrace.ir import (
    ArchNode,
    ArchTraceIR,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
)
from archtrace.web_bundle import embed_source_files


def _graph() -> ArchTraceIR:
    span = SourceSpan(path="models/policy.py", start_line=2, end_line=3)
    evidence = Evidence(
        id="evidence.source",
        kind=EvidenceKind.SOURCE,
        status=FactStatus.OBSERVED,
        source=span,
    )
    node = ArchNode(
        id="module.policy",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="Policy",
        source=[span],
        evidence_ids=[evidence.id],
    )
    return ArchTraceIR(
        project=ProjectInfo(name="bundle-test"),
        nodes=[node],
        evidence=[evidence],
    )


def test_embed_source_files_is_non_destructive(tmp_path: Path) -> None:
    source = tmp_path / "models" / "policy.py"
    source.parent.mkdir()
    source.write_text("class Policy:\n    pass\n", encoding="utf-8")
    graph = _graph()

    bundled = embed_source_files(graph, tmp_path)

    assert bundled.nodes == graph.nodes
    assert bundled.edges == graph.edges
    assert bundled.claims == graph.claims
    web = bundled.metadata["web"]
    assert web["source_files"]["models/policy.py"].startswith("class Policy")
    assert graph.metadata == {}


def test_embed_source_files_rejects_paths_outside_root(tmp_path: Path) -> None:
    graph = _graph().model_copy(deep=True)
    graph.nodes[0].source = [SourceSpan(path="../secret.txt", start_line=1)]

    bundled = embed_source_files(graph, tmp_path)

    web = bundled.metadata["web"]
    assert web["source_files"] == {}
    assert web["source_bundle"]["skipped"][0]["reason"] == "outside_source_root"
