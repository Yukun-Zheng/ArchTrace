from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from archtrace.ingest.repository import build_static_ir, scan_repository
from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    Claim,
    ClaimScope,
    Conflict,
    ConflictKind,
    CoverageRecord,
    CoverageStatus,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
    TraceRun,
    load_atir_json,
)


def test_source_span_rejects_reversed_range() -> None:
    with pytest.raises(ValidationError):
        SourceSpan(path="model.py", start_line=10, end_line=4)


def test_atir_rejects_unknown_edge_endpoint() -> None:
    with pytest.raises(ValidationError):
        ArchTraceIR(
            project=ProjectInfo(name="demo"),
            nodes=[
                ArchNode(
                    id="a",
                    level=NodeLevel.MODULE,
                    kind=NodeKind.MODULE,
                    identity_kind=IdentityKind.DEFINITION,
                    label="A",
                )
            ],
            edges=[
                ArchEdge(
                    id="e",
                    source="a",
                    target="missing",
                    kind=EdgeKind.DATA,
                )
            ],
        )


def test_repository_scan_builds_valid_static_ir(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text(
        """
import torch
from torch import nn

class TinyPolicy(nn.Module):
    def forward(self, x):
        return x
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text(
        """
from model import TinyPolicy

if __name__ == "__main__":
    model = TinyPolicy()
""".strip(),
        encoding="utf-8",
    )

    summary = scan_repository(tmp_path)
    assert len(summary.python_files) == 2
    assert summary.likely_entrypoints[0].path == "train.py"

    graph = build_static_ir(summary)
    assert graph.schema_version == "0.2"
    tiny_policy = next(node for node in graph.nodes if node.label == "TinyPolicy")
    assert tiny_policy.identity_kind == IdentityKind.DEFINITION
    assert any(edge.kind == EdgeKind.CONTAINS for edge in graph.edges)

    round_trip = ArchTraceIR.model_validate_json(graph.model_dump_json())
    assert round_trip.project.name == tmp_path.name


def test_occurrence_requires_run() -> None:
    with pytest.raises(ValidationError):
        ArchNode(
            id="call.0",
            level=NodeLevel.MODULE,
            kind=NodeKind.MODULE,
            identity_kind=IdentityKind.OCCURRENCE,
            label="SharedBlock call",
        )


def test_shared_definition_keeps_repeated_occurrences_distinct() -> None:
    definition = ArchNode(
        id="module.shared",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="SharedBlock",
    )
    first = ArchNode(
        id="call.run0.0",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.OCCURRENCE,
        label="SharedBlock #0",
        definition_id=definition.id,
        run_id="run.0",
        occurrence_index=0,
    )
    second = ArchNode(
        id="call.run0.1",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.OCCURRENCE,
        label="SharedBlock #1",
        definition_id=definition.id,
        run_id="run.0",
        occurrence_index=1,
    )

    graph = ArchTraceIR(
        project=ProjectInfo(name="shared"),
        runs=[TraceRun(id="run.0")],
        nodes=[definition, first, second],
    )

    occurrences = [node for node in graph.nodes if node.identity_kind == IdentityKind.OCCURRENCE]
    assert [node.id for node in occurrences] == ["call.run0.0", "call.run0.1"]
    assert {node.definition_id for node in occurrences} == {"module.shared"}


def test_definition_id_must_point_to_definition_node() -> None:
    with pytest.raises(ValidationError):
        ArchTraceIR(
            project=ProjectInfo(name="invalid"),
            runs=[TraceRun(id="run.0")],
            nodes=[
                ArchNode(
                    id="semantic.block",
                    level=NodeLevel.SEMANTIC,
                    kind=NodeKind.SEMANTIC_COMPONENT,
                    identity_kind=IdentityKind.GROUP,
                    label="Block",
                ),
                ArchNode(
                    id="call.0",
                    level=NodeLevel.MODULE,
                    kind=NodeKind.MODULE,
                    identity_kind=IdentityKind.OCCURRENCE,
                    label="Call",
                    definition_id="semantic.block",
                    run_id="run.0",
                ),
            ],
        )


def test_claim_conflict_and_coverage_can_coexist() -> None:
    module = ArchNode(
        id="module.vision",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="VisionEncoder",
    )
    author_evidence = Evidence(
        id="e.author",
        kind=EvidenceKind.AUTHOR,
        status=FactStatus.DECLARED,
        description="README says the vision encoder is frozen.",
    )
    runtime_evidence = Evidence(
        id="e.runtime",
        kind=EvidenceKind.RUNTIME,
        status=FactStatus.OBSERVED,
        run_id="run.0",
        description="Runtime observed trainable vision parameters.",
    )
    declared = Claim(
        id="claim.frozen.author",
        subject_id=module.id,
        predicate="frozen",
        value=True,
        evidence_ids=[author_evidence.id],
        status=FactStatus.DECLARED,
    )
    observed = Claim(
        id="claim.frozen.runtime",
        subject_id=module.id,
        predicate="frozen",
        value=False,
        evidence_ids=[runtime_evidence.id],
        status=FactStatus.OBSERVED,
        scope=ClaimScope(run_ids=["run.0"]),
    )

    graph = ArchTraceIR(
        project=ProjectInfo(name="claims"),
        runs=[TraceRun(id="run.0")],
        nodes=[module],
        evidence=[author_evidence, runtime_evidence],
        claims=[declared, observed],
        conflicts=[
            Conflict(
                id="conflict.vision.freeze",
                claim_ids=[declared.id, observed.id],
                kind=ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH,
            )
        ],
        coverage=[
            CoverageRecord(
                id="coverage.vision",
                subject_id=module.id,
                status=CoverageStatus.ALWAYS_OBSERVED,
                considered_run_ids=["run.0"],
                observed_run_ids=["run.0"],
                evidence_ids=[runtime_evidence.id],
            )
        ],
    )

    assert len(graph.conflicts) == 1
    assert graph.coverage[0].status == CoverageStatus.ALWAYS_OBSERVED


def test_sometimes_observed_requires_proper_subset() -> None:
    with pytest.raises(ValidationError):
        CoverageRecord(
            id="coverage.bad",
            subject_id="node.x",
            status=CoverageStatus.SOMETIMES_OBSERVED,
            considered_run_ids=["run.0"],
            observed_run_ids=["run.0"],
        )


def test_v01_payload_migrates_without_inventing_runtime_occurrences() -> None:
    old_payload = {
        "schema_version": "0.1",
        "project": {"name": "legacy"},
        "nodes": [
            {
                "id": "module.0",
                "level": "module",
                "kind": "module",
                "label": "LegacyModule",
            },
            {
                "id": "source.0",
                "level": "source",
                "kind": "source",
                "label": "model.py",
            },
        ],
        "edges": [],
        "evidence": [],
        "runs": [],
        "metadata": {},
    }

    graph = load_atir_json(json.dumps(old_payload))

    assert graph.schema_version == "0.2"
    assert graph.nodes[0].identity_kind == IdentityKind.DEFINITION
    assert graph.nodes[1].identity_kind == IdentityKind.SOURCE
    assert not any(node.identity_kind == IdentityKind.OCCURRENCE for node in graph.nodes)
    assert graph.metadata["schema_migrations"] == [{"from": "0.1", "to": "0.2"}]
