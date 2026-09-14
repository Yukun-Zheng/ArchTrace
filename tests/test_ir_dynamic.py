from __future__ import annotations

import pytest
from pydantic import ValidationError

from archtrace.ir import (
    ArchNode,
    ArchTraceIR,
    CoverageRecord,
    CoverageStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    TraceRun,
)


def test_loop_preserves_each_execution_occurrence() -> None:
    loop_definition = ArchNode(
        id="module.denoise_step",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="DenoiseStep",
    )
    iterations = [
        ArchNode(
            id=f"call.run0.denoise.{index}",
            level=NodeLevel.MODULE,
            kind=NodeKind.MODULE,
            identity_kind=IdentityKind.OCCURRENCE,
            label=f"DenoiseStep iteration {index}",
            definition_id=loop_definition.id,
            run_id="run.0",
            occurrence_index=index,
        )
        for index in range(4)
    ]

    graph = ArchTraceIR(
        project=ProjectInfo(name="loop"),
        runs=[TraceRun(id="run.0")],
        nodes=[loop_definition, *iterations],
    )

    assert len(graph.nodes) == 5
    assert [node.occurrence_index for node in graph.nodes[1:]] == [0, 1, 2, 3]
    assert {node.definition_id for node in graph.nodes[1:]} == {loop_definition.id}


def test_branch_coverage_is_scoped_across_runs() -> None:
    optional_branch = ArchNode(
        id="module.depth_branch",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="OptionalDepthBranch",
    )

    graph = ArchTraceIR(
        project=ProjectInfo(name="branches"),
        runs=[TraceRun(id="run.rgb"), TraceRun(id="run.rgbd")],
        nodes=[optional_branch],
        coverage=[
            CoverageRecord(
                id="coverage.depth_branch",
                subject_id=optional_branch.id,
                status=CoverageStatus.SOMETIMES_OBSERVED,
                considered_run_ids=["run.rgb", "run.rgbd"],
                observed_run_ids=["run.rgbd"],
                conditions={"depth_enabled": "depends_on_run_config"},
            )
        ],
    )

    record = graph.coverage[0]
    assert record.status == CoverageStatus.SOMETIMES_OBSERVED
    assert record.observed_run_ids == ["run.rgbd"]


def test_record_ids_are_globally_unique() -> None:
    node = ArchNode(
        id="shared.id",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.DEFINITION,
        label="Block",
    )

    with pytest.raises(ValidationError, match="duplicate record id"):
        ArchTraceIR(
            project=ProjectInfo(name="collision"),
            runs=[TraceRun(id="shared.id")],
            nodes=[node],
        )
