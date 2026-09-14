"""Minimal ATIR v0.2 example: shared execution + contradictory claims."""

from archtrace.ir import (
    ArchNode,
    ArchTraceIR,
    Claim,
    ClaimScope,
    Conflict,
    ConflictKind,
    CoverageRecord,
    CoverageStatus,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    TraceRun,
)

run = TraceRun(id="run.0", entrypoint="demo.py", framework="pytorch")

shared_definition = ArchNode(
    id="module.shared",
    level=NodeLevel.MODULE,
    kind=NodeKind.MODULE,
    identity_kind=IdentityKind.DEFINITION,
    label="SharedBlock",
)

calls = [
    ArchNode(
        id=f"call.run0.{index}",
        level=NodeLevel.MODULE,
        kind=NodeKind.MODULE,
        identity_kind=IdentityKind.OCCURRENCE,
        label=f"SharedBlock call #{index}",
        definition_id=shared_definition.id,
        run_id=run.id,
        occurrence_index=index,
    )
    for index in range(2)
]

author_evidence = Evidence(
    id="e.author",
    kind=EvidenceKind.AUTHOR,
    status=FactStatus.DECLARED,
    description="README declares that SharedBlock is frozen.",
)
runtime_evidence = Evidence(
    id="e.runtime",
    kind=EvidenceKind.RUNTIME,
    status=FactStatus.OBSERVED,
    run_id=run.id,
    description="Runtime observed trainable parameters for SharedBlock.",
)

author_claim = Claim(
    id="claim.frozen.author",
    subject_id=shared_definition.id,
    predicate="frozen",
    value=True,
    evidence_ids=[author_evidence.id],
    status=FactStatus.DECLARED,
)
runtime_claim = Claim(
    id="claim.frozen.runtime",
    subject_id=shared_definition.id,
    predicate="frozen",
    value=False,
    evidence_ids=[runtime_evidence.id],
    status=FactStatus.OBSERVED,
    scope=ClaimScope(run_ids=[run.id]),
)

graph = ArchTraceIR(
    project=ProjectInfo(name="atir-v02-example"),
    runs=[run],
    nodes=[shared_definition, *calls],
    evidence=[author_evidence, runtime_evidence],
    claims=[author_claim, runtime_claim],
    conflicts=[
        Conflict(
            id="conflict.freeze",
            claim_ids=[author_claim.id, runtime_claim.id],
            kind=ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH,
        )
    ],
    coverage=[
        CoverageRecord(
            id="coverage.shared",
            subject_id=shared_definition.id,
            status=CoverageStatus.ALWAYS_OBSERVED,
            considered_run_ids=[run.id],
            observed_run_ids=[run.id],
            evidence_ids=[runtime_evidence.id],
        )
    ],
)

print(graph.model_dump_json(indent=2))
