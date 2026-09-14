from __future__ import annotations

from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    ConflictKind,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
    TensorSpec,
)
from archtrace.semantics import (
    ContextSnippet,
    PaperViewPolicy,
    SemanticPhase,
    SemanticRole,
    add_author_claims,
    project_paper_view,
    recover_semantics,
    select_semantic_components,
)


def _vision_graph(requires_grad: bool | None) -> ArchTraceIR:
    module_evidence = Evidence(
        id="evidence.module",
        kind=EvidenceKind.STATIC,
        status=FactStatus.OBSERVED,
        source=SourceSpan(
            path="models/vision.py",
            start_line=10,
            end_line=20,
            symbol="VisionEncoder",
        ),
    )
    parameter_evidence = Evidence(
        id="evidence.parameter",
        kind=EvidenceKind.RUNTIME,
        status=FactStatus.OBSERVED,
        description="Parameter state observed during runtime capture.",
    )
    return ArchTraceIR(
        project=ProjectInfo(name="claim-fixture"),
        evidence=[module_evidence, parameter_evidence],
        nodes=[
            ArchNode(
                id="vision",
                level=NodeLevel.MODULE,
                kind=NodeKind.MODULE,
                identity_kind=IdentityKind.DEFINITION,
                label="VisionEncoder",
                evidence_ids=[module_evidence.id],
                source=[module_evidence.source],
            ),
            ArchNode(
                id="vision.weight",
                level=NodeLevel.OPERATION,
                kind=NodeKind.PARAMETER,
                identity_kind=IdentityKind.STATE,
                label="vision.weight",
                role="pytorch_parameter",
                parent_ids=["vision"],
                evidence_ids=[parameter_evidence.id],
                tensor=TensorSpec(
                    shape=[64, 64],
                    dtype="float32",
                    requires_grad=requires_grad,
                ),
            ),
        ],
    )


def _role_graph(
    nodes: list[tuple[str, str, NodeKind]],
    edges: list[tuple[str, str]] | None = None,
) -> ArchTraceIR:
    evidence: list[Evidence] = []
    arch_nodes: list[ArchNode] = []
    for index, (node_id, label, kind) in enumerate(nodes):
        evidence_id = f"evidence.{index}"
        span = SourceSpan(
            path=f"models/{node_id}.py",
            start_line=index + 1,
            end_line=index + 1,
            symbol=label,
        )
        evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.STATIC,
                status=FactStatus.OBSERVED,
                source=span,
            )
        )
        arch_nodes.append(
            ArchNode(
                id=node_id,
                level=NodeLevel.MODULE,
                kind=kind,
                identity_kind=IdentityKind.DEFINITION,
                label=label,
                evidence_ids=[evidence_id],
                source=[span],
            )
        )
    arch_edges = [
        ArchEdge(
            id=f"edge.{index}",
            source=source,
            target=target,
            kind=EdgeKind.DATA,
        )
        for index, (source, target) in enumerate(edges or [])
    ]
    return ArchTraceIR(
        project=ProjectInfo(name="claim-fixture"),
        nodes=arch_nodes,
        edges=arch_edges,
        evidence=evidence,
    )


def test_author_frozen_claim_conflicts_with_runtime_trainable_parameter() -> None:
    semantic = recover_semantics(_vision_graph(requires_grad=True))
    checked = add_author_claims(
        semantic,
        [
            ContextSnippet(
                path="README.md",
                text="The vision encoder is frozen during training.",
            )
        ],
    )

    author_frozen = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.author") and claim.predicate == "frozen"
    )
    implementation_frozen = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.implementation") and claim.predicate == "frozen"
    )
    assert author_frozen.value is True
    assert implementation_frozen.value is False
    assert author_frozen.metadata["implementation_check"] == "contradicted"
    assert any(
        conflict.kind == ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH
        and set(conflict.claim_ids) == {author_frozen.id, implementation_frozen.id}
        for conflict in checked.conflicts
    )


def test_frozen_claim_without_mechanical_trainability_stays_unresolved() -> None:
    semantic = recover_semantics(_vision_graph(requires_grad=None))
    checked = add_author_claims(
        semantic,
        [
            ContextSnippet(
                path="README.md",
                text="The vision encoder is frozen during training.",
            )
        ],
    )

    author_frozen = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.author") and claim.predicate == "frozen"
    )
    assert author_frozen.metadata["implementation_check"] == "unresolved"
    assert not [
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.implementation") and claim.predicate == "frozen"
    ]
    assert not checked.conflicts


def test_author_component_claim_is_supported_by_recovered_fusion() -> None:
    semantic = recover_semantics(
        _role_graph(
            [
                ("vision", "VisionEncoder", NodeKind.MODULE),
                ("language", "LanguageEncoder", NodeKind.MODULE),
                ("fusion", "CrossModalFusion", NodeKind.MODULE),
            ],
            [("vision", "fusion"), ("language", "fusion")],
        )
    )
    checked = add_author_claims(
        semantic,
        [
            ContextSnippet(
                path="README.md",
                text="The model uses a cross-modal fusion module.",
            )
        ],
    )

    author_claim = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.author")
        and claim.predicate == "component_present"
        and claim.metadata["semantic_role"] == SemanticRole.MULTIMODAL_FUSION.value
    )
    assert author_claim.value is True
    assert author_claim.metadata["implementation_check"] == "supported"
    assert not checked.conflicts


def test_unverified_author_component_becomes_declaration_only_and_not_paper_fact() -> None:
    semantic = recover_semantics(
        _role_graph([("vision", "VisionEncoder", NodeKind.MODULE)])
    )
    checked = add_author_claims(
        semantic,
        [
            ContextSnippet(
                path="README.md",
                text="A planner handles long-horizon decisions.",
            )
        ],
    )

    planner_claim = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.author")
        and claim.metadata["semantic_role"] == SemanticRole.PLANNER.value
    )
    declaration = next(
        node for node in checked.nodes if node.id == planner_claim.subject_id
    )
    assert declaration.attributes["declaration_only"] is True
    assert planner_claim.metadata["implementation_check"] == "unresolved"
    assert not checked.conflicts

    paper = project_paper_view(checked)
    assert SemanticRole.PLANNER.value not in {node.role for node in paper.nodes}


def test_negative_author_component_claim_conflicts_with_recovered_component() -> None:
    semantic = recover_semantics(
        _role_graph([("planner", "TrajectoryPlanner", NodeKind.MODULE)])
    )
    checked = add_author_claims(
        semantic,
        [
            ContextSnippet(
                path="README.md",
                text="The model does not use a planner.",
            )
        ],
    )

    author_claim = next(
        claim
        for claim in checked.claims
        if claim.id.startswith("claim.author")
        and claim.predicate == "component_present"
    )
    assert author_claim.value is False
    assert author_claim.metadata["implementation_check"] == "contradicted"
    assert len(checked.conflicts) == 1


def test_training_and_inference_semantic_views_share_both_phase_components() -> None:
    semantic = recover_semantics(
        _role_graph(
            [
                ("backbone", "TransformerBackbone", NodeKind.MODULE),
                ("loss", "PolicyLoss", NodeKind.LOSS),
                ("action", "ActionHead", NodeKind.MODULE),
            ],
            [("backbone", "loss"), ("backbone", "action")],
        )
    )

    training = select_semantic_components(semantic, phase=SemanticPhase.TRAINING)
    inference = select_semantic_components(semantic, phase=SemanticPhase.INFERENCE)
    training_roles = {node.role for node in training}
    inference_roles = {node.role for node in inference}

    assert SemanticRole.TRANSFORMER_BACKBONE.value in training_roles
    assert SemanticRole.TRANSFORMER_BACKBONE.value in inference_roles
    assert SemanticRole.LOSS.value in training_roles
    assert SemanticRole.LOSS.value not in inference_roles
    assert SemanticRole.ACTION_HEAD.value not in training_roles
    assert SemanticRole.ACTION_HEAD.value in inference_roles

    training_paper = project_paper_view(
        semantic,
        policy=PaperViewPolicy(phases=(SemanticPhase.TRAINING,)),
    )
    inference_paper = project_paper_view(
        semantic,
        policy=PaperViewPolicy(phases=(SemanticPhase.INFERENCE,)),
    )
    assert SemanticRole.LOSS.value in {node.role for node in training_paper.nodes}
    assert SemanticRole.ACTION_HEAD.value not in {
        node.role for node in training_paper.nodes
    }
    assert SemanticRole.ACTION_HEAD.value in {
        node.role for node in inference_paper.nodes
    }
    assert SemanticRole.LOSS.value not in {
        node.role for node in inference_paper.nodes
    }
