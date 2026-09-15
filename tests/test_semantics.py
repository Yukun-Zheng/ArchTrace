from __future__ import annotations

from pathlib import Path
from time import perf_counter

from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
)
from archtrace.semantics import (
    Modality,
    SemanticOverride,
    SemanticRole,
    collect_repository_context,
    project_paper_view,
    recover_semantics,
)


def _graph(
    nodes: list[tuple[str, str, NodeKind]],
    edges: list[tuple[str, str]],
) -> ArchTraceIR:
    evidence: list[Evidence] = []
    arch_nodes: list[ArchNode] = []
    for index, (node_id, label, kind) in enumerate(nodes):
        evidence_id = f"evidence.node.{index}"
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
                source=[span],
                evidence_ids=[evidence_id],
            )
        )

    arch_edges = [
        ArchEdge(
            id=f"edge.data.{index}",
            source=source,
            target=target,
            kind=EdgeKind.DATA,
        )
        for index, (source, target) in enumerate(edges)
    ]
    return ArchTraceIR(
        project=ProjectInfo(name="semantic-fixture"),
        nodes=arch_nodes,
        edges=arch_edges,
        evidence=evidence,
    )


def _semantic_by_role(graph: ArchTraceIR, role: SemanticRole) -> ArchNode:
    return next(
        node for node in graph.nodes if node.level == NodeLevel.SEMANTIC and node.role == role.value
    )


def test_cnn_classifier_recovers_semantics_and_grounded_paper_edge() -> None:
    source = _graph(
        [
            ("vision", "ResNetBackbone", NodeKind.MODULE),
            ("head", "ClassifierHead", NodeKind.MODULE),
        ],
        [("vision", "head")],
    )
    recovered = recover_semantics(source)

    vision = _semantic_by_role(recovered, SemanticRole.VISION_ENCODER)
    head = _semantic_by_role(recovered, SemanticRole.CLASSIFIER_HEAD)
    assert vision.attributes["member_ids"] == ["vision"]
    assert head.attributes["member_ids"] == ["head"]
    assert "vision" in vision.attributes["modalities"]

    paper = project_paper_view(recovered)
    paper_edges = {(edge.source, edge.target): edge for edge in paper.edges}
    edge = paper_edges[(vision.id, head.id)]
    assert edge.evidence_edge_ids == ["edge.data.0"]


def test_transformer_roles_are_deterministic() -> None:
    source = _graph(
        [
            ("tokens", "ImageTokenizer", NodeKind.MODULE),
            ("backbone", "TransformerEncoder", NodeKind.MODULE),
            ("head", "ClassificationHead", NodeKind.MODULE),
        ],
        [("tokens", "backbone"), ("backbone", "head")],
    )
    first = recover_semantics(source)
    second = recover_semantics(source)

    assert _semantic_by_role(first, SemanticRole.TOKENIZER)
    assert _semantic_by_role(first, SemanticRole.TRANSFORMER_BACKBONE)
    assert _semantic_by_role(first, SemanticRole.CLASSIFIER_HEAD)
    assert project_paper_view(first) == project_paper_view(second)


def test_multimodal_vla_propagates_modalities_through_fusion_to_action() -> None:
    source = _graph(
        [
            ("vision", "VisionEncoder", NodeKind.MODULE),
            ("language", "LanguageEncoder", NodeKind.MODULE),
            ("fusion", "CrossModalFusion", NodeKind.MODULE),
            ("denoiser", "DiffusionDenoiser", NodeKind.MODULE),
            ("action", "ActionHead", NodeKind.MODULE),
        ],
        [
            ("vision", "fusion"),
            ("language", "fusion"),
            ("fusion", "denoiser"),
            ("denoiser", "action"),
        ],
    )
    recovered = recover_semantics(source)

    fusion = _semantic_by_role(recovered, SemanticRole.MULTIMODAL_FUSION)
    denoiser = _semantic_by_role(recovered, SemanticRole.DIFFUSION_DENOISER)
    action = _semantic_by_role(recovered, SemanticRole.ACTION_HEAD)
    assert fusion.attributes["confidence"] >= 0.9
    assert {"vision", "language"} <= set(fusion.attributes["modalities"])
    assert {"vision", "language", "action"} <= set(action.attributes["modalities"])
    assert denoiser.attributes["phase"] == "both"
    assert action.attributes["phase"] == "inference"


def test_agent_environment_loop_recovers_policy_controller_environment() -> None:
    source = _graph(
        [
            ("policy", "RobotPolicy", NodeKind.MODULE),
            ("controller", "LowLevelController", NodeKind.MODULE),
            ("environment", "RobotEnvironment", NodeKind.ENVIRONMENT),
            ("state", "RobotState", NodeKind.MODULE),
        ],
        [
            ("policy", "controller"),
            ("controller", "environment"),
            ("environment", "state"),
            ("state", "policy"),
        ],
    )
    recovered = recover_semantics(source)

    assert _semantic_by_role(recovered, SemanticRole.POLICY)
    assert _semantic_by_role(recovered, SemanticRole.CONTROLLER)
    environment = _semantic_by_role(recovered, SemanticRole.ENVIRONMENT)
    assert environment.attributes["phase"] == "inference"


def test_unknown_block_stays_unclassified_instead_of_getting_a_fake_role() -> None:
    source = _graph(
        [("mystery", "MysteryBlock7", NodeKind.MODULE)],
        [],
    )
    recovered = recover_semantics(source)

    assert not [node for node in recovered.nodes if node.level == NodeLevel.SEMANTIC]
    entry = recovered.metadata["semantic"]["unclassified"]["mystery"]
    assert entry["status"] == "unknown"
    assert entry["candidates"] == []


def test_user_override_preserves_machine_hypotheses() -> None:
    source = _graph(
        [("state", "StateEncoder", NodeKind.MODULE)],
        [],
    )
    recovered = recover_semantics(
        source,
        overrides=[
            SemanticOverride(
                member_ids=("state",),
                role=SemanticRole.WORLD_MODEL,
                label="Learned Dynamics",
                modalities=(Modality.STATE,),
                reason="validated by experiment and author annotation",
            )
        ],
    )

    semantic = _semantic_by_role(recovered, SemanticRole.WORLD_MODEL)
    assert semantic.label == "Learned Dynamics"
    assert semantic.attributes["inference_backend"] == "user_override"
    assert any(
        item["role"] == SemanticRole.GENERIC_ENCODER.value
        for item in semantic.attributes["machine_hypotheses"]
    )
    evidence = next(item for item in recovered.evidence if item.id in semantic.evidence_ids)
    assert evidence.kind == EvidenceKind.USER
    assert evidence.status == FactStatus.CORRECTED


def test_semantic_recovery_does_not_fabricate_mechanical_data_edges() -> None:
    source = _graph(
        [
            ("vision", "VisionEncoder", NodeKind.MODULE),
            ("head", "ActionHead", NodeKind.MODULE),
        ],
        [("vision", "head")],
    )
    recovered = recover_semantics(source)

    original_data = {edge.id for edge in source.edges if edge.kind == EdgeKind.DATA}
    recovered_data = {edge.id for edge in recovered.edges if edge.kind == EdgeKind.DATA}
    assert recovered_data == original_data
    original_edge_ids = {edge.id for edge in source.edges}
    new_edges = [edge for edge in recovered.edges if edge.id not in original_edge_ids]
    assert new_edges
    assert all(edge.kind == EdgeKind.CONTAINS for edge in new_edges)


def test_loss_phase_is_training() -> None:
    source = _graph(
        [("loss", "PolicyLoss", NodeKind.LOSS)],
        [],
    )
    recovered = recover_semantics(source)
    loss = _semantic_by_role(recovered, SemanticRole.LOSS)
    assert loss.attributes["phase"] == "training"


def test_repository_context_collects_readme_then_docs_deterministically(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("# Demo\nVision policy", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text(
        "Cross-modal fusion details",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("not selected", encoding="utf-8")

    snippets = collect_repository_context(tmp_path)
    assert [snippet.path for snippet in snippets] == [
        "README.md",
        "docs/architecture.md",
    ]
    assert "Vision policy" in snippets[0].text


def test_large_semantic_recovery_remains_bounded() -> None:
    size = 4000
    source = _graph(
        [
            (f"vision_{index}", f"VisionEncoder{index}", NodeKind.MODULE)
            for index in range(size)
        ],
        [(f"vision_{index}", f"vision_{index + 1}") for index in range(size - 1)],
    )
    started = perf_counter()
    recovered = recover_semantics(source)
    elapsed = perf_counter() - started
    semantic_nodes = [node for node in recovered.nodes if node.level == NodeLevel.SEMANTIC]
    assert len(semantic_nodes) == size
    assert elapsed < 8.0
