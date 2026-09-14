from __future__ import annotations

from pathlib import Path

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
    TensorSpec,
    TraceRun,
)
from archtrace.query.explorer import ExplorerIndex
from archtrace.query.explorer_utils import make_key, parse_key
from archtrace.query.models import (
    ExplorerLevel,
    ExplorerState,
    LineageDirection,
)
from archtrace.query.source import source_pane
from archtrace.query.state import decode_explorer_state, encode_explorer_state
from archtrace.semantics import recover_semantics


def _fixture_graph() -> ArchTraceIR:
    evidence: list[Evidence] = []

    def node(
        node_id: str,
        label: str,
        *,
        level: NodeLevel,
        kind: NodeKind,
        identity: IdentityKind,
        line: int,
        run_id: str | None = None,
        definition_id: str | None = None,
        occurrence_index: int | None = None,
        tensor: TensorSpec | None = None,
    ) -> ArchNode:
        evidence_id = f"evidence.{node_id}"
        span = SourceSpan(
            path="models.py",
            start_line=line,
            end_line=line,
            symbol=label,
        )
        evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.RUNTIME if run_id else EvidenceKind.STATIC,
                status=FactStatus.OBSERVED,
                source=span,
                run_id=run_id,
            )
        )
        return ArchNode(
            id=node_id,
            level=level,
            kind=kind,
            identity_kind=identity,
            label=label,
            run_id=run_id,
            definition_id=definition_id,
            occurrence_index=occurrence_index,
            source=[span],
            evidence_ids=[evidence_id],
            tensor=tensor,
        )

    nodes = [
        node(
            "module.vision",
            "VisionEncoder",
            level=NodeLevel.MODULE,
            kind=NodeKind.MODULE,
            identity=IdentityKind.DEFINITION,
            line=2,
        ),
        node(
            "module.action",
            "ActionHead",
            level=NodeLevel.MODULE,
            kind=NodeKind.MODULE,
            identity=IdentityKind.DEFINITION,
            line=8,
        ),
    ]

    for run_name, suffix, start_line in (
        ("run.a", "a", 3),
        ("run.b", "b", 5),
    ):
        nodes.extend(
            [
                node(
                    f"input.{suffix}",
                    "rgb_input",
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.INPUT,
                    identity=IdentityKind.VALUE,
                    line=start_line,
                    run_id=run_name,
                    tensor=TensorSpec(
                        shape=[1, 3, 224, 224],
                        dtype="float32",
                        device="cpu",
                        semantics=["rgb", "vision"],
                    ),
                ),
                node(
                    f"op.vision.{suffix}",
                    "VisionKernel",
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.OPERATION,
                    identity=IdentityKind.OCCURRENCE,
                    line=start_line,
                    run_id=run_name,
                    definition_id="module.vision",
                    occurrence_index=0,
                ),
                node(
                    f"feature.{suffix}",
                    "visual_features",
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.TENSOR,
                    identity=IdentityKind.VALUE,
                    line=start_line,
                    run_id=run_name,
                    tensor=TensorSpec(
                        shape=[1, 196, 768],
                        dtype="float32",
                        device="cpu",
                        semantics=["visual_tokens"],
                    ),
                ),
                node(
                    f"op.action.{suffix}",
                    "ActionKernel",
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.OPERATION,
                    identity=IdentityKind.OCCURRENCE,
                    line=start_line + 6,
                    run_id=run_name,
                    definition_id="module.action",
                    occurrence_index=0,
                ),
                node(
                    f"output.{suffix}",
                    "robot_action",
                    level=NodeLevel.OPERATION,
                    kind=NodeKind.OUTPUT,
                    identity=IdentityKind.VALUE,
                    line=start_line + 6,
                    run_id=run_name,
                    tensor=TensorSpec(
                        shape=[1, 7],
                        dtype="float32",
                        device="cpu",
                        semantics=["action"],
                    ),
                ),
            ]
        )

    edges: list[ArchEdge] = []

    def add_edge(
        edge_id: str,
        source: str,
        target: str,
        kind: EdgeKind,
    ) -> None:
        edges.append(
            ArchEdge(
                id=edge_id,
                source=source,
                target=target,
                kind=kind,
            )
        )

    for suffix in ("a", "b"):
        add_edge(
            f"contains.vision.{suffix}",
            "module.vision",
            f"op.vision.{suffix}",
            EdgeKind.CONTAINS,
        )
        add_edge(
            f"contains.vision.input.{suffix}",
            f"op.vision.{suffix}",
            f"input.{suffix}",
            EdgeKind.CONTAINS,
        )
        add_edge(
            f"contains.vision.feature.{suffix}",
            f"op.vision.{suffix}",
            f"feature.{suffix}",
            EdgeKind.CONTAINS,
        )
        add_edge(
            f"contains.action.{suffix}",
            "module.action",
            f"op.action.{suffix}",
            EdgeKind.CONTAINS,
        )
        add_edge(
            f"contains.action.output.{suffix}",
            f"op.action.{suffix}",
            f"output.{suffix}",
            EdgeKind.CONTAINS,
        )
        add_edge(
            f"flow.consume.vision.{suffix}",
            f"input.{suffix}",
            f"op.vision.{suffix}",
            EdgeKind.CONSUMES,
        )
        add_edge(
            f"flow.produce.feature.{suffix}",
            f"op.vision.{suffix}",
            f"feature.{suffix}",
            EdgeKind.PRODUCES,
        )
        add_edge(
            f"flow.consume.action.{suffix}",
            f"feature.{suffix}",
            f"op.action.{suffix}",
            EdgeKind.CONSUMES,
        )
        add_edge(
            f"flow.produce.action.{suffix}",
            f"op.action.{suffix}",
            f"output.{suffix}",
            EdgeKind.PRODUCES,
        )

    graph = ArchTraceIR(
        project=ProjectInfo(name="explorer-fixture"),
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        runs=[
            TraceRun(id="run.a", framework="pytorch"),
            TraceRun(id="run.b", framework="pytorch"),
        ],
    )
    return recover_semantics(graph)


def _semantic_id(graph: ArchTraceIR, role: str) -> str:
    return next(
        node.id for node in graph.nodes if node.level == NodeLevel.SEMANTIC and node.role == role
    )


def test_paper_to_semantic_to_module_drilldown_and_breadcrumbs() -> None:
    graph = _fixture_graph()
    index = ExplorerIndex(graph)
    paper = index.paper_slice()
    assert {node.role for node in paper.nodes} >= {"vision_encoder", "action_head"}

    vision_semantic = _semantic_id(graph, "vision_encoder")
    paper_key = make_key(ExplorerLevel.PAPER, vision_semantic)
    l1 = index.expand(paper_key)
    assert {node.level for node in l1.nodes} == {
        ExplorerLevel.PAPER,
        ExplorerLevel.SEMANTIC,
    }

    semantic_key = make_key(ExplorerLevel.SEMANTIC, vision_semantic)
    l2 = index.expand(semantic_key)
    assert any(node.entity_id == "module.vision" for node in l2.nodes)

    module_key = make_key(ExplorerLevel.MODULE, "module.vision")
    l3 = index.expand(module_key, run_id="run.a")
    assert any(node.entity_id == "op.vision.a" for node in l3.nodes)
    assert not any(node.entity_id == "op.vision.b" for node in l3.nodes)

    output_key = make_key(ExplorerLevel.OPERATION, "output.a")
    crumbs = index.breadcrumbs(output_key)
    assert crumbs[0].level == ExplorerLevel.PAPER
    assert crumbs[1].level == ExplorerLevel.SEMANTIC
    assert crumbs[-1].key == output_key
    assert any(crumb.label == "ActionHead" for crumb in crumbs)


def test_search_and_run_filter_distinguish_dynamic_occurrences() -> None:
    index = ExplorerIndex(_fixture_graph())
    run_a = index.search("ActionKernel", run_id="run.a")
    run_b = index.search("ActionKernel", run_id="run.b")
    assert [hit.entity_id for hit in run_a] == ["op.action.a"]
    assert [hit.entity_id for hit in run_b] == ["op.action.b"]
    assert index.available_runs() == ["run.a", "run.b"]


def test_lineage_respects_run_and_bounds() -> None:
    graph = _fixture_graph()
    index = ExplorerIndex(graph)
    input_key = make_key(ExplorerLevel.OPERATION, "input.a")
    downstream = index.lineage(
        input_key,
        direction=LineageDirection.DOWNSTREAM,
        run_id="run.a",
    )
    assert make_key(ExplorerLevel.OPERATION, "output.a") in downstream.node_keys
    assert make_key(ExplorerLevel.OPERATION, "output.b") not in downstream.node_keys
    assert "flow.consume.action.a" in downstream.edge_ids

    bounded = index.lineage(
        input_key,
        direction=LineageDirection.DOWNSTREAM,
        run_id="run.a",
        max_nodes=2,
    )
    assert bounded.truncated is True
    assert len(bounded.node_keys) == 2


def test_semantic_lineage_highlights_underlying_mechanical_path() -> None:
    graph = _fixture_graph()
    index = ExplorerIndex(graph)
    vision_id = _semantic_id(graph, "vision_encoder")
    result = index.lineage(
        make_key(ExplorerLevel.SEMANTIC, vision_id),
        direction=LineageDirection.DOWNSTREAM,
        run_id="run.a",
    )
    assert make_key(ExplorerLevel.OPERATION, "op.action.a") in result.node_keys
    assert make_key(ExplorerLevel.OPERATION, "op.action.b") not in result.node_keys


def test_source_pane_returns_exact_span_and_local_context(tmp_path: Path) -> None:
    graph = _fixture_graph()
    (tmp_path / "models.py").write_text(
        "\n".join(f"line {index}" for index in range(1, 20)) + "\n",
        encoding="utf-8",
    )
    key = make_key(ExplorerLevel.OPERATION, "op.action.a")
    pane = source_pane(graph, key, root=tmp_path, context_lines=1)
    assert len(pane.excerpts) == 1
    excerpt = pane.excerpts[0]
    assert excerpt.start_line == 9
    assert excerpt.excerpt_start_line == 8
    assert excerpt.excerpt_end_line == 10
    assert excerpt.text == "line 8\nline 9\nline 10"


def test_deep_link_state_round_trip_and_key_round_trip() -> None:
    selected = make_key(ExplorerLevel.OPERATION, "op:action/a")
    state = ExplorerState(
        view=ExplorerLevel.OPERATION,
        selected_key=selected,
        run_id="run.a",
        expanded_keys=[
            make_key(ExplorerLevel.MODULE, "module.action"),
            make_key(ExplorerLevel.OPERATION, "op.action.a"),
        ],
        search="action kernel",
    )
    encoded = encode_explorer_state(state)
    assert decode_explorer_state(encoded) == state
    assert parse_key(selected) == (ExplorerLevel.OPERATION, "op:action/a")


def test_large_graph_search_returns_bounded_result_without_materializing_slice() -> None:
    nodes = [
        ArchNode(
            id=f"node.{index}",
            level=NodeLevel.OPERATION,
            kind=NodeKind.OPERATION,
            identity_kind=IdentityKind.DEFINITION,
            label=f"Operator{index}",
        )
        for index in range(10_000)
    ]
    graph = ArchTraceIR(
        project=ProjectInfo(name="large"),
        nodes=nodes,
    )
    index = ExplorerIndex(graph)
    hits = index.search("Operator9999", limit=5)
    assert len(hits) == 1
    assert hits[0].entity_id == "node.9999"
    empty_slice = index.expand(
        make_key(ExplorerLevel.OPERATION, "node.9999"),
        limit=20,
    )
    assert len(empty_slice.nodes) == 1
    assert empty_slice.total_candidates == 0
