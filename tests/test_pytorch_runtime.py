from __future__ import annotations

import pytest

from archtrace.ir import EdgeKind, IdentityKind, NodeKind
from archtrace.runtime import trace_model

torch = pytest.importorskip("torch")


class SharedBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return torch.relu(self.linear(x))


class SharedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = SharedBlock()
        self.left = shared
        self.right = shared

    def forward(self, x):
        return self.right(self.left(x))


class InplaceModel(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y.add_(1)
        return y


def test_runtime_trace_preserves_shared_module_occurrences() -> None:
    torch.manual_seed(7)
    model = SharedModel()
    x = torch.randn(2, 4)
    expected = model(x)

    result = trace_model(
        model,
        (x,),
        run_id="run.shared",
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir

    assert torch.allclose(result.output, expected)
    assert graph.runs[0].id == "run.shared"
    assert graph.metadata["runtime_backend"] == "pytorch_module_hooks"
    assert graph.metadata["operator_dispatch"] is True

    definitions = {
        node.id: node for node in graph.nodes if node.identity_kind == IdentityKind.DEFINITION
    }
    shared_definition = next(node for node in definitions.values() if node.label == "SharedBlock")
    assert shared_definition.attributes["aliases"] == ["model.left", "model.right"]

    shared_calls = [
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.OCCURRENCE
        and node.definition_id == shared_definition.id
    ]
    assert [node.occurrence_index for node in shared_calls] == [0, 1]
    assert all(node.run_id == "run.shared" for node in shared_calls)

    input_nodes = [
        node
        for node in graph.nodes
        if node.kind == NodeKind.INPUT and node.identity_kind == IdentityKind.VALUE
    ]
    output_nodes = [
        node
        for node in graph.nodes
        if node.kind == NodeKind.OUTPUT and node.identity_kind == IdentityKind.VALUE
    ]
    assert len(input_nodes) == 1
    assert len(output_nodes) == 1
    assert input_nodes[0].tensor is not None
    assert input_nodes[0].tensor.shape == [2, 4]
    assert output_nodes[0].tensor is not None
    assert output_nodes[0].tensor.shape == [2, 4]

    assert any(edge.kind == EdgeKind.CONSUMES for edge in graph.edges)
    assert any(edge.kind == EdgeKind.PRODUCES for edge in graph.edges)
    assert any(edge.kind == EdgeKind.CONTAINS for edge in graph.edges)


def test_operator_dispatch_records_aten_occurrences_and_parameter_flow() -> None:
    model = SharedModel()
    result = trace_model(
        model,
        (torch.ones(2, 4),),
        run_id="run.ops",
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir

    op_definitions = [
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION
        and node.role == "pytorch_operator_definition"
    ]
    op_occurrences = [
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.OCCURRENCE and node.role == "pytorch_operator_call"
    ]
    assert op_definitions
    assert op_occurrences
    assert all(node.run_id == "run.ops" for node in op_occurrences)
    assert any("relu" in node.label for node in op_definitions)

    occurrences_by_definition: dict[str, list[int | None]] = {}
    for node in op_occurrences:
        assert node.definition_id is not None
        occurrences_by_definition.setdefault(node.definition_id, []).append(node.occurrence_index)
    assert any(indices == [0, 1] for indices in occurrences_by_definition.values())

    state_nodes = [node for node in graph.nodes if node.identity_kind == IdentityKind.STATE]
    weight = next(
        node
        for node in state_nodes
        if node.role == "pytorch_parameter" and node.label.endswith("linear.weight")
    )
    assert weight.tensor is not None
    assert weight.tensor.shape == [4, 4]
    assert weight.attributes["aliases"] == [
        "model.left.linear.weight",
        "model.right.linear.weight",
    ]

    op_ids = {node.id for node in op_occurrences}
    assert any(
        edge.kind == EdgeKind.CONSUMES and edge.source == weight.id and edge.target in op_ids
        for edge in graph.edges
    )


def test_inplace_operator_creates_a_new_value_version() -> None:
    model = InplaceModel()
    result = trace_model(
        model,
        (torch.zeros(2, 3),),
        run_id="run.inplace",
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir

    op_definitions = {
        node.id: node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION
        and node.role == "pytorch_operator_definition"
    }
    add_call = next(
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.OCCURRENCE
        and node.role == "pytorch_operator_call"
        and node.definition_id is not None
        and "add_" in op_definitions[node.definition_id].label
    )

    consumed = {
        edge.source
        for edge in graph.edges
        if edge.kind == EdgeKind.CONSUMES and edge.target == add_call.id
    }
    produced = {
        edge.target
        for edge in graph.edges
        if edge.kind == EdgeKind.PRODUCES and edge.source == add_call.id
    }
    assert consumed
    assert produced
    assert consumed.isdisjoint(produced)
    assert any(
        edge.kind == EdgeKind.DERIVED_FROM and edge.source in consumed and edge.target in produced
        for edge in graph.edges
    )


def test_runtime_trace_records_source_and_framework_metadata() -> None:
    model = SharedModel()
    result = trace_model(
        model,
        (torch.ones(1, 4),),
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir

    assert graph.runs[0].framework == "pytorch"
    assert graph.runs[0].framework_version
    assert graph.runs[0].python_version

    shared_definition = next(
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION and node.label == "SharedBlock"
    )
    assert shared_definition.source
    assert shared_definition.source[0].symbol is not None
    assert "SharedBlock.forward" in shared_definition.source[0].symbol
