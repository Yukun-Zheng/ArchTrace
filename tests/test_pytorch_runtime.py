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


def test_runtime_trace_preserves_shared_module_occurrences() -> None:
    torch.manual_seed(7)
    model = SharedModel()
    x = torch.randn(2, 4)
    expected = model(x)

    result = trace_model(model, (x,), run_id="run.shared")
    graph = result.ir

    assert torch.allclose(result.output, expected)
    assert graph.runs[0].id == "run.shared"
    assert graph.metadata["runtime_backend"] == "pytorch_module_hooks"

    definitions = {
        node.id: node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION
    }
    shared_definition = next(
        node for node in definitions.values() if node.label == "SharedBlock"
    )
    assert shared_definition.attributes["aliases"] == ["model.left", "model.right"]

    shared_calls = [
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.OCCURRENCE
        and node.definition_id == shared_definition.id
    ]
    assert [node.occurrence_index for node in shared_calls] == [0, 1]
    assert all(node.run_id == "run.shared" for node in shared_calls)

    input_nodes = [node for node in graph.nodes if node.kind == NodeKind.INPUT]
    output_nodes = [node for node in graph.nodes if node.kind == NodeKind.OUTPUT]
    assert len(input_nodes) == 1
    assert len(output_nodes) == 1
    assert input_nodes[0].tensor is not None
    assert input_nodes[0].tensor.shape == [2, 4]
    assert output_nodes[0].tensor is not None
    assert output_nodes[0].tensor.shape == [2, 4]

    assert any(edge.kind == EdgeKind.CONSUMES for edge in graph.edges)
    assert any(edge.kind == EdgeKind.PRODUCES for edge in graph.edges)
    assert any(edge.kind == EdgeKind.CONTAINS for edge in graph.edges)


def test_runtime_trace_records_source_and_framework_metadata() -> None:
    model = SharedModel()
    result = trace_model(model, (torch.ones(1, 4),))
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
