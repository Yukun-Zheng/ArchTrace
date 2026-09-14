from __future__ import annotations

import pytest

from archtrace.ir import IdentityKind
from archtrace.query import (
    find_runtime_operator_path,
    runtime_input_ids,
    runtime_output_ids,
)
from archtrace.runtime import trace_model

torch = pytest.importorskip("torch")


class AcceptanceMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_fine_grained_operator_lineage_connects_input_to_output() -> None:
    result = trace_model(
        AcceptanceMLP(),
        (torch.randn(3, 8),),
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir
    inputs = runtime_input_ids(graph)
    outputs = runtime_output_ids(graph)

    assert len(inputs) == 1
    assert len(outputs) == 1
    path = find_runtime_operator_path(graph, inputs[0], outputs[0])
    assert path is not None
    assert len(path) >= 5

    nodes = {node.id: node for node in graph.nodes}
    assert any(
        nodes[node_id].identity_kind == IdentityKind.OCCURRENCE
        and nodes[node_id].role == "pytorch_operator_call"
        for node_id in path
    )
    assert not any(
        nodes[node_id].role == "pytorch_module_call" for node_id in path
    )


def test_native_transformer_encoder_layer_traces_end_to_end() -> None:
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    x = torch.randn(2, 4, 8)
    result = trace_model(
        layer,
        (x,),
        run_id="run.transformer",
        capture_fx=False,
        capture_export=False,
    )
    graph = result.ir

    assert tuple(result.output.shape) == (2, 4, 8)
    op_calls = [
        node
        for node in graph.nodes
        if node.identity_kind == IdentityKind.OCCURRENCE
        and node.role == "pytorch_operator_call"
    ]
    assert len(op_calls) >= 10

    labels = {
        node.label
        for node in graph.nodes
        if node.identity_kind == IdentityKind.DEFINITION
        and node.role == "pytorch_operator_definition"
    }
    assert any("softmax" in label for label in labels)
    assert any(
        token in label
        for label in labels
        for token in ("mm", "addmm", "bmm", "scaled_dot_product")
    )

    inputs = runtime_input_ids(graph)
    outputs = runtime_output_ids(graph)
    assert len(inputs) == 1
    assert len(outputs) == 1
    assert find_runtime_operator_path(graph, inputs[0], outputs[0]) is not None
