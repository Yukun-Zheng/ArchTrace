from __future__ import annotations

import math

import pytest

from archtrace.ir import EdgeKind, IdentityKind
from archtrace.runtime import trace_model

torch = pytest.importorskip("torch")


class StructuredMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TinyCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.head(x.flatten(1))


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(8, 8, bias=False)
        self.k = torch.nn.Linear(8, 8, bias=False)
        self.v = torch.nn.Linear(8, 8, bias=False)
        self.out = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(8)
        weights = torch.softmax(scores, dim=-1)
        return self.out(torch.matmul(weights, v))


class BranchModel(torch.nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:
            return x + 1
        return x - 1


class LoopModel(torch.nn.Module):
    def forward(self, x):
        for _ in range(3):
            x = torch.relu(x + 1)
        return x


def _reports(graph):
    return {item["backend"]: item for item in graph.metadata["structured_captures"]}


def test_fx_and_export_are_independent_aligned_evidence_sources() -> None:
    result = trace_model(StructuredMLP(), (torch.ones(2, 4),), run_id="run.structured")
    graph = result.ir
    reports = _reports(graph)

    assert reports["fx"]["status"] == "success"
    assert reports["export"]["status"] == "success"
    assert reports["fx"]["nodes"] > 0
    assert reports["export"]["nodes"] > 0
    assert reports["fx"]["aligned_nodes"] > 0
    assert reports["export"]["aligned_nodes"] > 0

    fx_nodes = [node for node in graph.nodes if node.role == "pytorch_fx_node"]
    export_nodes = [node for node in graph.nodes if node.role == "pytorch_export_node"]
    assert fx_nodes
    assert export_nodes
    assert any(edge.kind == EdgeKind.DATA for edge in graph.edges)
    assert any(
        edge.kind == EdgeKind.DERIVED_FROM
        and edge.source.startswith(("structured.fx.", "structured.export."))
        for edge in graph.edges
    )


def test_structured_backend_failure_does_not_break_runtime_trace(monkeypatch) -> None:
    def fail_fx(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("intentional fx failure")

    monkeypatch.setattr(torch.fx, "symbolic_trace", fail_fx)
    result = trace_model(
        StructuredMLP(),
        (torch.ones(1, 4),),
        capture_export=False,
    )
    graph = result.ir
    reports = _reports(graph)

    assert reports["fx"]["status"] == "failed"
    assert reports["fx"]["error_type"] == "RuntimeError"
    assert "intentional fx failure" in reports["fx"]["error"]
    assert any(
        node.identity_kind == IdentityKind.OCCURRENCE and node.role == "pytorch_operator_call"
        for node in graph.nodes
    )


def test_cnn_runtime_trace_reaches_output_through_tensor_flow() -> None:
    result = trace_model(
        TinyCNN(),
        (torch.randn(2, 1, 4, 4),),
        capture_fx=False,
        capture_export=False,
    )
    assert tuple(result.output.shape) == (2, 3)
    assert any("convolution" in node.label or "conv" in node.label for node in result.ir.nodes)


def test_attention_style_model_records_matmul_and_softmax() -> None:
    result = trace_model(
        TinyAttention(),
        (torch.randn(2, 5, 8),),
        capture_fx=False,
        capture_export=False,
    )
    labels = {node.label for node in result.ir.nodes if node.role == "pytorch_operator_definition"}
    assert any("matmul" in label or "bmm" in label for label in labels)
    assert any("softmax" in label for label in labels)
    assert tuple(result.output.shape) == (2, 5, 8)


def test_data_dependent_branch_records_only_observed_branch() -> None:
    result = trace_model(
        BranchModel(),
        (torch.ones(2, 2),),
        capture_fx=False,
        capture_export=False,
    )
    labels = [node.label for node in result.ir.nodes if node.role == "pytorch_operator_definition"]
    assert torch.equal(result.output, torch.full((2, 2), 2.0))
    assert any("add" in label for label in labels)
    assert not any("sub" in label for label in labels)


def test_loop_keeps_three_concrete_operator_occurrences() -> None:
    result = trace_model(
        LoopModel(),
        (torch.zeros(1),),
        capture_fx=False,
        capture_export=False,
    )
    occurrences = [node for node in result.ir.nodes if node.role == "pytorch_operator_call"]
    by_definition: dict[str, list[int | None]] = {}
    for node in occurrences:
        assert node.definition_id is not None
        by_definition.setdefault(node.definition_id, []).append(node.occurrence_index)
    assert any(indices == [0, 1, 2] for indices in by_definition.values())
