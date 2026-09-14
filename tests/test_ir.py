from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from archtrace.ingest.repository import build_static_ir, scan_repository
from archtrace.ir import (
    ArchEdge,
    ArchNode,
    ArchTraceIR,
    EdgeKind,
    NodeKind,
    NodeLevel,
    ProjectInfo,
    SourceSpan,
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
    assert graph.schema_version == "0.1"
    assert any(node.label == "TinyPolicy" for node in graph.nodes)
    assert any(edge.kind == EdgeKind.CONTAINS for edge in graph.edges)

    round_trip = ArchTraceIR.model_validate_json(graph.model_dump_json())
    assert round_trip.project.name == tmp_path.name
