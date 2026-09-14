"""Canonical ArchTrace Intermediate Representation (ATIR).

ATIR intentionally separates *mechanics* from *interpretation* while keeping both
connected through explicit evidence. Runtime observations, static inferences,
author declarations, semantic annotations, and user corrections can coexist
without one silently overwriting another.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeLevel(StrEnum):
    PAPER = "paper"
    SEMANTIC = "semantic"
    MODULE = "module"
    OPERATION = "operation"
    SOURCE = "source"


class NodeKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    SEMANTIC_COMPONENT = "semantic_component"
    MODULE = "module"
    OPERATION = "operation"
    TENSOR = "tensor"
    PARAMETER = "parameter"
    SOURCE = "source"
    CONFIG = "config"
    CONTROL = "control"
    DATASET = "dataset"
    LOSS = "loss"
    ENVIRONMENT = "environment"
    OTHER = "other"


class EdgeKind(StrEnum):
    DATA = "data"
    CONTROL = "control"
    CONTAINS = "contains"
    CALLS = "calls"
    IMPLEMENTS = "implements"
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    ALIAS = "alias"
    READS = "reads"
    WRITES = "writes"
    DERIVED_FROM = "derived_from"
    NEXT = "next"


class EvidenceKind(StrEnum):
    RUNTIME = "runtime"
    STATIC = "static"
    SOURCE = "source"
    AUTHOR = "author"
    SEMANTIC = "semantic"
    USER = "user"
    IMPORTED = "imported"


class FactStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    DECLARED = "declared"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    start_column: int | None = Field(default=None, ge=0)
    end_column: int | None = Field(default=None, ge=0)
    symbol: str | None = None

    @model_validator(mode="after")
    def validate_line_range(self) -> SourceSpan:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class TensorSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    shape: list[int | str | None] | None = None
    dtype: str | None = None
    device: str | None = None
    layout: str | None = None
    requires_grad: bool | None = None
    semantics: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    kind: EvidenceKind
    status: FactStatus
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str | None = None
    source: SourceSpan | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    level: NodeLevel
    kind: NodeKind
    label: str
    role: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    source: list[SourceSpan] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    tensor: TensorSpec | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArchEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    tensor: TensorSpec | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    entrypoint: str | None = None
    argv: list[str] = Field(default_factory=list)
    config_paths: list[str] = Field(default_factory=list)
    framework: str | None = None
    framework_version: str | None = None
    python_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    root: str | None = None
    repository_url: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchTraceIR(BaseModel):
    """One canonical evidence graph from which all ArchTrace views are projected."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    project: ProjectInfo
    nodes: list[ArchNode] = Field(default_factory=list)
    edges: list[ArchEdge] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    runs: list[TraceRun] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph_references(self) -> ArchTraceIR:
        node_ids = [node.id for node in self.nodes]
        evidence_ids = [item.id for item in self.evidence]
        run_ids = [run.id for run in self.runs]

        _require_unique("node", node_ids)
        _require_unique("evidence", evidence_ids)
        _require_unique("run", run_ids)
        _require_unique("edge", [edge.id for edge in self.edges])

        node_set = set(node_ids)
        evidence_set = set(evidence_ids)
        run_set = set(run_ids)

        for node in self.nodes:
            _require_known(node.parent_ids, node_set, f"node {node.id} parent")
            _require_known(node.evidence_ids, evidence_set, f"node {node.id} evidence")

        for edge in self.edges:
            _require_known([edge.source, edge.target], node_set, f"edge {edge.id} endpoint")
            _require_known(edge.evidence_ids, evidence_set, f"edge {edge.id} evidence")

        for item in self.evidence:
            if item.run_id is not None and item.run_id not in run_set:
                raise ValueError(f"evidence {item.id} references unknown run {item.run_id}")

        return self


def _require_unique(kind: str, values: list[str]) -> None:
    duplicates = {value for value in values if values.count(value) > 1}
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {kind} id(s): {joined}")


def _require_known(values: list[str], known: set[str], description: str) -> None:
    missing = sorted(set(values) - known)
    if missing:
        raise ValueError(f"{description} references unknown id(s): {', '.join(missing)}")
