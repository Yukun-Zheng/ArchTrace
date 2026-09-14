"""Frontend-neutral models for the hierarchical ArchTrace explorer."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExplorerLevel(StrEnum):
    PAPER = "paper"
    SEMANTIC = "semantic"
    MODULE = "module"
    OPERATION = "operation"
    SOURCE = "source"


class LineageDirection(StrEnum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


class ExplorerSourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int
    end_line: int | None = None
    start_column: int | None = None
    end_column: int | None = None
    symbol: str | None = None


class ExplorerNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    entity_id: str
    level: ExplorerLevel
    label: str
    kind: str
    role: str | None = None
    identity_kind: str | None = None
    child_count: int = Field(default=0, ge=0)
    run_id: str | None = None
    occurrence_index: int | None = None
    definition_id: str | None = None
    source: list[ExplorerSourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplorerEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    source_key: str
    target_key: str
    kind: str
    label: str | None = None
    entity_edge_ids: list[str] = Field(default_factory=list)


class ExplorerSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_key: str | None = None
    nodes: list[ExplorerNode] = Field(default_factory=list)
    edges: list[ExplorerEdge] = Field(default_factory=list)
    truncated: bool = False
    total_candidates: int = Field(default=0, ge=0)


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    entity_id: str
    label: str
    level: ExplorerLevel
    kind: str
    role: str | None = None
    source_path: str | None = None
    score: int = Field(ge=0)


class LineageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_key: str
    direction: LineageDirection
    node_keys: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    truncated: bool = False


class SourceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    text: str | None = None
    excerpt_start_line: int | None = None
    excerpt_end_line: int | None = None


class SourcePane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    excerpts: list[SourceExcerpt] = Field(default_factory=list)


class Breadcrumb(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    level: ExplorerLevel


class ExplorerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view: ExplorerLevel = ExplorerLevel.PAPER
    selected_key: str | None = None
    run_id: str | None = None
    expanded_keys: list[str] = Field(default_factory=list)
    search: str | None = None
