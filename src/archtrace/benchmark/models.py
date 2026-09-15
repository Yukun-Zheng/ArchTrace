"""Versioned models for ArchTrace real-world benchmarks."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class BenchmarkMode(StrEnum):
    STATIC = "static"
    RUNTIME = "runtime"
    HYBRID = "hybrid"


class BenchmarkTier(StrEnum):
    CALIBRATION = "calibration"
    RESEARCH = "research"
    SYSTEM = "system"


class BenchmarkStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class FailureSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class FailureCategory(StrEnum):
    MISSING_REPOSITORY = "missing_repository"
    REVISION_MISMATCH = "revision_mismatch"
    PARSE_ERROR = "parse_error"
    DYNAMIC_CALL = "dynamic_call"
    UNRESOLVED_CALL = "unresolved_call"
    CONFIG_DYNAMIC = "config_dynamic"
    CONFIG_UNRESOLVED = "config_unresolved"
    SUBMODULE_UNMATERIALIZED = "submodule_unmaterialized"
    ANALYSIS_EXCEPTION = "analysis_exception"


class BenchmarkCase(BaseModel):
    id: str
    repository: str
    revision: str = Field(min_length=7)
    branch: str | None = None
    tier: BenchmarkTier = BenchmarkTier.RESEARCH
    modes: list[BenchmarkMode] = Field(default_factory=lambda: [BenchmarkMode.STATIC])
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkManifest(BaseModel):
    schema_version: str = "1"
    cases: list[BenchmarkCase]

    def case(self, case_id: str) -> BenchmarkCase:
        matches = [case for case in self.cases if case.id == case_id]
        if len(matches) != 1:
            raise KeyError(f"benchmark case {case_id!r} not found uniquely")
        return matches[0]


class BenchmarkFailure(BaseModel):
    category: FailureCategory
    severity: FailureSeverity
    message: str
    count: int = 1
    examples: list[str] = Field(default_factory=list)


class StaticBenchmarkMetrics(BaseModel):
    python_files: int
    parse_errors: int
    parse_success_rate: float
    symbols: int
    calls: int
    local_calls: int
    external_calls: int
    dynamic_calls: int
    unresolved_calls: int
    call_resolution_rate: float
    config_entries: int
    config_references: int
    dynamic_config_references: int = 0
    unresolved_config_references: int
    entrypoints: int
    dataflow_edges: int
    atir_nodes: int
    atir_edges: int
    source_span_nodes: int
    source_span_coverage: float
    semantic_components: int
    semantic_unclassified_nodes: int
    semantic_coverage: float
    declared_submodules: int = 0
    unmaterialized_submodules: int = 0


class BenchmarkResult(BaseModel):
    schema_version: str = "1"
    case_id: str
    repository: str
    expected_revision: str
    actual_revision: str | None = None
    analyzer_revision: str | None = None
    analyzer_dirty: bool | None = None
    mode: BenchmarkMode
    status: BenchmarkStatus
    elapsed_seconds: float
    metrics: StaticBenchmarkMetrics | None = None
    failures: list[BenchmarkFailure] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
