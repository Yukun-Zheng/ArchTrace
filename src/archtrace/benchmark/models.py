"""Versioned models for ArchTrace real-world benchmarks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

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
    RUNTIME_SPEC_INVALID = "runtime_spec_invalid"
    RUNTIME_UNSUPPORTED_FRAMEWORK = "runtime_unsupported_framework"
    RUNTIME_DEPENDENCY_MISSING = "runtime_dependency_missing"
    RUNTIME_IMPORT_FAILURE = "runtime_import_failure"
    RUNTIME_CONSTRUCTION_FAILURE = "runtime_construction_failure"
    RUNTIME_EXECUTION_FAILURE = "runtime_execution_failure"
    RUNTIME_OUTPUT_MISMATCH = "runtime_output_mismatch"
    RUNTIME_ENVIRONMENT_MISMATCH = "runtime_environment_mismatch"
    RUNTIME_OVERLAY_MISMATCH = "runtime_overlay_mismatch"
    HYBRID_ALIGNMENT_GAP = "hybrid_alignment_gap"
    HYBRID_RECONCILIATION_FAILURE = "hybrid_reconciliation_failure"
    SYSTEM_RUNTIME_LAUNCH_FAILURE = "system_runtime_launch_failure"
    SYSTEM_RUNTIME_EXECUTION_FAILURE = "system_runtime_execution_failure"
    SYSTEM_RUNTIME_TRACE_INVALID = "system_runtime_trace_invalid"
    SYSTEM_RUNTIME_RESOURCE_MISMATCH = "system_runtime_resource_mismatch"


class BenchmarkCase(BaseModel):
    id: str
    repository: str
    revision: str = Field(min_length=7)
    branch: str | None = None
    tier: BenchmarkTier = BenchmarkTier.RESEARCH
    modes: list[BenchmarkMode] = Field(default_factory=lambda: [BenchmarkMode.STATIC])
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    runtime_spec: str | None = None
    runtime_environment: str | None = None
    system_spec: str | None = None


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


class RuntimeValueSpec(BaseModel):
    kind: Literal["tensor", "scalar", "mapping", "sequence", "object"] = "tensor"
    shape: list[int] = Field(default_factory=list)
    dtype: str = "float32"
    generator: Literal["randn", "zeros", "ones", "values"] = "randn"
    values: list[int | float | bool] = Field(default_factory=list)
    value: int | float | bool | str | None = None
    requires_grad: bool = False
    items: dict[str, RuntimeValueSpec] = Field(default_factory=dict)
    elements: list[RuntimeValueSpec] = Field(default_factory=list)
    sequence_type: Literal["list", "tuple"] = "list"
    module: str | None = None
    symbol: str | None = None
    constructor_args: list[RuntimeValueSpec] = Field(default_factory=list)
    constructor_kwargs: dict[str, RuntimeValueSpec] = Field(default_factory=dict)


class RuntimePackageRequirement(BaseModel):
    distribution: str
    version: str


class RuntimeSourceOverlay(BaseModel):
    repository_path: str
    module: str
    require_private_copy: bool = False


class RuntimeEnvironmentSpec(BaseModel):
    schema_version: str = "1"
    python: str | None = None
    packages: list[RuntimePackageRequirement] = Field(default_factory=list)
    overlays: list[RuntimeSourceOverlay] = Field(default_factory=list)


class RuntimeTargetSpec(BaseModel):
    schema_version: str = "1"
    framework: str = "pytorch"
    import_root: str = "."
    module: str
    symbol: str
    constructor_kwargs: dict[str, Any] = Field(default_factory=dict)
    args: list[RuntimeValueSpec] = Field(default_factory=list)
    kwargs: dict[str, RuntimeValueSpec] = Field(default_factory=dict)
    required_imports: list[str] = Field(default_factory=list)
    seed: int = 7
    eval_mode: bool = True
    no_grad: bool = True
    capture_operators: bool = True
    capture_fx: bool = False
    capture_export: bool = False
    torch_num_threads: int | None = Field(default=None, ge=1)
    torch_num_interop_threads: int | None = Field(default=None, ge=1)
    expected_output_shapes: list[list[int]] = Field(default_factory=list)


class PythonProbeSpec(BaseModel):
    module: str
    symbol: str
    role: str = "python"
    boundary: str = "local"


class SystemRuntimeResource(BaseModel):
    path: str
    kind: Literal["file", "directory"] = "file"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PythonSystemSpec(BaseModel):
    schema_version: str = "1"
    framework: Literal["python"] = "python"
    scenario: str
    python_paths: list[str] = Field(default_factory=lambda: ["."])
    probes: list[PythonProbeSpec]
    required_imports: list[str] = Field(default_factory=list)
    resources: list[SystemRuntimeResource] = Field(default_factory=list)


class SystemRuntimeBenchmarkMetrics(BaseModel):
    metric_kind: Literal["system_runtime"] = "system_runtime"
    trace_seconds: float
    runtime_nodes: int
    runtime_edges: int
    definitions: int
    source_definitions: int
    external_definitions: int
    occurrences: int
    values: int
    tensor_values: int
    tensor_spec_coverage: float
    adapter_occurrences: int
    transport_occurrences: int
    environment_occurrences: int
    consumes_edges: int
    produces_edges: int
    calls_edges: int
    next_edges: int


class StaticBenchmarkMetrics(BaseModel):
    metric_kind: Literal["static"] = "static"
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


class RuntimeBenchmarkMetrics(BaseModel):
    metric_kind: Literal["runtime"] = "runtime"
    trace_seconds: float
    runtime_nodes: int
    runtime_edges: int
    definitions: int
    module_definitions: int
    operator_definitions: int
    occurrences: int
    values: int
    states: int
    target_source_definitions: int
    target_operator_source_definitions: int = 0
    tensor_nodes: int
    tensor_spec_nodes: int
    tensor_spec_coverage: float
    consumes_edges: int
    produces_edges: int
    derived_from_edges: int
    parameter_count: int
    output_tensor_count: int


class HybridBenchmarkMetrics(BaseModel):
    metric_kind: Literal["hybrid"] = "hybrid"
    reconciliation_seconds: float
    static_nodes: int
    static_edges: int
    runtime_nodes: int
    runtime_edges: int
    merged_nodes: int
    merged_edges: int
    target_runtime_definitions: int
    aligned_target_runtime_definitions: int
    target_alignment_rate: float
    target_operator_definitions: int = 0
    source_alignments: int
    alias_edges: int
    coverage_records: int
    always_observed: int
    sometimes_observed: int
    static_reachable_unobserved: int


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
    metrics: (
        StaticBenchmarkMetrics
        | RuntimeBenchmarkMetrics
        | SystemRuntimeBenchmarkMetrics
        | HybridBenchmarkMetrics
        | None
    ) = None
    failures: list[BenchmarkFailure] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
