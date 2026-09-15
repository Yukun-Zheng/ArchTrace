"""Real-world benchmark support for ArchTrace."""

from archtrace.benchmark.manifest import load_benchmark_manifest
from archtrace.benchmark.models import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkManifest,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkTier,
    FailureCategory,
    FailureSeverity,
    HybridBenchmarkMetrics,
    PythonProbeSpec,
    PythonSystemSpec,
    RuntimeBenchmarkMetrics,
    RuntimeEnvironmentSpec,
    RuntimePackageRequirement,
    RuntimeSourceOverlay,
    RuntimeTargetSpec,
    RuntimeValueSpec,
    StaticBenchmarkMetrics,
    SystemRuntimeBenchmarkMetrics,
    SystemRuntimeResource,
)
from archtrace.benchmark.report import render_markdown_report
from archtrace.benchmark.runner import run_static_benchmark, write_benchmark_result
from archtrace.benchmark.runtime import (
    load_runtime_environment,
    load_runtime_spec,
    run_hybrid_benchmark,
    run_runtime_benchmark,
)
from archtrace.benchmark.system import (
    load_system_spec,
    run_system_benchmark,
    run_system_hybrid_benchmark,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkFailure",
    "BenchmarkManifest",
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkStatus",
    "BenchmarkTier",
    "FailureCategory",
    "FailureSeverity",
    "HybridBenchmarkMetrics",
    "PythonProbeSpec",
    "PythonSystemSpec",
    "RuntimeBenchmarkMetrics",
    "RuntimeEnvironmentSpec",
    "RuntimePackageRequirement",
    "RuntimeSourceOverlay",
    "RuntimeTargetSpec",
    "RuntimeValueSpec",
    "StaticBenchmarkMetrics",
    "SystemRuntimeBenchmarkMetrics",
    "SystemRuntimeResource",
    "load_benchmark_manifest",
    "load_runtime_environment",
    "load_runtime_spec",
    "load_system_spec",
    "render_markdown_report",
    "run_hybrid_benchmark",
    "run_runtime_benchmark",
    "run_system_benchmark",
    "run_system_hybrid_benchmark",
    "run_static_benchmark",
    "write_benchmark_result",
]
