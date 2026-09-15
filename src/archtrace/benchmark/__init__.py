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
    StaticBenchmarkMetrics,
)
from archtrace.benchmark.report import render_markdown_report
from archtrace.benchmark.runner import run_static_benchmark, write_benchmark_result

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
    "StaticBenchmarkMetrics",
    "load_benchmark_manifest",
    "render_markdown_report",
    "run_static_benchmark",
    "write_benchmark_result",
]
