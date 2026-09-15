from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from archtrace.benchmark import (  # noqa: E402
    BenchmarkCase,
    BenchmarkStatus,
    FailureCategory,
    HybridBenchmarkMetrics,
    RuntimeBenchmarkMetrics,
    RuntimeTargetSpec,
    RuntimeValueSpec,
    run_hybrid_benchmark,
    run_runtime_benchmark,
)


def _write_tiny_model(root: Path) -> None:
    (root / "tiny_model.py").write_text(
        "import torch.nn as nn\n\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self, dim=3):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(dim, dim)\n\n"
        "    def forward(self, x):\n"
        "        return self.proj(x)\n",
        encoding="utf-8",
    )


def _spec() -> RuntimeTargetSpec:
    return RuntimeTargetSpec(
        module="tiny_model",
        symbol="Tiny",
        constructor_kwargs={"dim": 3},
        args=[RuntimeValueSpec(shape=[2, 3], dtype="float32", generator="randn")],
        expected_output_shapes=[[2, 3]],
        capture_fx=False,
        capture_export=False,
    )


def test_runtime_benchmark_executes_real_pytorch_module(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path)
    case = BenchmarkCase(id="tiny", repository="local", revision="1234567")
    result = run_runtime_benchmark(case, tmp_path, _spec())
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, RuntimeBenchmarkMetrics)
    assert result.metrics.runtime_nodes > 0
    assert result.metrics.target_source_definitions == 1
    assert result.metrics.output_tensor_count == 1
    assert result.metadata["output_shapes"] == [[2, 3]]


def test_hybrid_benchmark_aligns_target_owned_runtime_definition(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path)
    case = BenchmarkCase(id="tiny", repository="local", revision="1234567")
    result = run_hybrid_benchmark(case, tmp_path, _spec())
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, HybridBenchmarkMetrics)
    assert result.metrics.target_runtime_definitions == 1
    assert result.metrics.aligned_target_runtime_definitions == 1
    assert result.metrics.target_alignment_rate == 1.0
    assert result.metrics.always_observed == 1


def test_runtime_benchmark_reports_missing_dependency(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path)
    case = BenchmarkCase(id="tiny", repository="local", revision="1234567")
    spec = _spec().model_copy(update={"required_imports": ["definitely_missing_archtrace_pkg"]})
    result = run_runtime_benchmark(case, tmp_path, spec)
    assert result.status == BenchmarkStatus.FAILED
    assert result.failures[-1].category == FailureCategory.RUNTIME_DEPENDENCY_MISSING


def test_runtime_constructor_values_support_imports_and_factories(tmp_path: Path) -> None:
    (tmp_path / "configured_model.py").write_text(
        "from dataclasses import dataclass\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "@dataclass\n"
        "class TinyConfig:\n"
        "    dim: int\n\n"
        "class TinyConfigured(nn.Module):\n"
        "    def __init__(self, config, dtype):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(config.dim, config.dim, dtype=dtype)\n\n"
        "    def forward(self, x):\n"
        "        return self.proj(x)\n",
        encoding="utf-8",
    )
    case = BenchmarkCase(id="configured", repository="local", revision="1234567")
    spec = RuntimeTargetSpec(
        module="configured_model",
        symbol="TinyConfigured",
        constructor_kwargs={
            "config": {
                "$kind": "construct",
                "module": "configured_model",
                "symbol": "TinyConfig",
                "kwargs": {"dim": 3},
            },
            "dtype": {"$kind": "import", "module": "torch", "symbol": "float32"},
        },
        args=[RuntimeValueSpec(shape=[2, 3], dtype="float32", generator="randn")],
        expected_output_shapes=[[2, 3]],
        capture_fx=False,
        capture_export=False,
    )
    result = run_runtime_benchmark(case, tmp_path, spec)
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, RuntimeBenchmarkMetrics)
    assert result.metrics.parameter_count == 12
