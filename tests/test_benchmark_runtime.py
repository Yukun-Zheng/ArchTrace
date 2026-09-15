from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from archtrace.benchmark import (  # noqa: E402
    BenchmarkCase,
    BenchmarkStatus,
    FailureCategory,
    HybridBenchmarkMetrics,
    RuntimeBenchmarkMetrics,
    RuntimeEnvironmentSpec,
    RuntimePackageRequirement,
    RuntimeSourceOverlay,
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



def test_runtime_values_support_nested_object_and_mapping(tmp_path: Path) -> None:
    (tmp_path / "structured.py").write_text(
        "import torch.nn as nn\n\n"
        "class Structured(nn.Module):\n"
        "    def forward(self, observation):\n"
        "        return observation.images['camera'] + observation.state\n",
        encoding="utf-8",
    )
    case = BenchmarkCase(id="structured", repository="local", revision="1234567")
    observation = RuntimeValueSpec(
        kind="object",
        module="types",
        symbol="SimpleNamespace",
        constructor_kwargs={
            "images": RuntimeValueSpec(
                kind="mapping",
                items={
                    "camera": RuntimeValueSpec(
                        shape=[2, 3], dtype="float32", generator="ones"
                    )
                },
            ),
            "state": RuntimeValueSpec(shape=[2, 3], dtype="float32", generator="zeros"),
        },
    )
    spec = RuntimeTargetSpec(
        module="structured",
        symbol="Structured",
        args=[observation],
        expected_output_shapes=[[2, 3]],
        capture_fx=False,
        capture_export=False,
    )
    result = run_runtime_benchmark(case, tmp_path, spec)
    assert result.status == BenchmarkStatus.COMPLETED
    assert result.metadata["output_shapes"] == [[2, 3]]


def test_runtime_environment_reports_version_mismatch(tmp_path: Path) -> None:
    _write_tiny_model(tmp_path)
    case = BenchmarkCase(id="tiny", repository="local", revision="1234567")
    environment = RuntimeEnvironmentSpec(
        packages=[RuntimePackageRequirement(distribution="archtrace", version="999.0")]
    )
    result = run_runtime_benchmark(case, tmp_path, _spec(), environment=environment)
    assert result.status == BenchmarkStatus.FAILED
    assert result.failures[-1].category == FailureCategory.RUNTIME_ENVIRONMENT_MISMATCH


def test_runtime_overlay_remaps_installed_source_back_to_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay_source = tmp_path / "overlay_src"
    installed_parent = tmp_path / "installed"
    installed_package = installed_parent / "mockoverlay"
    overlay_source.mkdir()
    installed_package.mkdir(parents=True)
    init_text = "from .block import ExternalBlock\n"
    block_text = (
        "import torch.nn as nn\n\n"
        "class ExternalBlock(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n"
    )
    for root in (overlay_source, installed_package):
        (root / "__init__.py").write_text(init_text, encoding="utf-8")
        (root / "block.py").write_text(block_text, encoding="utf-8")
    (tmp_path / "overlay_model.py").write_text(
        "import torch.nn as nn\n"
        "from mockoverlay import ExternalBlock\n\n"
        "class OverlayModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.block = ExternalBlock()\n\n"
        "    def forward(self, x):\n"
        "        return self.block(x)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(installed_parent))
    case = BenchmarkCase(id="overlay", repository="local", revision="1234567")
    environment = RuntimeEnvironmentSpec(
        overlays=[RuntimeSourceOverlay(repository_path="overlay_src", module="mockoverlay")]
    )
    spec = RuntimeTargetSpec(
        module="overlay_model",
        symbol="OverlayModel",
        args=[RuntimeValueSpec(shape=[2, 3], generator="zeros")],
        expected_output_shapes=[[2, 3]],
        capture_fx=False,
        capture_export=False,
    )
    try:
        result = run_hybrid_benchmark(case, tmp_path, spec, environment=environment)
    finally:
        sys.modules.pop("mockoverlay", None)
        sys.modules.pop("mockoverlay.block", None)
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, HybridBenchmarkMetrics)
    assert result.metrics.target_runtime_definitions == 2
    assert result.metrics.aligned_target_runtime_definitions == 2
    assert result.metrics.target_operator_definitions >= 1
    assert result.metrics.target_alignment_rate == 1.0
    environment_meta = result.metadata["environment"]
    assert isinstance(environment_meta, dict)
    overlays = environment_meta["overlays"]
    assert isinstance(overlays, list)
    assert overlays[0]["matched_files"] == 2



def test_runtime_overlay_rejects_shared_hardlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay_source = tmp_path / "overlay_src"
    installed_parent = tmp_path / "installed"
    installed_package = installed_parent / "hardoverlay"
    overlay_source.mkdir()
    installed_package.mkdir(parents=True)
    (overlay_source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    os.link(overlay_source / "__init__.py", installed_package / "__init__.py")
    _write_tiny_model(tmp_path)
    monkeypatch.syspath_prepend(str(installed_parent))
    environment = RuntimeEnvironmentSpec(
        overlays=[
            RuntimeSourceOverlay(
                repository_path="overlay_src",
                module="hardoverlay",
                require_private_copy=True,
            )
        ]
    )
    case = BenchmarkCase(id="tiny", repository="local", revision="1234567")
    result = run_runtime_benchmark(case, tmp_path, _spec(), environment=environment)
    assert result.status == BenchmarkStatus.FAILED
    assert result.failures[-1].category == FailureCategory.RUNTIME_OVERLAY_MISMATCH
    assert "shared hardlinks" in result.failures[-1].message
