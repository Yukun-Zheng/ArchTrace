from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from archtrace.benchmark import (
    BenchmarkCase,
    BenchmarkStatus,
    FailureCategory,
    HybridBenchmarkMetrics,
    SystemRuntimeBenchmarkMetrics,
    SystemRuntimeResource,
    load_system_spec,
    run_system_benchmark,
    run_system_hybrid_benchmark,
)


def _write_system_fixture(root: Path) -> tuple[Path, Path]:
    (root / "system_target.py").write_text(
        "def adapt(observation):\n"
        "    return {'state': observation['state'], 'ready': True}\n\n"
        "def action_adapter(action):\n"
        "    return action['joint']\n",
        encoding="utf-8",
    )
    scenario = root / "scenario.py"
    scenario.write_text(
        "def run(capture, *, repository_root, runtime_cwd):\n"
        "    import system_target\n"
        "    observation = capture.record_boundary(\n"
        "        'sim_obs', output={'state': [0.0, 1.0]}, "
        "role='environment', boundary='simulator')\n"
        "    adapted = system_target.adapt(observation)\n"
        "    response = capture.record_boundary(\n"
        "        'rpc', inputs={'observation': adapted}, "
        "output={'joint': [1.0, 2.0]}, role='transport', boundary='rpc')\n"
        "    action = system_target.action_adapter(response)\n"
        "    capture.record_boundary(\n"
        "        'actuate', inputs={'action': action}, output=True, "
        "role='environment', boundary='simulator')\n"
        "    return {'action': action}\n",
        encoding="utf-8",
    )
    spec = root / "system.json"
    spec.write_text(
        json.dumps(
            {
                "scenario": "scenario.py",
                "python_paths": ["."],
                "probes": [
                    {"module": "system_target", "symbol": "adapt", "role": "adapter"},
                    {
                        "module": "system_target",
                        "symbol": "action_adapter",
                        "role": "adapter",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return spec, scenario


def test_system_benchmark_runs_in_target_python(tmp_path: Path) -> None:
    spec_path, _ = _write_system_fixture(tmp_path)
    spec = load_system_spec(spec_path)
    case = BenchmarkCase(id="system", repository="local", revision="1234567")
    result = run_system_benchmark(
        case,
        tmp_path,
        spec,
        spec_path=spec_path,
        target_python=sys.executable,
    )
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, SystemRuntimeBenchmarkMetrics)
    assert result.metrics.source_definitions == 2
    assert result.metrics.external_definitions == 3
    assert result.metrics.transport_occurrences == 1
    assert result.metrics.environment_occurrences == 2
    assert result.metadata["scenario_output"] == {"action": [1.0, 2.0]}


def test_system_hybrid_aligns_probed_source_definitions(tmp_path: Path) -> None:
    spec_path, _ = _write_system_fixture(tmp_path)
    spec = load_system_spec(spec_path)
    case = BenchmarkCase(id="system", repository="local", revision="1234567")
    result = run_system_hybrid_benchmark(
        case,
        tmp_path,
        spec,
        spec_path=spec_path,
        target_python=sys.executable,
    )
    assert result.status == BenchmarkStatus.COMPLETED
    assert isinstance(result.metrics, HybridBenchmarkMetrics)
    assert result.metrics.target_runtime_definitions == 2
    assert result.metrics.aligned_target_runtime_definitions == 2
    assert result.metrics.target_alignment_rate == 1.0
    assert result.metadata["unaligned_target_definitions"] == []

def test_system_benchmark_verifies_runtime_resource_fingerprint(tmp_path: Path) -> None:
    spec_path, _ = _write_system_fixture(tmp_path)
    resource = tmp_path / "resource.json"
    resource.write_text('{"ready": true}\n', encoding="utf-8")
    expected = hashlib.sha256(resource.read_bytes()).hexdigest()
    spec = load_system_spec(spec_path).model_copy(
        update={
            "resources": [
                SystemRuntimeResource(
                    path="resource.json", kind="file", sha256=expected
                )
            ]
        }
    )
    case = BenchmarkCase(id="system", repository="local", revision="1234567")
    result = run_system_benchmark(
        case, tmp_path, spec, spec_path=spec_path, target_python=sys.executable
    )
    assert result.status == BenchmarkStatus.COMPLETED
    resources = result.metadata["runtime_resources"]
    assert isinstance(resources, list)
    assert resources == [
        {
            "path": "resource.json",
            "kind": "file",
            "sha256": expected,
            "size_bytes": resource.stat().st_size,
        }
    ]


def test_system_benchmark_rejects_runtime_resource_hash_mismatch(tmp_path: Path) -> None:
    spec_path, _ = _write_system_fixture(tmp_path)
    (tmp_path / "resource.json").write_text("different\n", encoding="utf-8")
    spec = load_system_spec(spec_path).model_copy(
        update={
            "resources": [
                SystemRuntimeResource(path="resource.json", sha256="0" * 64)
            ]
        }
    )
    case = BenchmarkCase(id="system", repository="local", revision="1234567")
    result = run_system_benchmark(
        case, tmp_path, spec, spec_path=spec_path, target_python=sys.executable
    )
    assert result.status == BenchmarkStatus.FAILED
    assert result.failures[-1].category == FailureCategory.SYSTEM_RUNTIME_RESOURCE_MISMATCH
