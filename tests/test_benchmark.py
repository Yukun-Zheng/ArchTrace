from __future__ import annotations

import json
from pathlib import Path

from archtrace.benchmark import (
    BenchmarkCase,
    BenchmarkMode,
    BenchmarkStatus,
    FailureCategory,
    load_benchmark_manifest,
    render_markdown_report,
    run_static_benchmark,
)


def _fixture(root: Path) -> None:
    (root / "model.py").write_text(
        "class Policy:\n"
        "    def encode(self, x):\n"
        "        return x\n\n"
        "    def __call__(self, x):\n"
        "        return self.encode(x)\n",
        encoding="utf-8",
    )
    (root / "train.py").write_text(
        "from model import Policy\n\n"
        "def main(raw):\n"
        "    policy = Policy()\n"
        "    encoded = policy.encode(raw)\n"
        "    return policy(encoded)\n\n"
        "if __name__ == '__main__':\n"
        "    main(None)\n",
        encoding="utf-8",
    )


def test_manifest_loads_pinned_cases(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        'schema_version = "1"\n[[case]]\nid = "toy"\n'
        'repository = "https://example.invalid/toy.git"\nrevision = "1234567"\n'
        'modes = ["static"]\n',
        encoding="utf-8",
    )
    loaded = load_benchmark_manifest(manifest)
    assert loaded.case("toy").revision == "1234567"
    assert loaded.case("toy").modes == [BenchmarkMode.STATIC]


def test_static_benchmark_records_coverage(tmp_path: Path) -> None:
    _fixture(tmp_path)
    case = BenchmarkCase(id="toy", repository="local", revision="1234567")
    result = run_static_benchmark(case, tmp_path, allow_revision_mismatch=True)
    assert result.status == BenchmarkStatus.COMPLETED
    assert result.metrics is not None
    assert result.metrics.python_files == 2
    assert result.metrics.parse_success_rate == 1.0
    assert result.metrics.atir_nodes > 0
    assert result.metrics.source_span_coverage > 0.0
    assert 0.0 <= result.metrics.semantic_coverage <= 1.0
    stage_seconds = result.metadata["stage_seconds"]
    assert isinstance(stage_seconds, dict)
    assert set(stage_seconds) == {"index", "atir", "semantic"}
    assert all(f.category != FailureCategory.ANALYSIS_EXCEPTION for f in result.failures)


def test_missing_repository_is_structured(tmp_path: Path) -> None:
    case = BenchmarkCase(id="missing", repository="local", revision="1234567")
    result = run_static_benchmark(case, tmp_path / "missing")
    assert result.status == BenchmarkStatus.FAILED
    assert result.failures[0].category == FailureCategory.MISSING_REPOSITORY


def test_report_is_deterministic_markdown(tmp_path: Path) -> None:
    _fixture(tmp_path)
    case = BenchmarkCase(id="toy", repository="local", revision="1234567")
    result = run_static_benchmark(case, tmp_path, allow_revision_mismatch=True)
    report = render_markdown_report([result])
    assert "# ArchTrace Benchmark Report" in report
    assert "| toy | completed |" in report
    json.loads(result.model_dump_json())


def test_result_schema_accepts_pre_dynamic_config_metric_snapshot() -> None:
    payload = {
        "case_id": "legacy",
        "repository": "local",
        "expected_revision": "1234567",
        "mode": "static",
        "status": "completed",
        "elapsed_seconds": 1.0,
        "metrics": {
            "python_files": 1,
            "parse_errors": 0,
            "parse_success_rate": 1.0,
            "symbols": 1,
            "calls": 0,
            "local_calls": 0,
            "external_calls": 0,
            "dynamic_calls": 0,
            "unresolved_calls": 0,
            "call_resolution_rate": 1.0,
            "config_entries": 0,
            "config_references": 0,
            "unresolved_config_references": 0,
            "entrypoints": 0,
            "dataflow_edges": 0,
            "atir_nodes": 1,
            "atir_edges": 0,
            "source_span_nodes": 1,
            "source_span_coverage": 1.0,
            "semantic_components": 0,
            "semantic_unclassified_nodes": 1,
            "semantic_coverage": 0.0,
        },
    }
    from archtrace.benchmark import BenchmarkResult

    result = BenchmarkResult.model_validate(payload)
    assert result.metrics is not None
    assert result.metrics.dynamic_config_references == 0
