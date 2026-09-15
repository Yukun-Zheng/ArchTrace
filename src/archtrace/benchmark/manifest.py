"""Manifest loading for real-world benchmark cases."""

from __future__ import annotations

import tomllib
from pathlib import Path

from archtrace.benchmark.models import BenchmarkManifest


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path)
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    return BenchmarkManifest.model_validate(
        {
            "schema_version": payload.get("schema_version", "1"),
            "cases": payload.get("case", []),
        }
    )
