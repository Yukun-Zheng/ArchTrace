"""Standalone Python 3.10-compatible driver for cross-runtime probe scenarios."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-cwd")
    args = parser.parse_args()

    repository = Path(args.repository).resolve()
    payload = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    runtime_cwd = Path(args.runtime_cwd).resolve() if args.runtime_cwd else repository
    for relative in reversed(payload.get("python_paths", ["."])):
        candidate = (repository / relative).resolve()
        if not candidate.is_relative_to(repository):
            raise RuntimeError(f"python path escapes repository: {relative}")
        sys.path.insert(0, str(candidate))
    os.chdir(runtime_cwd)

    for module_name in payload.get("required_imports", []):
        importlib.import_module(module_name)

    agent = _load_module("archtrace_probe_agent_target", Path(args.agent).resolve())
    probes = [
        agent.ProbeTarget(
            item["module"],
            item["symbol"],
            item.get("role", "python"),
            item.get("boundary", "local"),
        )
        for item in payload.get("probes", [])
    ]
    capture = agent.RawProbeCapture(
        probes,
        run_id="run.system.0",
        repository_root=repository,
    )
    capture.install()
    try:
        scenario = _load_module("archtrace_system_scenario", Path(args.scenario).resolve())
        output = scenario.run(
            capture,
            repository_root=repository,
            runtime_cwd=runtime_cwd,
        )
    finally:
        capture.remove()
    raw = capture.to_dict()
    raw["metadata"]["scenario_output"] = output
    raw["metadata"]["scenario_path"] = str(Path(args.scenario).resolve())
    raw["metadata"]["runtime_cwd"] = str(runtime_cwd)
    raw["metadata"]["target_python"] = sys.executable
    Path(args.output).write_text(json.dumps(raw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
