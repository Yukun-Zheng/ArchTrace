"""Stdlib-only smoke test for the Python 3.10 cross-runtime probe agent."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    agent_path = root / "src/archtrace/runtime/python_probe_agent.py"
    with tempfile.TemporaryDirectory(prefix="archtrace-probe-compat-") as temporary:
        target_root = Path(temporary)
        (target_root / "compat_target.py").write_text(
            "def adapt(value):\n"
            "    return {'value': value['value'] + 1}\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(target_root))
        spec = importlib.util.spec_from_file_location("archtrace_probe_agent_compat", agent_path)
        assert spec is not None and spec.loader is not None
        agent = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = agent
        spec.loader.exec_module(agent)
        target = importlib.import_module("compat_target")
        capture = agent.RawProbeCapture(
            [agent.ProbeTarget("compat_target", "adapt", role="adapter")],
            repository_root=target_root,
        )
        capture.install()
        try:
            source = capture.record_boundary(
                "source",
                output={"value": 1},
                role="environment",
                boundary="external",
            )
            output = target.adapt(source)
            capture.record_boundary(
                "sink",
                inputs={"value": output},
                output=True,
                role="environment",
                boundary="external",
            )
        finally:
            capture.remove()
            sys.path.remove(str(target_root))
            sys.modules.pop("compat_target", None)
            sys.modules.pop(spec.name, None)
        raw = capture.to_dict()
        assert output == {"value": 2}
        assert len(raw["definitions"]) == 3
        assert len(raw["occurrences"]) == 3
        assert all(item["status"] == "returned" for item in raw["occurrences"])
        assert any(edge["kind"] == "consumes" for edge in raw["edges"])
        assert any(edge["kind"] == "produces" for edge in raw["edges"])
    print("python-probe-agent compatibility smoke: PASS")


if __name__ == "__main__":
    main()
