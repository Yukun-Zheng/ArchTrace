from __future__ import annotations

import importlib
import sys
from pathlib import Path

from archtrace.ir import EdgeKind, IdentityKind
from archtrace.runtime import PythonProbeTarget, trace_python_target


def _write_pipeline(root: Path) -> None:
    (root / "probe_pipeline.py").write_text(
        "class Client:\n"
        "    def __init__(self):\n"
        "        self.obs = None\n"
        "    def call(self, func_name, obs=None):\n"
        "        if func_name == 'update_obs':\n"
        "            self.obs = obs\n"
        "            return None\n"
        "        return [{'joint': [1.0, 2.0]}]\n\n"
        "class Env:\n"
        "    def __init__(self):\n"
        "        self.actions = []\n"
        "    def get_obs(self):\n"
        "        return {'state': [0.0, 0.0], 'rgb': [[1, 2]]}\n"
        "    def take_action(self, action):\n"
        "        self.actions.append(action)\n"
        "        return True\n\n"
        "def obs_adapter(observation):\n"
        "    return {'vision': observation['rgb'], 'state': observation['state']}\n\n"
        "def normalize_action(chunk):\n"
        "    return chunk[0]\n\n"
        "def action_adapter(action):\n"
        "    return action['joint']\n\n"
        "def run(env, client):\n"
        "    observation = env.get_obs()\n"
        "    adapted = obs_adapter(observation)\n"
        "    client.call('update_obs', obs=adapted)\n"
        "    chunk = client.call('get_action')\n"
        "    action = action_adapter(normalize_action(chunk))\n"
        "    env.take_action(action)\n"
        "    return action\n",
        encoding="utf-8",
    )


def test_python_probe_captures_system_boundaries_and_payload_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    _write_pipeline(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("probe_pipeline")
    env = module.Env()
    client = module.Client()
    probes = [
        PythonProbeTarget("probe_pipeline", "Env.get_obs", role="environment"),
        PythonProbeTarget("probe_pipeline", "obs_adapter", role="adapter"),
        PythonProbeTarget("probe_pipeline", "Client.call", role="transport", boundary="rpc"),
        PythonProbeTarget("probe_pipeline", "normalize_action", role="adapter"),
        PythonProbeTarget("probe_pipeline", "action_adapter", role="adapter"),
        PythonProbeTarget("probe_pipeline", "Env.take_action", role="environment"),
    ]
    try:
        result = trace_python_target(
            PythonProbeTarget("probe_pipeline", "run", role="entrypoint"),
            (env, client),
            probes=probes,
            project_name="probe-pipeline",
            repository_root=tmp_path,
        )
    finally:
        sys.modules.pop("probe_pipeline", None)

    assert result.output == [1.0, 2.0]
    assert env.actions == [[1.0, 2.0]]
    graph = result.ir
    definitions = [node for node in graph.nodes if node.identity_kind == IdentityKind.DEFINITION]
    occurrences = [node for node in graph.nodes if node.identity_kind == IdentityKind.OCCURRENCE]
    values = [node for node in graph.nodes if node.identity_kind == IdentityKind.VALUE]
    assert len(definitions) == 7
    assert len(occurrences) == 8  # entrypoint + six boundaries, Client.call twice
    assert values
    assert all(node.source and node.source[0].path == "probe_pipeline.py" for node in definitions)
    assert any(node.attributes.get("boundary") == "rpc" for node in occurrences)

    consumes = [edge for edge in graph.edges if edge.kind == EdgeKind.CONSUMES]
    produces = [edge for edge in graph.edges if edge.kind == EdgeKind.PRODUCES]
    next_edges = [edge for edge in graph.edges if edge.kind == EdgeKind.NEXT]
    calls = [edge for edge in graph.edges if edge.kind == EdgeKind.CALLS]
    assert consumes and produces
    assert len(next_edges) == len(occurrences) - 1
    assert calls

    produced_values = {edge.target for edge in produces}
    consumed_values = {edge.source for edge in consumes}
    assert produced_values & consumed_values  # concrete objects flow across probe boundaries


def test_python_probe_restores_original_callables(tmp_path: Path, monkeypatch) -> None:
    _write_pipeline(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("probe_pipeline")
    original = module.obs_adapter
    try:
        trace_python_target(
            PythonProbeTarget("probe_pipeline", "run"),
            (module.Env(), module.Client()),
            probes=[PythonProbeTarget("probe_pipeline", "obs_adapter")],
            repository_root=tmp_path,
        )
        assert module.obs_adapter is original
    finally:
        sys.modules.pop("probe_pipeline", None)


def test_raw_probe_agent_round_trips_to_atir(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    from archtrace.runtime.python_probe import raw_python_trace_to_atir

    _write_pipeline(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    agent_path = Path(__file__).parents[1] / "src/archtrace/runtime/python_probe_agent.py"
    spec = importlib.util.spec_from_file_location("archtrace_probe_agent_test", agent_path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    module = importlib.import_module("probe_pipeline")
    capture = agent.RawProbeCapture(
        [
            agent.ProbeTarget("probe_pipeline", "obs_adapter", role="adapter"),
            agent.ProbeTarget("probe_pipeline", "action_adapter", role="adapter"),
        ],
        repository_root=tmp_path,
    )
    capture.install()
    try:
        obs = capture.record_boundary(
            "simulator_observation",
            output={"state": [0.0, 0.0], "rgb": [[1, 2]]},
            role="environment",
            boundary="simulator",
        )
        adapted = module.obs_adapter(obs)
        action = capture.record_boundary(
            "policy_rpc",
            inputs={"observation": adapted},
            output={"joint": [1.0, 2.0]},
            role="transport",
            boundary="rpc",
        )
        output = module.action_adapter(action)
    finally:
        capture.remove()
        sys.modules.pop("probe_pipeline", None)
        sys.modules.pop(spec.name, None)
    assert output == [1.0, 2.0]
    raw = capture.to_dict()
    graph = raw_python_trace_to_atir(raw, project_name="raw-probe", repository_root=tmp_path)
    assert graph.metadata["runtime_backend"] == "python_probe_agent"
    assert len([n for n in graph.nodes if n.identity_kind == IdentityKind.DEFINITION]) == 4
    assert any(n.attributes.get("boundary") == "rpc" for n in graph.nodes)
    assert any(edge.kind == EdgeKind.CONSUMES for edge in graph.edges)
