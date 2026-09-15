from __future__ import annotations

from pathlib import Path

from archtrace.ir import EdgeKind, NodeKind
from archtrace.static import (
    CallResolution,
    RepositoryScanPolicy,
    index_repository,
    repository_index_to_atir,
)


def _write_fixture(root: Path) -> None:
    (root / "pipeline.py").write_text(
        """
def preprocess(raw):
    return raw


def loss_fn(pred, target):
    return pred
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "train.py").write_text(
        """
import argparse
import torch
from pipeline import loss_fn, preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10)


def main(raw, target, model):
    x = preprocess(raw)
    y = model(x)
    loss = loss_fn(y, target)
    return loss


if __name__ == "__main__":
    main(None, None, lambda value: value)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "config.toml").write_text(
        """
[train]
lr = 0.001
epochs = 20
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "policy.yaml").write_text(
        """
model:
  hidden: 128
optimizer:
  name: adam
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_repository_index_resolves_local_calls_and_keeps_dynamic_calls(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    index = index_repository(tmp_path)

    tracked = {"preprocess", "model", "loss_fn"}
    calls = {call.callee_text: call for call in index.calls if call.callee_text in tracked}
    assert calls["preprocess"].resolution == CallResolution.LOCAL
    assert calls["preprocess"].resolved_symbol_id is not None
    assert calls["loss_fn"].resolution == CallResolution.LOCAL
    assert calls["loss_fn"].resolved_symbol_id is not None
    assert calls["model"].resolution == CallResolution.DYNAMIC
    assert calls["model"].resolved_symbol_id is None

    links = {
        (link.producer_call_id, link.consumer_call_id, link.variable) for link in index.dataflow
    }
    assert (calls["preprocess"].id, calls["model"].id, "x") in links
    assert (calls["model"].id, calls["loss_fn"].id, "y") in links


def test_repository_index_discovers_configs_and_explains_entrypoint_score(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    index = index_repository(tmp_path)

    config = {(entry.kind, entry.key): entry.value for entry in index.config_entries}
    assert config[("toml", "train.lr")] == 0.001
    assert config[("toml", "train.epochs")] == 20
    assert config[("yaml", "model.hidden")] == 128
    assert config[("yaml", "optimizer.name")] == "adam"
    assert config[("argparse_argument", "--epochs")] == 10

    assert index.entrypoints
    candidate = index.entrypoints[0]
    assert candidate.path == "train.py"
    assert candidate.score >= 10
    assert any("__main__" in reason for reason in candidate.reasons)
    assert any("entrypoint filename" in reason for reason in candidate.reasons)


def test_static_index_normalizes_into_valid_atir_with_call_and_data_edges(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    index = index_repository(tmp_path)
    graph = repository_index_to_atir(index)

    call_nodes = [node for node in graph.nodes if node.role == "python_call_site"]
    assert call_nodes
    assert any(node.label == "preprocess" for node in call_nodes)
    assert any(
        node.label == "model" and node.attributes["resolution"] == "dynamic" for node in call_nodes
    )

    assert any(edge.kind == EdgeKind.CALLS for edge in graph.edges)
    data_edges = [edge for edge in graph.edges if edge.kind == EdgeKind.DATA]
    assert {edge.label for edge in data_edges} >= {"x", "y"}

    config_nodes = [node for node in graph.nodes if node.kind == NodeKind.CONFIG]
    assert any(node.label == "train.lr" for node in config_nodes)
    assert graph.metadata["static_backend"] == "python_ast_repository_index"
    assert graph.metadata["entrypoints"][0]["path"] == "train.py"


def test_repository_index_excludes_vendored_code_by_default(tmp_path: Path) -> None:
    (tmp_path / "owned.py").write_text("def owned():\n    return 1\n", encoding="utf-8")
    vendor = tmp_path / "third_party"
    vendor.mkdir()
    (vendor / "dependency.py").write_text(
        "def dependency():\n    return 2\n", encoding="utf-8"
    )

    default_index = index_repository(tmp_path)
    assert {item.path for item in default_index.files} == {"owned.py"}

    full_index = index_repository(
        tmp_path, scan_policy=RepositoryScanPolicy(include_vendored=True)
    )
    assert {item.path for item in full_index.files} == {
        "owned.py",
        "third_party/dependency.py",
    }


def test_repository_index_classifies_python_builtins_as_external(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "def main(values):\n    return len(list(values))\n", encoding="utf-8"
    )
    index = index_repository(tmp_path)
    calls = {call.callee_text: call for call in index.calls}
    assert calls["len"].resolution == CallResolution.EXTERNAL
    assert calls["len"].resolved_target == "builtins.len"
    assert calls["list"].resolution == CallResolution.EXTERNAL
    assert calls["list"].resolved_target == "builtins.list"


def test_repository_index_resolves_inherited_self_and_super_methods(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(
        "class Base:\n"
        "    def move(self, value):\n"
        "        return value\n",
        encoding="utf-8",
    )
    (tmp_path / "child.py").write_text(
        "from base import Base\n\n"
        "class Child(Base):\n"
        "    def run(self, value):\n"
        "        first = self.move(value)\n"
        "        return super().move(first)\n",
        encoding="utf-8",
    )
    index = index_repository(tmp_path)
    calls = [call for call in index.calls if call.callee_text in {"self.move", "super.move"}]
    assert len(calls) == 2
    assert all(call.resolution == CallResolution.LOCAL for call in calls)
    assert all(call.resolved_target == "base.Base.move" for call in calls)
    graph = repository_index_to_atir(index)
    child = next(node for node in graph.nodes if node.label == "Child")
    assert child.attributes["bases"] == ["Base"]


def test_repository_index_treats_nested_git_repo_as_boundary(tmp_path: Path) -> None:
    (tmp_path / "owned.py").write_text("def owned():\n    return 1\n", encoding="utf-8")
    nested = tmp_path / "component"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: /tmp/component.git\n", encoding="utf-8")
    (nested / "foreign.py").write_text("def foreign():\n    return 2\n", encoding="utf-8")

    default_index = index_repository(tmp_path)
    assert {item.path for item in default_index.files} == {"owned.py"}

    full_index = index_repository(
        tmp_path, scan_policy=RepositoryScanPolicy(include_nested_repositories=True)
    )
    assert {item.path for item in full_index.files} == {"owned.py", "component/foreign.py"}
