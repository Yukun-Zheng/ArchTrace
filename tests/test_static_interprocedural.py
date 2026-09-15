from __future__ import annotations

from pathlib import Path

from archtrace.ir import EdgeKind
from archtrace.static import (
    BoundaryFlowKind,
    ConfigReferenceStatus,
    RepositoryIndex,
    analyze_interprocedural_flow,
    index_repository,
    repository_index_to_atir,
    resolve_config_references,
)


def _write_pipeline_fixture(root: Path) -> None:
    (root / "pipeline.py").write_text(
        """
def normalize(raw):
    return raw


def preprocess(raw):
    cleaned = normalize(raw)
    return cleaned


def postprocess(pred):
    adjusted = normalize(pred)
    return adjusted


def loss_fn(pred, target):
    return pred
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "train.py").write_text(
        """
import hydra
from omegaconf import OmegaConf
from pipeline import loss_fn, postprocess, preprocess


@hydra.main(config_path="conf", config_name="train")
def main(raw, target, model):
    cfg = OmegaConf.load("conf/train.yaml")
    x = preprocess(raw)
    y = model(x)
    z = postprocess(y)
    loss = loss_fn(z, target)
    return loss
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "conf").mkdir()
    (root / "conf" / "train.yaml").write_text(
        """
model:
  hidden: 128
train:
  epochs: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_interprocedural_flow_crosses_argument_and_return_boundaries(
    tmp_path: Path,
) -> None:
    _write_pipeline_fixture(tmp_path)
    index = index_repository(tmp_path)
    links = analyze_interprocedural_flow(index)

    calls = {call.callee_text: call for call in index.calls}
    normalize_calls = [call for call in index.calls if call.callee_text == "normalize"]
    preprocess_normalize = next(
        call
        for call in normalize_calls
        if "preprocess" in _caller_qualname(index, call.caller_symbol_id)
    )

    assert any(
        link.producer_call_id == calls["preprocess"].id
        and link.consumer_call_id == preprocess_normalize.id
        and link.kind == BoundaryFlowKind.ARGUMENT
        and link.metadata["formal_parameter"] == "raw"
        for link in links
    )
    assert any(
        link.producer_call_id == preprocess_normalize.id
        and link.consumer_call_id == calls["preprocess"].id
        and link.kind == BoundaryFlowKind.RETURN
        and link.variable == "cleaned"
        for link in links
    )


def test_static_atir_contains_interprocedural_and_config_reference_edges(
    tmp_path: Path,
) -> None:
    _write_pipeline_fixture(tmp_path)
    graph = repository_index_to_atir(index_repository(tmp_path))

    data_edges = [edge for edge in graph.edges if edge.kind == EdgeKind.DATA]
    flow_kinds = {edge.attributes.get("flow_kind") for edge in data_edges}
    assert "argument_into_callee" in flow_kinds
    assert "return_from_callee" in flow_kinds

    reads = [edge for edge in graph.edges if edge.kind == EdgeKind.READS]
    assert reads
    config_targets = {
        node.id for node in graph.nodes if node.attributes.get("path") == "conf/train.yaml"
    }
    assert config_targets
    assert any(edge.target in config_targets for edge in reads)

    reference_nodes = [
        node
        for node in graph.nodes
        if node.role in {"config_omegaconf_load", "config_hydra_entrypoint"}
    ]
    assert reference_nodes
    assert all(node.attributes["reference_status"] == "resolved" for node in reference_nodes)


def test_unresolved_config_reference_remains_explicit(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text(
        """
from omegaconf import OmegaConf


def main():
    return OmegaConf.load("conf/missing.yaml")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    graph = repository_index_to_atir(index_repository(tmp_path))
    reference = next(node for node in graph.nodes if node.role == "config_omegaconf_load")
    assert reference.attributes["reference_status"] == "unresolved"
    assert reference.attributes["target_entry_ids"] == []
    assert not any(
        edge.kind == EdgeKind.READS and edge.source == reference.id for edge in graph.edges
    )


def _caller_qualname(index: RepositoryIndex, symbol_id: str) -> str:
    return next(symbol.qualname for symbol in index.symbols if symbol.id == symbol_id)


def test_hydra_pathlib_expression_resolves_without_execution(tmp_path: Path) -> None:
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "train.yaml").write_text("model:\n  hidden: 64\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import pathlib\nimport hydra\n\n"
        "@hydra.main(\n"
        "    version_base=None,\n"
        "    config_path=str(pathlib.Path(__file__).parent.joinpath('conf')),\n"
        "    config_name=pathlib.Path(__file__).stem,\n"
        ")\n"
        "def main(cfg):\n    return cfg\n",
        encoding="utf-8",
    )
    index = index_repository(tmp_path)
    entry_by_id = {entry.id: entry for entry in index.config_entries}
    reference = next(
        item
        for item in resolve_config_references(index)
        if entry_by_id[item.source_entry_id].kind == "hydra_entrypoint"
    )
    assert reference.status == ConfigReferenceStatus.RESOLVED
    target_paths = {
        entry.path for entry in index.config_entries if entry.id in reference.target_entry_ids
    }
    assert target_paths == {"conf/train.yaml"}


def test_dynamic_omegaconf_path_is_not_reported_as_missing_file(tmp_path: Path) -> None:
    (tmp_path / "loader.py").write_text(
        "from omegaconf import OmegaConf\n\n"
        "def load(cfg_path):\n    return OmegaConf.load(cfg_path)\n",
        encoding="utf-8",
    )
    index = index_repository(tmp_path)
    reference = next(iter(resolve_config_references(index)))
    assert reference.status == ConfigReferenceStatus.DYNAMIC
    assert reference.candidate_paths == []
    assert reference.target_entry_ids == []
