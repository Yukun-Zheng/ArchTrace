"""Resolve static configuration references without executing repository code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from archtrace.static.python_index import ConfigEntry, RepositoryIndex


class ConfigReferenceStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(slots=True)
class ConfigReference:
    source_entry_id: str
    status: ConfigReferenceStatus
    candidate_paths: list[str] = field(default_factory=list)
    target_entry_ids: list[str] = field(default_factory=list)


def resolve_config_references(index: RepositoryIndex) -> list[ConfigReference]:
    """Resolve OmegaConf and Hydra file references against indexed config files."""

    file_entries: dict[str, list[ConfigEntry]] = {}
    for entry in index.config_entries:
        if entry.kind in {"json", "toml", "yaml"}:
            file_entries.setdefault(_normalize_path(entry.path), []).append(entry)

    references: list[ConfigReference] = []
    for entry in index.config_entries:
        if entry.kind == "omegaconf_load":
            candidates = _omegaconf_candidates(entry)
        elif entry.kind == "hydra_entrypoint":
            candidates = _hydra_candidates(entry)
        else:
            continue

        targets: list[str] = []
        for candidate in candidates:
            targets.extend(item.id for item in file_entries.get(candidate, []))
        targets = sorted(set(targets))
        status = (
            ConfigReferenceStatus.RESOLVED
            if targets
            else ConfigReferenceStatus.UNRESOLVED
        )
        references.append(
            ConfigReference(
                source_entry_id=entry.id,
                status=status,
                candidate_paths=candidates,
                target_entry_ids=targets,
            )
        )
    return references


def _omegaconf_candidates(entry: ConfigEntry) -> list[str]:
    if not isinstance(entry.value, str) or not entry.value:
        return []
    return [_relative_candidate(entry.path, entry.value)]


def _hydra_candidates(entry: ConfigEntry) -> list[str]:
    if not isinstance(entry.value, dict):
        return []

    raw_path = entry.value.get("config_path")
    raw_name = entry.value.get("config_name")
    if not isinstance(raw_name, str) or not raw_name:
        return []
    if raw_path is not None and not isinstance(raw_path, str):
        return []

    base = Path(entry.path).parent
    if isinstance(raw_path, str) and raw_path:
        base = base / raw_path

    name = Path(raw_name)
    names = (
        [name]
        if name.suffix
        else [Path(f"{raw_name}.yaml"), Path(f"{raw_name}.yml")]
    )
    return sorted({_normalize_path((base / item).as_posix()) for item in names})


def _relative_candidate(source_path: str, referenced_path: str) -> str:
    path = Path(referenced_path)
    if path.is_absolute():
        return _normalize_path(path.as_posix().lstrip("/"))
    return _normalize_path((Path(source_path).parent / path).as_posix())


def _normalize_path(value: str) -> str:
    parts: list[str] = []
    for part in Path(value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)
