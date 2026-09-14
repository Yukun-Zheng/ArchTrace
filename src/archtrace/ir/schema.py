"""Canonical ArchTrace Intermediate Representation (ATIR) v0.2.

ATIR separates mechanics from interpretation while preserving explicit evidence.
Definitions, concrete execution occurrences, runtime values, semantic groups,
claims, conflicts, and multi-run coverage are represented independently and
linked through validated identifiers.
"""

from __future__ import annotations

import json
from copy import deepcopy
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CURRENT_SCHEMA_VERSION: Literal["0.2"] = "0.2"


class NodeLevel(StrEnum):
    PAPER = "paper"
    SEMANTIC = "semantic"
    MODULE = "module"
    OPERATION = "operation"
    SOURCE = "source"


class NodeKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    SEMANTIC_COMPONENT = "semantic_component"
    MODULE = "module"
    OPERATION = "operation"
    TENSOR = "tensor"
    PARAMETER = "parameter"
    SOURCE = "source"
    CONFIG = "config"
    CONTROL = "control"
    DATASET = "dataset"
    LOSS = "loss"
    ENVIRONMENT = "environment"
    OTHER = "other"


class IdentityKind(StrEnum):
    """How a node exists, independently from its architectural role."""

    GROUP = "group"
    DEFINITION = "definition"
    OCCURRENCE = "occurrence"
    VALUE = "value"
    STATE = "state"
    SOURCE = "source"


class EdgeKind(StrEnum):
    DATA = "data"
    CONTROL = "control"
    CONTAINS = "contains"
    CALLS = "calls"
    IMPLEMENTS = "implements"
    INSTANCE_OF = "instance_of"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    ALIAS = "alias"
    READS = "reads"
    WRITES = "writes"
    DERIVED_FROM = "derived_from"
    NEXT = "next"


class EvidenceKind(StrEnum):
    RUNTIME = "runtime"
    STATIC = "static"
    SOURCE = "source"
    AUTHOR = "author"
    SEMANTIC = "semantic"
    USER = "user"
    IMPORTED = "imported"


class FactStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    DECLARED = "declared"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


class CoverageStatus(StrEnum):
    ALWAYS_OBSERVED = "always_observed"
    SOMETIMES_OBSERVED = "sometimes_observed"
    STATIC_REACHABLE_UNOBSERVED = "static_reachable_unobserved"
    UNREACHABLE_UNDER_CONFIGURATION = "unreachable_under_configuration"
    UNRESOLVED = "unresolved"


class ConflictKind(StrEnum):
    CONTRADICTORY_CLAIMS = "contradictory_claims"
    AUTHOR_IMPLEMENTATION_MISMATCH = "author_implementation_mismatch"
    STATIC_RUNTIME_MISMATCH = "static_runtime_mismatch"
    COVERAGE_MISMATCH = "coverage_mismatch"
    OTHER = "other"


class ConflictStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)
    start_column: int | None = Field(default=None, ge=0)
    end_column: int | None = Field(default=None, ge=0)
    symbol: str | None = None

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class TensorSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    shape: list[int | str | None] | None = None
    dtype: str | None = None
    device: str | None = None
    layout: str | None = None
    requires_grad: bool | None = None
    semantics: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    kind: EvidenceKind
    status: FactStatus
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str | None = None
    source: SourceSpan | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    level: NodeLevel
    kind: NodeKind
    identity_kind: IdentityKind = IdentityKind.GROUP
    label: str
    role: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    definition_id: str | None = None
    run_id: str | None = None
    occurrence_index: int | None = Field(default=None, ge=0)
    source: list[SourceSpan] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    tensor: TensorSpec | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.identity_kind == IdentityKind.OCCURRENCE and self.run_id is None:
            raise ValueError("execution occurrence nodes require run_id")
        if self.occurrence_index is not None and self.identity_kind != IdentityKind.OCCURRENCE:
            raise ValueError("occurrence_index is only valid for occurrence nodes")
        if self.definition_id is not None and self.identity_kind != IdentityKind.OCCURRENCE:
            raise ValueError("definition_id is only valid for occurrence nodes")
        return self


class ArchEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    kind: EdgeKind
    label: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    tensor: TensorSpec | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    entrypoint: str | None = None
    argv: list[str] = Field(default_factory=list)
    config_paths: list[str] = Field(default_factory=list)
    framework: str | None = None
    framework_version: str | None = None
    python_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimScope(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_ids: list[str] = Field(default_factory=list)
    conditions: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """An evidence-backed proposition about an ATIR entity.

    Exactly one of ``object_id`` and ``value`` must be supplied. Literal null is
    intentionally not a claim object; use FactStatus.UNKNOWN for unknown facts.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    value: Any | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    status: FactStatus
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    scope: ClaimScope = Field(default_factory=ClaimScope)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_object(self) -> Self:
        has_entity = self.object_id is not None
        has_value = self.value is not None
        if has_entity == has_value:
            raise ValueError("claim requires exactly one of object_id or non-null value")
        return self


class Conflict(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    claim_ids: list[str] = Field(min_length=2)
    kind: ConflictKind = ConflictKind.CONTRADICTORY_CLAIMS
    status: ConflictStatus = ConflictStatus.OPEN
    description: str | None = None
    resolution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_claim_ids(self) -> Self:
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError("conflict claim_ids must be unique")
        if self.status == ConflictStatus.RESOLVED and not self.resolution:
            raise ValueError("resolved conflicts require a resolution")
        return self


class CoverageRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    subject_id: str
    status: CoverageStatus
    considered_run_ids: list[str] = Field(default_factory=list)
    observed_run_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    conditions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        considered = set(self.considered_run_ids)
        observed = set(self.observed_run_ids)
        if not observed <= considered:
            raise ValueError("observed_run_ids must be a subset of considered_run_ids")

        if self.status == CoverageStatus.ALWAYS_OBSERVED:
            if not considered or observed != considered:
                raise ValueError("always_observed requires all considered runs to be observed")
        elif self.status == CoverageStatus.SOMETIMES_OBSERVED:
            if not observed or observed == considered:
                raise ValueError("sometimes_observed requires a non-empty proper observed subset")
        elif (
            self.status
            in {
                CoverageStatus.STATIC_REACHABLE_UNOBSERVED,
                CoverageStatus.UNREACHABLE_UNDER_CONFIGURATION,
            }
            and observed
        ):
            raise ValueError(f"{self.status.value} requires no observed runs")
        return self


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    root: str | None = None
    repository_url: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArchTraceIR(BaseModel):
    """One canonical evidence graph from which all ArchTrace views are projected."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.2"] = CURRENT_SCHEMA_VERSION
    project: ProjectInfo
    nodes: list[ArchNode] = Field(default_factory=list)
    edges: list[ArchEdge] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    runs: list[TraceRun] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    coverage: list[CoverageRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph_references(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        evidence_ids = [item.id for item in self.evidence]
        run_ids = [run.id for run in self.runs]
        claim_ids = [claim.id for claim in self.claims]
        conflict_ids = [conflict.id for conflict in self.conflicts]
        coverage_ids = [record.id for record in self.coverage]

        _require_unique(
            "record",
            node_ids + edge_ids + evidence_ids + run_ids + claim_ids + conflict_ids + coverage_ids,
        )

        node_set = set(node_ids)
        edge_set = set(edge_ids)
        entity_set = node_set | edge_set | set(run_ids)
        evidence_set = set(evidence_ids)
        run_set = set(run_ids)
        claim_set = set(claim_ids)
        nodes_by_id = {node.id: node for node in self.nodes}

        for node in self.nodes:
            _require_known(node.parent_ids, node_set, f"node {node.id} parent")
            _require_known(node.evidence_ids, evidence_set, f"node {node.id} evidence")
            if node.run_id is not None:
                _require_known([node.run_id], run_set, f"node {node.id} run")
            if node.definition_id is not None:
                _require_known([node.definition_id], node_set, f"node {node.id} definition")
                definition = nodes_by_id[node.definition_id]
                if definition.identity_kind != IdentityKind.DEFINITION:
                    raise ValueError(
                        f"node {node.id} definition_id must reference a definition node"
                    )

        for edge in self.edges:
            _require_known([edge.source, edge.target], node_set, f"edge {edge.id} endpoint")
            _require_known(edge.evidence_ids, evidence_set, f"edge {edge.id} evidence")

        for item in self.evidence:
            if item.run_id is not None:
                _require_known([item.run_id], run_set, f"evidence {item.id} run")

        for claim in self.claims:
            _require_known([claim.subject_id], entity_set, f"claim {claim.id} subject")
            if claim.object_id is not None:
                _require_known([claim.object_id], entity_set, f"claim {claim.id} object")
            _require_known(claim.evidence_ids, evidence_set, f"claim {claim.id} evidence")
            _require_known(claim.scope.run_ids, run_set, f"claim {claim.id} scope run")

        for conflict in self.conflicts:
            _require_known(conflict.claim_ids, claim_set, f"conflict {conflict.id} claim")

        for record in self.coverage:
            _require_known([record.subject_id], entity_set, f"coverage {record.id} subject")
            _require_known(
                record.considered_run_ids,
                run_set,
                f"coverage {record.id} considered run",
            )
            _require_known(
                record.observed_run_ids,
                run_set,
                f"coverage {record.id} observed run",
            )
            _require_known(record.evidence_ids, evidence_set, f"coverage {record.id} evidence")

        return self


def migrate_atir_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a serialized ATIR payload to the current schema.

    v0.1 did not distinguish definition/occurrence/value identity and had no
    claim/conflict/coverage records. Migration therefore infers only identity
    classes that are mechanically safe from the existing node kind/level. It
    never invents runtime occurrences or semantic claims.
    """

    version = str(payload.get("schema_version", "0.1"))
    if version == CURRENT_SCHEMA_VERSION:
        return deepcopy(payload)
    if version != "0.1":
        raise ValueError(f"unsupported ATIR schema version: {version}")

    migrated = deepcopy(payload)
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    migrated.setdefault("claims", [])
    migrated.setdefault("conflicts", [])
    migrated.setdefault("coverage", [])

    nodes = migrated.get("nodes", [])
    if isinstance(nodes, list):
        for raw_node in nodes:
            if isinstance(raw_node, dict) and "identity_kind" not in raw_node:
                raw_node["identity_kind"] = _infer_v01_identity(raw_node)

    metadata = migrated.setdefault("metadata", {})
    if isinstance(metadata, dict):
        migrations = metadata.setdefault("schema_migrations", [])
        if isinstance(migrations, list):
            migrations.append({"from": "0.1", "to": CURRENT_SCHEMA_VERSION})
    return migrated


def load_atir_json(text: str) -> ArchTraceIR:
    """Load JSON, migrate supported historical schemas, and validate ATIR."""

    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("ATIR root must be a JSON object")
    return ArchTraceIR.model_validate(migrate_atir_payload(raw))


def _infer_v01_identity(node: dict[str, Any]) -> str:
    kind = str(node.get("kind", "other"))
    level = str(node.get("level", "semantic"))
    if kind == NodeKind.SOURCE.value or level == NodeLevel.SOURCE.value:
        return IdentityKind.SOURCE.value
    if kind in {NodeKind.MODULE.value, NodeKind.OPERATION.value}:
        return IdentityKind.DEFINITION.value
    if kind in {NodeKind.TENSOR.value, NodeKind.INPUT.value, NodeKind.OUTPUT.value}:
        return IdentityKind.VALUE.value
    if kind in {NodeKind.PARAMETER.value, NodeKind.CONFIG.value}:
        return IdentityKind.STATE.value
    return IdentityKind.GROUP.value


def _require_unique(kind: str, values: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {kind} id(s): {joined}")


def _require_known(values: list[str], known: set[str], description: str) -> None:
    missing = sorted(set(values) - known)
    if missing:
        raise ValueError(f"{description} references unknown id(s): {', '.join(missing)}")
