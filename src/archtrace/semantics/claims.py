"""Deterministic author-claim extraction and implementation checking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from archtrace.ir import (
    ArchNode,
    ArchTraceIR,
    Claim,
    Conflict,
    ConflictKind,
    ConflictStatus,
    EdgeKind,
    Evidence,
    EvidenceKind,
    FactStatus,
    IdentityKind,
    NodeKind,
    NodeLevel,
    SourceSpan,
)
from archtrace.semantics.context import ContextSnippet
from archtrace.semantics.ontology import SemanticRole, role_spec


class ClaimCheckStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class AuthorDeclaration:
    role: SemanticRole
    predicate: str
    value: bool
    path: str
    line: int
    excerpt: str


_ROLE_PATTERNS: tuple[tuple[SemanticRole, re.Pattern[str]], ...] = (
    (
        SemanticRole.VISION_ENCODER,
        re.compile(
            r"\b(?:vision|visual|image)[\s_-]*(?:encoder|backbone)\b|视觉编码器",
            re.IGNORECASE,
        ),
    ),
    (
        SemanticRole.LANGUAGE_ENCODER,
        re.compile(
            r"\b(?:language|text)[\s_-]*(?:encoder|model|backbone)\b|语言编码器",
            re.IGNORECASE,
        ),
    ),
    (
        SemanticRole.MULTIMODAL_FUSION,
        re.compile(
            r"\b(?:multi[\s_-]*modal|cross[\s_-]*modal)[\s_-]*fusion\b"
            r"|\bcross[\s_-]*attention\b|多模态融合|跨模态融合",
            re.IGNORECASE,
        ),
    ),
    (
        SemanticRole.TRANSFORMER_BACKBONE,
        re.compile(r"\btransformer\b|Transformer主干", re.IGNORECASE),
    ),
    (
        SemanticRole.DIFFUSION_DENOISER,
        re.compile(r"\b(?:diffusion|denoiser|denoising)\b|扩散模型|去噪器", re.IGNORECASE),
    ),
    (
        SemanticRole.WORLD_MODEL,
        re.compile(r"\bworld[\s_-]*model\b|世界模型", re.IGNORECASE),
    ),
    (
        SemanticRole.MEMORY,
        re.compile(r"\b(?:memory|memory[\s_-]*bank)\b|记忆模块", re.IGNORECASE),
    ),
    (
        SemanticRole.ACTION_HEAD,
        re.compile(
            r"\baction[\s_-]*(?:head|decoder|projector)\b|动作头|动作解码器",
            re.IGNORECASE,
        ),
    ),
    (
        SemanticRole.PLANNER,
        re.compile(r"\b(?:planner|planning[\s_-]*module)\b|规划器|规划模块", re.IGNORECASE),
    ),
    (
        SemanticRole.CONTROLLER,
        re.compile(r"\bcontroller\b|控制器", re.IGNORECASE),
    ),
    (
        SemanticRole.ENVIRONMENT,
        re.compile(r"\b(?:environment|env[\s_-]*wrapper)\b|环境模块", re.IGNORECASE),
    ),
    (
        SemanticRole.TOKENIZER,
        re.compile(r"\btokenizer\b|分词器|标记器", re.IGNORECASE),
    ),
    (
        SemanticRole.POLICY,
        re.compile(r"\bpolicy(?:[\s_-]*(?:network|model))?\b|策略网络", re.IGNORECASE),
    ),
)

_FROZEN_TRUE = re.compile(
    r"\b(?:frozen|freeze|freezes|freezed|fixed[\s_-]*weights?)\b|冻结|固定参数",
    re.IGNORECASE,
)
_FROZEN_FALSE = re.compile(
    r"\b(?:trainable|unfrozen|unfreeze|fine[\s_-]*tun(?:e|ed|ing)?)\b|可训练|微调",
    re.IGNORECASE,
)


def add_author_claims(
    graph: ArchTraceIR,
    snippets: list[ContextSnippet],
) -> ArchTraceIR:
    """Add conservative author declarations and compare provable implementation facts."""

    result = graph.model_copy(deep=True)
    used_ids = _all_record_ids(result)
    implementation_cache: dict[tuple[str, str], Claim | None] = {}

    for declaration in extract_author_declarations(snippets):
        author_evidence_id = _next_id("evidence.author", used_ids)
        author_evidence = Evidence(
            id=author_evidence_id,
            kind=EvidenceKind.AUTHOR,
            status=FactStatus.DECLARED,
            confidence=1.0,
            description="Author declaration recovered from repository documentation.",
            source=SourceSpan(
                path=declaration.path,
                start_line=declaration.line,
                end_line=declaration.line,
            ),
            metadata={
                "excerpt": declaration.excerpt,
                "semantic_role": declaration.role.value,
                "predicate": declaration.predicate,
            },
        )
        result.evidence.append(author_evidence)

        subject, implementation_subject = _resolve_subject(
            result,
            declaration.role,
            author_evidence_id,
            declaration,
            used_ids,
        )
        author_claim = Claim(
            id=_next_id("claim.author", used_ids),
            subject_id=subject.id,
            predicate=declaration.predicate,
            value=declaration.value,
            evidence_ids=[author_evidence_id],
            status=FactStatus.DECLARED,
            confidence=1.0,
            metadata={
                "semantic_role": declaration.role.value,
                "source_path": declaration.path,
                "source_line": declaration.line,
                "implementation_check": ClaimCheckStatus.UNRESOLVED.value,
            },
        )
        result.claims.append(author_claim)

        if implementation_subject is None:
            continue

        cache_key = (implementation_subject.id, declaration.predicate)
        if cache_key not in implementation_cache:
            implementation_cache[cache_key] = _implementation_claim(
                result,
                implementation_subject,
                declaration.predicate,
                used_ids,
            )
            implementation_claim = implementation_cache[cache_key]
            if implementation_claim is not None:
                result.claims.append(implementation_claim)
        else:
            implementation_claim = implementation_cache[cache_key]

        if implementation_claim is None:
            continue

        if implementation_claim.value == author_claim.value:
            author_claim.metadata["implementation_check"] = ClaimCheckStatus.SUPPORTED.value
            author_claim.metadata["implementation_claim_id"] = implementation_claim.id
            continue

        author_claim.metadata["implementation_check"] = ClaimCheckStatus.CONTRADICTED.value
        author_claim.metadata["implementation_claim_id"] = implementation_claim.id
        result.conflicts.append(
            Conflict(
                id=_next_id("conflict.author_implementation", used_ids),
                claim_ids=[author_claim.id, implementation_claim.id],
                kind=ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH,
                status=ConflictStatus.OPEN,
                description=(
                    f"Author claim {declaration.predicate}={declaration.value!r} "
                    f"contradicts implementation evidence "
                    f"{declaration.predicate}={implementation_claim.value!r}."
                ),
                metadata={"semantic_role": declaration.role.value},
            )
        )

    result.metadata["author_claims"] = {
        "backend": "deterministic_document_claims_v0",
        "claim_count": sum(
            1 for claim in result.claims if claim.id.startswith("claim.author.")
        ),
        "conflict_count": sum(
            1
            for conflict in result.conflicts
            if conflict.kind == ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH
        ),
    }
    return ArchTraceIR.model_validate(result.model_dump(mode="python"))


def extract_author_declarations(
    snippets: list[ContextSnippet],
) -> list[AuthorDeclaration]:
    """Extract only explicit, line-local component and frozen/trainable claims."""

    declarations: list[AuthorDeclaration] = []
    seen: set[tuple[str, int, SemanticRole, str, bool]] = set()
    for snippet in snippets:
        for line_number, raw_line in enumerate(snippet.text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            for role, pattern in _ROLE_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue

                presence = _presence_value(line, match)
                key = (snippet.path, line_number, role, "component_present", presence)
                if key not in seen:
                    seen.add(key)
                    declarations.append(
                        AuthorDeclaration(
                            role=role,
                            predicate="component_present",
                            value=presence,
                            path=snippet.path,
                            line=line_number,
                            excerpt=line[:500],
                        )
                    )

                frozen = _frozen_value(line)
                if frozen is None:
                    continue
                key = (snippet.path, line_number, role, "frozen", frozen)
                if key in seen:
                    continue
                seen.add(key)
                declarations.append(
                    AuthorDeclaration(
                        role=role,
                        predicate="frozen",
                        value=frozen,
                        path=snippet.path,
                        line=line_number,
                        excerpt=line[:500],
                    )
                )
    return declarations


def _resolve_subject(
    graph: ArchTraceIR,
    role: SemanticRole,
    evidence_id: str,
    declaration: AuthorDeclaration,
    used_ids: set[str],
) -> tuple[ArchNode, ArchNode | None]:
    implementation_nodes = sorted(
        (
            node
            for node in graph.nodes
            if node.level == NodeLevel.SEMANTIC
            and node.role == role.value
            and node.attributes.get("declaration_only") is not True
        ),
        key=lambda node: node.id,
    )
    if len(implementation_nodes) == 1:
        return implementation_nodes[0], implementation_nodes[0]

    spec = role_spec(role)
    declaration_node = ArchNode(
        id=_next_id("semantic.author_declaration", used_ids),
        level=NodeLevel.SEMANTIC,
        kind=NodeKind.SEMANTIC_COMPONENT,
        identity_kind=IdentityKind.GROUP,
        label=spec.display_label,
        role=role.value,
        source=[
            SourceSpan(
                path=declaration.path,
                start_line=declaration.line,
                end_line=declaration.line,
            )
        ],
        evidence_ids=[evidence_id],
        attributes={
            "declaration_only": True,
            "member_ids": [],
            "candidate_implementation_subject_ids": [
                node.id for node in implementation_nodes
            ],
        },
    )
    graph.nodes.append(declaration_node)
    return declaration_node, None


def _implementation_claim(
    graph: ArchTraceIR,
    semantic_node: ArchNode,
    predicate: str,
    used_ids: set[str],
) -> Claim | None:
    if predicate == "component_present":
        confidence = _semantic_confidence(semantic_node)
        evidence_ids = _semantic_support_evidence(semantic_node)
        return Claim(
            id=_next_id("claim.implementation", used_ids),
            subject_id=semantic_node.id,
            predicate=predicate,
            value=True,
            evidence_ids=evidence_ids,
            status=FactStatus.INFERRED,
            confidence=confidence,
            metadata={"basis": "semantic_component_recovered"},
        )

    if predicate != "frozen":
        return None

    trainability = _frozen_from_mechanics(graph, semantic_node)
    if trainability is None:
        return None
    frozen, evidence_ids = trainability
    return Claim(
        id=_next_id("claim.implementation", used_ids),
        subject_id=semantic_node.id,
        predicate="frozen",
        value=frozen,
        evidence_ids=evidence_ids,
        status=FactStatus.OBSERVED,
        confidence=1.0,
        metadata={"basis": "explicit_parameter_trainability"},
    )


def _frozen_from_mechanics(
    graph: ArchTraceIR,
    semantic_node: ArchNode,
) -> tuple[bool, list[str]] | None:
    related = _related_mechanical_nodes(graph, semantic_node)
    votes: list[tuple[bool, list[str]]] = []
    for node in related:
        explicit_frozen = node.attributes.get("frozen")
        if isinstance(explicit_frozen, bool):
            votes.append((explicit_frozen, list(node.evidence_ids)))

        trainable = node.attributes.get("trainable")
        if isinstance(trainable, bool):
            votes.append((not trainable, list(node.evidence_ids)))

        requires_grad = node.attributes.get("requires_grad")
        if isinstance(requires_grad, bool):
            votes.append((not requires_grad, list(node.evidence_ids)))

        if (
            node.kind == NodeKind.PARAMETER
            and node.tensor is not None
            and node.tensor.requires_grad is not None
        ):
            votes.append((not node.tensor.requires_grad, list(node.evidence_ids)))

    if not votes:
        return None
    values = {value for value, _ in votes}
    if len(values) != 1:
        return None
    evidence_ids = sorted(
        {
            evidence_id
            for _, vote_evidence in votes
            for evidence_id in vote_evidence
        }
    )
    return next(iter(values)), evidence_ids


def _related_mechanical_nodes(
    graph: ArchTraceIR,
    semantic_node: ArchNode,
) -> list[ArchNode]:
    raw_members = semantic_node.attributes.get("member_ids", [])
    member_ids = {
        item for item in raw_members if isinstance(item, str)
    } if isinstance(raw_members, list) else set()
    if not member_ids:
        return []

    node_by_id = {node.id: node for node in graph.nodes}
    children: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.kind == EdgeKind.CONTAINS:
            children.setdefault(edge.source, set()).add(edge.target)
    for node in graph.nodes:
        for parent_id in node.parent_ids:
            children.setdefault(parent_id, set()).add(node.id)

    related_ids = set(member_ids)
    queue = list(sorted(member_ids))
    while queue:
        current = queue.pop(0)
        for child_id in sorted(children.get(current, set())):
            if child_id in related_ids:
                continue
            related_ids.add(child_id)
            queue.append(child_id)

    return [
        node_by_id[node_id]
        for node_id in sorted(related_ids)
        if node_id in node_by_id and node_by_id[node_id].level != NodeLevel.SEMANTIC
    ]


def _semantic_support_evidence(node: ArchNode) -> list[str]:
    raw = node.attributes.get("support_evidence_ids", [])
    support = [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []
    return sorted(set([*node.evidence_ids, *support]))


def _semantic_confidence(node: ArchNode) -> float:
    value = node.attributes.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.5


def _presence_value(line: str, match: re.Match[str]) -> bool:
    before = line[: match.start()].lower()[-80:]
    after = line[match.end() :].lower()[:80]
    negative_before = re.search(
        r"(?:without|no|does\s+not\s+use|doesn't\s+use|do\s+not\s+use|"
        r"not\s+using|不使用|没有|无)\s*(?:an?\s+|the\s+)?$",
        before,
    )
    negative_after = re.match(
        r"\s*(?:is\s+not\s+used|is\s+absent|is\s+disabled|未使用|不存在)",
        after,
    )
    return negative_before is None and negative_after is None


def _frozen_value(line: str) -> bool | None:
    frozen = _FROZEN_TRUE.search(line) is not None
    trainable = _FROZEN_FALSE.search(line) is not None
    if frozen == trainable:
        return None
    return frozen


def _all_record_ids(graph: ArchTraceIR) -> set[str]:
    return {
        record.id
        for group in (
            graph.nodes,
            graph.edges,
            graph.evidence,
            graph.runs,
            graph.claims,
            graph.conflicts,
            graph.coverage,
        )
        for record in group
    }


def _next_id(prefix: str, used_ids: set[str]) -> str:
    index = 0
    while f"{prefix}.{index:07d}" in used_ids:
        index += 1
    value = f"{prefix}.{index:07d}"
    used_ids.add(value)
    return value
