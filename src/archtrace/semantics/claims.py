"""Deterministic author-claim extraction and implementation checking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

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


_ROLE_PATTERNS: tuple[tuple[SemanticRole, str], ...] = (
    (
        SemanticRole.VISION_ENCODER,
        r"\b(?:vision|visual|image)[\s_-]*(?:encoder|backbone)\b|视觉编码器",
    ),
    (
        SemanticRole.LANGUAGE_ENCODER,
        r"\b(?:language|text)[\s_-]*(?:encoder|model|backbone)\b|语言编码器",
    ),
    (
        SemanticRole.MULTIMODAL_FUSION,
        r"\b(?:multi[\s_-]*modal|cross[\s_-]*modal)[\s_-]*fusion\b"
        r"|\bcross[\s_-]*attention\b|多模态融合|跨模态融合",
    ),
    (
        SemanticRole.TRANSFORMER_BACKBONE,
        r"\btransformer\b|Transformer主干",
    ),
    (
        SemanticRole.DIFFUSION_DENOISER,
        r"\b(?:diffusion|denoiser|denoising)\b|扩散模型|去噪器",
    ),
    (SemanticRole.WORLD_MODEL, r"\bworld[\s_-]*model\b|世界模型"),
    (SemanticRole.MEMORY, r"\b(?:memory|memory[\s_-]*bank)\b|记忆模块"),
    (
        SemanticRole.ACTION_HEAD,
        r"\baction[\s_-]*(?:head|decoder|projector)\b|动作头|动作解码器",
    ),
    (
        SemanticRole.PLANNER,
        r"\b(?:planner|planning[\s_-]*module)\b|规划器|规划模块",
    ),
    (SemanticRole.CONTROLLER, r"\bcontroller\b|控制器"),
    (
        SemanticRole.ENVIRONMENT,
        r"\b(?:environment|env[\s_-]*wrapper)\b|环境模块",
    ),
    (SemanticRole.TOKENIZER, r"\btokenizer\b|分词器|标记器"),
    (
        SemanticRole.POLICY,
        r"\bpolicy(?:[\s_-]*(?:network|model))?\b|策略网络",
    ),
)
_PATTERNS = tuple(
    (role, re.compile(pattern, re.IGNORECASE))
    for role, pattern in _ROLE_PATTERNS
)
_FROZEN = re.compile(
    r"\b(?:frozen|freeze|freezes|fixed[\s_-]*weights?)\b|冻结|固定参数",
    re.IGNORECASE,
)
_TRAINABLE = re.compile(
    r"\b(?:trainable|unfrozen|unfreeze|fine[\s_-]*tun(?:e|ed|ing)?)\b"
    r"|可训练|微调",
    re.IGNORECASE,
)


def add_author_claims(
    graph: ArchTraceIR,
    snippets: list[ContextSnippet],
) -> ArchTraceIR:
    result = graph.model_copy(deep=True)
    used = _ids(result)
    cache: dict[tuple[str, str], Claim | None] = {}

    for declaration in extract_author_declarations(snippets):
        evidence_id = _next("evidence.author", used)
        result.evidence.append(
            Evidence(
                id=evidence_id,
                kind=EvidenceKind.AUTHOR,
                status=FactStatus.DECLARED,
                description=(
                    "Author declaration recovered from repository documentation."
                ),
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
        )
        subject, implementation = _resolve_subject(
            result,
            declaration,
            evidence_id,
            used,
        )
        author = Claim(
            id=_next("claim.author", used),
            subject_id=subject.id,
            predicate=declaration.predicate,
            value=declaration.value,
            evidence_ids=[evidence_id],
            status=FactStatus.DECLARED,
            metadata={
                "semantic_role": declaration.role.value,
                "source_path": declaration.path,
                "source_line": declaration.line,
                "implementation_check": ClaimCheckStatus.UNRESOLVED.value,
            },
        )
        result.claims.append(author)
        if implementation is None:
            continue

        key = (implementation.id, declaration.predicate)
        if key not in cache:
            cache[key] = _implementation_claim(
                result,
                implementation,
                declaration.predicate,
                used,
            )
            if cache[key] is not None:
                result.claims.append(cache[key])
        impl = cache[key]
        if impl is None:
            continue

        author.metadata["implementation_claim_id"] = impl.id
        if impl.value == author.value:
            author.metadata["implementation_check"] = (
                ClaimCheckStatus.SUPPORTED.value
            )
            continue

        author.metadata["implementation_check"] = (
            ClaimCheckStatus.CONTRADICTED.value
        )
        result.conflicts.append(
            Conflict(
                id=_next("conflict.author_implementation", used),
                claim_ids=[author.id, impl.id],
                kind=ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH,
                status=ConflictStatus.OPEN,
                description=(
                    f"Author {author.predicate}={author.value!r} contradicts "
                    f"implementation {impl.value!r}."
                ),
                metadata={"semantic_role": declaration.role.value},
            )
        )

    result.metadata["author_claims"] = {
        "backend": "deterministic_document_claims_v0",
        "claim_count": sum(
            claim.id.startswith("claim.author.")
            for claim in result.claims
        ),
        "conflict_count": sum(
            conflict.kind == ConflictKind.AUTHOR_IMPLEMENTATION_MISMATCH
            for conflict in result.conflicts
        ),
    }
    return ArchTraceIR.model_validate(result.model_dump(mode="python"))


def extract_author_declarations(
    snippets: list[ContextSnippet],
) -> list[AuthorDeclaration]:
    declarations: list[AuthorDeclaration] = []
    seen: set[tuple[str, int, SemanticRole, str, bool]] = set()

    for snippet in snippets:
        for line_number, raw in enumerate(
            snippet.text.splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line:
                continue
            for role, pattern in _PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue
                _append(
                    declarations,
                    seen,
                    snippet.path,
                    line_number,
                    line,
                    role,
                    "component_present",
                    _presence(line, match),
                )
                frozen = _frozen_value(line)
                if frozen is not None:
                    _append(
                        declarations,
                        seen,
                        snippet.path,
                        line_number,
                        line,
                        role,
                        "frozen",
                        frozen,
                    )
    return declarations


def _append(
    output: list[AuthorDeclaration],
    seen: set[tuple[str, int, SemanticRole, str, bool]],
    path: str,
    line: int,
    excerpt: str,
    role: SemanticRole,
    predicate: str,
    value: bool,
) -> None:
    key = (path, line, role, predicate, value)
    if key in seen:
        return
    seen.add(key)
    output.append(
        AuthorDeclaration(
            role=role,
            predicate=predicate,
            value=value,
            path=path,
            line=line,
            excerpt=excerpt[:500],
        )
    )


def _resolve_subject(
    graph: ArchTraceIR,
    declaration: AuthorDeclaration,
    evidence_id: str,
    used: set[str],
) -> tuple[ArchNode, ArchNode | None]:
    matches = sorted(
        (
            node
            for node in graph.nodes
            if node.level == NodeLevel.SEMANTIC
            and node.role == declaration.role.value
            and node.attributes.get("declaration_only") is not True
        ),
        key=lambda node: node.id,
    )
    if len(matches) == 1:
        return matches[0], matches[0]

    node = ArchNode(
        id=_next("semantic.author_declaration", used),
        level=NodeLevel.SEMANTIC,
        kind=NodeKind.SEMANTIC_COMPONENT,
        identity_kind=IdentityKind.GROUP,
        label=role_spec(declaration.role).display_label,
        role=declaration.role.value,
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
                item.id for item in matches
            ],
        },
    )
    graph.nodes.append(node)
    return node, None


def _implementation_claim(
    graph: ArchTraceIR,
    semantic: ArchNode,
    predicate: str,
    used: set[str],
) -> Claim | None:
    if predicate == "component_present":
        return Claim(
            id=_next("claim.implementation", used),
            subject_id=semantic.id,
            predicate=predicate,
            value=True,
            evidence_ids=_support(semantic),
            status=FactStatus.INFERRED,
            confidence=_confidence(semantic),
            metadata={"basis": "semantic_component_recovered"},
        )
    if predicate != "frozen":
        return None

    frozen = _frozen_from_mechanics(graph, semantic)
    if frozen is None:
        return None
    value, evidence_ids = frozen
    return Claim(
        id=_next("claim.implementation", used),
        subject_id=semantic.id,
        predicate="frozen",
        value=value,
        evidence_ids=evidence_ids,
        status=FactStatus.OBSERVED,
        metadata={"basis": "explicit_parameter_trainability"},
    )


def _frozen_from_mechanics(
    graph: ArchTraceIR,
    semantic: ArchNode,
) -> tuple[bool, list[str]] | None:
    votes: list[tuple[bool, list[str]]] = []
    for node in _related(graph, semantic):
        for key, invert in (
            ("frozen", False),
            ("trainable", True),
            ("requires_grad", True),
        ):
            value = node.attributes.get(key)
            if isinstance(value, bool):
                frozen = (not value) if invert else value
                votes.append((frozen, list(node.evidence_ids)))
        if (
            node.kind == NodeKind.PARAMETER
            and node.tensor is not None
            and node.tensor.requires_grad is not None
        ):
            votes.append(
                (
                    not node.tensor.requires_grad,
                    list(node.evidence_ids),
                )
            )

    if not votes or len({value for value, _ in votes}) != 1:
        return None
    evidence_ids = sorted(
        {
            evidence_id
            for _, ids in votes
            for evidence_id in ids
        }
    )
    return votes[0][0], evidence_ids


def _related(
    graph: ArchTraceIR,
    semantic: ArchNode,
) -> list[ArchNode]:
    raw = semantic.attributes.get("member_ids", [])
    members = (
        {item for item in raw if isinstance(item, str)}
        if isinstance(raw, list)
        else set()
    )
    by_id = {node.id: node for node in graph.nodes}
    children: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.kind == EdgeKind.CONTAINS:
            children.setdefault(edge.source, set()).add(edge.target)
    for node in graph.nodes:
        for parent in node.parent_ids:
            children.setdefault(parent, set()).add(node.id)

    related = set(members)
    queue = sorted(members)
    while queue:
        current = queue.pop(0)
        for child in sorted(children.get(current, set())):
            if child in related:
                continue
            related.add(child)
            queue.append(child)

    return [
        by_id[item]
        for item in sorted(related)
        if item in by_id
        and by_id[item].level != NodeLevel.SEMANTIC
    ]


def _support(node: ArchNode) -> list[str]:
    raw = node.attributes.get("support_evidence_ids", [])
    extra = (
        [item for item in raw if isinstance(item, str)]
        if isinstance(raw, list)
        else []
    )
    return sorted(set([*node.evidence_ids, *extra]))


def _confidence(node: ArchNode) -> float:
    value = node.attributes.get("confidence")
    if not isinstance(value, (int, float)):
        return 0.5
    return max(0.0, min(1.0, float(value)))


def _presence(
    line: str,
    match: re.Match[str],
) -> bool:
    before = line[: match.start()].lower()[-80:]
    after = line[match.end() :].lower()[:80]
    neg_before = re.search(
        r"(?:without|no|does\s+not\s+use|doesn't\s+use|do\s+not\s+use|"
        r"not\s+using|不使用|没有|无)\s*(?:an?\s+|the\s+)?$",
        before,
    )
    neg_after = re.match(
        r"\s*(?:is\s+not\s+used|is\s+absent|is\s+disabled|未使用|不存在)",
        after,
    )
    return neg_before is None and neg_after is None


def _frozen_value(line: str) -> bool | None:
    frozen = _FROZEN.search(line) is not None
    trainable = _TRAINABLE.search(line) is not None
    if frozen == trainable:
        return None
    return frozen


def _ids(graph: ArchTraceIR) -> set[str]:
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


def _next(prefix: str, used: set[str]) -> str:
    index = 0
    while f"{prefix}.{index:07d}" in used:
        index += 1
    value = f"{prefix}.{index:07d}"
    used.add(value)
    return value
