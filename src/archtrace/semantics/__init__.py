"""Evidence-grounded semantic architecture recovery."""

from archtrace.semantics.claims import (
    AuthorDeclaration,
    ClaimCheckStatus,
    add_author_claims,
    extract_author_declarations,
)
from archtrace.semantics.context import ContextSnippet, collect_repository_context
from archtrace.semantics.engine import (
    SemanticHypothesis,
    SemanticOverride,
    recover_semantics,
)
from archtrace.semantics.ontology import (
    Modality,
    RoleSpec,
    SemanticPhase,
    SemanticRole,
    role_spec,
)
from archtrace.semantics.projection import (
    PaperEdge,
    PaperNode,
    PaperView,
    PaperViewPolicy,
    project_paper_view,
    select_semantic_components,
)

__all__ = [
    "AuthorDeclaration",
    "ClaimCheckStatus",
    "ContextSnippet",
    "Modality",
    "PaperEdge",
    "PaperNode",
    "PaperView",
    "PaperViewPolicy",
    "RoleSpec",
    "SemanticHypothesis",
    "SemanticOverride",
    "SemanticPhase",
    "SemanticRole",
    "add_author_claims",
    "collect_repository_context",
    "extract_author_declarations",
    "project_paper_view",
    "recover_semantics",
    "role_spec",
    "select_semantic_components",
]
