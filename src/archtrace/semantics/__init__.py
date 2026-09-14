"""Evidence-grounded semantic architecture recovery."""

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
)

__all__ = [
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
    "collect_repository_context",
    "project_paper_view",
    "recover_semantics",
    "role_spec",
]
