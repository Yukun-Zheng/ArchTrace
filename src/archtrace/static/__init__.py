"""Repository-scale static analysis."""

from archtrace.static.config_resolution import (
    ConfigReference,
    ConfigReferenceStatus,
    resolve_config_references,
)
from archtrace.static.interprocedural import (
    BoundaryFlowKind,
    BoundaryFlowLink,
    analyze_interprocedural_flow,
)
from archtrace.static.python_index import (
    CallResolution,
    CallSite,
    ConfigEntry,
    DataFlowLink,
    EntrypointCandidate,
    ImportBinding,
    PythonFileIndex,
    PythonSymbol,
    RepositoryIndex,
    SymbolKind,
    index_repository,
)
from archtrace.static.to_ir import repository_index_to_atir

__all__ = [
    "BoundaryFlowKind",
    "BoundaryFlowLink",
    "CallResolution",
    "CallSite",
    "ConfigEntry",
    "ConfigReference",
    "ConfigReferenceStatus",
    "DataFlowLink",
    "EntrypointCandidate",
    "ImportBinding",
    "PythonFileIndex",
    "PythonSymbol",
    "RepositoryIndex",
    "SymbolKind",
    "analyze_interprocedural_flow",
    "index_repository",
    "repository_index_to_atir",
    "resolve_config_references",
]
