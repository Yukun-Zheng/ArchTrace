"""Repository-scale static analysis."""

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
    "CallResolution",
    "CallSite",
    "ConfigEntry",
    "DataFlowLink",
    "EntrypointCandidate",
    "ImportBinding",
    "PythonFileIndex",
    "PythonSymbol",
    "RepositoryIndex",
    "SymbolKind",
    "index_repository",
    "repository_index_to_atir",
]
