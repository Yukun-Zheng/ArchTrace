"""Repository-scale Python source indexing for static ArchTrace analysis."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from archtrace.ir import SourceSpan

_IGNORED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "site-packages",
    "node_modules",
}
_ENTRYPOINT_NAMES = {
    "main.py",
    "train.py",
    "eval.py",
    "evaluate.py",
    "inference.py",
    "demo.py",
    "run.py",
    "app.py",
}
_FRAMEWORK_IMPORTS = {"torch", "jax", "tensorflow"}


class SymbolKind(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class CallResolution(StrEnum):
    LOCAL = "local"
    EXTERNAL = "external"
    DYNAMIC = "dynamic"
    UNRESOLVED = "unresolved"


@dataclass(slots=True)
class PythonSymbol:
    id: str
    module: str
    qualname: str
    name: str
    kind: SymbolKind
    span: SourceSpan
    parent_id: str | None = None


@dataclass(slots=True)
class ImportBinding:
    module: str
    local_name: str
    imported_module: str
    imported_name: str | None
    span: SourceSpan


@dataclass(slots=True)
class CallSite:
    id: str
    module: str
    caller_symbol_id: str
    callee_text: str
    span: SourceSpan
    order: int
    result_targets: list[str] = field(default_factory=list)
    argument_names: list[str] = field(default_factory=list)
    resolution: CallResolution = CallResolution.UNRESOLVED
    resolved_symbol_id: str | None = None
    resolved_target: str | None = None


@dataclass(slots=True)
class DataFlowLink:
    producer_call_id: str
    consumer_call_id: str
    variable: str


@dataclass(slots=True)
class ConfigEntry:
    id: str
    kind: str
    path: str
    key: str
    value: Any
    span: SourceSpan
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntrypointCandidate:
    path: str
    score: int
    reasons: list[str]


@dataclass(slots=True)
class PythonFileIndex:
    path: str
    module: str
    symbols: list[PythonSymbol] = field(default_factory=list)
    imports: list[ImportBinding] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    config_entries: list[ConfigEntry] = field(default_factory=list)
    entrypoint_reasons: list[tuple[int, str]] = field(default_factory=list)
    parse_error: str | None = None


@dataclass(slots=True)
class RepositoryIndex:
    root: Path
    files: list[PythonFileIndex]
    symbols: list[PythonSymbol]
    imports: list[ImportBinding]
    calls: list[CallSite]
    dataflow: list[DataFlowLink]
    config_entries: list[ConfigEntry]
    entrypoints: list[EntrypointCandidate]


def index_repository(root: str | Path) -> RepositoryIndex:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    files: list[PythonFileIndex] = []
    for path in sorted(root_path.rglob("*.py")):
        if _ignored(path):
            continue
        files.append(_index_python_file(root_path, path))

    symbols = [symbol for file in files for symbol in file.symbols]
    imports = [binding for file in files for binding in file.imports]
    calls = [call for file in files for call in file.calls]
    _resolve_calls(files, symbols, calls)
    dataflow = _build_dataflow(calls)
    config_entries = [entry for file in files for entry in file.config_entries]
    config_entries.extend(_index_config_files(root_path))
    entrypoints = _entrypoint_candidates(files)

    return RepositoryIndex(
        root=root_path,
        files=files,
        symbols=symbols,
        imports=imports,
        calls=calls,
        dataflow=dataflow,
        config_entries=config_entries,
        entrypoints=entrypoints,
    )


def _index_python_file(root: Path, path: Path) -> PythonFileIndex:
    relative = path.relative_to(root).as_posix()
    module = _module_name(relative)
    source = path.read_text(encoding="utf-8", errors="replace")
    index = PythonFileIndex(path=relative, module=module)
    if path.name.lower() in _ENTRYPOINT_NAMES:
        index.entrypoint_reasons.append((4, f"common entrypoint filename: {path.name}"))

    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        index.parse_error = f"{exc.msg} at line {exc.lineno}"
        return index

    analyzer = _FileAnalyzer(index, tree)
    analyzer.run()
    return index


class _FileAnalyzer(ast.NodeVisitor):
    def __init__(self, index: PythonFileIndex, tree: ast.Module) -> None:
        self.index = index
        self.tree = tree
        self._scope: list[PythonSymbol] = []
        self._class_names: list[str] = []
        self._result_targets: list[str] = []
        self._call_order = 0
        self._imports_framework = False

    def run(self) -> None:
        module_span = SourceSpan(
            path=self.index.path,
            start_line=1,
            end_line=max(
                (getattr(node, "end_lineno", 1) or 1 for node in self.tree.body),
                default=1,
            ),
            symbol=self.index.module,
        )
        module_symbol = PythonSymbol(
            id=_symbol_id(self.index.path, self.index.module, 1),
            module=self.index.module,
            qualname=self.index.module,
            name=self.index.module.rsplit(".", 1)[-1],
            kind=SymbolKind.MODULE,
            span=module_span,
        )
        self.index.symbols.append(module_symbol)
        self._scope.append(module_symbol)
        self.visit(self.tree)
        self._scope.pop()
        if self._imports_framework:
            self.index.entrypoint_reasons.append((1, "imports an ML framework"))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", 1)[0]
            self.index.imports.append(
                ImportBinding(
                    module=self.index.module,
                    local_name=local_name,
                    imported_module=alias.name,
                    imported_name=None,
                    span=_span(self.index.path, node, f"import {alias.name}"),
                )
            )
            if alias.name.split(".", 1)[0] in _FRAMEWORK_IMPORTS:
                self._imports_framework = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported_module = _absolute_import_module(
            self.index.module,
            node.module,
            node.level,
        )
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.index.imports.append(
                ImportBinding(
                    module=self.index.module,
                    local_name=local_name,
                    imported_module=imported_module,
                    imported_name=alias.name,
                    span=_span(
                        self.index.path,
                        node,
                        f"from {imported_module} import {alias.name}",
                    ),
                )
            )
            if imported_module.split(".", 1)[0] in _FRAMEWORK_IMPORTS:
                self._imports_framework = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = self._scope[-1]
        qualname = _child_qualname(parent, node.name)
        symbol = PythonSymbol(
            id=_symbol_id(self.index.path, qualname, node.lineno),
            module=self.index.module,
            qualname=qualname,
            name=node.name,
            kind=SymbolKind.CLASS,
            span=_span(self.index.path, node, qualname),
            parent_id=parent.id,
        )
        self.index.symbols.append(symbol)
        self._scope.append(symbol)
        self._class_names.append(node.name)
        for item in node.body:
            self.visit(item)
        self._class_names.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._scope[-1]
        qualname = _child_qualname(parent, node.name)
        kind = SymbolKind.METHOD if self._class_names else SymbolKind.FUNCTION
        symbol = PythonSymbol(
            id=_symbol_id(self.index.path, qualname, node.lineno),
            module=self.index.module,
            qualname=qualname,
            name=node.name,
            kind=kind,
            span=_span(self.index.path, node, qualname),
            parent_id=parent.id,
        )
        self.index.symbols.append(symbol)
        self._scan_decorators(node)
        self._scope.append(symbol)
        for item in node.body:
            self.visit(item)
        self._scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = _assignment_names(node.targets)
        if isinstance(node.value, ast.Call):
            previous = self._result_targets
            self._result_targets = targets
            self.visit(node.value)
            self._result_targets = previous
        else:
            self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            return
        targets = _assignment_names([node.target])
        if isinstance(node.value, ast.Call):
            previous = self._result_targets
            self._result_targets = targets
            self.visit(node.value)
            self._result_targets = previous
        else:
            self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node.test):
            self.index.entrypoint_reasons.append((6, "contains __main__ guard"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _expr_text(node.func)
        caller = self._scope[-1]
        call = CallSite(
            id=_call_id(self.index.path, node, self._call_order),
            module=self.index.module,
            caller_symbol_id=caller.id,
            callee_text=callee,
            span=_span(self.index.path, node, callee),
            order=self._call_order,
            result_targets=list(self._result_targets),
            argument_names=sorted(_call_argument_names(node)),
        )
        self._call_order += 1
        self.index.calls.append(call)
        self._scan_call_config(node, callee)

        previous = self._result_targets
        self._result_targets = []
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._result_targets = previous

    def _scan_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            name = _expr_text(decorator.func)
            if name.endswith("hydra.main") or name == "hydra.main":
                metadata = {
                    keyword.arg: _literal_or_text(keyword.value)
                    for keyword in decorator.keywords
                    if keyword.arg is not None
                }
                self.index.config_entries.append(
                    ConfigEntry(
                        id=f"config.python.{self.index.path}.{node.lineno}.hydra",
                        kind="hydra_entrypoint",
                        path=self.index.path,
                        key=node.name,
                        value=metadata,
                        span=_span(self.index.path, decorator, "hydra.main"),
                    )
                )
                self.index.entrypoint_reasons.append((4, "contains @hydra.main entrypoint"))

    def _scan_call_config(self, node: ast.Call, callee: str) -> None:
        if callee.endswith(".add_argument") and node.args:
            key = _literal_or_text(node.args[0])
            if isinstance(key, str):
                metadata = {
                    keyword.arg: _literal_or_text(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                self.index.config_entries.append(
                    ConfigEntry(
                        id=f"config.python.{self.index.path}.{node.lineno}.{self._call_order}",
                        kind="argparse_argument",
                        path=self.index.path,
                        key=key,
                        value=metadata.get("default"),
                        span=_span(self.index.path, node, callee),
                        metadata=metadata,
                    )
                )
                self.index.entrypoint_reasons.append((1, "defines argparse options"))
        if callee in {"OmegaConf.load", "omegaconf.OmegaConf.load"} and node.args:
            self.index.config_entries.append(
                ConfigEntry(
                    id=f"config.python.{self.index.path}.{node.lineno}.omegaconf",
                    kind="omegaconf_load",
                    path=self.index.path,
                    key="config_path",
                    value=_literal_or_text(node.args[0]),
                    span=_span(self.index.path, node, callee),
                )
            )


def _resolve_calls(
    files: list[PythonFileIndex],
    symbols: list[PythonSymbol],
    calls: list[CallSite],
) -> None:
    symbol_by_id = {symbol.id: symbol for symbol in symbols}
    full_symbols: dict[str, PythonSymbol] = {}
    for symbol in symbols:
        if symbol.kind == SymbolKind.MODULE:
            full_symbols[symbol.module] = symbol
        else:
            local_qualname = _local_qualname(symbol)
            full_symbols[f"{symbol.module}.{local_qualname}"] = symbol

    imports_by_module = {
        file.module: {binding.local_name: binding for binding in file.imports} for file in files
    }

    for call in calls:
        caller = symbol_by_id[call.caller_symbol_id]
        imports = imports_by_module.get(call.module, {})
        target, resolution = _resolve_callee_target(
            call.callee_text,
            caller,
            imports,
            full_symbols,
        )
        call.resolved_target = target
        call.resolution = resolution
        if target is not None and target in full_symbols:
            call.resolved_symbol_id = full_symbols[target].id
            call.resolution = CallResolution.LOCAL


def _resolve_callee_target(
    callee: str,
    caller: PythonSymbol,
    imports: dict[str, ImportBinding],
    symbols: dict[str, PythonSymbol],
) -> tuple[str | None, CallResolution]:
    if not callee:
        return None, CallResolution.DYNAMIC

    if "." not in callee:
        same_module = f"{caller.module}.{callee}"
        if same_module in symbols:
            return same_module, CallResolution.LOCAL
        binding = imports.get(callee)
        if binding is not None:
            target = binding.imported_module
            if binding.imported_name is not None:
                target = f"{target}.{binding.imported_name}"
            return target, (CallResolution.LOCAL if target in symbols else CallResolution.EXTERNAL)
        return None, CallResolution.DYNAMIC

    head, tail = callee.split(".", 1)
    if head == "self" and caller.kind == SymbolKind.METHOD:
        class_name = _caller_class_name(caller)
        if class_name is not None:
            target = f"{caller.module}.{class_name}.{tail}"
            if target in symbols:
                return target, CallResolution.LOCAL
        return None, CallResolution.DYNAMIC

    binding = imports.get(head)
    if binding is not None:
        base = binding.imported_module
        if binding.imported_name is not None:
            base = f"{base}.{binding.imported_name}"
        target = f"{base}.{tail}"
        return target, (CallResolution.LOCAL if target in symbols else CallResolution.EXTERNAL)

    return None, CallResolution.DYNAMIC


def _build_dataflow(calls: list[CallSite]) -> list[DataFlowLink]:
    grouped: dict[str, list[CallSite]] = {}
    for call in calls:
        grouped.setdefault(call.caller_symbol_id, []).append(call)

    links: list[DataFlowLink] = []
    for scope_calls in grouped.values():
        scope_calls.sort(
            key=lambda item: (
                item.span.start_line,
                item.span.start_column or 0,
                item.order,
            )
        )
        last_producer: dict[str, str] = {}
        for call in scope_calls:
            for name in call.argument_names:
                producer = last_producer.get(name)
                if producer is not None and producer != call.id:
                    links.append(
                        DataFlowLink(
                            producer_call_id=producer,
                            consumer_call_id=call.id,
                            variable=name,
                        )
                    )
            for target in call.result_targets:
                last_producer[target] = call.id
    return links


def _entrypoint_candidates(files: list[PythonFileIndex]) -> list[EntrypointCandidate]:
    candidates: list[EntrypointCandidate] = []
    for file in files:
        if not file.entrypoint_reasons:
            continue
        reasons = [reason for _, reason in file.entrypoint_reasons]
        score = sum(points for points, _ in file.entrypoint_reasons)
        candidates.append(EntrypointCandidate(path=file.path, score=score, reasons=reasons))
    return sorted(candidates, key=lambda item: (-item.score, item.path))


def _index_config_files(root: Path) -> list[ConfigEntry]:
    entries: list[ConfigEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _ignored(path):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".json", ".toml", ".yaml", ".yml"}:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                entries.extend(_flatten_config(relative, "json", data))
            elif suffix == ".toml":
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                entries.extend(_flatten_config(relative, "toml", data))
            else:
                entries.extend(_scan_yaml(relative, path.read_text(encoding="utf-8")))
        except (OSError, ValueError, tomllib.TOMLDecodeError):
            continue
    return entries


def _flatten_config(
    path: str,
    kind: str,
    value: Any,
    prefix: str = "",
) -> list[ConfigEntry]:
    if isinstance(value, dict):
        entries: list[ConfigEntry] = []
        for key, item in value.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            entries.extend(_flatten_config(path, kind, item, child))
        return entries
    return [
        ConfigEntry(
            id=f"config.file.{safe_id(path)}.{safe_id(prefix or 'root')}",
            kind=kind,
            path=path,
            key=prefix or "root",
            value=value,
            span=SourceSpan(path=path, start_line=1),
        )
    ]


def _scan_yaml(path: str, source: str) -> list[ConfigEntry]:
    entries: list[ConfigEntry] = []
    stack: list[tuple[int, str]] = []
    for line_number, raw in enumerate(source.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([A-Za-z0-9_.-]+)\s*:\s*(.*?)\s*$", raw)
        if match is None:
            continue
        indent = len(match.group(1).replace("\t", "    "))
        key = match.group(2)
        raw_value = match.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        prefix = ".".join(item[1] for item in stack)
        full_key = key if not prefix else f"{prefix}.{key}"
        if raw_value:
            entries.append(
                ConfigEntry(
                    id=f"config.file.{safe_id(path)}.{line_number}.{safe_id(full_key)}",
                    kind="yaml",
                    path=path,
                    key=full_key,
                    value=_parse_yaml_scalar(raw_value),
                    span=SourceSpan(
                        path=path,
                        start_line=line_number,
                        end_line=line_number,
                    ),
                )
            )
        else:
            stack.append((indent, key))
    return entries


def _parse_yaml_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _module_name(relative: str) -> str:
    path = Path(relative)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__root__"


def _absolute_import_module(current_module: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = current_module.split(".")
    package = parts[:-1]
    trim = max(level - 1, 0)
    if trim:
        package = package[:-trim] if trim <= len(package) else []
    if module:
        package.extend(module.split("."))
    return ".".join(package)


def _child_qualname(parent: PythonSymbol, name: str) -> str:
    if parent.kind == SymbolKind.MODULE:
        return name
    return f"{parent.qualname}.{name}"


def _local_qualname(symbol: PythonSymbol) -> str:
    return symbol.qualname


def _caller_class_name(symbol: PythonSymbol) -> str | None:
    parts = symbol.qualname.split(".")
    return parts[-2] if len(parts) >= 2 else None


def _span(path: str, node: ast.AST, symbol: str | None = None) -> SourceSpan:
    return SourceSpan(
        path=path,
        start_line=getattr(node, "lineno", 1),
        end_line=getattr(node, "end_lineno", None),
        start_column=getattr(node, "col_offset", None),
        end_column=getattr(node, "end_col_offset", None),
        symbol=symbol,
    )


def _symbol_id(path: str, qualname: str, line: int) -> str:
    return f"symbol.{safe_id(path)}.{safe_id(qualname)}.{line}"


def _call_id(path: str, node: ast.Call, order: int) -> str:
    return f"callsite.{safe_id(path)}.{node.lineno}.{node.col_offset}.{order}"


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "root"


def _expr_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expr_text(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _expr_text(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_text(node.value)
    return "<dynamic>"


def _assignment_names(targets: list[ast.expr]) -> list[str]:
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(_assignment_names(list(target.elts)))
    return names


def _call_argument_names(node: ast.Call) -> set[str]:
    names: set[str] = set()
    for value in [*node.args, *(keyword.value for keyword in node.keywords)]:
        for child in ast.walk(value):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _literal_or_text(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _expr_text(node)


def _is_main_guard(node: ast.expr) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left = node.left
    right = node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _ignored(path: Path) -> bool:
    return any(part in _IGNORED_PARTS for part in path.parts)
