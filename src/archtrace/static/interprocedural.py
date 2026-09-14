"""Conservative interprocedural data-flow recovery for the Python static index."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from archtrace.static.python_index import (
    CallSite,
    PythonSymbol,
    RepositoryIndex,
    SymbolKind,
)


class BoundaryFlowKind(StrEnum):
    ARGUMENT = "argument_into_callee"
    RETURN = "return_from_callee"


@dataclass(slots=True)
class BoundaryFlowLink:
    producer_call_id: str
    consumer_call_id: str
    variable: str
    kind: BoundaryFlowKind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _FunctionFacts:
    symbol: PythonSymbol
    parameters: list[str]
    return_names: list[tuple[int, str]] = field(default_factory=list)
    return_calls: list[tuple[int, int]] = field(default_factory=list)


def analyze_interprocedural_flow(index: RepositoryIndex) -> list[BoundaryFlowLink]:
    """Recover direct argument and return flow across resolved local calls."""

    facts, ast_calls = _collect_function_facts(index)
    calls_by_id = {call.id: call for call in index.calls}
    calls_by_scope: dict[str, list[CallSite]] = {}
    for call in index.calls:
        calls_by_scope.setdefault(call.caller_symbol_id, []).append(call)

    links: list[BoundaryFlowLink] = []
    seen: set[tuple[str, str, str, BoundaryFlowKind]] = set()

    for outer in index.calls:
        target_id = outer.resolved_symbol_id
        if target_id is None or target_id not in facts:
            continue
        outer_ast = ast_calls.get(outer.id)
        if outer_ast is None:
            continue

        target = facts[target_id]
        bindings = _argument_bindings(outer_ast, target.parameters)
        for inner in calls_by_scope.get(target_id, []):
            for formal_name, actual_name in bindings.items():
                if formal_name not in inner.argument_names:
                    continue
                _append_link(
                    links,
                    seen,
                    BoundaryFlowLink(
                        producer_call_id=outer.id,
                        consumer_call_id=inner.id,
                        variable=actual_name,
                        kind=BoundaryFlowKind.ARGUMENT,
                        metadata={
                            "formal_parameter": formal_name,
                            "callee_symbol_id": target_id,
                        },
                    ),
                )

        for line, returned_name in target.return_names:
            producer = _latest_producer_before(
                calls_by_scope.get(target_id, []),
                returned_name,
                line,
            )
            if producer is None:
                continue
            _append_link(
                links,
                seen,
                BoundaryFlowLink(
                    producer_call_id=producer.id,
                    consumer_call_id=outer.id,
                    variable=returned_name,
                    kind=BoundaryFlowKind.RETURN,
                    metadata={"callee_symbol_id": target_id},
                ),
            )

        for line, column in target.return_calls:
            producer = _call_at_location(
                calls_by_scope.get(target_id, []),
                line,
                column,
            )
            if producer is None or producer.id not in calls_by_id:
                continue
            _append_link(
                links,
                seen,
                BoundaryFlowLink(
                    producer_call_id=producer.id,
                    consumer_call_id=outer.id,
                    variable="<return>",
                    kind=BoundaryFlowKind.RETURN,
                    metadata={"callee_symbol_id": target_id},
                ),
            )

    return links


def _collect_function_facts(
    index: RepositoryIndex,
) -> tuple[dict[str, _FunctionFacts], dict[str, ast.Call]]:
    symbols = {
        (symbol.span.path, symbol.span.start_line): symbol
        for symbol in index.symbols
        if symbol.kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
    }
    indexed_calls = {
        (
            call.span.path,
            call.span.start_line,
            call.span.start_column or 0,
        ): call
        for call in index.calls
    }
    facts: dict[str, _FunctionFacts] = {}
    ast_calls: dict[str, ast.Call] = {}

    for file in index.files:
        path = index.root / file.path
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"),
                filename=file.path,
            )
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = symbols.get((file.path, node.lineno))
                if symbol is None:
                    continue
                facts[symbol.id] = _function_facts(symbol, node)

            if isinstance(node, ast.Call):
                call = indexed_calls.get(
                    (
                        file.path,
                        getattr(node, "lineno", 1),
                        getattr(node, "col_offset", 0),
                    )
                )
                if call is not None:
                    ast_calls[call.id] = node

    return facts, ast_calls


def _function_facts(
    symbol: PythonSymbol,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _FunctionFacts:
    parameters = [argument.arg for argument in [*node.args.posonlyargs, *node.args.args]]
    if node.args.vararg is not None:
        parameters.append(node.args.vararg.arg)
    parameters.extend(argument.arg for argument in node.args.kwonlyargs)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg.arg)

    facts = _FunctionFacts(symbol=symbol, parameters=parameters)
    collector = _ReturnCollector(facts)
    for statement in node.body:
        collector.visit(statement)
    return facts


class _ReturnCollector(ast.NodeVisitor):
    def __init__(self, facts: _FunctionFacts) -> None:
        self.facts = facts

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            return
        if isinstance(node.value, ast.Name):
            self.facts.return_names.append((node.lineno, node.value.id))
        elif isinstance(node.value, ast.Call):
            self.facts.return_calls.append(
                (
                    getattr(node.value, "lineno", node.lineno),
                    getattr(node.value, "col_offset", 0),
                )
            )


def _argument_bindings(node: ast.Call, parameters: list[str]) -> dict[str, str]:
    usable = list(parameters)
    if usable and usable[0] in {"self", "cls"}:
        usable = usable[1:]

    bindings: dict[str, str] = {}
    for formal, actual in zip(usable, node.args, strict=False):
        if isinstance(actual, ast.Name):
            bindings[formal] = actual.id

    for keyword in node.keywords:
        if keyword.arg is not None and isinstance(keyword.value, ast.Name):
            bindings[keyword.arg] = keyword.value.id
    return bindings


def _latest_producer_before(
    calls: list[CallSite],
    variable: str,
    line: int,
) -> CallSite | None:
    candidates = [
        call for call in calls if variable in call.result_targets and call.span.start_line <= line
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda call: (
            call.span.start_line,
            call.span.start_column or 0,
            call.order,
        ),
    )


def _call_at_location(
    calls: list[CallSite],
    line: int,
    column: int,
) -> CallSite | None:
    for call in calls:
        if call.span.start_line == line and (call.span.start_column or 0) == column:
            return call
    return None


def _append_link(
    links: list[BoundaryFlowLink],
    seen: set[tuple[str, str, str, BoundaryFlowKind]],
    link: BoundaryFlowLink,
) -> None:
    key = (
        link.producer_call_id,
        link.consumer_call_id,
        link.variable,
        link.kind,
    )
    if key in seen:
        return
    seen.add(key)
    links.append(link)
