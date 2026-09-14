"""Public PyTorch runtime tracing API."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from archtrace.ir import ArchTraceIR
from archtrace.runtime._pytorch_capture import PyTorchModuleCapture
from archtrace.runtime._pytorch_dispatch import make_dispatch_mode
from archtrace.runtime._pytorch_structured import capture_structured_graphs


@dataclass(slots=True)
class PyTorchTraceResult:
    """One concrete model execution and its normalized ATIR graph."""

    ir: ArchTraceIR
    output: Any


def trace_model(
    model: Any,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    run_id: str = "run.0",
    project_name: str | None = None,
    capture_operators: bool = True,
    capture_fx: bool = True,
    capture_export: bool = True,
) -> PyTorchTraceResult:
    """Execute a ``torch.nn.Module`` once and capture multiple evidence layers.

    Runtime hooks recover module hierarchy and concrete call occurrences;
    ``TorchDispatchMode`` records concrete ``aten.*`` operations and tensor/state
    flow. FX and ``torch.export`` are then attempted as non-fatal independent
    structural evidence sources and aligned back to runtime definitions.
    """

    torch = _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("trace_model expects an instance of torch.nn.Module")

    actual_kwargs = {} if kwargs is None else dict(kwargs)
    capture = PyTorchModuleCapture(
        model,
        torch=torch,
        run_id=run_id,
        project_name=project_name or type(model).__name__,
    )
    dispatch_context = (
        make_dispatch_mode(capture, torch) if capture_operators else nullcontext()
    )

    capture.install()
    try:
        with dispatch_context:
            output = model(*args, **actual_kwargs)
    finally:
        capture.remove()

    structured_reports = capture_structured_graphs(
        capture,
        model,
        args,
        actual_kwargs,
        torch=torch,
        capture_fx=capture_fx,
        capture_export=capture_export,
    )

    graph = capture.finish()
    graph.metadata["operator_dispatch"] = capture_operators
    graph.metadata["structured_captures"] = [
        report.to_dict() for report in structured_reports
    ]
    graph = ArchTraceIR.model_validate(graph.model_dump())
    return PyTorchTraceResult(ir=graph, output=output)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch tracing requires the optional dependency. "
            "Install with `pip install -e '.[pytorch]'`."
        ) from exc
