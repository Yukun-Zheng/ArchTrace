"""Public PyTorch runtime tracing API."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from archtrace.ir import ArchTraceIR
from archtrace.runtime._pytorch_capture import PyTorchModuleCapture
from archtrace.runtime._pytorch_dispatch import make_dispatch_mode


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
) -> PyTorchTraceResult:
    """Execute a ``torch.nn.Module`` once and capture runtime flow into ATIR.

    Module hooks recover hierarchy and repeated/shared module calls. By default,
    a ``TorchDispatchMode`` additionally records concrete ``aten.*`` operator
    occurrences, state/tensor dependencies, outputs, and basic in-place value
    versioning. Both layers normalize into one ATIR graph.
    """

    torch = _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("trace_model expects an instance of torch.nn.Module")

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
            output = model(*args, **({} if kwargs is None else dict(kwargs)))
    finally:
        capture.remove()

    graph = capture.finish()
    graph.metadata["operator_dispatch"] = capture_operators
    return PyTorchTraceResult(ir=graph, output=output)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch tracing requires the optional dependency. "
            "Install with `pip install -e '.[pytorch]'`."
        ) from exc
