"""Public PyTorch runtime tracing API."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from archtrace.ir import ArchTraceIR
from archtrace.runtime._pytorch_capture import PyTorchModuleCapture


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
) -> PyTorchTraceResult:
    """Execute a ``torch.nn.Module`` once and capture module/tensor runtime flow.

    M1 initially captures module definitions, concrete call occurrences, nesting,
    tensor inputs/outputs, and source provenance. Operator-dispatch and FX/export
    evidence will enrich the same ATIR identities rather than form separate graphs.
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
    capture.install()
    try:
        output = model(*args, **({} if kwargs is None else dict(kwargs)))
    finally:
        capture.remove()

    return PyTorchTraceResult(ir=capture.finish(), output=output)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch tracing requires the optional dependency. "
            "Install with `pip install -e '.[pytorch]'`."
        ) from exc
