"""PyTorch adapter helpers kept independent from a hard torch import."""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator
from typing import Any

from archtrace.ir import SourceSpan, TensorSpec


def named_modules(model: Any) -> Iterator[tuple[str, Any]]:
    """Yield aliases too when the installed PyTorch supports it."""

    try:
        yield from model.named_modules(remove_duplicate=False)
    except TypeError:
        yield from model.named_modules()


def iter_tensors(value: Any, torch: Any) -> Iterator[Any]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item, torch)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item, torch)


def tensor_spec(tensor: Any) -> TensorSpec:
    shape: list[int | str | None] = []
    for dimension in tensor.shape:
        try:
            shape.append(int(dimension))
        except (TypeError, ValueError):
            shape.append(str(dimension))
    return TensorSpec(
        shape=shape,
        dtype=str(tensor.dtype).removeprefix("torch."),
        device=str(tensor.device),
        layout=str(tensor.layout).removeprefix("torch."),
        requires_grad=bool(tensor.requires_grad),
    )


def source_span(callable_obj: Any) -> SourceSpan | None:
    try:
        path = inspect.getsourcefile(callable_obj)
        lines, start_line = inspect.getsourcelines(callable_obj)
    except (OSError, TypeError):
        return None
    if path is None:
        return None
    symbol = getattr(callable_obj, "__qualname__", getattr(callable_obj, "__name__", None))
    return SourceSpan(
        path=path,
        start_line=start_line,
        end_line=start_line + max(len(lines) - 1, 0),
        symbol=str(symbol) if symbol is not None else None,
    )


def safe_fragment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return normalized or "root"
