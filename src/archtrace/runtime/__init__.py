"""Runtime capture backends."""

from archtrace.runtime.pytorch import PyTorchTraceResult, trace_model

__all__ = ["PyTorchTraceResult", "trace_model"]
