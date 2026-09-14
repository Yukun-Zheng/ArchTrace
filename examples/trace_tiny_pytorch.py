"""Run a real PyTorch model once and print runtime-grounded ATIR."""

import torch

from archtrace.runtime import trace_model


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 8),
        )
        self.shared = block

    def forward(self, x):
        x = self.shared(x)
        return self.shared(x)


model = TinyPolicy()
result = trace_model(model, (torch.randn(2, 8),), run_id="run.demo")
print(result.ir.model_dump_json(indent=2))
