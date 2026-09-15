# ArchTrace Real-World Benchmark

ArchTrace is evaluated as a reconstruction system, not only as a visualizer. The benchmark measures whether research repositories can be normalized into trustworthy ATIR while preserving explicit uncertainty.

## Principles

1. Every external repository is pinned to an exact commit.
2. Static analysis never imports or executes target repository code.
3. Runtime and hybrid measurements are reported separately from static measurements.
4. Dynamic or unresolved facts are benchmark outcomes, not reasons to fabricate structure.
5. Repository-specific hacks are prohibited unless generalized into a documented language/framework mechanism.
6. Metrics remain separate; there is deliberately no single vanity score.

## Current sequence

| Case | Initial mode | Purpose |
| --- | --- | --- |
| DP3 | static → runtime | point-cloud encoder + diffusion policy |
| Diffusion Policy | static → runtime | canonical robotics diffusion pipeline |
| openpi | static → runtime | large VLA / π0 family |
| RDT | static → runtime | diffusion transformer + multimodal attention |
| V-JEPA2 | static → runtime | video/world-model visual representation |
| RoboTwin | static → runtime | system-level simulator/controller/policy integration |

RoboTwin submodules such as XPolicyLab are tracked explicitly: an unmaterialized submodule is a benchmark warning rather than silently disappearing from the analyzed architecture.

## Static metrics

The first layer records Python parse success, call-resolution distribution, configuration resolution, entrypoints, static data-flow edges, ATIR size, source-span coverage, semantic coverage, and structured failure categories. These are coverage/calibration metrics, not substitutes for later hand-authored precision/recall ground truth.

## CLI

```bash
archtrace benchmark list benchmarks/manifest.toml
archtrace benchmark static benchmarks/manifest.toml dp3 /path/to/3D-Diffusion-Policy \
  -o benchmarks/results/dp3.static.json
archtrace benchmark report benchmarks/results/dp3.static.json \
  benchmarks/results/diffusion-policy.static.json \
  -o benchmarks/results/REPORT.md
```

Revision verification is mandatory by default. An explicit mismatch override exists only for exploratory diagnosis; those runs are not comparable with pinned scorecards.
