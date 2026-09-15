# ArchTrace Benchmark Report

| Case | Status | Parse | Call resolution | Semantic coverage | Nodes | Edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dp3 | completed | 100.0% | 64.9% | 98.0% | 7221 | 12080 |
| diffusion-policy | completed | 100.0% | 66.4% | 93.9% | 19523 | 40838 |
| openpi | completed | 100.0% | 71.4% | 57.5% | 6035 | 12380 |
| rdt | completed | 100.0% | 76.9% | 67.8% | 8682 | 16124 |
| vjepa2 | completed | 100.0% | 59.3% | 78.7% | 18982 | 20623 |
| robotwin | completed | 100.0% | 58.3% | 64.2% | 12281 | 24204 |

## Failure taxonomy

- **dp3**
  - `dynamic_call` (warning, n=817): Call targets depend on runtime values and remain dynamic.
  - `config_dynamic` (warning, n=2): Configuration path/name depends on runtime values.
- **diffusion-policy**
  - `dynamic_call` (warning, n=2884): Call targets depend on runtime values and remain dynamic.
  - `config_dynamic` (warning, n=5): Configuration path/name depends on runtime values.
- **openpi**
  - `dynamic_call` (warning, n=1187): Call targets depend on runtime values and remain dynamic.
  - `submodule_unmaterialized` (warning, n=2): Git submodules are declared but not materialized in this checkout.
- **rdt**
  - `dynamic_call` (warning, n=1040): Call targets depend on runtime values and remain dynamic.
- **vjepa2**
  - `dynamic_call` (warning, n=1952): Call targets depend on runtime values and remain dynamic.
- **robotwin**
  - `dynamic_call` (warning, n=2925): Call targets depend on runtime values and remain dynamic.
