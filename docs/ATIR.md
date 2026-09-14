# ATIR v0.1 — ArchTrace Intermediate Representation

ATIR is the canonical evidence graph of ArchTrace. It exists so that repository analysis, runtime tracing, semantic interpretation, interactive visualization, publication rendering, and model comparison operate on the **same underlying facts**.

## 1. Non-negotiable invariants

1. **Every view is a projection of ATIR.** The paper view is not maintained separately from the runtime graph.
2. **Mechanical facts and semantic interpretation are separate.** A `matmul` remains a `matmul` even if it is grouped into `Cross-Attention`.
3. **Every non-trivial claim can carry evidence.** Evidence records where a fact came from and how certain it is.
4. **A runtime trace is one observed execution, not the whole program.** Multiple runs may coexist.
5. **Hierarchy is reversible.** High-level nodes point to the lower-level nodes they contain; users can always drill down.
6. **Unknown is a valid value.** ArchTrace must not invent shape, semantics, control flow, or source provenance to make a graph look complete.
7. **Human corrections are overlays with provenance.** They must not destroy the original automated evidence.

## 2. Evidence taxonomy

ATIR distinguishes how a fact entered the graph:

| Evidence kind | Meaning | Typical source |
|---|---|---|
| `runtime` | directly observed during execution | hooks, operator dispatch, profiler, exported program |
| `static` | inferred from program structure without executing it | AST, symbol table, data-flow, call graph |
| `source` | literal identity/provenance in source | file, line, callable, config declaration |
| `author` | explicitly stated by repository/paper authors | README, paper, comments, config names |
| `semantic` | interpreted architectural meaning | deterministic rules or semantic model |
| `user` | human correction/annotation | editor overlay |
| `imported` | fact from an external graph format | ONNX, FX/export, external tracer |

Every evidence record also has a fact status:

- `observed`
- `inferred`
- `declared`
- `corrected`
- `unknown`

Confidence is numeric but **confidence does not turn inference into observation**.

## 3. Abstraction levels

### L0 `paper`

Minimal communicative architecture. This level is intentionally selective and optimized for explanation/publication.

### L1 `semantic`

Meaningful mechanisms and roles: encoders, fusion, memory, policy, dynamics model, planner, decoder, loss, controller, environment, etc.

### L2 `module`

Framework/module hierarchy and parameterized reusable components.

### L3 `operation`

Source-visible operations and tensors. The target atomic boundary is a source-visible tensor-producing transformation rather than a compiler/CUDA instruction.

### L4 `source`

Source files/spans, call sites, configuration definitions, runtime call stacks, and concrete run metadata.

The hierarchy is not assumed to be a strict tree. Shared modules, tied parameters, reused functions, recurrent execution, and multi-role components require a graph.

## 4. Node identity

A stable node ID identifies an entity *within one ATIR document*. Runtime occurrences that execute the same source/module multiple times should be representable independently from the reusable definition.

Planned identity split for v0.2:

```text
Definition       Execution occurrence       Value
module.foo  ───→ call.run3.0042 ─────────→ tensor.run3.0088
```

This avoids collapsing loops and repeated/shared modules.

## 5. Tensor representation

A tensor/value may carry:

```yaml
shape: [B, T, 256, 1024]
dtype: float16
device: cuda:0
layout: contiguous
requires_grad: true
semantics:
  - visual_tokens
```

Dimensions may be concrete integers, symbolic strings, or unknown (`null`).

Future versions will add:

- stride/storage/alias identity;
- named dimensions and coordinate semantics;
- ragged/sparse/quantized values;
- structured values (`dict`, tuple, dataclass, PyTree);
- distribution/random-variable metadata;
- mutation/version relationships.

## 6. Edge semantics

ATIR does not use one generic arrow. Edges encode relations such as:

- `data`: value/information flow;
- `control`: execution/control dependency;
- `contains`: hierarchy/grouping;
- `calls`: source/runtime invocation;
- `implements`: lower-level entities implementing a higher-level concept;
- `parameter`: parameter dependency;
- `gradient`: backward/gradient flow;
- `alias`: values sharing identity/storage;
- `reads` / `writes`: state/config/resource access;
- `derived_from`: inference/projection relationship;
- `next`: temporal/environment transition.

Graph views choose which edge classes to show rather than changing the underlying graph.

## 7. Multi-run model

Dynamic architectures require multiple executions. `TraceRun` records the entrypoint, arguments, configuration, framework versions, and arbitrary environment metadata.

The same ATIR project may therefore contain:

```text
run.train.seed0
run.eval.task_pick_place
run.eval.task_handover
run.eval.occluded_case
```

Coverage is accumulated across runs. ArchTrace must be able to distinguish:

- path observed in every run;
- path observed in some runs;
- statically reachable but never observed;
- unreachable/dead under current configuration;
- unresolved because analysis is incomplete.

## 8. Semantic aggregation

Semantic nodes are evidence-backed groupings over mechanics:

```text
Semantic: Cross-Modal Fusion
    contains/implements
        Module: model.fusion
        Module: model.cross_attention
            contains
                op.823 layer_norm
                op.824 linear
                op.825 reshape
                op.826 attention
                ...
```

A semantic engine may propose such a grouping, but it must retain:

- supporting source/runtime nodes;
- evidence records;
- confidence;
- competing interpretations when necessary.

## 9. Contradictions

Contradiction is expected rather than treated as corruption.

Example:

```text
README:       "frozen vision encoder"
config:       freeze_vision=false
runtime:      vision parameters require_grad=True
optimizer:    vision parameters are present
```

ATIR should preserve all four facts and expose a contradiction to higher-level analysis. Planned v0.2 introduces explicit `Claim` and `Conflict` entities.

## 10. Serialization

The first canonical representation is versioned JSON generated by Pydantic models. The extension used by the CLI is currently:

```text
*.atir.json
```

The schema must remain independent of any visualization library and any one ML framework.

## 11. Query contract

Everything ArchTrace ultimately does should be expressible as queries/projections over ATIR, for example:

- trace all paths from `RGB` to `Action`;
- find every shape-changing operation;
- find fusion points between vision and language lineages;
- show all operations implemented by `Cross-Modal Fusion`;
- find semantic nodes unsupported by runtime evidence;
- find author claims contradicted by execution;
- compare modality bottlenecks between two models;
- render a paper view with at most N major nodes;
- jump from any visual node to exact source spans.

This queryability is why ATIR, rather than a renderer or tracer, is the center of ArchTrace.
