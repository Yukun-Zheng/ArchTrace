# ATIR v0.2 — ArchTrace Intermediate Representation

ATIR is the canonical evidence graph of ArchTrace. Repository analysis, runtime tracing, semantic interpretation, interactive visualization, publication rendering, and model comparison must all operate on this common representation rather than maintaining unrelated graphs.

## 1. Core invariants

1. **Every view is a projection of ATIR.** Paper, semantic, module, tensor/op, and source views are different projections of the same evidence graph.
2. **Mechanics and interpretation remain separate.** A low-level operation never disappears merely because it is grouped into a semantic component.
3. **Definition is not execution.** Reusable code/module/operator definitions are distinct from each concrete runtime occurrence.
4. **Execution is not value.** A call/op occurrence is distinct from the tensors or other values it consumes and produces.
5. **Runtime is evidence for one run, not proof of every possible path.** Multi-run coverage is explicit.
6. **Claims carry provenance.** Author statements, static inference, runtime observation, semantic interpretation, and user correction can coexist.
7. **Contradictions are preserved.** Conflicting claims are represented, not silently overwritten.
8. **Unknown is valid.** ArchTrace must not fabricate missing mechanics to complete a diagram.
9. **Hierarchy is reversible.** Any high-level component can be expanded back to underlying definitions, occurrences, values, and source spans.
10. **Record IDs are globally unique within a document.** Cross-references therefore remain unambiguous.

## 2. Identity model

ATIR v0.2 introduces `IdentityKind`:

| identity | meaning |
|---|---|
| `group` | semantic/paper/repository aggregation |
| `definition` | reusable module/function/operator definition |
| `occurrence` | one concrete execution occurrence in one run |
| `value` | runtime or symbolic value/tensor |
| `state` | parameter, buffer, config, mutable state |
| `source` | source-code artifact/span container |

The central distinction is:

```text
Definition                  Execution occurrence                 Value
module.shared ────────────→ call.run0.0042 ───────────────────→ tensor.run0.0088
      │                              │                                  │
models/block.py:31          run=run.0, occurrence=3           [B, 256, 1024]
```

A shared block called ten times therefore remains **one definition plus ten occurrences**, not one misleading runtime node.

Occurrence nodes may contain:

```yaml
identity_kind: occurrence
definition_id: module.shared
run_id: run.0
occurrence_index: 3
```

`definition_id` is validated to point to a `definition` node.

## 3. Architectural levels

Identity and abstraction level are orthogonal.

### L0 `paper`
Communication-oriented architecture: major inputs, proposed mechanisms, outputs, and the relationships worth showing in a paper main figure.

### L1 `semantic`
Meaningful roles such as vision encoder, multimodal fusion, memory, world model, policy backbone, action head, loss, planner, controller, or environment.

### L2 `module`
Framework/module hierarchy, repeated blocks, shared parameters, and module-level I/O.

### L3 `operation`
Source-visible tensor-producing operations and runtime values. This is ArchTrace's intended mechanical atomic boundary.

### L4 `source`
Files, spans, symbols, call sites, configs, call stacks, and execution metadata.

## 4. Evidence taxonomy

Evidence records state **how a fact entered the graph**:

| kind | meaning |
|---|---|
| `runtime` | directly observed during execution |
| `static` | inferred from code without executing it |
| `source` | literal source/config provenance |
| `author` | declared by paper/repository authors |
| `semantic` | architectural interpretation |
| `user` | explicit human correction/annotation |
| `imported` | imported from an external graph/tracer |

Every evidence item also carries a fact status:

- `observed`
- `inferred`
- `declared`
- `corrected`
- `unknown`

Confidence does not change epistemic type: a 0.99 inference is still not an observation.

## 5. Claims

A `Claim` is a proposition about an ATIR entity:

```yaml
id: claim.vision.frozen.author
subject_id: module.vision
predicate: frozen
value: true
status: declared
evidence_ids:
  - evidence.readme.freeze
confidence: 1.0
```

A claim may point to another ATIR entity via `object_id` instead of a literal `value`. Exactly one is required.

Scopes can bind a claim to particular runs or conditions:

```yaml
scope:
  run_ids: [run.eval.0]
  conditions:
    config: configs/eval.yaml
```

This is essential because many architecture facts are configuration- or path-dependent.

## 6. Conflicts

Contradictions are first-class:

```text
README claim:  VisionEncoder.frozen = true
runtime claim: VisionEncoder.frozen = false
optimizer:     VisionEncoder parameters present
```

ATIR preserves the claims and creates a `Conflict` linking them. Conflict kinds include:

- contradictory claims;
- author/implementation mismatch;
- static/runtime mismatch;
- coverage mismatch;
- other.

A conflict remains `open`, may be explicitly `accepted`, or becomes `resolved` with a recorded resolution. Resolution never deletes the original evidence.

## 7. Multi-run coverage

Dynamic programs require more than one trace. `CoverageRecord` distinguishes:

- `always_observed`
- `sometimes_observed`
- `static_reachable_unobserved`
- `unreachable_under_configuration`
- `unresolved`

Example:

```yaml
subject_id: module.optional_depth_branch
status: sometimes_observed
considered_run_ids: [run.0, run.1, run.2]
observed_run_ids: [run.1]
```

ATIR validates that observed runs are a subset of considered runs and that strong statuses such as `always_observed` are internally consistent.

This makes statements such as "the model uses this branch" precise rather than binary and misleading.

## 8. Tensor/value representation

`TensorSpec` currently records:

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

The next value-schema extension will cover:

- PyTrees / dict / tuple / dataclass structures;
- scalar and non-tensor values;
- stride/storage identity and aliasing;
- mutation/version chains;
- sparse, ragged, and quantized values;
- named dimensions and coordinate semantics;
- distributions and stochastic values.

These extensions must preserve the definition/occurrence/value identity split introduced in v0.2.

## 9. Edge semantics

ATIR does not use one generic arrow. Relations include:

- `data`
- `control`
- `contains`
- `calls`
- `implements`
- `instance_of`
- `produces`
- `consumes`
- `parameter`
- `gradient`
- `alias`
- `reads` / `writes`
- `derived_from`
- `next`

Views choose which relations to render; they do not rewrite the underlying evidence graph.

## 10. v0.1 → v0.2 migration

`load_atir_json()` accepts v0.1 JSON and upgrades it before validation.

Migration is intentionally conservative:

- source nodes → `source`;
- module/operation nodes → `definition`;
- tensor/input/output nodes → `value`;
- parameter/config nodes → `state`;
- other nodes → `group`;
- empty claims/conflicts/coverage collections are added;
- **no runtime occurrence is invented**.

The payload records:

```yaml
metadata:
  schema_migrations:
    - from: "0.1"
      to: "0.2"
```

Unknown future schema versions are rejected rather than guessed.

## 11. What v0.2 now makes possible

A future PyTorch backend can normalize a run as:

```text
module.block                  # reusable definition
    │
    ├── call.run0.001         # occurrence 1
    │      ├── consumes → tensor.run0.010
    │      └── produces → tensor.run0.011
    │
    └── call.run0.017         # same definition, occurrence 2
           ├── consumes → tensor.run0.080
           └── produces → tensor.run0.081
```

A semantic engine can then group these into:

```text
Semantic: Cross-Modal Fusion
  implements → module.block
```

without collapsing the mechanics.

An author declaration can coexist with runtime truth:

```text
claim.author.freeze = true
claim.runtime.freeze = false
              │
              └── conflict.author_implementation
```

and several traces can quantify which paths were actually exercised.

## 12. Query contract

ArchTrace should ultimately answer queries over ATIR such as:

- show every execution occurrence of this shared definition;
- trace a concrete tensor from RGB input to action output;
- show all definitions that produced a value in run X;
- find operations that are statically reachable but never observed;
- find claims unsupported or contradicted by runtime evidence;
- find modality fusion points;
- compare shape-changing bottlenecks across two models;
- determine whether a supposedly frozen backbone received gradients;
- project a paper figure with at most N semantic nodes;
- jump from any rendered component to exact source evidence.

ATIR is therefore not a renderer format. It is the **mechanical and epistemic substrate** of ArchTrace.
