# ArchTrace

> **From research code to a source-grounded, explorable model architecture.**

**Live explorer:** https://archtrace.vercel.app

ArchTrace is an open-source system for reconstructing, tracing, understanding, and visualizing machine-learning architectures directly from research repositories.

The long-term goal is deliberately broader than a model visualizer. ArchTrace aims to unify **repository understanding, static analysis, runtime tracing, tensor/data-flow reconstruction, semantic architecture recovery, source-code grounding, interactive exploration, cross-model comparison, and publication-quality figure generation** in one system.

## Vision

Given a research repository, an entry point/configuration, and optionally a sample input or checkpoint, ArchTrace should be able to answer all of the following from one unified representation:

- What is the paper/model doing semantically?
- What is the actual execution path for this run?
- How does every tensor change in shape, dtype, device, and meaning?
- Which modules and operators consume and produce each value?
- Where in the source code did every operation originate?
- How do low-level operations aggregate into modules and semantic components?
- How does information flow from raw input to final output/action/loss?
- Which parts are static, conditional, repeated, shared, stochastic, or data-dependent?
- What should the architecture look like as a paper figure?
- How does this architecture differ structurally and semantically from another model?

The end state is a **zoomable architecture microscope**: start from a clean paper-level main figure and continuously drill down until the smallest source-visible tensor-producing operation, while preserving semantic meaning and provenance at every level.

## Core principle: one graph, many views

ArchTrace does not treat the paper diagram, module tree, runtime graph, tensor graph, and source map as unrelated artifacts. They are projections of one canonical intermediate representation:

```text
Repository + Config + Entrypoint + Runtime Evidence + Paper/README Context
                              │
                              ▼
                     ArchTrace Unified IR
                              │
        ┌─────────────┬───────┼────────┬──────────────┐
        ▼             ▼       ▼        ▼              ▼
    Paper View   Semantic   Module   Tensor/Op      Source
                 View       View     View           View
        │             │       │        │              │
        └─────────────┴───────┴────────┴──────────────┘
                              │
                              ▼
                  Interactive / SVG / PDF /
                 PPTX / draw.io / JSON export
```

## Target abstraction levels

ArchTrace uses explicit levels instead of flattening everything into one unreadable graph.

### L0 — Paper view

The clean architecture figure a paper would show: major inputs, proposed components, key information flows, outputs, and training/inference distinctions.

### L1 — Semantic architecture

Components such as `Vision Encoder`, `Language Encoder`, `Cross-Modal Fusion`, `World Model`, `Policy Backbone`, `Action Head`, `Loss`, `Memory`, `Planner`, or domain-specific concepts inferred from code and repository context.

### L2 — Module architecture

Framework modules and submodules, repeated blocks, parameter sharing, containment, branch structure, and module-level I/O.

### L3 — Tensor / operation graph

Every source-visible tensor-producing operation and transformation, including shape, dtype, device, aliasing, parameter use, control-flow context, and producer/consumer relationships.

### L4 — Source / runtime provenance

Exact file, line, callable, call stack, runtime metadata, timing/profile information, configuration origin, and evidence supporting each reconstructed edge or semantic claim.

Optional performance backends may descend further into compiler/kernel traces, but **the canonical atomic unit of ArchTrace is the source-visible tensor-producing operation**, not a CUDA instruction.

## What ArchTrace unifies

ArchTrace is designed to absorb the strongest ideas from several existing classes of tools without becoming coupled to one of them:

- runtime neural-network tracing and activation provenance;
- FX/export/ONNX-style graph capture and shape propagation;
- repository-scale static analysis, AST/call/import graphs, and config resolution;
- semantic repository understanding and architecture recovery;
- hierarchical large-graph exploration;
- publication-quality architecture layout and export;
- feature-level and tensor-level data-flow analysis;
- architecture comparison across models, repositories, runs, and checkpoints.

Backends should be adapters. **ArchTrace IR is the product.**

## Planned architecture

```text
archtrace/
├── ingest/          # repository, paper/readme, config, entrypoint discovery
├── static/          # AST, imports, symbols, call graph, config/data-flow analysis
├── runtime/         # execution tracing and framework adapters
├── tensor/          # tensor identity, shape/dtype/device/alias/data-flow tracking
├── semantics/       # semantic role inference and evidence-grounded aggregation
├── ir/              # canonical graph schema, provenance, validation, serialization
├── align/           # static/runtime/semantic/source reconciliation
├── views/           # L0-L4 graph projections
├── layout/          # hierarchical graph layout and visual grammar
├── exporters/       # web, JSON, SVG, PDF, PPTX, draw.io, graph formats
├── compare/         # architecture and information-flow comparison
├── cli/             # `archtrace analyze`, `trace`, `view`, `compare`, `export`
└── server/          # interactive explorer API
```

A separate web application will render the IR as a progressively explorable graph rather than embedding visualization assumptions into the analyzer.

## The Unified IR

Every important fact in ArchTrace should be represented together with **provenance and confidence**. A semantic node is not merely an LLM-generated label, and a runtime edge is not assumed to describe all possible executions.

Illustrative object:

```yaml
node:
  id: semantic.multimodal_fusion
  level: semantic
  role: multimodal_fusion
  label: Cross-Modal Fusion

  contains:
    modules:
      - model.fusion
      - model.cross_attention
    operations:
      - op.823
      - op.824
      - op.1049

  inputs:
    - tensor: vision_tokens
      shape: [B, 256, 1024]
      semantics: visual_tokens
    - tensor: language_tokens
      shape: [B, 32, 1024]
      semantics: language_tokens

  outputs:
    - tensor: fused_tokens
      shape: [B, 288, 1024]

  source:
    - models/policy.py:141-203

  evidence:
    - runtime_trace: run.001
    - static_analysis: ast.391
    - repository_text: readme.section.4

  confidence: 0.97
```

This evidence model is essential: ArchTrace should distinguish **observed**, **statically inferred**, **semantically inferred**, **declared by authors**, and **unknown** facts.

## Analysis pipeline

```text
1. Repository ingestion
        ↓
2. Entrypoint/config/dependency discovery
        ↓
3. Static repository analysis
        ↓
4. Runtime capture (when runnable)
        ↓
5. Tensor and operator reconstruction
        ↓
6. Source provenance linking
        ↓
7. Static/runtime graph alignment
        ↓
8. Semantic component recovery
        ↓
9. Hierarchical aggregation into L0-L4
        ↓
10. Validation + uncertainty labeling
        ↓
11. Interactive visualization / publication export / comparison
```

ArchTrace must still produce a useful graph when runtime execution is unavailable. Static-only and hybrid modes are first-class modes, not degraded afterthoughts.

## CLI target

```bash
# inspect a repository and discover likely entrypoints/configs
archtrace inspect ./repo

# static + semantic analysis
archtrace analyze ./repo

# execute and capture one concrete path
archtrace trace ./repo \
  --entry scripts/eval.py \
  --config configs/task.yaml

# launch the hierarchical explorer
archtrace view .archtrace/project.atir

# produce a publication-oriented main figure
archtrace export .archtrace/project.atir --view paper --format svg

# compare two architectures or executions
archtrace compare model_a.atir model_b.atir
```

Eventually, a remote repository should be sufficient input:

```bash
archtrace analyze https://github.com/owner/research-repo
```

## First-class use cases

### 1. Read a paper implementation

Clone/import a repository and navigate from the claimed architecture down to the exact code and tensor transformations that implement it.

### 2. Verify a paper figure against code

Compare author claims and diagrams with observed/static implementation evidence. Highlight omitted paths, unused modules, shape-changing bottlenecks, and discrepancies.

### 3. Generate the architecture figure

Produce an editable, source-grounded paper figure whose components are traceable back to the implementation rather than hallucinated from a README.

### 4. Understand information flow

Track where visual, language, proprioceptive, geometric, temporal, latent, action, reward, and loss information originates, transforms, merges, branches, disappears, or reaches the output.

### 5. Compare research models

Normalize multiple repositories into the same IR and compare architecture, fusion points, bottlenecks, token/tensor transformations, parameter sharing, gradient paths, and modality-to-output influence.

### 6. Debug and audit

Click any node to see runtime observations, source lines, call stacks, configs, shapes, device placement, and downstream consumers.

## Design constraints

1. **Source-grounded by default.** Every factual edge should have inspectable evidence.
2. **Runtime is evidence, not truth about all paths.** A trace represents a concrete execution unless coverage proves otherwise.
3. **Semantics never overwrite mechanics.** Semantic grouping must remain reversible to lower-level nodes.
4. **No lossy hierarchy.** L0/L1/L2 nodes retain links to all underlying operations and sources.
5. **Dynamic models are first-class.** Loops, branches, recursion, shared modules, stochastic paths, multimodal pipelines, and agent/environment loops cannot be flattened away.
6. **Repository-level scope.** Preprocessing, datasets, wrappers, action decoding, losses, controllers, and environment interactions matter as much as `model.forward()`.
7. **Framework adapters, framework-neutral IR.** PyTorch first; JAX, TensorFlow/Keras, ONNX, exported graphs, and other systems can follow.
8. **Human-correctable.** Users can rename/group/reclassify nodes and preserve those corrections as overlays.
9. **LLMs assist semantics, never fabricate mechanics.** Unknown information remains unknown.
10. **Publication and exploration are projections of the same evidence graph.**

## Roadmap

### M0 — Specification and minimal vertical slice

- define ArchTrace IR (ATIR) v0.1;
- provenance/confidence model;
- repository scanner;
- Python/PyTorch source indexing;
- minimal PyTorch runtime tracer adapter;
- tensor shape/dtype/device flow;
- source-line mapping;
- hierarchical JSON explorer prototype;
- golden test repositories.

### M1 — PyTorch architecture reconstruction

- module + op + tensor graphs;
- FX / `torch.export` / runtime-trace reconciliation;
- loops, shared modules, branches, nested structures;
- config and entrypoint resolution;
- backward/gradient graph optional capture;
- deterministic trace manifests;
- L2/L3/L4 views.

### M2 — Semantic architecture recovery

- repository-level context retrieval;
- semantic role ontology;
- evidence-grounded component naming/grouping;
- modality/data lineage;
- training-vs-inference views;
- confidence and contradiction handling;
- L0/L1 projection.

### M3 — Architecture explorer

- infinite hierarchical drill-down;
- expand/collapse semantic → module → op → source;
- graph search and path queries;
- input/output lineage highlighting;
- shape and metadata overlays;
- code pane synchronized with graph selection;
- trace/run selector and dynamic-path comparison.

### M4 — Publication figure engine

- scientific visual grammar;
- semantic layout templates;
- automatic abstraction/detail selection;
- overview + detailed proposed-block composition;
- SVG/PDF/PPTX/draw.io export;
- fully editable labels, grouping, and layout constraints.

### M5 — Repository-scale and cross-model intelligence

- preprocessing-to-output full-pipeline tracing;
- datasets/configs/controllers/environment loops;
- architecture diff;
- modality-flow comparison;
- paper-vs-code verification;
- multi-run path coverage;
- scalable graph storage/query engine.

### M6 — Multi-framework ecosystem

- JAX/Flax;
- TensorFlow/Keras;
- ONNX and compiler/export formats;
- plugin SDK for custom frameworks and scientific domains.

## Initial integration philosophy

We will evaluate and, where technically/licensing-wise appropriate, integrate or adapt ideas/backends from projects in the following categories: runtime tracers (e.g. TorchLens-like capabilities), PyTorch FX/export graph capture, Model Explorer-style hierarchical visualization, repository/semantic graph systems, and scientific architecture-diagram generators.

The goal is **not** to vendor unrelated projects into one monorepo. The goal is to define a stronger canonical model and make external analyzers/renderers replaceable adapters around it.

## Status

ArchTrace is in **product alpha**. M0–M3 are implemented: ATIR v0.2, PyTorch runtime tracing, repository-scale static analysis, evidence-grounded semantic recovery, and the hierarchical Web explorer are all integrated on `main`.

The active phase is **M3.5: real-world benchmark and torture testing**. Pinned public research repositories are analyzed without importing target code, and failures are recorded as structured uncertainty rather than hidden behind successful-looking diagrams. The initial matrix covers DP3, Diffusion Policy, openpi/π0, RDT, V-JEPA2, and RoboTwin. See `docs/BENCHMARK.md` for methodology and `docs/PRODUCT.md` for the live product workflow.

## License

License selection is pending dependency/license review before external code is incorporated.
