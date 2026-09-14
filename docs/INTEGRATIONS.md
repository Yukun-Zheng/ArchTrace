# Integration Map

ArchTrace is not intended to be a wrapper around one existing visualizer. It is a unifying architecture with a framework-neutral IR at the center.

This document records how the major project categories discussed during project inception map into ArchTrace.

## 1. Runtime tracing / activation provenance

Representative capability: TorchLens-style runtime capture.

ArchTrace should absorb the following ideas through runtime adapters:

- concrete execution DAG;
- every tensor-producing operation;
- activation/gradient provenance;
- module containment;
- source/call-stack identity;
- shape, dtype, device, parameters, timing, RNG/control context;
- repeated and dynamic execution.

What ArchTrace adds:

- repository-scale context outside `forward()`;
- multi-run coverage;
- semantic aggregation;
- author/static/runtime contradiction handling;
- publication and comparison projections.

Integration boundary:

```text
runtime backend
    ↓
RuntimeObservation stream
    ↓
ATIR normalization
```

External tracing libraries must never become the canonical data model.

## 2. PyTorch graph capture

Representative capabilities: `torch.fx`, `torch.export`, profiler/operator interception, graph/module hooks.

ArchTrace should use multiple capture mechanisms because no single mechanism covers all research code.

Planned backend ensemble:

```text
AST/static analysis ───────────┐
torch.fx ──────────────────────┤
torch.export ──────────────────┤
runtime dispatch/hooks ────────┤──→ alignment → ATIR
profiler traces ────────────────┤
autograd/backward metadata ────┤
external tracer adapters ──────┘
```

The alignment layer resolves which observations describe the same module, operation definition, execution occurrence, or tensor/value.

## 3. Interactive large-graph exploration

Representative capability: Google Model Explorer-style hierarchical visualization.

ArchTrace explorer requirements:

- hierarchy-aware expansion/collapse;
- very large graph rendering;
- synchronized source-code pane;
- metadata overlays;
- search/filter/path queries;
- input/output lineage highlighting;
- semantic → module → operation → source drill-down;
- run/path selector;
- side-by-side and overlay architecture diff;
- user-correctable grouping and labels.

The web application consumes ATIR projections via a stable API. Renderer-specific IDs/layout state must not leak into canonical IR.

## 4. Repository semantic understanding

Representative capabilities: GitDiagram-, DeepWiki-, and semantic-flow-graph-style repository understanding.

ArchTrace needs repository-level semantics for things that runtime model tracers miss:

- entrypoints and configuration composition;
- datasets and preprocessing;
- augmentation;
- tokenization;
- environment wrappers;
- controllers and action post-processing;
- training loops and losses;
- evaluation/rollout code;
- feature-level intent and architectural roles;
- paper/README terminology.

ArchTrace differs from general repository diagrams by forcing semantic interpretation to align with mechanical evidence.

A semantic label should eventually be inspectable as:

```text
"Cross-Modal Fusion"
  ├─ author evidence: README / paper terminology
  ├─ static evidence: class + calls + data-flow
  ├─ runtime evidence: tensors from two modality lineages merge here
  └─ underlying nodes: module.*, op.*, tensor.*
```

## 5. Code-to-architecture visualization

Representative capability: ModelViz-style direct code/graph reconstruction.

Useful ideas:

- fast first graph from code;
- FX-first with static fallback;
- shape propagation;
- branch topology;
- developer-friendly IDE interaction.

ArchTrace should support this as a fast mode while retaining the deeper evidence graph for full analysis.

## 6. Publication-quality architecture generation

Representative capability: ML architecture diagram generation skills/tools.

ArchTrace publication engine requirements:

- semantic scientific visual grammar;
- high-level overall architecture plus detailed proposed block;
- automatic detail budgeting;
- repeated-block compression without losing traceability;
- modality-aware visual conventions;
- train/inference distinction;
- tensor shape annotations where useful;
- editable output;
- SVG, PDF, PPTX, draw.io and structured JSON export;
- every displayed component traceable back to ATIR evidence.

The publication renderer must not use an LLM to invent missing mechanical details. Unknowns remain unknown or are omitted explicitly.

## 7. Cross-model mechanism analysis

This is a core ArchTrace capability that emerges only after unification.

Normalize different repositories into ATIR, then ask architecture-level research questions such as:

- Where does each modality enter?
- Where do modalities first interact?
- Which spatial/temporal dimensions are collapsed and when?
- What information bottlenecks exist between perception and action?
- Which visual tokens can reach the action/loss path?
- How many transformations separate a modality from the final output?
- Are backbones frozen in declaration, optimizer configuration, and actual gradient flow?
- Which components are shared/reused/recurrent?
- Which claimed components are inactive in observed runs?
- How do two versions differ structurally and semantically?

This turns ArchTrace from a visualizer into **architecture reverse engineering and mechanism analysis infrastructure**.

## 8. Integration rule

Before incorporating external code, perform a dependency and license review. Prefer, in order:

1. native ArchTrace implementation of a general technique;
2. optional adapter around a compatible dependency;
3. interoperable import/export format;
4. vendoring/forking only when justified and license-compatible.

Do not copy implementations merely because their functionality is desired. Capabilities are unified at the architecture/IR level first.

## 9. Planned adapter namespaces

```text
archtrace/runtime/pytorch/
    hooks.py
    dispatch.py
    fx.py
    export.py
    profiler.py
    torchlens_adapter.py      # optional

archtrace/importers/
    onnx.py
    exported_program.py
    external_graph.py

archtrace/semantics/
    ontology.py
    rules.py
    grounding.py
    llm_backend.py            # optional provider interface

archtrace/exporters/
    web.py
    svg.py
    pdf.py
    pptx.py
    drawio.py
```

Names indicate intended boundaries, not a commitment to a specific dependency.

## 10. What success looks like

A user supplies a real research repository and can move continuously through:

```text
Paper claim
  ↓
Paper-level architecture
  ↓
Semantic component
  ↓
Framework module
  ↓
Concrete operation occurrence
  ↓
Tensor/value transformation
  ↓
Source line + configuration + runtime evidence
```

and then reverse the direction without losing identity or provenance.

That continuity is the defining feature of ArchTrace.
