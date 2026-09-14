# ArchTrace Product

**Live product:** https://archtrace.vercel.app

ArchTrace is a source-grounded architecture microscope for research code. The product consumes validated ATIR and keeps mechanical facts separate from presentation state.

## Product workflow

1. Analyze a repository and produce ATIR.
2. Optionally create a browser-ready bundle with source text:

   ```bash
   archtrace web-bundle .archtrace/project.atir.json \
     --source-root /path/to/repository \
     -o .archtrace/project.web.atir.json
   ```

3. Open the live explorer and load the ATIR JSON.

## Explorer capabilities

- L0 paper architecture → L1 semantic component → L2 module → L3 operator/tensor → L4 exact source span.
- Run and training/inference phase filtering.
- Upstream/downstream mechanical lineage highlighting.
- Search over semantic roles, modules, operators, tensors, shapes, modalities, and source paths.
- Repeated execution collapse (`×N`) while preserving occurrence identity.
- Source, tensor metadata, evidence, claims, conflicts, and coverage inspection.
- Non-destructive local annotations and display-name overlays.
- Architecture diff view.
- URL deep links, selection history, focus mode, zoom/pan, and minimap.

## Invariants

The browser is a projection client. It does not invent mechanical edges, tensor shapes, operations, source locations, or execution occurrences, and presentation overlays do not mutate ATIR facts.
