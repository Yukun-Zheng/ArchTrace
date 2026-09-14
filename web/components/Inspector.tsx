"use client";

import type { ReactNode } from "react";
import type { AtirProject, OverlayRecord, VisibleNode } from "@/lib/types";
import type { ProjectIndex } from "@/lib/project";
import { sourceExcerpt } from "@/lib/project";

interface InspectorProps {
  project: AtirProject;
  index: ProjectIndex;
  selected: VisibleNode | null;
  overlay: OverlayRecord;
  lineageCount: number;
  onOverlay: (next: OverlayRecord) => void;
  onLineage: (direction: "upstream" | "downstream") => void;
  onClearLineage: () => void;
}

function Badge({ children }: { children: ReactNode }) {
  return <span className="badge">{children}</span>;
}

export function Inspector({
  project,
  index,
  selected,
  overlay,
  lineageCount,
  onOverlay,
  onLineage,
  onClearLineage,
}: InspectorProps) {
  if (!selected) {
    return (
      <aside className="inspector empty-panel">
        <div className="empty-icon">⌁</div>
        <strong>Select a component</strong>
        <p>Click any block to inspect source, tensor metadata, evidence, claims, and conflicts.</p>
      </aside>
    );
  }

  if (selected.synthetic === "repeat") {
    return (
      <aside className="inspector">
        <div className="panel-heading">
          <div>
            <div className="eyebrow">Repeated execution group</div>
            <h2>{selected.label}</h2>
          </div>
        </div>
        <p className="muted">This is a presentation-only collapse. The underlying occurrences remain distinct ATIR nodes.</p>
        <div className="member-list">
          {(selected.repeatMembers ?? []).map((id) => <code key={id}>{id}</code>)}
        </div>
      </aside>
    );
  }

  const entityId = selected.entityId;
  const node = entityId ? index.nodes.get(entityId) : undefined;
  if (!node) return <aside className="inspector empty-panel">No mechanical entity is attached to this view node.</aside>;

  const evidence = (node.evidence_ids ?? []).map((id) => index.evidence.get(id)).filter(Boolean);
  const claims = index.claimsBySubject.get(node.id) ?? [];
  const conflicts = claims.flatMap((claim) => index.conflictsByClaim.get(claim.id) ?? []);
  const excerpt = sourceExcerpt(project, node);
  const tensor = node.tensor;

  return (
    <aside className="inspector">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">{node.level} · {node.identity_kind ?? "group"}</div>
          <h2>{overlay.alias?.trim() || node.label}</h2>
        </div>
        {node.role ? <Badge>{node.role}</Badge> : null}
      </div>

      <div className="inspector-actions">
        <button onClick={() => onLineage("upstream")}>↑ Upstream</button>
        <button onClick={() => onLineage("downstream")}>↓ Downstream</button>
        {lineageCount ? <button className="ghost" onClick={onClearLineage}>Clear · {lineageCount}</button> : null}
      </div>

      <section>
        <div className="section-title">Presentation overlay</div>
        <label className="field">
          <span>Display name</span>
          <input
            value={overlay.alias ?? ""}
            placeholder={node.label}
            onChange={(event) => onOverlay({ ...overlay, alias: event.target.value })}
          />
        </label>
        <label className="field">
          <span>Research note</span>
          <textarea
            value={overlay.note ?? ""}
            placeholder="Non-destructive annotation…"
            onChange={(event) => onOverlay({ ...overlay, note: event.target.value })}
          />
        </label>
        <p className="microcopy">Saved locally in your browser. Mechanical ATIR remains unchanged.</p>
      </section>

      <section>
        <div className="section-title">Identity</div>
        <dl className="kv-grid">
          <dt>ID</dt><dd><code>{node.id}</code></dd>
          <dt>Kind</dt><dd>{node.kind}</dd>
          {node.run_id ? <><dt>Run</dt><dd><code>{node.run_id}</code></dd></> : null}
          {node.definition_id ? <><dt>Definition</dt><dd><code>{node.definition_id}</code></dd></> : null}
          {node.occurrence_index !== undefined && node.occurrence_index !== null ? <><dt>Occurrence</dt><dd>#{node.occurrence_index}</dd></> : null}
        </dl>
      </section>

      {tensor ? (
        <section>
          <div className="section-title">Tensor</div>
          <div className="tensor-card">
            <strong>{tensor.shape ? `[${tensor.shape.map((value) => value ?? "?").join(", ")}]` : "shape unknown"}</strong>
            <span>{[tensor.dtype, tensor.device].filter(Boolean).join(" · ") || "dtype/device unknown"}</span>
            {(tensor.semantics ?? []).length ? <div className="badge-row">{tensor.semantics?.map((item) => <Badge key={item}>{item}</Badge>)}</div> : null}
          </div>
        </section>
      ) : null}

      <section>
        <div className="section-title">Source</div>
        {excerpt ? (
          <div className="source-card">
            <div className="source-path">{excerpt.path}:{node.source?.[0]?.start_line}</div>
            {excerpt.lines.length ? (
              <pre>{excerpt.lines.map((line) => (
                <div key={line.number} className={line.active ? "source-line active" : "source-line"}>
                  <span>{String(line.number).padStart(4, " ")}</span><code>{line.text || " "}</code>
                </div>
              ))}</pre>
            ) : <p>Source text is not embedded in this ATIR. The exact span is still preserved.</p>}
          </div>
        ) : <p className="muted">No source span attached.</p>}
      </section>

      <section>
        <div className="section-title">Evidence · {evidence.length}</div>
        <div className="stack-list">
          {evidence.length ? evidence.map((item) => item ? (
            <div className="evidence-card" key={item.id}>
              <div><Badge>{item.kind}</Badge> <Badge>{item.status}</Badge></div>
              <strong>{item.description || item.id}</strong>
              <code>{item.id}</code>
            </div>
          ) : null) : <p className="muted">No direct evidence records.</p>}
        </div>
      </section>

      {claims.length ? (
        <section>
          <div className="section-title">Claims · {claims.length}</div>
          <div className="stack-list">
            {claims.map((claim) => (
              <div className="evidence-card" key={claim.id}>
                <strong>{claim.predicate} = {JSON.stringify(claim.value)}</strong>
                <span>{claim.status} · confidence {claim.confidence ?? 1}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {conflicts.length ? (
        <section>
          <div className="section-title danger">Conflicts · {conflicts.length}</div>
          <div className="stack-list">
            {conflicts.map((conflict) => (
              <div className="conflict-card" key={conflict.id}>
                <strong>{conflict.kind}</strong>
                <p>{conflict.description || "Evidence sources disagree."}</p>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </aside>
  );
}
