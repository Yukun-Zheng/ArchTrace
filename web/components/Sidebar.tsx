import type { RefObject } from "react";
import type {
  AtirNode,
  AtirProject,
  DiffSummary,
  ExplorerMode,
  OverlayMap,
  SearchResult,
} from "@/lib/types";

interface SidebarProps {
  project: AtirProject;
  mode: ExplorerMode;
  stats: { semantic: number; nodes: number; edges: number };
  query: string;
  searchInput: RefObject<HTMLInputElement | null>;
  searchResults: SearchResult[];
  semanticNodes: AtirNode[];
  overlays: OverlayMap;
  selectedEntityId: string | null;
  expanded: Set<string>;
  baseline: AtirProject | null;
  diff?: DiffSummary;
  onQuery: (value: string) => void;
  onReveal: (entityId: string) => void;
  onSelectPaper: (id: string) => void;
  onTogglePaper: (id: string) => void;
  onUseDemoComparison: () => void;
  onExportOverlays: () => void;
  onReset: () => void;
}

export function Sidebar({
  project,
  mode,
  stats,
  query,
  searchInput,
  searchResults,
  semanticNodes,
  overlays,
  selectedEntityId,
  expanded,
  baseline,
  diff,
  onQuery,
  onReveal,
  onSelectPaper,
  onTogglePaper,
  onUseDemoComparison,
  onExportOverlays,
  onReset,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="project-card">
        <div className="eyebrow">Project</div>
        <h1>{project.project.name}</h1>
        <div className="stat-row">
          <span>{stats.semantic} semantic</span>
          <span>{stats.nodes} nodes</span>
          <span>{stats.edges} edges</span>
        </div>
      </div>

      <div className="search-box">
        <span>⌕</span>
        <input
          ref={searchInput}
          value={query}
          onChange={(event) => onQuery(event.target.value)}
          placeholder="Search role, tensor, source…  /"
        />
        {query ? <button onClick={() => onQuery("")}>×</button> : null}
      </div>

      {query ? (
        <div className="sidebar-section scroll-list">
          <div className="section-title">Search · {searchResults.length}</div>
          {searchResults.map((result) => (
            <button className="list-item" key={result.id} onClick={() => onReveal(result.id)}>
              <strong>{overlays[result.id]?.alias || result.label}</strong>
              <span>{result.role || result.kind}{result.source ? ` · ${result.source}` : ""}</span>
            </button>
          ))}
          {!searchResults.length ? <p className="muted">No matching ATIR entities.</p> : null}
        </div>
      ) : mode === "diff" ? (
        <div className="sidebar-section scroll-list">
          <div className="section-title">Architecture diff</div>
          {!baseline ? (
            <div className="diff-empty">
              <p>Load a second ATIR or use the built-in candidate to compare semantic architecture.</p>
              <button onClick={onUseDemoComparison}>Compare demo v2</button>
            </div>
          ) : (
            <>
              <div className="diff-summary">
                <span className="diff-added">+{diff?.added.size ?? 0}</span>
                <span className="diff-removed">−{diff?.removed.size ?? 0}</span>
                <span className="diff-changed">~{diff?.changed.size ?? 0}</span>
              </div>
              {diff?.rows.map((row) => (
                <button
                  className={`list-item diff-row ${row.status}`}
                  key={row.key}
                  onClick={() => row.current && onReveal(row.current.id)}
                >
                  <strong>{row.current?.label || row.baseline?.label}</strong>
                  <span>{row.status} · {row.current?.role || row.baseline?.role}</span>
                </button>
              ))}
            </>
          )}
        </div>
      ) : (
        <div className="sidebar-section scroll-list">
          <div className="section-title">Paper architecture</div>
          {semanticNodes.map((node) => (
            <button
              className={`list-item ${selectedEntityId === node.id ? "selected" : ""}`}
              key={node.id}
              onClick={() => onSelectPaper(`paper:${node.id}`)}
              onDoubleClick={() => onTogglePaper(`paper:${node.id}`)}
            >
              <strong>{overlays[node.id]?.alias || node.label}</strong>
              <span>{node.role}{expanded.has(node.id) ? " · expanded" : ""}</span>
            </button>
          ))}
        </div>
      )}

      <div className="sidebar-footer">
        <button className="ghost" onClick={onExportOverlays}>Export overlays</button>
        <button className="ghost" onClick={onReset}>Reset demo</button>
      </div>
    </aside>
  );
}
