import type { AtirNode, OverlayMap } from "@/lib/types";

interface CanvasHeaderProps {
  breadcrumbs: AtirNode[];
  overlays: OverlayMap;
  canBack: boolean;
  canForward: boolean;
  focusMode: boolean;
  hasSelection: boolean;
  onBack: () => void;
  onForward: () => void;
  onReveal: (id: string) => void;
  onFocus: () => void;
}

export function CanvasHeader({
  breadcrumbs,
  overlays,
  canBack,
  canForward,
  focusMode,
  hasSelection,
  onBack,
  onForward,
  onReveal,
  onFocus,
}: CanvasHeaderProps) {
  return (
    <div className="canvas-toolbar">
      <div className="canvas-nav">
        <button className="nav-button" disabled={!canBack} onClick={onBack} aria-label="Back in selection history">←</button>
        <button className="nav-button" disabled={!canForward} onClick={onForward} aria-label="Forward in selection history">→</button>
        <div className="breadcrumbs">
          {breadcrumbs.length ? breadcrumbs.map((node, index) => (
            <span key={node.id} className="breadcrumb-item">
              {index ? <span>›</span> : null}
              <button onClick={() => onReveal(node.id)}>{overlays[node.id]?.alias || node.label}</button>
            </span>
          )) : (
            <>
              <span>L0 Paper</span><span>›</span><span>L1 Semantic</span>
              <span>›</span><span>L2 Module</span><span>›</span>
              <span>L3 Op/Tensor</span><span>›</span><span>L4 Source</span>
            </>
          )}
        </div>
      </div>
      <div className="canvas-actions">
        <button className={focusMode ? "active" : ""} onClick={onFocus} disabled={!hasSelection}>Focus</button>
        <span className="canvas-hint">double-click to drill down · click to inspect</span>
      </div>
    </div>
  );
}
