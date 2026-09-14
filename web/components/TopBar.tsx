import type { RefObject } from "react";
import type { AtirProject, ExplorerMode, PhaseFilter } from "@/lib/types";

interface TopBarProps {
  project: AtirProject;
  mode: ExplorerMode;
  phase: PhaseFilter;
  runId: string | null;
  projectInput: RefObject<HTMLInputElement | null>;
  baselineInput: RefObject<HTMLInputElement | null>;
  onMode: (mode: ExplorerMode) => void;
  onPhase: (phase: PhaseFilter) => void;
  onRun: (runId: string | null) => void;
  onReset: () => void;
  onLoad: (file: File, target: "project" | "baseline") => Promise<void>;
}

export function TopBar({
  project,
  mode,
  phase,
  runId,
  projectInput,
  baselineInput,
  onMode,
  onPhase,
  onRun,
  onReset,
  onLoad,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="brand" onClick={onReset} role="button" tabIndex={0}>
        <div className="brand-mark">A</div>
        <div><strong>ArchTrace</strong><span>architecture microscope</span></div>
      </div>
      <div className="segmented">
        <button className={mode === "explore" ? "active" : ""} onClick={() => onMode("explore")}>Explore</button>
        <button className={mode === "diff" ? "active" : ""} onClick={() => onMode("diff")}>Diff</button>
      </div>
      <div className="topbar-spacer" />
      <label className="compact-select">
        <span>Phase</span>
        <select value={phase} onChange={(event) => onPhase(event.target.value as PhaseFilter)}>
          <option value="all">All</option>
          <option value="training">Training</option>
          <option value="inference">Inference</option>
        </select>
      </label>
      <label className="compact-select">
        <span>Run</span>
        <select value={runId ?? ""} onChange={(event) => onRun(event.target.value || null)}>
          <option value="">All runs</option>
          {(project.runs ?? []).map((run) => <option value={run.id} key={run.id}>{run.id}</option>)}
        </select>
      </label>
      <button className="secondary" onClick={() => projectInput.current?.click()}>Open ATIR</button>
      <button className="secondary" onClick={() => baselineInput.current?.click()}>Compare</button>
      <input
        ref={projectInput}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void onLoad(file, "project");
          event.currentTarget.value = "";
        }}
      />
      <input
        ref={baselineInput}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void onLoad(file, "baseline");
          event.currentTarget.value = "";
        }}
      />
    </header>
  );
}
