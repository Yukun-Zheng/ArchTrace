"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GraphCanvas } from "./GraphCanvas";
import { Inspector } from "./Inspector";
import { TopBar } from "./TopBar";
import { Sidebar } from "./Sidebar";
import { CanvasHeader } from "./CanvasHeader";
import { demoComparisonProject, demoProject } from "@/lib/demo";
import {
  ancestorsToReveal,
  breadcrumbPath,
  buildIndex,
  buildVisibleGraph,
  diffProjects,
  focusVisibleGraph,
  lineage,
  parseProject,
  projectStats,
  searchProject,
} from "@/lib/project";
import type {
  AtirProject,
  ExplorerMode,
  LineageDirection,
  OverlayMap,
  OverlayRecord,
  PhaseFilter,
  VisibleNode,
} from "@/lib/types";

const EMPTY_LINEAGE = { nodes: new Set<string>(), edges: new Set<string>() };

function cloneSet<T>(source: Set<T>): Set<T> {
  return new Set(source);
}

function overlayStorageKey(project: AtirProject): string {
  return `archtrace.overlays.${project.project.name}`;
}

function semanticVisibleInPhase(
  node: AtirProject["nodes"][number],
  phase: PhaseFilter,
): boolean {
  if (node.level !== "semantic" || node.attributes?.declaration_only === true) {
    return false;
  }
  if (phase === "all") return true;
  const value = node.attributes?.phase;
  return value === undefined || value === null || value === "both" || value === phase;
}

export function ExplorerApp() {
  const [project, setProject] = useState<AtirProject>(demoProject);
  const [baseline, setBaseline] = useState<AtirProject | null>(null);
  const [mode, setMode] = useState<ExplorerMode>("explore");
  const [phase, setPhase] = useState<PhaseFilter>("all");
  const [runId, setRunId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [repeatExpanded, setRepeatExpanded] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [overlays, setOverlays] = useState<OverlayMap>({});
  const [lineageState, setLineageState] = useState(EMPTY_LINEAGE);
  const [focusMode, setFocusMode] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [urlHydrated, setUrlHydrated] = useState(false);
  const [, setHistoryVersion] = useState(0);
  const projectInput = useRef<HTMLInputElement>(null);
  const baselineInput = useRef<HTMLInputElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const historyRef = useRef<string[]>([]);
  const historyCursorRef = useRef(-1);

  const index = useMemo(() => buildIndex(project), [project]);
  const stats = useMemo(() => projectStats(project), [project]);
  const diff = useMemo(
    () => baseline ? diffProjects(project, baseline) : undefined,
    [baseline, project],
  );
  const graph = useMemo(
    () => buildVisibleGraph(
      project.nodes,
      index,
      expanded,
      repeatExpanded,
      runId,
      phase,
      overlays,
      mode === "diff" ? diff : undefined,
    ),
    [project.nodes, index, expanded, repeatExpanded, runId, phase, overlays, diff, mode],
  );
  const renderedGraph = useMemo(
    () => focusMode && selectedId ? focusVisibleGraph(graph, selectedId, 2) : graph,
    [focusMode, graph, selectedId],
  );
  const searchResults = useMemo(
    () => searchProject(project, query, runId),
    [project, query, runId],
  );
  const selectedVisible = useMemo<VisibleNode | null>(
    () => graph.nodes.find(
      (node) => node.id === selectedId || node.entityId === selectedId,
    ) ?? null,
    [graph.nodes, selectedId],
  );
  const selectedEntityId = selectedVisible?.entityId ?? (
    selectedId && !selectedId.startsWith("repeat:")
      ? selectedId.replace(/^paper:/, "")
      : null
  );
  const selectedOverlay = selectedEntityId ? overlays[selectedEntityId] ?? {} : {};
  const breadcrumbNodes = useMemo(
    () => selectedEntityId ? breadcrumbPath(index, selectedEntityId) : [],
    [index, selectedEntityId],
  );
  const canHistoryBack = historyCursorRef.current > 0;
  const canHistoryForward = (
    historyCursorRef.current >= 0
    && historyCursorRef.current < historyRef.current.length - 1
  );

  const selectNode = useCallback((id: string | null, record = true) => {
    setSelectedId(id);
    if (!id || !record) return;
    const current = historyRef.current.slice(0, historyCursorRef.current + 1);
    if (current[current.length - 1] !== id) current.push(id);
    historyRef.current = current.slice(-80);
    historyCursorRef.current = historyRef.current.length - 1;
    setHistoryVersion((version) => version + 1);
  }, []);

  const moveHistory = useCallback((delta: -1 | 1) => {
    const next = historyCursorRef.current + delta;
    if (next < 0 || next >= historyRef.current.length) return;
    historyCursorRef.current = next;
    setSelectedId(historyRef.current[next]);
    setHistoryVersion((version) => version + 1);
  }, []);

  const resetHistory = useCallback(() => {
    historyRef.current = [];
    historyCursorRef.current = -1;
    setHistoryVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(overlayStorageKey(project));
      setOverlays(stored ? JSON.parse(stored) as OverlayMap : {});
    } catch {
      setOverlays({});
    }
  }, [project]);

  useEffect(() => {
    if (!urlHydrated) return;
    try {
      window.localStorage.setItem(overlayStorageKey(project), JSON.stringify(overlays));
    } catch {
      // Local storage can be disabled; overlays still work for the current session.
    }
  }, [overlays, project, urlHydrated]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const rawMode = params.get("mode");
    const rawPhase = params.get("phase");
    const rawRun = params.get("run");
    const rawNode = params.get("node");
    const rawQuery = params.get("q");
    const rawExpanded = params.get("exp");
    if (rawMode === "diff") setMode("diff");
    if (rawPhase === "training" || rawPhase === "inference") setPhase(rawPhase);
    if (rawRun) setRunId(rawRun);
    if (rawNode) setSelectedId(rawNode);
    if (rawQuery) setQuery(rawQuery);
    if (rawExpanded) setExpanded(new Set(rawExpanded.split(",").filter(Boolean)));
    if (params.get("focus") === "1") setFocusMode(true);
    setUrlHydrated(true);
  }, []);

  useEffect(() => {
    if (!urlHydrated || !selectedId || selectedId.startsWith("repeat:")) return;
    const entityId = selectedId.replace(/^paper:/, "");
    if (!index.nodes.has(entityId)) return;
    setExpanded((current) => {
      const next = cloneSet(current);
      for (const id of ancestorsToReveal(index, entityId)) next.add(id);
      return next;
    });
  }, [index, selectedId, urlHydrated]);

  useEffect(() => {
    if (!urlHydrated) return;
    const params = new URLSearchParams();
    if (mode !== "explore") params.set("mode", mode);
    if (phase !== "all") params.set("phase", phase);
    if (runId) params.set("run", runId);
    if (selectedId) params.set("node", selectedId);
    if (query) params.set("q", query);
    if (expanded.size) params.set("exp", [...expanded].sort().join(","));
    if (focusMode) params.set("focus", "1");
    const next = params.toString();
    window.history.replaceState(
      null,
      "",
      next ? `?${next}` : window.location.pathname,
    );
  }, [expanded, focusMode, mode, phase, query, runId, selectedId, urlHydrated]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        event.key === "/"
        && !(event.target instanceof HTMLInputElement)
        && !(event.target instanceof HTMLTextAreaElement)
      ) {
        event.preventDefault();
        searchInput.current?.focus();
      }
      if (event.key === "Escape") {
        selectNode(null, false);
        setLineageState(EMPTY_LINEAGE);
        setFocusMode(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectNode]);

  const showNotice = useCallback((message: string) => {
    setNotice(message);
    window.setTimeout(() => setNotice(null), 2600);
  }, []);

  const loadFile = useCallback(async (
    file: File,
    target: "project" | "baseline",
  ) => {
    try {
      const raw = JSON.parse(await file.text()) as unknown;
      const parsed = parseProject(raw);
      if (target === "project") {
        setProject(parsed);
        setRunId(null);
        selectNode(null, false);
        resetHistory();
        setExpanded(new Set());
        setRepeatExpanded(new Set());
        setLineageState(EMPTY_LINEAGE);
        setFocusMode(false);
        showNotice(`Loaded ${parsed.project.name}`);
      } else {
        setBaseline(parsed);
        setMode("diff");
        showNotice(`Comparison loaded: ${parsed.project.name}`);
      }
    } catch (error) {
      showNotice(
        error instanceof Error ? error.message : "Could not parse ATIR JSON.",
      );
    }
  }, [resetHistory, selectNode, showNotice]);

  const revealEntity = useCallback((entityId: string) => {
    const node = index.nodes.get(entityId);
    if (!node) return;
    if (node.run_id) setRunId(node.run_id);
    setExpanded((current) => {
      const next = cloneSet(current);
      for (const id of ancestorsToReveal(index, entityId)) next.add(id);
      return next;
    });
    selectNode(node.level === "semantic" ? `paper:${node.id}` : node.id);
  }, [index, selectNode]);

  const toggleExpansion = useCallback((displayId: string) => {
    if (displayId.startsWith("repeat:")) {
      setRepeatExpanded((current) => {
        const next = cloneSet(current);
        if (next.has(displayId)) next.delete(displayId);
        else next.add(displayId);
        return next;
      });
      return;
    }
    const entityId = displayId.replace(/^paper:/, "");
    setExpanded((current) => {
      const next = cloneSet(current);
      if (next.has(entityId)) next.delete(entityId);
      else next.add(entityId);
      return next;
    });
  }, []);

  const updateOverlay = useCallback((next: OverlayRecord) => {
    if (!selectedEntityId) return;
    setOverlays((current) => ({ ...current, [selectedEntityId]: next }));
  }, [selectedEntityId]);

  const runLineage = useCallback((direction: LineageDirection) => {
    if (!selectedEntityId) return;
    setLineageState(lineage(index, selectedEntityId, direction, runId));
  }, [index, runId, selectedEntityId]);

  const exportOverlays = useCallback(() => {
    const payload = JSON.stringify(
      { project: project.project.name, overlays },
      null,
      2,
    );
    const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = (
      `${project.project.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}-overlays.json`
    );
    anchor.click();
    URL.revokeObjectURL(url);
  }, [overlays, project.project.name]);

  const resetDemo = useCallback(() => {
    setProject(demoProject);
    setBaseline(null);
    setMode("explore");
    setPhase("all");
    setRunId(null);
    selectNode(null, false);
    resetHistory();
    setExpanded(new Set());
    setRepeatExpanded(new Set());
    setQuery("");
    setLineageState(EMPTY_LINEAGE);
    setFocusMode(false);
    showNotice("Demo restored");
  }, [resetHistory, selectNode, showNotice]);

  const semanticNodes = project.nodes.filter(
    (node) => semanticVisibleInPhase(node, phase),
  );

  return (
    <main
      className="workbench"
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (file) void loadFile(file, "project");
      }}
    >
      <TopBar
        project={project}
        mode={mode}
        phase={phase}
        runId={runId}
        projectInput={projectInput}
        baselineInput={baselineInput}
        onMode={setMode}
        onPhase={setPhase}
        onRun={setRunId}
        onReset={resetDemo}
        onLoad={loadFile}
      />

      <Sidebar
        project={project}
        mode={mode}
        stats={stats}
        query={query}
        searchInput={searchInput}
        searchResults={searchResults}
        semanticNodes={semanticNodes}
        overlays={overlays}
        selectedEntityId={selectedEntityId}
        expanded={expanded}
        baseline={baseline}
        diff={diff}
        onQuery={setQuery}
        onReveal={revealEntity}
        onSelectPaper={selectNode}
        onTogglePaper={toggleExpansion}
        onUseDemoComparison={() => setBaseline(demoComparisonProject)}
        onExportOverlays={exportOverlays}
        onReset={resetDemo}
      />

      <section className="canvas-shell">
        <CanvasHeader
          breadcrumbs={breadcrumbNodes}
          overlays={overlays}
          canBack={canHistoryBack}
          canForward={canHistoryForward}
          focusMode={focusMode}
          hasSelection={Boolean(selectedId)}
          onBack={() => moveHistory(-1)}
          onForward={() => moveHistory(1)}
          onReveal={revealEntity}
          onFocus={() => setFocusMode((value) => !value)}
        />
        <div className="canvas">
          <GraphCanvas
            graph={renderedGraph}
            selectedId={selectedId}
            lineageNodes={lineageState.nodes}
            lineageEdges={lineageState.edges}
            onSelect={(id) => selectNode(id)}
            onToggle={toggleExpansion}
          />
        </div>
        <div className="statusbar">
          <span>{renderedGraph.nodes.length} rendered / {stats.nodes} ATIR nodes</span>
          <span>
            {lineageState.nodes.size
              ? `${lineageState.nodes.size} lineage nodes highlighted`
              : "bounded hierarchical rendering"}
          </span>
          <span>ATIR {project.schema_version ?? "unknown"}</span>
        </div>
      </section>

      <Inspector
        project={project}
        index={index}
        selected={selectedVisible}
        overlay={selectedOverlay}
        lineageCount={lineageState.nodes.size}
        onOverlay={updateOverlay}
        onLineage={runLineage}
        onClearLineage={() => setLineageState(EMPTY_LINEAGE)}
      />

      {dragging ? (
        <div className="drop-overlay">
          <div>
            <strong>Drop ATIR JSON</strong>
            <span>ArchTrace will open it locally in your browser.</span>
          </div>
        </div>
      ) : null}
      {notice ? <div className="toast">{notice}</div> : null}
    </main>
  );
}
