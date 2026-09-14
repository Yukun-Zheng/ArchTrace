"use client";

import { useMemo } from "react";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import type { VisibleGraph, VisibleNode } from "@/lib/types";

interface GraphCanvasProps {
  graph: VisibleGraph;
  selectedId: string | null;
  lineageNodes: Set<string>;
  lineageEdges: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
}

function nodeTone(node: VisibleNode): string {
  if (node.diffStatus === "added") return "var(--diff-added)";
  if (node.diffStatus === "changed") return "var(--diff-changed)";
  if (node.synthetic === "repeat") return "var(--repeat)";
  if (node.kind === "paper_component") return "var(--paper)";
  if (node.kind === "module") return "var(--module)";
  if (node.kind === "operation") return "var(--operation)";
  if (["tensor", "input", "output", "parameter"].includes(node.kind)) return "var(--tensor)";
  return "var(--surface-strong)";
}

function layoutGraph(graph: VisibleGraph): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const paperNodes = graph.nodes.filter((node) => node.depth === 0);
  const paperIds = new Set(paperNodes.map((node) => node.id));
  const ranks = new Map(paperNodes.map((node) => [node.id, 0]));
  const paperEdges = graph.edges.filter((edge) => paperIds.has(edge.source) && paperIds.has(edge.target));
  for (let pass = 0; pass < Math.max(1, paperNodes.length); pass += 1) {
    let changed = false;
    for (const edge of paperEdges) {
      const sourceRank = ranks.get(edge.source) ?? 0;
      const targetRank = ranks.get(edge.target) ?? 0;
      if (sourceRank + 1 > targetRank && sourceRank + 1 < paperNodes.length) {
        ranks.set(edge.target, sourceRank + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const byRank = new Map<number, VisibleNode[]>();
  for (const node of paperNodes) {
    const rank = ranks.get(node.id) ?? 0;
    if (!byRank.has(rank)) byRank.set(rank, []);
    byRank.get(rank)?.push(node);
  }
  for (const [rank, nodes] of byRank) {
    nodes.sort((a, b) => a.label.localeCompare(b.label));
    nodes.forEach((node, index) => {
      positions.set(node.id, { x: rank * 280, y: (index - (nodes.length - 1) / 2) * 150 });
    });
  }

  const drillEdges = graph.edges.filter((edge) => edge.kind === "drilldown");
  for (let depth = 1; depth <= 8; depth += 1) {
    const nodes = graph.nodes.filter((node) => node.depth === depth);
    const childrenByParent = new Map<string, VisibleNode[]>();
    for (const node of nodes) {
      const parent = drillEdges.find((edge) => edge.target === node.id)?.source;
      if (!parent) continue;
      if (!childrenByParent.has(parent)) childrenByParent.set(parent, []);
      childrenByParent.get(parent)?.push(node);
    }
    for (const [parentId, children] of childrenByParent) {
      const parent = positions.get(parentId) ?? { x: 0, y: 0 };
      children.sort((a, b) => a.label.localeCompare(b.label));
      children.forEach((node, index) => {
        positions.set(node.id, {
          x: parent.x + 290,
          y: parent.y + (index - (children.length - 1) / 2) * 115,
        });
      });
    }
  }
  graph.nodes.forEach((node, index) => {
    if (!positions.has(node.id)) positions.set(node.id, { x: node.depth * 280, y: index * 110 });
  });
  return positions;
}

export function GraphCanvas({
  graph,
  selectedId,
  lineageNodes,
  lineageEdges,
  onSelect,
  onToggle,
}: GraphCanvasProps) {
  const positions = useMemo(() => layoutGraph(graph), [graph]);

  const nodes = useMemo<Node[]>(
    () => graph.nodes.map((node) => {
      const active = selectedId === node.id || (node.entityId ? lineageNodes.has(node.entityId) : lineageNodes.has(node.id));
      const metadata = [node.role, node.runId].filter(Boolean).join(" · ");
      return {
        id: node.id,
        position: positions.get(node.id) ?? { x: 0, y: 0 },
        data: {
          label: (
            <div className="graph-node-content">
              <div className="graph-node-label">{node.label}</div>
              {metadata ? <div className="graph-node-meta">{metadata}</div> : null}
              {node.synthetic === "repeat" ? <div className="graph-node-hint">double-click to expand</div> : null}
            </div>
          ),
        },
        className: `arch-node ${active ? "is-active" : ""}`,
        style: {
          width: node.kind === "paper_component" ? 210 : 190,
          borderColor: active ? "var(--accent)" : nodeTone(node),
          background: "var(--surface)",
          boxShadow: active ? "0 0 0 2px color-mix(in srgb, var(--accent) 28%, transparent)" : undefined,
        },
      };
    }),
    [graph.nodes, lineageNodes, positions, selectedId],
  );

  const edges = useMemo<Edge[]>(
    () => graph.edges.map((edge) => {
      const highlighted = edge.mechanicalEdgeIds.some((id) => lineageEdges.has(id));
      const synthetic = edge.synthetic || edge.kind === "drilldown";
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        animated: highlighted,
        markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
        style: {
          stroke: highlighted ? "var(--accent)" : synthetic ? "var(--edge-soft)" : "var(--edge)",
          strokeWidth: highlighted ? 2.4 : synthetic ? 1 : 1.4,
          strokeDasharray: synthetic ? "5 5" : undefined,
        },
      };
    }),
    [graph.edges, lineageEdges],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1.1 }}
      minZoom={0.08}
      maxZoom={2.2}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
      onNodeClick={(_, node) => onSelect(node.id)}
      onNodeDoubleClick={(_, node) => onToggle(node.id)}
      proOptions={{ hideAttribution: true }}
    >
      <MiniMap pannable zoomable className="arch-minimap" />
      <Controls showInteractive={false} />
      <Background gap={26} size={1} />
    </ReactFlow>
  );
}
