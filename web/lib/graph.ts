import type { AtirNode, DiffSummary, OverlayMap, PhaseFilter, VisibleEdge, VisibleGraph, VisibleNode } from "./types";
import { FLOW_KINDS, type ProjectIndex, semanticAllowed, topSemanticMembers, visibleInRun } from "./atir";

interface ChildDescriptor {
  id: string;
  synthetic?: "repeat";
  members?: string[];
  label?: string;
}

function paperEdges(index: ProjectIndex, semanticIds: Set<string>): VisibleEdge[] {
  const pairs = new Map<string, { source: string; target: string; ids: string[]; kinds: Set<string> }>();
  for (const edge of index.edges) {
    if (!FLOW_KINDS.has(edge.kind)) continue;
    const source = index.memberSemantic.get(edge.source);
    const target = index.memberSemantic.get(edge.target);
    if (!source || !target || source === target) continue;
    if (!semanticIds.has(source) || !semanticIds.has(target)) continue;
    const key = `${source}→${target}`;
    if (!pairs.has(key)) {
      pairs.set(key, { source, target, ids: [], kinds: new Set() });
    }
    const item = pairs.get(key)!;
    item.ids.push(edge.id);
    item.kinds.add(edge.kind);
  }
  return [...pairs.entries()].map(([key, item]) => ({
    id: `paper:${key}`,
    source: `paper:${item.source}`,
    target: `paper:${item.target}`,
    kind: item.kinds.size === 1 ? [...item.kinds][0] : "data",
    mechanicalEdgeIds: [...new Set(item.ids)].sort(),
    synthetic: true,
  }));
}

function childDescriptors(
  index: ProjectIndex,
  entityId: string,
  runId: string | null,
  repeatExpanded: Set<string>,
): ChildDescriptor[] {
  const node = index.nodes.get(entityId);
  if (!node) return [];
  let ids: string[];
  if (node.level === "semantic") {
    ids = topSemanticMembers(index, entityId, runId);
  } else {
    ids = [...(index.children.get(entityId) ?? [])]
      .filter((id) => {
        const child = index.nodes.get(id);
        return child ? visibleInRun(child, runId) : false;
      })
      .sort();
  }
  if (!ids.length && node.level === "operation") {
    ids = [...new Set([
      ...(index.flowOut.get(entityId) ?? []).map((edge) => edge.target),
      ...(index.flowIn.get(entityId) ?? []).map((edge) => edge.source),
    ])]
      .filter((id) => {
        const child = index.nodes.get(id);
        return child ? visibleInRun(child, runId) : false;
      })
      .sort();
  }

  const groups = new Map<string, string[]>();
  const singles: string[] = [];
  for (const id of ids) {
    const child = index.nodes.get(id);
    const key = child?.definition_id ? `definition:${child.definition_id}` : null;
    if (!key) singles.push(id);
    else {
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)?.push(id);
    }
  }

  const result: ChildDescriptor[] = singles.map((id) => ({ id }));
  for (const [key, members] of groups) {
    if (members.length < 3) {
      result.push(...members.map((id) => ({ id })));
      continue;
    }
    const repeatId = `repeat:${entityId}:${key}`;
    if (repeatExpanded.has(repeatId)) {
      result.push(...members.map((id) => ({ id })));
      continue;
    }
    const first = index.nodes.get(members[0]);
    const definition = first?.definition_id
      ? index.nodes.get(first.definition_id)
      : undefined;
    result.push({
      id: repeatId,
      synthetic: "repeat",
      members,
      label: `${definition?.label ?? first?.label ?? "Repeated block"} ×${members.length}`,
    });
  }
  return result.sort((a, b) => a.id.localeCompare(b.id));
}

function displayLabel(node: AtirNode, overlays: OverlayMap): string {
  return overlays[node.id]?.alias?.trim() || node.label;
}

function entityNode(
  node: AtirNode,
  depth: number,
  overlays: OverlayMap,
  diff?: DiffSummary,
): VisibleNode {
  let diffStatus: VisibleNode["diffStatus"];
  if (diff?.added.has(node.id)) diffStatus = "added";
  else if (diff?.changed.has(node.id)) diffStatus = "changed";
  else if (diff?.same.has(node.id)) diffStatus = "same";
  return {
    id: node.level === "semantic" && depth === 0 ? `paper:${node.id}` : node.id,
    entityId: node.id,
    label: displayLabel(node, overlays),
    kind: node.level === "semantic" && depth === 0 ? "paper_component" : node.kind,
    role: node.role,
    depth,
    runId: node.run_id,
    diffStatus,
  };
}

const ROLE_ORDER = [
  "dataset", "preprocessor", "vision_encoder", "language_encoder",
  "proprioception_encoder", "point_cloud_encoder", "tokenizer",
  "multimodal_fusion", "transformer_backbone", "diffusion_denoiser",
  "world_model", "memory", "policy", "action_head", "classifier_head",
  "planner", "loss", "controller", "environment",
];

function semanticPriority(node: AtirNode): number {
  const index = ROLE_ORDER.indexOf(node.role ?? "");
  return index === -1 ? 100 : index;
}

function dedupeEdges(edges: VisibleEdge[]): VisibleEdge[] {
  const map = new Map<string, VisibleEdge>();
  for (const edge of edges) if (!map.has(edge.id)) map.set(edge.id, edge);
  return [...map.values()];
}

export function buildVisibleGraph(
  projectNodes: AtirNode[],
  index: ProjectIndex,
  expanded: Set<string>,
  repeatExpanded: Set<string>,
  runId: string | null,
  phase: PhaseFilter,
  overlays: OverlayMap,
  diff?: DiffSummary,
): VisibleGraph {
  const semanticNodes = projectNodes
    .filter((node) => semanticAllowed(node, phase))
    .sort((a, b) => semanticPriority(a) - semanticPriority(b) || a.id.localeCompare(b.id));
  const semanticIds = new Set(semanticNodes.map((node) => node.id));
  const nodes = semanticNodes.map((node) => entityNode(node, 0, overlays, diff));
  const edges = paperEdges(index, semanticIds);
  const seen = new Set(nodes.map((node) => node.id));

  const recurse = (parentDisplayId: string, entityId: string, depth: number) => {
    if (!expanded.has(entityId)) return;
    for (const child of childDescriptors(index, entityId, runId, repeatExpanded)) {
      if (child.synthetic === "repeat") {
        if (!seen.has(child.id)) {
          seen.add(child.id);
          nodes.push({
            id: child.id,
            label: child.label ?? "Repeated block",
            kind: "repeat_group",
            depth,
            synthetic: "repeat",
            repeatMembers: child.members,
          });
        }
        edges.push({
          id: `drill:${parentDisplayId}:${child.id}`,
          source: parentDisplayId,
          target: child.id,
          kind: "drilldown",
          mechanicalEdgeIds: [],
          synthetic: true,
        });
        continue;
      }
      const node = index.nodes.get(child.id);
      if (!node || !visibleInRun(node, runId)) continue;
      if (!seen.has(node.id)) {
        seen.add(node.id);
        nodes.push(entityNode(node, depth, overlays, diff));
      }
      edges.push({
        id: `drill:${parentDisplayId}:${node.id}`,
        source: parentDisplayId,
        target: node.id,
        kind: "drilldown",
        mechanicalEdgeIds: [],
        synthetic: true,
      });
      recurse(node.id, node.id, depth + 1);
    }
  };

  for (const semantic of semanticNodes) {
    recurse(`paper:${semantic.id}`, semantic.id, 1);
  }

  const visibleEntities = new Set(
    nodes.map((node) => node.entityId).filter((id): id is string => Boolean(id)),
  );
  for (const edge of index.edges) {
    if (!visibleEntities.has(edge.source) || !visibleEntities.has(edge.target)) continue;
    if (edge.kind === "contains") continue;
    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      kind: edge.kind,
      mechanicalEdgeIds: [edge.id],
    });
  }
  return { nodes, edges: dedupeEdges(edges) };
}

export function focusVisibleGraph(
  graph: VisibleGraph,
  selectedId: string,
  radius = 2,
): VisibleGraph {
  const selected = graph.nodes.find(
    (node) => node.id === selectedId || node.entityId === selectedId,
  );
  if (!selected) return graph;
  const adjacency = new Map<string, Set<string>>();
  for (const edge of graph.edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, new Set());
    if (!adjacency.has(edge.target)) adjacency.set(edge.target, new Set());
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }
  const keep = new Set([selected.id]);
  let frontier = new Set([selected.id]);
  for (let depth = 0; depth < radius; depth += 1) {
    const next = new Set<string>();
    for (const id of frontier) {
      for (const neighbor of adjacency.get(id) ?? []) {
        if (keep.has(neighbor)) continue;
        keep.add(neighbor);
        next.add(neighbor);
      }
    }
    frontier = next;
  }
  return {
    nodes: graph.nodes.filter((node) => keep.has(node.id)),
    edges: graph.edges.filter(
      (edge) => keep.has(edge.source) && keep.has(edge.target),
    ),
  };
}
