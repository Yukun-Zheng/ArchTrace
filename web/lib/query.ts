import type { AtirNode, AtirProject, LineageDirection, SearchResult } from "./types";
import type { ProjectIndex } from "./atir";
import { visibleInRun } from "./atir";

export function searchProject(
  project: AtirProject,
  query: string,
  runId: string | null,
): SearchResult[] {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!tokens.length) return [];
  const normalized = query.trim().toLowerCase();
  const results: SearchResult[] = [];
  for (const node of project.nodes) {
    if (!visibleInRun(node, runId)) continue;
    const paths = (node.source ?? [])
      .map((span) => `${span.path} ${span.symbol ?? ""}`)
      .join(" ");
    const tensor = node.tensor
      ? `${node.tensor.dtype ?? ""} ${node.tensor.device ?? ""} ${(node.tensor.semantics ?? []).join(" ")} ${JSON.stringify(node.tensor.shape ?? [])}`
      : "";
    const text = `${node.id} ${node.label} ${node.role ?? ""} ${node.kind} ${paths} ${tensor}`.toLowerCase();
    if (!tokens.every((token) => text.includes(token))) continue;
    let score = 10 * tokens.length;
    if (node.label.toLowerCase() === normalized || node.id.toLowerCase() === normalized) {
      score += 100;
    } else if (node.label.toLowerCase().startsWith(normalized)) {
      score += 60;
    }
    if ((node.role ?? "").toLowerCase().includes(normalized)) score += 40;
    results.push({
      id: node.id,
      label: node.label,
      role: node.role,
      kind: node.kind,
      source: node.source?.[0]?.path,
      score,
    });
  }
  return results
    .sort((a, b) => b.score - a.score || a.label.localeCompare(b.label))
    .slice(0, 80);
}

export function lineage(
  index: ProjectIndex,
  originId: string,
  direction: LineageDirection,
  runId: string | null,
  maxNodes = 2500,
): { nodes: Set<string>; edges: Set<string> } {
  const originNode = index.nodes.get(originId);
  const origins = originNode?.level === "semantic"
    ? [...(index.semanticMembers.get(originId) ?? [])]
    : [originId];
  const visited = new Set(origins);
  const edgeIds = new Set<string>();
  const queue = [...origins];
  while (queue.length && visited.size < maxNodes) {
    const current = queue.shift()!;
    const adjacent = direction === "downstream"
      ? index.flowOut.get(current) ?? []
      : index.flowIn.get(current) ?? [];
    for (const edge of adjacent) {
      const nextId = direction === "downstream" ? edge.target : edge.source;
      const node = index.nodes.get(nextId);
      if (!node || !visibleInRun(node, runId)) continue;
      edgeIds.add(edge.id);
      if (!visited.has(nextId)) {
        visited.add(nextId);
        queue.push(nextId);
      }
    }
  }
  return { nodes: visited, edges: edgeIds };
}

export function ancestorsToReveal(
  index: ProjectIndex,
  entityId: string,
): Set<string> {
  const result = new Set<string>();
  const semantic = index.memberSemantic.get(entityId);
  if (semantic) result.add(semantic);
  let current = entityId;
  const guard = new Set<string>();
  while (!guard.has(current)) {
    guard.add(current);
    const parent = [...(index.parents.get(current) ?? [])].sort()[0];
    if (!parent) break;
    result.add(parent);
    current = parent;
  }
  return result;
}

export function sourceFile(project: AtirProject, path: string): string | null {
  const web = project.metadata?.web;
  if (!web || typeof web !== "object") return null;
  const files = (web as { source_files?: unknown }).source_files;
  if (!files || typeof files !== "object") return null;
  const value = (files as Record<string, unknown>)[path];
  return typeof value === "string" ? value : null;
}

export function sourceExcerpt(
  project: AtirProject,
  node: AtirNode,
  context = 6,
): {
  path: string;
  start: number;
  lines: Array<{ number: number; text: string; active: boolean }>;
} | null {
  const span = node.source?.[0];
  if (!span) return null;
  const content = sourceFile(project, span.path);
  if (!content) return { path: span.path, start: span.start_line, lines: [] };
  const all = content.split("\n");
  const start = Math.max(1, span.start_line - context);
  const endLine = span.end_line ?? span.start_line;
  const end = Math.min(all.length, endLine + context);
  const lines = [];
  for (let number = start; number <= end; number += 1) {
    lines.push({
      number,
      text: all[number - 1] ?? "",
      active: number >= span.start_line && number <= endLine,
    });
  }
  return { path: span.path, start, lines };
}

export function breadcrumbPath(
  index: ProjectIndex,
  entityId: string,
): AtirNode[] {
  const result: AtirNode[] = [];
  const semanticId = index.memberSemantic.get(entityId);
  if (semanticId) {
    const semantic = index.nodes.get(semanticId);
    if (semantic) result.push(semantic);
  }
  const chain: AtirNode[] = [];
  let current = entityId;
  const seen = new Set<string>();
  while (!seen.has(current)) {
    seen.add(current);
    const node = index.nodes.get(current);
    if (node) chain.push(node);
    const parent = [...(index.parents.get(current) ?? [])].sort()[0];
    if (!parent) break;
    current = parent;
  }
  chain.reverse();
  for (const node of chain) {
    if (!result.some((item) => item.id === node.id)) result.push(node);
  }
  return result;
}
