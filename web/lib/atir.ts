import type {
  AtirClaim,
  AtirConflict,
  AtirEdge,
  AtirNode,
  AtirProject,
  PhaseFilter,
} from "./types";

export const FLOW_KINDS = new Set([
  "data",
  "consumes",
  "produces",
  "derived_from",
  "alias",
  "next",
  "parameter",
]);

export interface ProjectIndex {
  nodes: Map<string, AtirNode>;
  edges: AtirEdge[];
  children: Map<string, Set<string>>;
  parents: Map<string, Set<string>>;
  semanticMembers: Map<string, Set<string>>;
  memberSemantic: Map<string, string>;
  flowOut: Map<string, AtirEdge[]>;
  flowIn: Map<string, AtirEdge[]>;
  evidence: Map<string, NonNullable<AtirProject["evidence"]>[number]>;
  claimsBySubject: Map<string, AtirClaim[]>;
  conflictsByClaim: Map<string, AtirConflict[]>;
}

export function parseProject(raw: unknown): AtirProject {
  if (!raw || typeof raw !== "object") {
    throw new Error("ATIR must be a JSON object.");
  }
  const candidate = raw as Partial<AtirProject>;
  if (!candidate.project || typeof candidate.project.name !== "string") {
    throw new Error("ATIR is missing project.name.");
  }
  if (!Array.isArray(candidate.nodes) || !Array.isArray(candidate.edges)) {
    throw new Error("ATIR requires nodes[] and edges[].");
  }
  for (const node of candidate.nodes) {
    if (!node || typeof node.id !== "string" || typeof node.label !== "string") {
      throw new Error("Every ATIR node requires id and label.");
    }
  }
  return {
    schema_version: candidate.schema_version,
    project: candidate.project,
    nodes: candidate.nodes,
    edges: candidate.edges,
    evidence: candidate.evidence ?? [],
    runs: candidate.runs ?? [],
    claims: candidate.claims ?? [],
    conflicts: candidate.conflicts ?? [],
    coverage: candidate.coverage ?? [],
    metadata: candidate.metadata ?? {},
  };
}

export function buildIndex(project: AtirProject): ProjectIndex {
  const nodes = new Map(project.nodes.map((node) => [node.id, node]));
  const children = new Map<string, Set<string>>();
  const parents = new Map<string, Set<string>>();
  const semanticMembers = new Map<string, Set<string>>();
  const memberSemantic = new Map<string, string>();
  const flowOut = new Map<string, AtirEdge[]>();
  const flowIn = new Map<string, AtirEdge[]>();

  const addChild = (parent: string, child: string) => {
    if (!children.has(parent)) children.set(parent, new Set());
    if (!parents.has(child)) parents.set(child, new Set());
    children.get(parent)?.add(child);
    parents.get(child)?.add(parent);
  };

  for (const node of project.nodes) {
    for (const parentId of node.parent_ids ?? []) {
      if (nodes.has(parentId)) addChild(parentId, node.id);
    }
  }

  for (const edge of project.edges) {
    if (edge.kind === "contains") {
      const source = nodes.get(edge.source);
      if (source?.level === "semantic") {
        if (!semanticMembers.has(source.id)) {
          semanticMembers.set(source.id, new Set());
        }
        semanticMembers.get(source.id)?.add(edge.target);
      } else if (nodes.has(edge.source) && nodes.has(edge.target)) {
        addChild(edge.source, edge.target);
      }
    }
    if (FLOW_KINDS.has(edge.kind)) {
      if (!flowOut.has(edge.source)) flowOut.set(edge.source, []);
      if (!flowIn.has(edge.target)) flowIn.set(edge.target, []);
      flowOut.get(edge.source)?.push(edge);
      flowIn.get(edge.target)?.push(edge);
    }
  }

  for (const node of project.nodes) {
    if (node.level !== "semantic") continue;
    const raw = node.attributes?.member_ids;
    if (!Array.isArray(raw)) continue;
    if (!semanticMembers.has(node.id)) semanticMembers.set(node.id, new Set());
    for (const value of raw) {
      if (typeof value === "string" && nodes.has(value)) {
        semanticMembers.get(node.id)?.add(value);
      }
    }
  }

  for (const [semanticId, members] of semanticMembers) {
    const queue = [...members];
    const visited = new Set<string>();
    while (queue.length) {
      const id = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      if (!memberSemantic.has(id)) memberSemantic.set(id, semanticId);
      for (const child of children.get(id) ?? []) queue.push(child);
    }
  }

  const evidence = new Map(
    (project.evidence ?? []).map((item) => [item.id, item]),
  );
  const claimsBySubject = new Map<string, AtirClaim[]>();
  for (const claim of project.claims ?? []) {
    if (!claimsBySubject.has(claim.subject_id)) {
      claimsBySubject.set(claim.subject_id, []);
    }
    claimsBySubject.get(claim.subject_id)?.push(claim);
  }
  const conflictsByClaim = new Map<string, AtirConflict[]>();
  for (const conflict of project.conflicts ?? []) {
    for (const claimId of conflict.claim_ids) {
      if (!conflictsByClaim.has(claimId)) {
        conflictsByClaim.set(claimId, []);
      }
      conflictsByClaim.get(claimId)?.push(conflict);
    }
  }

  return {
    nodes,
    edges: project.edges,
    children,
    parents,
    semanticMembers,
    memberSemantic,
    flowOut,
    flowIn,
    evidence,
    claimsBySubject,
    conflictsByClaim,
  };
}

export function semanticAllowed(node: AtirNode, phase: PhaseFilter): boolean {
  if (node.level !== "semantic") return false;
  if (node.attributes?.declaration_only === true) return false;
  if (phase === "all") return true;
  const value = node.attributes?.phase;
  return value == null || value === "both" || value === phase;
}

export function visibleInRun(node: AtirNode, runId: string | null): boolean {
  return runId === null || node.run_id == null || node.run_id === runId;
}

export function topSemanticMembers(
  index: ProjectIndex,
  semanticId: string,
  runId: string | null,
): string[] {
  const members = index.semanticMembers.get(semanticId) ?? new Set<string>();
  return [...members]
    .filter((id) => {
      const node = index.nodes.get(id);
      if (!node || !visibleInRun(node, runId)) return false;
      const parentIds = index.parents.get(id) ?? new Set<string>();
      return ![...parentIds].some((parent) => members.has(parent));
    })
    .sort();
}
