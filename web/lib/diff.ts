import type { AtirNode, AtirProject, DiffSummary } from "./types";

function semanticKey(node: AtirNode): string {
  return `${node.role ?? "unknown"}::${node.label.toLowerCase()}`;
}

function semanticFingerprint(node: AtirNode): string {
  const attrs = node.attributes ?? {};
  return JSON.stringify({
    role: node.role,
    label: node.label,
    phase: attrs.phase ?? null,
    modalities: attrs.modalities ?? [],
    members: Array.isArray(attrs.member_ids) ? attrs.member_ids.length : 0,
  });
}

export function diffProjects(
  current: AtirProject,
  baseline: AtirProject,
): DiffSummary {
  const currentSemantic = current.nodes.filter(
    (node) => node.level === "semantic" && node.attributes?.declaration_only !== true,
  );
  const baselineSemantic = baseline.nodes.filter(
    (node) => node.level === "semantic" && node.attributes?.declaration_only !== true,
  );
  const currentByKey = new Map(
    currentSemantic.map((node) => [semanticKey(node), node]),
  );
  const baselineByKey = new Map(
    baselineSemantic.map((node) => [semanticKey(node), node]),
  );
  const keys = new Set([...currentByKey.keys(), ...baselineByKey.keys()]);
  const added = new Set<string>();
  const removed = new Set<string>();
  const changed = new Set<string>();
  const same = new Set<string>();
  const rows: DiffSummary["rows"] = [];

  for (const key of [...keys].sort()) {
    const currentNode = currentByKey.get(key);
    const baselineNode = baselineByKey.get(key);
    let status: "added" | "removed" | "changed" | "same";
    if (currentNode && !baselineNode) {
      status = "added";
      added.add(currentNode.id);
    } else if (!currentNode && baselineNode) {
      status = "removed";
      removed.add(baselineNode.id);
    } else if (
      currentNode
      && baselineNode
      && semanticFingerprint(currentNode) !== semanticFingerprint(baselineNode)
    ) {
      status = "changed";
      changed.add(currentNode.id);
    } else {
      status = "same";
      if (currentNode) same.add(currentNode.id);
    }
    rows.push({ key, current: currentNode, baseline: baselineNode, status });
  }
  return { added, removed, changed, same, rows };
}

export function projectStats(project: AtirProject): {
  nodes: number;
  edges: number;
  semantic: number;
  runs: number;
  conflicts: number;
} {
  return {
    nodes: project.nodes.length,
    edges: project.edges.length,
    semantic: project.nodes.filter((node) => node.level === "semantic").length,
    runs: project.runs?.length ?? 0,
    conflicts: project.conflicts?.length ?? 0,
  };
}
