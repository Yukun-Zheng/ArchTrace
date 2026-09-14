export type NodeLevel = "paper" | "semantic" | "module" | "operation" | "source";

export type IdentityKind = "group" | "definition" | "occurrence" | "value" | "state" | "source";

export interface SourceSpan {
  path: string;
  start_line: number;
  end_line?: number | null;
  start_column?: number | null;
  end_column?: number | null;
  symbol?: string | null;
}

export interface TensorSpec {
  shape?: Array<number | string | null> | null;
  dtype?: string | null;
  device?: string | null;
  layout?: string | null;
  requires_grad?: boolean | null;
  semantics?: string[];
  [key: string]: unknown;
}

export interface AtirNode {
  id: string;
  level: NodeLevel;
  kind: string;
  identity_kind?: IdentityKind;
  label: string;
  role?: string | null;
  parent_ids?: string[];
  definition_id?: string | null;
  run_id?: string | null;
  occurrence_index?: number | null;
  source?: SourceSpan[];
  evidence_ids?: string[];
  tensor?: TensorSpec | null;
  attributes?: Record<string, unknown>;
}

export interface AtirEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  label?: string | null;
  evidence_ids?: string[];
  tensor?: TensorSpec | null;
  attributes?: Record<string, unknown>;
}

export interface AtirEvidence {
  id: string;
  kind: string;
  status: string;
  confidence?: number;
  description?: string | null;
  source?: SourceSpan | null;
  run_id?: string | null;
  metadata?: Record<string, unknown>;
}

export interface AtirRun {
  id: string;
  entrypoint?: string | null;
  argv?: string[];
  config_paths?: string[];
  framework?: string | null;
  framework_version?: string | null;
  python_version?: string | null;
  metadata?: Record<string, unknown>;
}

export interface AtirClaim {
  id: string;
  subject_id: string;
  predicate: string;
  object_id?: string | null;
  value?: unknown;
  evidence_ids?: string[];
  status: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
}

export interface AtirConflict {
  id: string;
  claim_ids: string[];
  kind: string;
  status: string;
  description?: string | null;
  metadata?: Record<string, unknown>;
}

export interface AtirProject {
  schema_version?: string;
  project: {
    name: string;
    root?: string | null;
    repository_url?: string | null;
    revision?: string | null;
    metadata?: Record<string, unknown>;
  };
  nodes: AtirNode[];
  edges: AtirEdge[];
  evidence?: AtirEvidence[];
  runs?: AtirRun[];
  claims?: AtirClaim[];
  conflicts?: AtirConflict[];
  coverage?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
}

export type PhaseFilter = "all" | "training" | "inference";
export type ExplorerMode = "explore" | "diff";
export type LineageDirection = "upstream" | "downstream";

export interface OverlayRecord {
  alias?: string;
  note?: string;
  group?: string;
}

export type OverlayMap = Record<string, OverlayRecord>;

export interface VisibleNode {
  id: string;
  entityId?: string;
  label: string;
  kind: string;
  role?: string | null;
  depth: number;
  runId?: string | null;
  synthetic?: "repeat";
  repeatMembers?: string[];
  diffStatus?: "added" | "removed" | "changed" | "same";
}

export interface VisibleEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  mechanicalEdgeIds: string[];
  synthetic?: boolean;
}

export interface VisibleGraph {
  nodes: VisibleNode[];
  edges: VisibleEdge[];
}

export interface SearchResult {
  id: string;
  label: string;
  role?: string | null;
  kind: string;
  source?: string;
  score: number;
}

export interface DiffSummary {
  added: Set<string>;
  removed: Set<string>;
  changed: Set<string>;
  same: Set<string>;
  rows: Array<{
    key: string;
    current?: AtirNode;
    baseline?: AtirNode;
    status: "added" | "removed" | "changed" | "same";
  }>;
}
