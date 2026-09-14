import type { AtirEdge, AtirNode, AtirProject } from "./types";

function semantic(
  id: string,
  label: string,
  role: string,
  members: string[],
  phase: "training" | "inference" | "both" = "both",
  modalities: string[] = [],
): AtirNode {
  return {
    id,
    level: "semantic",
    kind: "semantic_component",
    identity_kind: "group",
    label,
    role,
    source: [],
    evidence_ids: [`evidence.${id}`],
    attributes: {
      confidence: 0.94,
      member_ids: members,
      phase,
      modalities,
      inference_backend: "deterministic_rules_v0",
    },
  };
}

function moduleNode(
  id: string,
  label: string,
  path: string,
  line: number,
  parentIds: string[] = [],
): AtirNode {
  return {
    id,
    level: "module",
    kind: "module",
    identity_kind: "definition",
    label,
    role: "pytorch_module_definition",
    parent_ids: parentIds,
    source: [{ path, start_line: line, end_line: line + 16, symbol: label }],
    evidence_ids: [`evidence.${id}`],
  };
}

function valueNode(
  id: string,
  label: string,
  shape: Array<number | string>,
  semantics: string[],
  runId = "run.demo",
): AtirNode {
  return {
    id,
    level: "operation",
    kind: label.includes("input") ? "input" : label.includes("output") ? "output" : "tensor",
    identity_kind: "value",
    label,
    role: "tensor",
    run_id: runId,
    tensor: {
      shape,
      dtype: "float32",
      device: "cuda:0",
      semantics,
    },
    evidence_ids: [`evidence.${id}`],
  };
}

function opNode(
  id: string,
  label: string,
  parent: string,
  line: number,
  runId = "run.demo",
): AtirNode {
  return {
    id,
    level: "operation",
    kind: "operation",
    identity_kind: "occurrence",
    label,
    role: "pytorch_operator_call",
    parent_ids: [parent],
    run_id: runId,
    occurrence_index: 0,
    source: [{ path: "models/policy.py", start_line: line, end_line: line, symbol: label }],
    evidence_ids: [`evidence.${id}`],
  };
}

function flow(
  id: string,
  source: string,
  target: string,
  kind = "data",
): AtirEdge {
  return { id, source, target, kind, evidence_ids: [`evidence.${id}`] };
}

const semanticNodes: AtirNode[] = [
  semantic("sem.vision", "Vision Encoder", "vision_encoder", ["mod.vision"], "both", ["vision"]),
  semantic("sem.language", "Language Encoder", "language_encoder", ["mod.language"], "both", ["language"]),
  semantic("sem.fusion", "Multimodal Fusion", "multimodal_fusion", ["mod.fusion"], "both", ["vision", "language"]),
  semantic("sem.backbone", "Transformer Backbone", "transformer_backbone", ["mod.transformer"], "both", ["latent"]),
  semantic("sem.action", "Action Head", "action_head", ["mod.action"], "inference", ["action"]),
  semantic("sem.loss", "Policy Loss", "loss", ["mod.loss"], "training"),
  semantic("sem.controller", "Low-level Controller", "controller", ["mod.controller"], "inference", ["action"]),
  semantic("sem.env", "Robot Environment", "environment", ["mod.env"], "inference", ["state"]),
];

const moduleNodes: AtirNode[] = [
  moduleNode("mod.vision", "VisionEncoder", "models/vision.py", 18),
  moduleNode("mod.language", "LanguageEncoder", "models/language.py", 11),
  moduleNode("mod.fusion", "CrossModalFusion", "models/policy.py", 42),
  moduleNode("mod.transformer", "TransformerBackbone", "models/policy.py", 71),
  moduleNode("mod.transformer_block", "TransformerBlock", "models/policy.py", 88, ["mod.transformer"]),
  moduleNode("mod.action", "ActionHead", "models/policy.py", 132),
  moduleNode("mod.loss", "PolicyLoss", "train/losses.py", 24),
  moduleNode("mod.controller", "LowLevelController", "robot/controller.py", 38),
  moduleNode("mod.env", "RobotEnvironment", "robot/environment.py", 17),
];

const repeatedBlocks: AtirNode[] = Array.from({ length: 6 }, (_, index) => ({
  id: `call.block.${index}`,
  level: "module",
  kind: "module",
  identity_kind: "occurrence",
  label: `block.${index}`,
  role: "pytorch_module_call",
  parent_ids: ["mod.transformer"],
  definition_id: "mod.transformer_block",
  run_id: "run.demo",
  occurrence_index: index,
  source: [{ path: "models/policy.py", start_line: 88, end_line: 112, symbol: "TransformerBlock.forward" }],
  evidence_ids: [`evidence.call.block.${index}`],
}));

const values: AtirNode[] = [
  valueNode("value.image.input", "image input", [1, 3, 224, 224], ["vision", "rgb"]),
  valueNode("value.text.input", "instruction input", [1, 32], ["language", "tokens"]),
  valueNode("value.visual.tokens", "visual tokens", [1, 196, 768], ["vision", "latent"]),
  valueNode("value.language.tokens", "language tokens", [1, 32, 768], ["language", "latent"]),
  valueNode("value.fused", "fused tokens", [1, 228, 768], ["vision", "language", "latent"]),
  valueNode("value.latent", "policy latent", [1, 228, 768], ["latent"]),
  valueNode("value.action.output", "action output", [1, 16, 14], ["action"]),
  valueNode("value.state", "robot state", [1, 48], ["state", "proprioception"]),
];

const operations: AtirNode[] = [
  opNode("op.vision.patch", "aten::conv2d", "mod.vision", 27),
  opNode("op.vision.norm", "aten::layer_norm", "mod.vision", 31),
  opNode("op.language.embed", "aten::embedding", "mod.language", 19),
  opNode("op.fusion.concat", "aten::cat", "mod.fusion", 51),
  opNode("op.fusion.proj", "aten::linear", "mod.fusion", 55),
  opNode("op.action.linear", "aten::linear", "mod.action", 141),
  opNode("op.controller.step", "controller.step", "mod.controller", 49),
];

const nodes = [...semanticNodes, ...moduleNodes, ...repeatedBlocks, ...values, ...operations];

const edges: AtirEdge[] = [
  flow("edge.paper.vision-fusion", "mod.vision", "mod.fusion"),
  flow("edge.paper.language-fusion", "mod.language", "mod.fusion"),
  flow("edge.paper.fusion-backbone", "mod.fusion", "mod.transformer"),
  flow("edge.paper.backbone-action", "mod.transformer", "mod.action"),
  flow("edge.paper.backbone-loss", "mod.transformer", "mod.loss"),
  flow("edge.paper.action-controller", "mod.action", "mod.controller"),
  flow("edge.paper.controller-env", "mod.controller", "mod.env"),
  flow("edge.1", "value.image.input", "op.vision.patch", "consumes"),
  flow("edge.2", "op.vision.patch", "op.vision.norm", "data"),
  flow("edge.3", "op.vision.norm", "value.visual.tokens", "produces"),
  flow("edge.4", "value.text.input", "op.language.embed", "consumes"),
  flow("edge.5", "op.language.embed", "value.language.tokens", "produces"),
  flow("edge.6", "value.visual.tokens", "op.fusion.concat", "consumes"),
  flow("edge.7", "value.language.tokens", "op.fusion.concat", "consumes"),
  flow("edge.8", "op.fusion.concat", "op.fusion.proj", "data"),
  flow("edge.9", "op.fusion.proj", "value.fused", "produces"),
  flow("edge.10", "value.fused", "call.block.0", "consumes"),
  ...Array.from({ length: 5 }, (_, index) => flow(`edge.block.${index}`, `call.block.${index}`, `call.block.${index + 1}`, "next")),
  flow("edge.16", "call.block.5", "value.latent", "produces"),
  flow("edge.17", "value.latent", "op.action.linear", "consumes"),
  flow("edge.18", "op.action.linear", "value.action.output", "produces"),
  flow("edge.19", "value.action.output", "op.controller.step", "consumes"),
  flow("edge.20", "op.controller.step", "mod.env", "data"),
  flow("edge.21", "mod.env", "value.state", "produces"),
  { id: "edge.contains.blockdef", source: "mod.transformer", target: "mod.transformer_block", kind: "contains" },
  ...repeatedBlocks.map((node, index) => ({ id: `edge.contains.block.${index}`, source: "mod.transformer", target: node.id, kind: "contains" })),
];

const evidence = nodes.map((node) => ({
  id: node.evidence_ids?.[0] ?? `evidence.${node.id}`,
  kind: node.run_id ? "runtime" : "static",
  status: node.run_id ? "observed" : "inferred",
  confidence: node.level === "semantic" ? 0.94 : 1,
  description: node.level === "semantic" ? "Evidence-grounded semantic grouping." : "Mechanical fact captured by ArchTrace.",
  source: node.source?.[0] ?? null,
  run_id: node.run_id ?? null,
}));

export const demoProject: AtirProject = {
  schema_version: "0.2",
  project: {
    name: "VLA Manipulation Policy",
    repository_url: "https://github.com/Yukun-Zheng/ArchTrace",
    revision: "demo",
  },
  nodes,
  edges,
  evidence,
  runs: [
    { id: "run.demo", entrypoint: "eval.py", framework: "pytorch", framework_version: "2.x" },
    { id: "run.alt", entrypoint: "eval.py", framework: "pytorch", framework_version: "2.x", metadata: { note: "alternate rollout" } },
  ],
  claims: [
    {
      id: "claim.author.vision_frozen",
      subject_id: "sem.vision",
      predicate: "frozen",
      value: true,
      evidence_ids: ["evidence.sem.vision"],
      status: "declared",
      confidence: 1,
      metadata: { implementation_check: "unresolved" },
    },
  ],
  conflicts: [],
  coverage: [],
  metadata: {
    semantic: { backend: "deterministic_rules_v0" },
    web: {
      source_files: {
        "models/vision.py": `class VisionEncoder(nn.Module):\n    def forward(self, image):\n        x = self.patch_embed(image)\n        x = self.norm(x)\n        return x.flatten(2).transpose(1, 2)\n`,
        "models/language.py": `class LanguageEncoder(nn.Module):\n    def forward(self, token_ids):\n        return self.embedding(token_ids)\n`,
        "models/policy.py": `class CrossModalFusion(nn.Module):\n    def forward(self, vision, language):\n        fused = torch.cat([vision, language], dim=1)\n        return self.proj(fused)\n\nclass TransformerBlock(nn.Module):\n    def forward(self, x):\n        x = x + self.attn(self.norm1(x))\n        return x + self.mlp(self.norm2(x))\n\nclass ActionHead(nn.Module):\n    def forward(self, latent):\n        return self.linear(latent[:, -16:])\n`,
        "train/losses.py": `class PolicyLoss(nn.Module):\n    def forward(self, prediction, target):\n        return F.mse_loss(prediction, target)\n`,
        "robot/controller.py": `class LowLevelController:\n    def step(self, action):\n        return self.robot.apply_action(action)\n`,
        "robot/environment.py": `class RobotEnvironment:\n    def step(self, action):\n        self.controller.step(action)\n        return self.observe()\n`,
      },
    },
  },
};

export const demoComparisonProject: AtirProject = {
  ...demoProject,
  project: { ...demoProject.project, name: "VLA Policy — candidate v2", revision: "demo-v2" },
  nodes: [
    ...demoProject.nodes.filter((node) => node.id !== "sem.loss" && node.id !== "mod.loss"),
    semantic("sem.world", "Predictive World Model", "world_model", ["mod.world"], "both", ["latent", "state"]),
    moduleNode("mod.world", "LatentWorldModel", "models/world_model.py", 12),
  ],
  edges: [
    ...demoProject.edges.filter((edge) => !edge.id.includes("backbone-loss")),
    flow("edge.paper.backbone-world", "mod.transformer", "mod.world"),
    flow("edge.paper.world-action", "mod.world", "mod.action"),
  ],
  metadata: {
    ...demoProject.metadata,
    web: {
      source_files: {
        ...((demoProject.metadata?.web as { source_files?: Record<string, string> })?.source_files ?? {}),
        "models/world_model.py": `class LatentWorldModel(nn.Module):\n    def forward(self, latent, state):\n        return self.transition(torch.cat([latent, state], dim=-1))\n`,
      },
    },
  },
};
