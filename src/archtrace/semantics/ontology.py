"""Deterministic semantic ontology used by ArchTrace M2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SemanticRole(StrEnum):
    VISION_ENCODER = "vision_encoder"
    LANGUAGE_ENCODER = "language_encoder"
    PROPRIOCEPTION_ENCODER = "proprioception_encoder"
    POINT_CLOUD_ENCODER = "point_cloud_encoder"
    GENERIC_ENCODER = "generic_encoder"
    TOKENIZER = "tokenizer"
    MULTIMODAL_FUSION = "multimodal_fusion"
    TRANSFORMER_BACKBONE = "transformer_backbone"
    DIFFUSION_DENOISER = "diffusion_denoiser"
    WORLD_MODEL = "world_model"
    MEMORY = "memory"
    POLICY = "policy"
    ACTION_HEAD = "action_head"
    CLASSIFIER_HEAD = "classifier_head"
    PLANNER = "planner"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    DATASET = "dataset"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    CONTROLLER = "controller"
    ENVIRONMENT = "environment"
    REWARD = "reward"
    GENERIC_DECODER = "generic_decoder"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    VISION = "vision"
    LANGUAGE = "language"
    PROPRIOCEPTION = "proprioception"
    DEPTH = "depth"
    POINT_CLOUD = "point_cloud"
    AUDIO = "audio"
    STATE = "state"
    LATENT = "latent"
    ACTION = "action"
    REWARD = "reward"


class SemanticPhase(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    role: SemanticRole
    display_label: str
    keywords: tuple[str, ...]
    phase: SemanticPhase = SemanticPhase.BOTH
    default_modalities: tuple[Modality, ...] = ()
    priority: int = 100


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(
        SemanticRole.VISION_ENCODER,
        "Vision Encoder",
        (
            "vision_encoder",
            "visual_encoder",
            "image_encoder",
            "vision_backbone",
            "image_backbone",
            "resnet",
            "dino",
            "clip_vision",
            "vit",
        ),
        default_modalities=(Modality.VISION,),
        priority=10,
    ),
    RoleSpec(
        SemanticRole.LANGUAGE_ENCODER,
        "Language Encoder",
        (
            "language_encoder",
            "text_encoder",
            "language_model",
            "text_model",
            "llm",
            "bert",
            "qwen",
        ),
        default_modalities=(Modality.LANGUAGE,),
        priority=11,
    ),
    RoleSpec(
        SemanticRole.PROPRIOCEPTION_ENCODER,
        "Proprioception Encoder",
        ("proprio_encoder", "proprioception_encoder", "proprioceptive_encoder"),
        default_modalities=(Modality.PROPRIOCEPTION,),
        priority=12,
    ),
    RoleSpec(
        SemanticRole.POINT_CLOUD_ENCODER,
        "Point-Cloud Encoder",
        ("point_cloud_encoder", "pointcloud_encoder", "pointnet", "point_transformer"),
        default_modalities=(Modality.POINT_CLOUD,),
        priority=13,
    ),
    RoleSpec(
        SemanticRole.TOKENIZER,
        "Tokenizer",
        ("tokenizer", "tokenization", "tokenize", "patch_embed", "patch_embedding"),
        priority=20,
    ),
    RoleSpec(
        SemanticRole.MULTIMODAL_FUSION,
        "Multimodal Fusion",
        (
            "multimodal_fusion",
            "cross_modal_fusion",
            "crossmodal_fusion",
            "cross_attention",
            "crossattention",
        ),
        priority=30,
    ),
    RoleSpec(
        SemanticRole.TRANSFORMER_BACKBONE,
        "Transformer Backbone",
        ("transformer", "attention_backbone", "transformer_backbone"),
        priority=40,
    ),
    RoleSpec(
        SemanticRole.DIFFUSION_DENOISER,
        "Diffusion Denoiser",
        (
            "diffusion_denoiser",
            "denoiser",
            "noise_predictor",
            "noise_prediction",
            "diffusion_model",
            "unet",
        ),
        priority=41,
    ),
    RoleSpec(
        SemanticRole.WORLD_MODEL,
        "World Model",
        ("world_model", "dynamics_model", "latent_dynamics", "transition_model"),
        priority=42,
    ),
    RoleSpec(
        SemanticRole.MEMORY,
        "Memory",
        ("memory", "memory_bank", "episodic_memory", "kv_cache"),
        priority=43,
    ),
    RoleSpec(
        SemanticRole.POLICY,
        "Policy",
        ("policy", "actor", "policy_network", "policy_model"),
        phase=SemanticPhase.INFERENCE,
        priority=50,
    ),
    RoleSpec(
        SemanticRole.ACTION_HEAD,
        "Action Head",
        (
            "action_head",
            "action_decoder",
            "action_projector",
            "action_prediction",
            "action_output",
        ),
        phase=SemanticPhase.INFERENCE,
        default_modalities=(Modality.ACTION,),
        priority=51,
    ),
    RoleSpec(
        SemanticRole.CLASSIFIER_HEAD,
        "Classifier Head",
        ("classifier", "classification_head", "classifier_head", "logits_head"),
        priority=52,
    ),
    RoleSpec(
        SemanticRole.PLANNER,
        "Planner",
        ("planner", "planning", "trajectory_planner", "motion_planner"),
        phase=SemanticPhase.INFERENCE,
        priority=53,
    ),
    RoleSpec(
        SemanticRole.LOSS,
        "Loss",
        ("loss", "criterion", "objective", "loss_fn"),
        phase=SemanticPhase.TRAINING,
        priority=60,
    ),
    RoleSpec(
        SemanticRole.OPTIMIZER,
        "Optimizer",
        ("optimizer", "adam", "adamw", "sgd", "lion"),
        phase=SemanticPhase.TRAINING,
        priority=61,
    ),
    RoleSpec(
        SemanticRole.DATASET,
        "Dataset",
        ("dataset", "dataloader", "data_loader", "sampler"),
        priority=70,
    ),
    RoleSpec(
        SemanticRole.PREPROCESSOR,
        "Preprocessor",
        ("preprocess", "preprocessor", "augmentation", "augment", "transform_input"),
        priority=71,
    ),
    RoleSpec(
        SemanticRole.POSTPROCESSOR,
        "Postprocessor",
        ("postprocess", "postprocessor", "decode_output"),
        priority=72,
    ),
    RoleSpec(
        SemanticRole.CONTROLLER,
        "Controller",
        ("controller", "control_policy", "low_level_control", "pid_controller"),
        phase=SemanticPhase.INFERENCE,
        default_modalities=(Modality.ACTION,),
        priority=73,
    ),
    RoleSpec(
        SemanticRole.ENVIRONMENT,
        "Environment",
        ("environment", "env_wrapper", "simulator", "simulation_env", "robot_env"),
        phase=SemanticPhase.INFERENCE,
        default_modalities=(Modality.STATE,),
        priority=74,
    ),
    RoleSpec(
        SemanticRole.REWARD,
        "Reward",
        ("reward", "reward_model", "reward_fn"),
        phase=SemanticPhase.TRAINING,
        default_modalities=(Modality.REWARD,),
        priority=75,
    ),
    RoleSpec(
        SemanticRole.GENERIC_ENCODER,
        "Encoder",
        ("encoder", "encode"),
        priority=90,
    ),
    RoleSpec(
        SemanticRole.GENERIC_DECODER,
        "Decoder",
        ("decoder", "decode"),
        priority=91,
    ),
)

ROLE_BY_VALUE = {spec.role: spec for spec in ROLE_SPECS}
_UNKNOWN_ROLE_SPEC = RoleSpec(
    role=SemanticRole.UNKNOWN,
    display_label="Unknown",
    keywords=(),
    priority=999,
)


MODALITY_KEYWORDS: dict[Modality, tuple[str, ...]] = {
    Modality.VISION: ("vision", "visual", "image", "rgb", "camera", "pixel", "resnet", "vit"),
    Modality.LANGUAGE: ("language", "text", "prompt", "instruction", "word", "llm", "bert", "qwen"),
    Modality.PROPRIOCEPTION: ("proprio", "joint_state", "joint_position", "joint_pos", "ee_pose"),
    Modality.DEPTH: ("depth", "depth_map", "rgbd"),
    Modality.POINT_CLOUD: ("point_cloud", "pointcloud", "pointnet", "point_transformer"),
    Modality.AUDIO: ("audio", "speech", "waveform", "microphone"),
    Modality.STATE: ("state", "observation", "robot_state"),
    Modality.LATENT: ("latent", "embedding", "hidden_state"),
    Modality.ACTION: ("action", "actuator", "motor_command"),
    Modality.REWARD: ("reward",),
}


def role_spec(role: SemanticRole) -> RoleSpec:
    return ROLE_BY_VALUE.get(role, _UNKNOWN_ROLE_SPEC)
