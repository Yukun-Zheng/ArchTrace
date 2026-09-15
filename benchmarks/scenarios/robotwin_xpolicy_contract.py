"""RoboTwin↔XPolicyLab adapter contract scenario (target-runtime code)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run(capture, *, repository_root: Path, runtime_cwd: Path):
    import scripts.eval_policy_xpolicylab as bridge

    source = Path(bridge.__file__).resolve()
    if not source.is_relative_to(Path(repository_root).resolve()):
        raise RuntimeError(f"RoboTwin bridge imported from wrong source tree: {source}")

    observation = {
        "observation": {
            "head_camera": {
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "extrinsic_cv": np.eye(4, dtype=np.float32),
                "shape": (8, 8),
            },
            "left_camera": {"rgb": np.ones((8, 8, 3), dtype=np.uint8), "shape": (8, 8)},
            "right_camera": {
                "rgb": np.full((8, 8, 3), 2, dtype=np.uint8),
                "shape": (8, 8),
            },
        },
        "joint_action": {
            "left_arm": np.linspace(0, 0.5, 6, dtype=np.float32),
            "left_gripper": 0.1,
            "right_arm": np.linspace(0.5, 1.0, 6, dtype=np.float32),
            "right_gripper": 0.2,
        },
        "endpose": {
            "left_endpose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "left_gripper": 0.1,
            "right_endpose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            "right_gripper": 0.2,
        },
    }
    observation = capture.record_boundary(
        "simulator_observation",
        output=observation,
        role="environment",
        boundary="simulator",
        stubbed=True,
    )
    policy_observation = bridge.robotwin_obs_to_xpolicylab(
        observation,
        instruction="pick object",
        env_idx=0,
        frequency=30,
        task_env=None,
    )
    policy_response = {
        "actions": [
            {
                "left_arm_joint_state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "left_ee_joint_state": [0.25],
                "right_arm_joint_state": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "right_ee_joint_state": [0.35],
                "action_type": "joint",
            }
        ]
    }
    policy_response = capture.record_boundary(
        "policy_rpc",
        inputs={"observation": policy_observation},
        output=policy_response,
        role="transport",
        boundary="websocket",
        stubbed=True,
    )
    chunk = bridge.normalize_action_chunk(policy_response)
    action = capture.record_boundary(
        "action_chunk_select",
        inputs={"chunk": chunk},
        output=chunk[0],
        role="adapter",
        boundary="selection",
        stubbed=False,
    )
    flat_action, action_type = bridge.xpolicylab_action_to_robotwin(
        action,
        action_type="joint",
        current_observation=observation,
    )
    capture.record_boundary(
        "simulator_actuation",
        inputs={"action": flat_action, "action_type": action_type},
        output={"accepted": True},
        role="environment",
        boundary="simulator",
        stubbed=True,
    )
    return {
        "bridge_source": str(source),
        "runtime_cwd": str(runtime_cwd),
        "vision_keys": sorted(policy_observation["vision"]),
        "state_keys": sorted(policy_observation["state"]),
        "action_shape": list(flat_action.shape),
        "action_type": action_type,
    }
