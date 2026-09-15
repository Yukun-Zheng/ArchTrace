"""One real RoboTwin SAPIEN episode step with a stubbed policy boundary."""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import numpy as np
import yaml


def _as_float(value):
    array = np.asarray(value).reshape(-1)
    return float(array[0])


def run(capture, *, repository_root: Path, runtime_cwd: Path):
    repository_root = Path(repository_root).resolve()
    runtime_cwd = Path(runtime_cwd).resolve()
    os.chdir(runtime_cwd)

    import envs._GLOBAL_CONFIGS as global_configs
    import scripts.eval_policy_xpolicylab as bridge
    from scripts.collect_data import get_embodiment_config

    # RoboTwin normally derives planner assets from the source checkout. For this
    # benchmark code remains pinned while large runtime assets live in runtime_cwd.
    global_configs.ROOT_PATH = str(runtime_cwd) + "/"

    task_module = importlib.import_module("envs.adjust_bottle")
    base_module = importlib.import_module("envs._base_task")
    for module in (task_module, base_module, bridge):
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(repository_root):
            raise RuntimeError(f"RoboTwin module imported from wrong source tree: {source}")

    cfg_root = repository_root / "env_cfg" / "task_config"
    args = yaml.safe_load((cfg_root / "demo_clean.yml").read_text(encoding="utf-8"))
    args.update(
        episode_num=1,
        language_num=10,
        task_name="adjust_bottle",
        task_config="archtrace_real_sapien",
        need_plan=False,
        save_data=False,
        render_freq=0,
        embodiment_name="aloha-agilex",
        eval_mode=True,
    )
    args["data_type"].update(
        rgb=True,
        depth=True,
        pointcloud=False,
        qpos=True,
        endpose=True,
    )
    embodiment = yaml.safe_load(
        (cfg_root / "_embodiment_config.yml").read_text(encoding="utf-8")
    )["aloha-agilex"]["file_path"]
    args.update(
        left_robot_file=embodiment,
        right_robot_file=embodiment,
        dual_arm_embodied=True,
        left_embodiment_config=get_embodiment_config(embodiment),
        right_embodiment_config=get_embodiment_config(embodiment),
    )

    task_env = task_module.adjust_bottle()
    setup_started = time.perf_counter()
    try:
        task_env.setup_demo(now_ep_num=0, seed=0, is_test=True, **args)
        setup_seconds = time.perf_counter() - setup_started
        task_env.set_instruction(instruction="adjust bottle")

        observation_before = task_env.get_obs()
        qpos_before = np.asarray(observation_before["joint_action"]["vector"], dtype=np.float64)
        rgb_before = np.asarray(observation_before["observation"]["head_camera"]["rgb"])
        depth_before = np.asarray(observation_before["observation"]["head_camera"]["depth"])

        policy_observation = bridge.robotwin_obs_to_xpolicylab(
            observation_before,
            instruction=task_env.get_instruction(),
            env_idx=0,
            frequency=30,
            task_env=task_env,
        )
        joint = observation_before["joint_action"]
        policy_response = {
            "actions": [
                {
                    "left_arm_joint_state": np.asarray(joint["left_arm"]).tolist(),
                    "left_ee_joint_state": [_as_float(joint["left_gripper"])],
                    "right_arm_joint_state": np.asarray(joint["right_arm"]).tolist(),
                    "right_ee_joint_state": [_as_float(joint["right_gripper"])],
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
        action_chunk = bridge.normalize_action_chunk(policy_response)
        flat_action, action_type = bridge.xpolicylab_action_to_robotwin(
            action_chunk[0],
            action_type="joint",
            current_observation=observation_before,
        )

        action_started = time.perf_counter()
        task_env.take_action(flat_action, action_type=action_type)
        action_seconds = time.perf_counter() - action_started
        observation_after = task_env.get_obs()
        qpos_after = np.asarray(observation_after["joint_action"]["vector"], dtype=np.float64)
        rgb_after = np.asarray(observation_after["observation"]["head_camera"]["rgb"])

        return {
            "task": "adjust_bottle",
            "seed": 0,
            "task_source": str(Path(task_module.__file__).resolve()),
            "base_source": str(Path(base_module.__file__).resolve()),
            "bridge_source": str(Path(bridge.__file__).resolve()),
            "resource_root": str(runtime_cwd),
            "setup_seconds": setup_seconds,
            "action_seconds": action_seconds,
            "take_action_count": int(task_env.take_action_cnt),
            "qpos_shape": list(qpos_before.shape),
            "qpos_delta_max": float(np.max(np.abs(qpos_after - qpos_before))),
            "head_rgb_shape": list(rgb_before.shape),
            "head_rgb_dtype": str(rgb_before.dtype),
            "head_rgb_std": float(rgb_before.std()),
            "head_rgb_changed_mae": float(
                np.abs(rgb_after.astype(np.float32) - rgb_before.astype(np.float32)).mean()
            ),
            "head_depth_shape": list(depth_before.shape),
            "head_depth_finite": bool(np.isfinite(depth_before).all()),
            "policy_vision_keys": sorted(policy_observation["vision"]),
            "policy_state_keys": sorted(policy_observation["state"]),
            "action_shape": list(np.asarray(flat_action).shape),
            "action_type": action_type,
            "eval_success": bool(task_env.eval_success),
        }
    finally:
        task_env.close_env(clear_cache=True)
