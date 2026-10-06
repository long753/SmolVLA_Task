import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from smolvla_task.controllers.panda_ik_controller import PandaIKController
from smolvla_task.envs.cube_tray_env import CubeTrayEnv


DATASET_FPS = 25
STATE_NAMES = [
    "joint1_rad",
    "joint2_rad",
    "joint3_rad",
    "joint4_rad",
    "joint5_rad",
    "joint6_rad",
    "joint7_rad",
    "gripper_finger_position_m",
]
ACTION_NAMES = [
    "joint1_target_rad",
    "joint2_target_rad",
    "joint3_target_rad",
    "joint4_target_rad",
    "joint5_target_rad",
    "joint6_target_rad",
    "joint7_target_rad",
    "gripper_command_0_255",
]


def make_dataset_features(
    image_height: int = 256,
    image_width: int = 256,
) -> dict[str, dict]:
    camera_feature = {
        "dtype": "video",
        "shape": (image_height, image_width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.images.front": dict(camera_feature),
        "observation.images.wrist": dict(camera_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ACTION_NAMES,
        },
        "episode_seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["seed"],
        },
    }


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_dataset(
    repo_id: str,
    root: str | Path,
    expected_episodes: int | None = None,
) -> LeRobotDataset:
    """Load a finalized dataset and validate its SmolVLA-facing contract."""
    dataset = LeRobotDataset(repo_id, root=root)
    expected_features = make_dataset_features()

    if dataset.fps != DATASET_FPS:
        raise AssertionError(
            f"Dataset FPS is {dataset.fps}, expected {DATASET_FPS}."
        )

    if expected_episodes is not None and dataset.num_episodes != expected_episodes:
        raise AssertionError(
            f"Dataset has {dataset.num_episodes} episodes, "
            f"expected {expected_episodes}."
        )

    for key, expected in expected_features.items():
        if key not in dataset.features:
            raise AssertionError(f"Missing dataset feature: {key}")
        actual = dataset.features[key]
        if actual["dtype"] != expected["dtype"]:
            raise AssertionError(
                f"Feature {key} dtype is {actual['dtype']}, "
                f"expected {expected['dtype']}."
            )
        if tuple(actual["shape"]) != tuple(expected["shape"]):
            raise AssertionError(
                f"Feature {key} shape is {actual['shape']}, "
                f"expected {expected['shape']}."
            )

    if dataset.num_frames == 0:
        raise AssertionError("Dataset contains no frames.")

    timeline = dataset.hf_dataset.select_columns(
        ["episode_index", "frame_index", "timestamp", "index"]
    ).with_format("numpy")[:]
    episode_indices = timeline["episode_index"]
    if not np.array_equal(timeline["index"], np.arange(dataset.num_frames)):
        raise AssertionError("Dataset index is not contiguous.")
    if np.any(np.diff(episode_indices) < 0):
        raise AssertionError("Episode rows are not stored contiguously.")
    if not np.array_equal(
        np.unique(episode_indices),
        np.arange(dataset.num_episodes),
    ):
        raise AssertionError("Episode indices are missing or non-contiguous.")

    for episode_index in range(dataset.num_episodes):
        episode_rows = episode_indices == episode_index
        frame_indices = timeline["frame_index"][episode_rows]
        timestamps = timeline["timestamp"][episode_rows]
        expected_frames = np.arange(frame_indices.size)
        if not np.array_equal(frame_indices, expected_frames):
            raise AssertionError(
                f"Frame indices are not contiguous in episode {episode_index}."
            )
        expected_timestamps = expected_frames / DATASET_FPS
        if not np.allclose(
            timestamps,
            expected_timestamps,
            rtol=0.0,
            atol=1e-6,
        ):
            raise AssertionError(
                f"Timestamps are not aligned to {DATASET_FPS} FPS "
                f"in episode {episode_index}."
            )

    sample_indices = sorted(
        {
            0,
            dataset.num_frames // 2,
            dataset.num_frames - 1,
        }
    )
    for index in sample_indices:
        sample = dataset[index]
        state = _as_numpy(sample["observation.state"])
        action = _as_numpy(sample["action"])
        front = _as_numpy(sample["observation.images.front"])
        wrist = _as_numpy(sample["observation.images.wrist"])

        if state.shape != (8,) or not np.isfinite(state).all():
            raise AssertionError(f"Invalid state at dataset index {index}.")
        if action.shape != (8,) or not np.isfinite(action).all():
            raise AssertionError(f"Invalid action at dataset index {index}.")
        if front.shape != (3, 256, 256):
            raise AssertionError(
                f"Invalid front image shape at index {index}: {front.shape}."
            )
        if wrist.shape != (3, 256, 256):
            raise AssertionError(
                f"Invalid wrist image shape at index {index}: {wrist.shape}."
            )
        if not 0.0 <= action[7] <= 255.0:
            raise AssertionError(
                f"Invalid gripper action at index {index}: {action[7]}."
            )

    return dataset


def replay_episode(
    repo_id: str,
    root: str | Path,
    episode_index: int,
) -> bool:
    """Replay recorded actuator targets without invoking the scripted expert."""
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=[episode_index],
    )
    if len(dataset) == 0:
        raise AssertionError(f"Episode {episode_index} contains no frames.")

    seed_value = _as_numpy(dataset[0]["episode_seed"]).reshape(-1)[0]
    seed = int(seed_value)
    env = CubeTrayEnv()

    try:
        controller = PandaIKController(
            env.model,
            env.data,
            ee_site_name="ee_site",
        )
        env.reset(seed=seed)
        controller.set_joint_target(
            env.data.qpos[env.arm_qpos_indices].copy()
        )
        controller.open_gripper()
        for _ in range(250):
            mujoco.mj_step(env.model, env.data)

        for sample in dataset:
            action = _as_numpy(sample["action"]).astype(
                np.float64,
                copy=False,
            )
            controller.set_joint_target(action[:7])
            controller.set_gripper(float(action[7]))
            for _ in range(20):
                mujoco.mj_step(env.model, env.data)

        required_stable_steps = int(
            np.ceil(2.0 / env.model.opt.timestep)
        )
        stable_steps = 0
        timeout_steps = int(
            np.ceil(4.0 / env.model.opt.timestep)
        )
        for _ in range(timeout_steps):
            mujoco.mj_step(env.model, env.data)
            if env.is_cube_stably_in_tray():
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= required_stable_steps:
                return True

        return False
    finally:
        env.close()
