import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controllers.panda_ik_controller import PandaIKController
from controllers.scripted_expert import ScriptedPickPlaceExpert
from envs.cube_tray_env import CubeTrayEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils.dataset_validation import (
    DATASET_FPS,
    make_dataset_features,
    replay_episode,
    validate_dataset,
)


DEFAULT_ROOT = PROJECT_ROOT / "datasets" / "smolvla_cube_tray"
DEFAULT_REPO_ID = "local/smolvla_cube_tray"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect successful MuJoCo pick-and-place episodes in LeRobot format."
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def create_or_resume_dataset(args, env):
    metadata_path = args.root / "meta" / "info.json"

    if args.resume:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: dataset metadata not found at {metadata_path}"
            )
        return LeRobotDataset.resume(
            args.repo_id,
            root=args.root,
            streaming_encoding=True,
        )

    if metadata_path.exists():
        raise FileExistsError(
            f"Dataset already exists at {args.root}. Use --resume to append."
        )

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=DATASET_FPS,
        robot_type="mujoco_panda",
        features=make_dataset_features(
            image_height=env.image_height,
            image_width=env.image_width,
        ),
        use_videos=True,
        streaming_encoding=True,
    )


def validate_resume_contract(dataset):
    expected = make_dataset_features()
    if dataset.fps != DATASET_FPS:
        raise ValueError(
            f"Existing dataset FPS is {dataset.fps}, expected {DATASET_FPS}."
        )

    for key, feature in expected.items():
        if key not in dataset.features:
            raise ValueError(f"Existing dataset is missing feature {key}.")
        existing = dataset.features[key]
        if existing["dtype"] != feature["dtype"]:
            raise ValueError(f"Existing feature {key} has incompatible dtype.")
        if tuple(existing["shape"]) != tuple(feature["shape"]):
            raise ValueError(f"Existing feature {key} has incompatible shape.")


def collect_episode(dataset, env, expert, seed):
    def add_control_frame(action, phase):
        del phase
        observation = env.get_observation()
        dataset.add_frame(
            {
                "observation.images.front": observation[
                    "observation.images.front"
                ],
                "observation.images.wrist": observation[
                    "observation.images.wrist"
                ],
                "observation.state": observation[
                    "observation.state"
                ],
                "action": action.astype(np.float32, copy=False),
                "episode_seed": np.array([seed], dtype=np.int64),
                "task": observation["task"],
            }
        )

    return expert.run_episode(
        seed=seed,
        control_callback=add_control_frame,
    )


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")

    args.root = args.root.resolve()
    env = CubeTrayEnv()
    dataset = None

    try:
        controller = PandaIKController(
            env.model,
            env.data,
            ee_site_name="ee_site",
        )
        expert = ScriptedPickPlaceExpert(env, controller)
        dataset = create_or_resume_dataset(args, env)
        validate_resume_contract(dataset)

        print(f"Dataset root: {args.root}")
        print(f"Repository id: {args.repo_id}")
        print(f"Existing episodes: {dataset.num_episodes}")
        print(f"Target episodes: {args.num_episodes}")

        while dataset.num_episodes < args.num_episodes:
            episode_index = dataset.num_episodes
            seed = args.start_seed + episode_index

            for attempt in range(1, args.max_retries + 1):
                result = collect_episode(
                    dataset,
                    env,
                    expert,
                    seed,
                )

                if result.success:
                    dataset.save_episode(parallel_encoding=True)
                    print(
                        f"episode={episode_index} seed={seed} "
                        f"frames={result.control_steps} "
                        f"xy_error={result.tray_xy_error:.6f} "
                        "result=PASS"
                    )
                    break

                dataset.clear_episode_buffer()
                print(
                    f"episode={episode_index} seed={seed} "
                    f"attempt={attempt}/{args.max_retries} "
                    f"result=FAIL reason={result.failure_reason}"
                )
            else:
                raise RuntimeError(
                    f"Seed {seed} failed after {args.max_retries} attempts."
                )
    except BaseException:
        if dataset is not None and dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        raise
    finally:
        try:
            if dataset is not None:
                dataset.finalize()
        finally:
            env.close()

    validated = validate_dataset(
        args.repo_id,
        args.root,
        expected_episodes=args.num_episodes,
    )
    replay_indices = sorted({0, validated.num_episodes - 1})
    for episode_index in replay_indices:
        if not replay_episode(
            args.repo_id,
            args.root,
            episode_index,
        ):
            raise RuntimeError(
                f"Action replay failed for episode {episode_index}."
            )
        print(f"replay episode={episode_index} result=PASS")

    print(
        f"Dataset validation PASS: episodes={validated.num_episodes}, "
        f"frames={validated.num_frames}, fps={validated.fps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
