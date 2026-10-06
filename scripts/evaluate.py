import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from smolvla_task.envs.cube_tray_env import CubeTrayEnv
from smolvla_task.utils.episode_monitor import EpisodeEventMonitor
from smolvla_task.utils.mujoco_video_recorder import MuJoCoVideoRecorder


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "smolvla_cube_tray"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


@dataclass(frozen=True)
class EvaluationResult:
    run_name: str
    policy_path: str
    seed: int
    success: bool
    termination_reason: str
    control_steps: int
    action_chunks: int
    simulated_seconds: float
    wall_seconds: float
    action_latencies_ms: list[float]
    inference_latencies_ms: list[float]
    clipped_action_steps: int
    stable_duration: float
    cube_lifted: bool
    cube_dropped: bool
    max_cube_height: float
    collision: bool
    collision_events: int
    collision_sim_steps: int
    initial_cube_position: list[float]
    final_cube_position: list[float]
    tray_position: list[float]
    tray_xy_error: float
    action_min: list[float]
    action_max: list[float]
    front_video: str | None
    wrist_video: str | None


def parse_bool(value):
    normalized_value = value.lower()
    if normalized_value == "true":
        return True
    if normalized_value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected either true or false.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one closed-loop SmolVLA episode in CubeTrayEnv."
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--dataset-repo-id", default="local/smolvla_cube_tray")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default="smolvla_base")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", type=parse_bool, default=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Resolve model and tokenizer only from the Hugging Face cache.",
    )
    return parser.parse_args()


def resolve_policy_path(policy_path: str, offline: bool):
    local_path = Path(policy_path).expanduser()
    if local_path.exists():
        return local_path.resolve()
    if not offline:
        return policy_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=policy_path,
            local_files_only=True,
        )
    )


def load_policy_stack(args):
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device}, but CUDA is unavailable.")

    dataset_root = args.dataset_root.resolve()
    metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=dataset_root,
    )
    policy_source = resolve_policy_path(args.policy_path, args.offline)
    policy_config = PreTrainedConfig.from_pretrained(policy_source)
    policy_config.pretrained_path = policy_source
    policy_config.input_features = None
    policy_config.output_features = None
    policy_config.device = args.device
    policy_config.n_action_steps = args.n_action_steps

    policy = make_policy(policy_config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        dataset_stats=metadata.stats,
    )
    policy.eval()
    return policy, preprocessor, postprocessor, policy_source


def to_policy_observation(observation):
    import torch

    policy_observation = {
        "observation.state": torch.from_numpy(
            observation["observation.state"]
        ),
        "task": observation["task"],
    }
    for camera in ("front", "wrist"):
        key = f"observation.images.{camera}"
        image = np.ascontiguousarray(
            observation[key].transpose(2, 0, 1)
        )
        policy_observation[key] = torch.from_numpy(image).float().div(255.0)
    return policy_observation


def create_recorders(env, run_dir, enabled):
    if not enabled:
        return [], None, None

    front_path = run_dir / "front.mp4"
    wrist_path = run_dir / "wrist.mp4"
    recorders = [
        MuJoCoVideoRecorder(env, front_path, camera_name="front"),
        MuJoCoVideoRecorder(env, wrist_path, camera_name="wrist"),
    ]
    return recorders, front_path, wrist_path


def run_episode(
    args,
    policy,
    preprocessor,
    postprocessor,
    policy_source,
    run_dir,
):
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = CubeTrayEnv()
    env.MAX_EPISODE_SECONDS = args.max_episode_seconds
    run_dir.mkdir(parents=True, exist_ok=True)
    recorders, front_path, wrist_path = create_recorders(
        env,
        run_dir,
        args.video,
    )
    event_monitor = None


    def on_simulation_step():
        for recorder in recorders:
            recorder.on_simulation_step()
        if event_monitor is not None:
            event_monitor.observe()

    actions = []
    action_latencies_ms = []
    inference_latencies_ms = []
    clipped_action_steps = 0
    info = None
    started_at = time.perf_counter()

    try:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        observation = env.reset_policy_episode(
            seed=args.seed,
            simulation_callback=on_simulation_step,
        )
        initial_cube_position = env.get_cube_position()
        event_monitor = EpisodeEventMonitor(
            env,
            initial_cube_z=float(initial_cube_position[2]),
        )

        terminated = False
        truncated = False
        while not (terminated or truncated):
            starts_action_chunk = len(actions) % args.n_action_steps == 0
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(args.device))
            action_started_at = time.perf_counter()
            batch = preprocessor(to_policy_observation(observation))
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action_numpy = action.detach().cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(args.device))
            action_latency_ms = (time.perf_counter() - action_started_at) * 1000.0
            action_latencies_ms.append(action_latency_ms)
            if starts_action_chunk:
                inference_latencies_ms.append(action_latency_ms)
            if action_numpy.shape != (1, env.ACTION_DIM):
                raise RuntimeError(
                    "SmolVLA returned action shape "
                    f"{action_numpy.shape}, expected (1, {env.ACTION_DIM})."
                )

            action_numpy = action_numpy[0].astype(np.float32, copy=False)
            actions.append(action_numpy.copy())
            observation, _, terminated, truncated, info = env.step(
                action_numpy,
                simulation_callback=on_simulation_step,
            )
            clipped_action_steps += int(info["action_clipped"])

            if getattr(args, "verbose", True) and (
                info["control_steps"] == 1
                or info["control_steps"] % 25 == 0
                or terminated
                or truncated
            ):
                cube = info["cube_position"]
                print(
                    f"step={info['control_steps']:03d} "
                    f"gripper={info['applied_action'][7]:7.2f} "
                    f"cube=({cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}) "
                    f"stable={info['stable_duration']:.3f}s"
                )
    finally:
        wall_seconds = time.perf_counter() - started_at
        for recorder in recorders:
            recorder.close()
        env.close()

    if info is None or not actions:
        raise RuntimeError("The policy episode produced no control steps.")

    action_array = np.stack(actions)
    success = bool(info["is_success"])
    control_period = env.SIM_STEPS_PER_CONTROL * env.model.opt.timestep
    result = EvaluationResult(
        run_name=args.run_name,
        policy_path=str(policy_source),
        seed=args.seed,
        success=success,
        termination_reason="success" if success else "timeout",
        control_steps=info["control_steps"],
        action_chunks=int(
            np.ceil(info["control_steps"] / args.n_action_steps)
        ),
        simulated_seconds=info["control_steps"] * control_period,
        wall_seconds=wall_seconds,
        action_latencies_ms=action_latencies_ms,
        inference_latencies_ms=inference_latencies_ms,
        clipped_action_steps=clipped_action_steps,
        stable_duration=float(info["stable_duration"]),
        cube_lifted=event_monitor.lifted,
        cube_dropped=event_monitor.dropped,
        max_cube_height=event_monitor.max_cube_z,
        collision=event_monitor.collision,
        collision_events=event_monitor.collision_events,
        collision_sim_steps=event_monitor.collision_sim_steps,
        initial_cube_position=initial_cube_position.tolist(),
        final_cube_position=info["cube_position"].tolist(),
        tray_position=info["tray_position"].tolist(),
        tray_xy_error=float(info["tray_xy_error"]),
        action_min=action_array.min(axis=0).tolist(),
        action_max=action_array.max(axis=0).tolist(),
        front_video=str(front_path) if front_path is not None else None,
        wrist_video=str(wrist_path) if wrist_path is not None else None,
    )
    return result


def print_result(result, metrics_path):
    print("\n=== SmolVLA episode result ===")
    print(f"run_name:            {result.run_name}")
    print(f"seed:                {result.seed}")
    print(f"success:             {result.success}")
    print(f"termination_reason:  {result.termination_reason}")
    print(f"control_steps:       {result.control_steps}")
    print(f"action_chunks:       {result.action_chunks}")
    print(f"simulated_seconds:   {result.simulated_seconds:.3f}")
    print(f"wall_seconds:        {result.wall_seconds:.3f}")
    print(
        "inference_latency:   "
        f"mean={np.mean(result.inference_latencies_ms):.2f}ms "
        f"p95={np.percentile(result.inference_latencies_ms, 95):.2f}ms"
    )
    print(f"clipped_action_steps:{result.clipped_action_steps:>8d}")
    print(f"stable_duration:     {result.stable_duration:.3f}")
    print(f"tray_xy_error:       {result.tray_xy_error:.6f}")
    print(f"cube_dropped:        {result.cube_dropped}")
    print(f"collision:           {result.collision}")
    print(f"metrics:             {metrics_path}")
    if result.front_video is not None:
        print(f"front_video:         {result.front_video}")
        print(f"wrist_video:         {result.wrist_video}")
    print(f"\nRESULT: {'SUCCESS' if result.success else 'TASK FAILURE'}")


def main():
    args = parse_args()
    if args.n_action_steps <= 0 or args.n_action_steps > 50:
        raise ValueError("--n-action-steps must be in [1, 50].")
    if args.max_episode_seconds <= 0:
        raise ValueError("--max-episode-seconds must be positive.")
    if not (args.dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {args.dataset_root}."
        )

    run_dir = args.output_dir / f"{args.run_name}_seed_{args.seed}"
    policy, preprocessor, postprocessor, policy_source = load_policy_stack(args)
    result = run_episode(
        args,
        policy,
        preprocessor,
        postprocessor,
        policy_source,
        run_dir,
    )

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    print_result(result, metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
