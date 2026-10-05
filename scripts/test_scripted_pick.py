import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controllers.panda_ik_controller import PandaIKController
from controllers.scripted_expert import ScriptedPickPlaceExpert
from envs.cube_tray_env import CubeTrayEnv
from utils.mujoco_video_recorder import MuJoCoVideoRecorder


def parse_bool(value):
    normalized_value = value.lower()
    if normalized_value == "true":
        return True
    if normalized_value == "false":
        return False
    raise argparse.ArgumentTypeError("--video must be either true or false")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one scripted Panda cube pick-and-place episode."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--video",
        type=parse_bool,
        default=False,
        metavar="true|false",
    )
    return parser.parse_args()


def print_result(result):
    print("\n=== Episode result ===")
    print(f"Seed:                  {result.seed}")
    print(f"Control steps:         {result.control_steps}")
    print(f"Settled cube position: {result.settled_cube_position}")
    print(f"Final cube position:   {result.final_cube_position}")
    print(f"Tray position:         {result.tray_position}")
    print(f"Tray XY error:         {result.tray_xy_error:.6f} m")
    print(f"Stable duration:       {result.stable_duration:.3f} s")
    if result.failure_reason is not None:
        print(f"Failure reason:        {result.failure_reason}")
    print(f"\nRESULT: {'PASS' if result.success else 'FAIL'}")


def main():
    args = parse_args()
    env = CubeTrayEnv()
    recorder = None

    try:
        controller = PandaIKController(
            env.model,
            env.data,
            ee_site_name="ee_site",
        )
        expert = ScriptedPickPlaceExpert(env, controller)

        if args.video:
            recorder = MuJoCoVideoRecorder(
                env,
                PROJECT_ROOT
                / "outputs"
                / "video"
                / f"pick_place_seed_{args.seed}.mp4",
            )

        simulation_callback = (
            recorder.on_simulation_step
            if recorder is not None
            else None
        )

        if args.headless:
            result = expert.run_episode(
                seed=args.seed,
                simulation_callback=simulation_callback,
            )
        else:
            with mujoco.viewer.launch_passive(
                env.model,
                env.data,
            ) as viewer:
                result = expert.run_episode(
                    seed=args.seed,
                    viewer=viewer,
                    simulation_callback=simulation_callback,
                    realtime=True,
                )

            while viewer.is_running():
                time.sleep(0.01)

        print_result(result)
    finally:
        try:
            if recorder is not None:
                recorder.close()
                print(
                    f"Video saved: {recorder.output_path.relative_to(PROJECT_ROOT)}"
                )
                print(
                    f"Video: {recorder.frame_count} frames, "
                    f"{recorder.fps} FPS, {recorder.duration:.3f} s"
                )
        finally:
            env.close()

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
