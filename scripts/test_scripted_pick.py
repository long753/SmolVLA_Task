import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controllers.panda_ik_controller import PandaIKController
from envs.cube_tray_env import CubeTrayEnv


SIM_STEPS_PER_CONTROL = 20
RESET_SETTLE_STEPS = 250
GRIPPER_CLOSE_STEPS = 300
POST_LIFT_STEPS = 100
GRIPPER_RELEASE_STEPS = 150
MAX_CONTROL_STEPS = 500

PRE_GRASP_HEIGHT = 0.12
LIFT_HEIGHT = 0.18
PRE_PLACE_HEIGHT = 0.12
PLACE_CLEARANCE = 0.005
MIN_SUCCESSFUL_LIFT = 0.10
SUCCESS_HOLD_SECONDS = 2.0
SUCCESS_TIMEOUT_SECONDS = 4.0


def step_simulation(env, steps, viewer=None):
    """Advance physics and periodically update an optional passive viewer."""
    remaining_steps = steps

    while remaining_steps > 0:
        batch_steps = min(SIM_STEPS_PER_CONTROL, remaining_steps)

        for _ in range(batch_steps):
            mujoco.mj_step(env.model, env.data)

        if viewer is not None:
            viewer.sync()
            time.sleep(batch_steps * env.model.opt.timestep)

        remaining_steps -= batch_steps


def move_to_pose(
    env,
    controller,
    target_position,
    target_rotation,
    phase_name,
    viewer=None,
):
    """Drive one Cartesian phase and return whether its pose was reached."""
    print(f"\n[{phase_name}] target position: {target_position}")

    for control_step in range(MAX_CONTROL_STEPS):
        position_error, orientation_error = controller.step_towards_pose(
            target_position,
            target_rotation,
        )

        if (
            position_error < controller.position_tolerance
            and orientation_error < controller.orientation_tolerance
        ):
            if viewer is not None:
                viewer.sync()

            print(
                f"[{phase_name}] reached in {control_step} control steps "
                f"(position error={position_error:.6f} m, "
                f"orientation error={orientation_error:.6f} rad)"
            )
            return True

        step_simulation(
            env,
            SIM_STEPS_PER_CONTROL,
            viewer,
        )

    current_position, current_rotation = controller.get_ee_pose()
    final_position_error = np.linalg.norm(
        target_position - current_position
    )
    final_orientation_error = np.linalg.norm(
        controller.compute_orientation_error(
            current_rotation,
            target_rotation,
        )
    )

    print(
        f"[{phase_name}] FAILED after {MAX_CONTROL_STEPS} control steps "
        f"(position error={final_position_error:.6f} m, "
        f"orientation error={final_orientation_error:.6f} rad)"
    )
    return False

def wait_for_stable_placement(env, viewer=None):
    """Require the released cube to remain stable in the tray for two seconds."""
    timestep = env.model.opt.timestep
    required_stable_steps = int(np.ceil(SUCCESS_HOLD_SECONDS / timestep))
    timeout_steps = int(np.ceil(SUCCESS_TIMEOUT_SECONDS / timestep))
    stable_steps = 0

    print(
        f"\n[verify-place] requiring {SUCCESS_HOLD_SECONDS:.1f} s "
        "of continuous stability"
    )

    for simulation_step in range(timeout_steps):
        mujoco.mj_step(env.model, env.data)

        if env.is_cube_stably_in_tray():
            stable_steps += 1
        else:
            stable_steps = 0

        if (
            viewer is not None
            and (simulation_step + 1) % SIM_STEPS_PER_CONTROL == 0
        ):
            viewer.sync()
            time.sleep(SIM_STEPS_PER_CONTROL * timestep)

        if stable_steps >= required_stable_steps:
            if viewer is not None:
                viewer.sync()
                time.sleep(SIM_STEPS_PER_CONTROL * timestep)

            stable_duration = stable_steps * timestep
            print(
                f"[verify-place] cube remained stable for "
                f"{stable_duration:.3f} s"
            )
            return True, stable_duration

    stable_duration = stable_steps * timestep
    print(
        f"[verify-place] FAILED after {SUCCESS_TIMEOUT_SECONDS:.1f} s "
        f"(last continuous stable duration={stable_duration:.3f} s)"
    )
    return False, stable_duration


def run_pick_and_place_sequence(env, controller, seed, viewer=None):
    """Execute a ground-truth scripted cube pick-and-place."""
    env.reset(seed=seed)

    # reset() writes qpos but MuJoCo control targets are still zero. Hold the
    # reset arm pose and command the gripper open before advancing physics.
    reset_arm_position = env.data.qpos[env.arm_qpos_indices].copy()
    controller.set_joint_target(reset_arm_position)
    controller.open_gripper()

    print(f"Seed: {seed}")
    print("\n[open] gripper command: 255")
    step_simulation(env, RESET_SETTLE_STEPS, viewer)

    # The cube starts slightly above the table in the XML. Read ground truth
    # only after it has settled so every waypoint uses its physical position.
    settled_cube_position = env.get_cube_position()
    _, grasp_rotation = controller.get_ee_pose()

    pre_grasp_position = settled_cube_position + np.array(
        [0.0, 0.0, PRE_GRASP_HEIGHT],
        dtype=np.float64,
    )
    grasp_position = settled_cube_position.copy()
    lift_position = settled_cube_position + np.array(
        [0.0, 0.0, LIFT_HEIGHT],
        dtype=np.float64,
    )

    print(f"[open] settled cube position: {settled_cube_position}")

    if not move_to_pose(
        env,
        controller,
        pre_grasp_position,
        grasp_rotation,
        "pre-grasp",
        viewer,
    ):
        print("\nRESULT: FAIL (pre-grasp pose was not reached)")
        return False

    if not move_to_pose(
        env,
        controller,
        grasp_position,
        grasp_rotation,
        "descend",
        viewer,
    ):
        print("\nRESULT: FAIL (grasp pose was not reached)")
        return False

    controller.close_gripper()
    print("\n[close] gripper command: 0")
    step_simulation(env, GRIPPER_CLOSE_STEPS, viewer)
    cube_position_after_close = env.get_cube_position()
    print(f"[close] cube position: {cube_position_after_close}")

    if not move_to_pose(
        env,
        controller,
        lift_position,
        grasp_rotation,
        "lift",
        viewer,
    ):
        print("\nRESULT: FAIL (lift pose was not reached)")
        return False

    step_simulation(env, POST_LIFT_STEPS, viewer)

    cube_position_after_lift = env.get_cube_position()
    ee_position_after_lift, _ = controller.get_ee_pose()
    lifted_height = (
        cube_position_after_lift[2]
        - settled_cube_position[2]
    )

    print(
        f"[verify-grasp] cube lifted {lifted_height:.6f} m "
        f"(required={MIN_SUCCESSFUL_LIFT:.6f} m)"
    )

    if lifted_height < MIN_SUCCESSFUL_LIFT:
        print("\nRESULT: FAIL (cube was not lifted)")
        return False

    # Account for the actual offset between the grasped cube and ee_site so
    # that the cube, rather than the end effector, is centered on the tray.
    carried_cube_offset = (
        cube_position_after_lift
        - ee_position_after_lift
    )

    tray_position = env.get_tray_position()
    tray_half_size = env.model.geom_size[env.tray_geom_id]
    cube_half_size = env.model.geom_size[env.cube_geom_id]

    desired_cube_position = tray_position.copy()
    desired_cube_position[2] += (
        tray_half_size[2]
        + cube_half_size[2]
        + PLACE_CLEARANCE
    )

    place_position = (
        desired_cube_position
        - carried_cube_offset
    )
    pre_place_position = place_position + np.array(
        [0.0, 0.0, PRE_PLACE_HEIGHT],
        dtype=np.float64,
    )

    print(f"[transport] tray position: {tray_position}")
    print(f"[transport] carried cube offset: {carried_cube_offset}")
    print(f"[transport] desired cube position: {desired_cube_position}")

    if not move_to_pose(
        env,
        controller,
        pre_place_position,
        grasp_rotation,
        "pre-place",
        viewer,
    ):
        print("\nRESULT: FAIL (pre-place pose was not reached)")
        return False

    if not move_to_pose(
        env,
        controller,
        place_position,
        grasp_rotation,
        "place",
        viewer,
    ):
        print("\nRESULT: FAIL (place pose was not reached)")
        return False

    controller.open_gripper()
    print("\n[release] gripper command: 255")
    step_simulation(env, GRIPPER_RELEASE_STEPS, viewer)

    if not move_to_pose(
        env,
        controller,
        pre_place_position,
        grasp_rotation,
        "retreat",
        viewer,
    ):
        print("\nRESULT: FAIL (retreat pose was not reached)")
        return False

    success, stable_duration = wait_for_stable_placement(
        env,
        viewer,
    )

    final_cube_position = env.get_cube_position()
    final_ee_position, _ = controller.get_ee_pose()
    finger_positions = env.data.qpos[env.finger_qpos_indices].copy()
    cube_velocity = env.get_cube_velocity()
    cube_inside_tray = env.is_cube_in_tray()
    tray_xy_error = np.linalg.norm(
        final_cube_position[:2]
        - tray_position[:2]
    )

    print("\n=== Final diagnostics ===")
    print(f"Settled cube position: {settled_cube_position}")
    print(f"Desired cube position: {desired_cube_position}")
    print(f"Final cube position:   {final_cube_position}")
    print(f"Final EE position:     {final_ee_position}")
    print(f"Finger positions:      {finger_positions}")
    print(f"Cube linear velocity:  {cube_velocity[:3]}")
    print(f"Cube angular velocity: {cube_velocity[3:]}")
    print(f"Tray XY error:         {tray_xy_error:.6f} m")
    print(f"Cube inside tray:      {cube_inside_tray}")
    print(f"Stable duration:       {stable_duration:.3f} s")
    print(f"\nRESULT: {'PASS' if success else 'FAIL'}")

    return success


def create_controller(env):
    """Create the controller."""
    return PandaIKController(
        env.model,
        env.data,
        ee_site_name="ee_site",
        damping=1e-3,
        step_size=0.5,
        max_joint_step=0.05,
        position_tolerance=0.005,
        orientation_tolerance=0.03,
        orientation_weight=1.0,
        gravity_compensation=True,
        gravity_compensation_scale=1.0,
        max_gravity_offset=0.02,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a ground-truth scripted Panda cube pick-and-place."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Cube reset seed (default: 42).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the passive MuJoCo viewer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = CubeTrayEnv()

    try:
        controller = create_controller(env)

        if args.headless:
            success = run_pick_and_place_sequence(
                env,
                controller,
                args.seed,
            )
        else:
            with mujoco.viewer.launch_passive(
                env.model,
                env.data,
            ) as viewer:
                success = run_pick_and_place_sequence(
                    env,
                    controller,
                    args.seed,
                    viewer,
                )
            # Handle.close() only requests that the viewer thread exits.
            # Wait for it before releasing the shared model and data.
            while viewer.is_running():
                time.sleep(0.01)
    finally:
        env.close()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
