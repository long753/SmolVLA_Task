from collections.abc import Callable
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from controllers.panda_ik_controller import PandaIKController
from envs.cube_tray_env import CubeTrayEnv


ControlCallback = Callable[[np.ndarray, str], None]
SimulationCallback = Callable[[], None]


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    seed: int
    failure_reason: str | None
    control_steps: int
    settled_cube_position: np.ndarray
    final_cube_position: np.ndarray
    tray_position: np.ndarray
    tray_xy_error: float
    stable_duration: float


class ScriptedPickPlaceExpert:
    """Ground-truth scripted expert for the Panda cube-to-tray task."""

    SIM_STEPS_PER_CONTROL = 20
    RESET_SETTLE_STEPS = 250
    CLOSE_CONTROL_STEPS = 15
    POST_LIFT_CONTROL_STEPS = 5
    RELEASE_CONTROL_STEPS = 8
    MAX_POSE_CONTROL_STEPS = 500

    PRE_GRASP_HEIGHT = 0.12
    LIFT_HEIGHT = 0.18
    PRE_PLACE_HEIGHT = 0.12
    PLACE_CLEARANCE = 0.005
    MIN_SUCCESSFUL_LIFT = 0.10

    SUCCESS_HOLD_SECONDS = 2.0
    SUCCESS_TIMEOUT_SECONDS = 4.0

    def __init__(
        self,
        env: CubeTrayEnv,
        controller: PandaIKController,
    ):
        self.env = env
        self.controller = controller
        self.phase = "idle"
        self.control_steps = 0

        self.gripper_actuator_id = mujoco.mj_name2id(
            self.env.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator8",
        )
        if self.gripper_actuator_id == -1:
            raise RuntimeError("Could not find gripper actuator 'actuator8'.")

    def get_action(self) -> np.ndarray:
        """Return the actuator command currently sent to the Panda."""
        action = np.empty(8, dtype=np.float32)
        action[:7] = self.env.data.ctrl[
            self.controller.arm_actuator_ids
        ]
        action[7] = self.env.data.ctrl[self.gripper_actuator_id]
        return action

    def _emit_control(
        self,
        phase: str,
        control_callback: ControlCallback | None,
    ) -> None:
        self.phase = phase
        self.control_steps += 1
        if control_callback is not None:
            control_callback(self.get_action(), phase)

    def _step_simulation(
        self,
        steps: int,
        viewer=None,
        simulation_callback: SimulationCallback | None = None,
        realtime: bool = False,
    ) -> None:
        remaining_steps = steps

        while remaining_steps > 0:
            batch_steps = min(
                self.SIM_STEPS_PER_CONTROL,
                remaining_steps,
            )

            for _ in range(batch_steps):
                mujoco.mj_step(self.env.model, self.env.data)
                if simulation_callback is not None:
                    simulation_callback()

            if viewer is not None:
                viewer.sync()
                if realtime:
                    time.sleep(batch_steps * self.env.model.opt.timestep)

            remaining_steps -= batch_steps

    def _hold_control(
        self,
        phase: str,
        control_ticks: int,
        control_callback: ControlCallback | None,
        viewer,
        simulation_callback: SimulationCallback | None,
        realtime: bool,
    ) -> None:
        for _ in range(control_ticks):
            self._emit_control(phase, control_callback)
            self._step_simulation(
                self.SIM_STEPS_PER_CONTROL,
                viewer,
                simulation_callback,
                realtime,
            )

    def _move_to_pose(
        self,
        phase: str,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        control_callback: ControlCallback | None,
        viewer,
        simulation_callback: SimulationCallback | None,
        realtime: bool,
    ) -> bool:
        for _ in range(self.MAX_POSE_CONTROL_STEPS):
            position_error, orientation_error = (
                self.controller.step_towards_pose(
                    target_position,
                    target_rotation,
                )
            )

            if (
                position_error < self.controller.position_tolerance
                and orientation_error < self.controller.orientation_tolerance
            ):
                return True

            self._emit_control(phase, control_callback)
            self._step_simulation(
                self.SIM_STEPS_PER_CONTROL,
                viewer,
                simulation_callback,
                realtime,
            )

        return False

    def _wait_for_stable_placement(
        self,
        viewer,
        simulation_callback: SimulationCallback | None,
        realtime: bool,
    ) -> tuple[bool, float]:
        timestep = self.env.model.opt.timestep
        required_stable_steps = int(
            np.ceil(self.SUCCESS_HOLD_SECONDS / timestep)
        )
        timeout_steps = int(
            np.ceil(self.SUCCESS_TIMEOUT_SECONDS / timestep)
        )
        stable_steps = 0

        self.phase = "verify-place"

        for simulation_step in range(timeout_steps):
            mujoco.mj_step(self.env.model, self.env.data)
            if simulation_callback is not None:
                simulation_callback()

            if self.env.is_cube_stably_in_tray():
                stable_steps += 1
            else:
                stable_steps = 0

            if (
                viewer is not None
                and (simulation_step + 1) % self.SIM_STEPS_PER_CONTROL == 0
            ):
                viewer.sync()
                if realtime:
                    time.sleep(self.SIM_STEPS_PER_CONTROL * timestep)

            if stable_steps >= required_stable_steps:
                return True, stable_steps * timestep

        return False, stable_steps * timestep

    def _result(
        self,
        seed: int,
        success: bool,
        failure_reason: str | None,
        settled_cube_position: np.ndarray,
        tray_position: np.ndarray,
        stable_duration: float = 0.0,
    ) -> EpisodeResult:
        final_cube_position = self.env.get_cube_position()
        tray_xy_error = float(
            np.linalg.norm(
                final_cube_position[:2] - tray_position[:2]
            )
        )
        self.phase = "done" if success else "failed"

        return EpisodeResult(
            success=success,
            seed=seed,
            failure_reason=failure_reason,
            control_steps=self.control_steps,
            settled_cube_position=settled_cube_position.copy(),
            final_cube_position=final_cube_position,
            tray_position=tray_position.copy(),
            tray_xy_error=tray_xy_error,
            stable_duration=stable_duration,
        )

    def run_episode(
        self,
        seed: int,
        control_callback: ControlCallback | None = None,
        viewer=None,
        simulation_callback: SimulationCallback | None = None,
        realtime: bool = False,
    ) -> EpisodeResult:
        """Run one episode and optionally expose every recorded control tick."""
        self.phase = "reset"
        self.control_steps = 0
        self.env.reset(seed=seed)

        reset_arm_position = self.env.data.qpos[
            self.env.arm_qpos_indices
        ].copy()
        self.controller.set_joint_target(reset_arm_position)
        self.controller.open_gripper()
        self._step_simulation(
            self.RESET_SETTLE_STEPS,
            viewer,
            simulation_callback,
            realtime,
        )

        settled_cube_position = self.env.get_cube_position()
        tray_position = self.env.get_tray_position()
        _, grasp_rotation = self.controller.get_ee_pose()

        pre_grasp_position = settled_cube_position + np.array(
            [0.0, 0.0, self.PRE_GRASP_HEIGHT],
            dtype=np.float64,
        )
        grasp_position = settled_cube_position.copy()
        lift_position = settled_cube_position + np.array(
            [0.0, 0.0, self.LIFT_HEIGHT],
            dtype=np.float64,
        )

        if not self._move_to_pose(
            "pre-grasp",
            pre_grasp_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "pre-grasp pose was not reached",
                settled_cube_position,
                tray_position,
            )

        if not self._move_to_pose(
            "descend",
            grasp_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "grasp pose was not reached",
                settled_cube_position,
                tray_position,
            )

        self.controller.close_gripper()
        self._hold_control(
            "close",
            self.CLOSE_CONTROL_STEPS,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        )

        if not self._move_to_pose(
            "lift",
            lift_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "lift pose was not reached",
                settled_cube_position,
                tray_position,
            )

        self._hold_control(
            "post-lift",
            self.POST_LIFT_CONTROL_STEPS,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        )

        cube_position_after_lift = self.env.get_cube_position()
        ee_position_after_lift, _ = self.controller.get_ee_pose()
        lifted_height = (
            cube_position_after_lift[2]
            - settled_cube_position[2]
        )
        if lifted_height < self.MIN_SUCCESSFUL_LIFT:
            return self._result(
                seed,
                False,
                "cube was not lifted",
                settled_cube_position,
                tray_position,
            )

        carried_cube_offset = (
            cube_position_after_lift
            - ee_position_after_lift
        )
        tray_half_size = self.env.model.geom_size[
            self.env.tray_geom_id
        ]
        cube_half_size = self.env.model.geom_size[
            self.env.cube_geom_id
        ]

        desired_cube_position = tray_position.copy()
        desired_cube_position[2] += (
            tray_half_size[2]
            + cube_half_size[2]
            + self.PLACE_CLEARANCE
        )
        place_position = (
            desired_cube_position
            - carried_cube_offset
        )
        pre_place_position = place_position + np.array(
            [0.0, 0.0, self.PRE_PLACE_HEIGHT],
            dtype=np.float64,
        )

        if not self._move_to_pose(
            "pre-place",
            pre_place_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "pre-place pose was not reached",
                settled_cube_position,
                tray_position,
            )

        if not self._move_to_pose(
            "place",
            place_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "place pose was not reached",
                settled_cube_position,
                tray_position,
            )

        self.controller.open_gripper()
        self._hold_control(
            "release",
            self.RELEASE_CONTROL_STEPS,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        )

        if not self._move_to_pose(
            "retreat",
            pre_place_position,
            grasp_rotation,
            control_callback,
            viewer,
            simulation_callback,
            realtime,
        ):
            return self._result(
                seed,
                False,
                "retreat pose was not reached",
                settled_cube_position,
                tray_position,
            )

        success, stable_duration = self._wait_for_stable_placement(
            viewer,
            simulation_callback,
            realtime,
        )
        failure_reason = None if success else "cube did not remain stable in the tray"

        return self._result(
            seed,
            success,
            failure_reason,
            settled_cube_position,
            tray_position,
            stable_duration,
        )
