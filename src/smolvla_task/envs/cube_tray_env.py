from pathlib import Path

import mujoco
import numpy as np


class CubeTrayEnv:
    """
    Minimal MuJoCo environment for the SmolVLA cube-to-tray task.

    Observation:
        observation.images.front : uint8 RGB image, (H, W, 3)
        observation.images.wrist : uint8 RGB image, (H, W, 3)
        observation.state        : float32 robot state, (8,)
        task                     : language instruction

    Current implementation:
        - Load and reset the Panda, cube, and tray scene
        - Randomize the cube position from a reproducible seed
        - Render synchronized front and wrist RGB observations
        - Execute the dataset's 8D joint/gripper action at 25 Hz
        - Terminate after a stable two-second placement or a 30-second timeout
    """

    TASK_INSTRUCTION = "Pick up the red cube and place it in the tray."

    ACTION_DIM = 8
    SIM_STEPS_PER_CONTROL = 20
    RESET_SETTLE_STEPS = 250
    MAX_EPISODE_SECONDS = 30.0
    SUCCESS_HOLD_SECONDS = 2.0

    def __init__(
        self,
        image_width=256,
        image_height=256,
        cube_x_range=(0.35, 0.50),
        cube_y_range=(-0.15, 0.15),
    ):

        package_dir = Path(__file__).resolve().parent
        self.scene_path = (
            package_dir
            / "assets"
            / "franka_emika_panda"
            / "task_scene.xml"
        )

        if not self.scene_path.exists():
            raise FileNotFoundError(
                f"MuJoCo scene not found:\n{self.scene_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.scene_path)
        )

        self.data = mujoco.MjData(self.model)

        self.image_width = image_width
        self.image_height = image_height

        self.renderer = mujoco.Renderer(
            self.model,
            height=image_height,
            width=image_width,
        )

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range

        self.cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_freejoint",
        )

        if self.cube_joint_id == -1:
            raise RuntimeError(
                "Could not find joint 'cube_freejoint'. "
                "Check task_scene.xml."
            )

        # qpos address of the cube free joint
        self.cube_qpos_adr = self.model.jnt_qposadr[
            self.cube_joint_id
        ]

        # qvel address of the cube free joint
        self.cube_qvel_adr = self.model.jnt_dofadr[
            self.cube_joint_id
        ]

        self.cube_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "red_cube_geom",
        )

        if self.cube_geom_id == -1:
            raise RuntimeError(
                "Could not find geom 'red_cube_geom'. "
                "Check task_scene.xml."
            )

        self.tray_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "tray_bottom",
        )

        if self.tray_geom_id == -1:
            raise RuntimeError(
                "Could not find geom 'tray_bottom'. "
                "Check task_scene.xml."
            )

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.arm_qpos_indices = []

        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id == -1:
                raise RuntimeError(
                    f"Could not find Panda joint '{name}'."
                )

            qpos_adr = self.model.jnt_qposadr[joint_id]

            self.arm_qpos_indices.append(qpos_adr)

        self.arm_qpos_indices = np.asarray(
            self.arm_qpos_indices,
            dtype=np.int32,
        )

        self.finger_joint_names = [
            "finger_joint1",
            "finger_joint2",
        ]

        self.finger_qpos_indices = []

        for name in self.finger_joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id == -1:
                raise RuntimeError(
                    f"Could not find finger joint '{name}'."
                )

            qpos_adr = self.model.jnt_qposadr[joint_id]

            self.finger_qpos_indices.append(qpos_adr)

        self.finger_qpos_indices = np.asarray(
            self.finger_qpos_indices,
            dtype=np.int32,
        )

        self.arm_actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"actuator{index}",
                )
                for index in range(1, 8)
            ],
            dtype=np.int32,
        )
        if np.any(self.arm_actuator_ids == -1):
            raise RuntimeError("Could not find all Panda arm actuators.")

        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator8",
        )
        if self.gripper_actuator_id == -1:
            raise RuntimeError("Could not find Panda gripper actuator 'actuator8'.")

        # A reasonable Panda home pose.
        # We can tune this later according to your table geometry.
        self.home_qpos = np.array(
            [
                0.0,
                -0.4,
                0.0,
                -2.2,
                0.0,
                1.8,
                0.8,
            ],
            dtype=np.float64,
        )

        self.rng = np.random.default_rng()
        self.policy_control_steps = 0
        self.policy_stable_steps = 0

        # Do one initial reset
        self.reset()


    def reset(self, seed=None):
        """
        Reset the simulation.

        Args:
            seed:
                If specified, cube randomization is reproducible.

        Returns:
            observation dictionary
        """

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset all MuJoCo state
        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # Reset Panda arm
        self.data.qpos[
            self.arm_qpos_indices
        ] = self.home_qpos


        # Panda fingers usually have ~0.04 m travel each.
        self.data.qpos[
            self.finger_qpos_indices
        ] = 0.04

        # Randomize cube position
        cube_x = self.rng.uniform(
            self.cube_x_range[0],
            self.cube_x_range[1],
        )

        cube_y = self.rng.uniform(
            self.cube_y_range[0],
            self.cube_y_range[1],
        )

        # Free joint qpos layout:
        #
        # [x, y, z, qw, qx, qy, qz]
        #
        # Keep z and orientation from the XML initial configuration.

        adr = self.cube_qpos_adr

        self.data.qpos[adr + 0] = cube_x
        self.data.qpos[adr + 1] = cube_y

        # Keep cube upright
        self.data.qpos[adr + 3 : adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

        self.data.qvel[:] = 0.0

        # Recompute kinematics
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.get_observation()
    

    def get_observation(self):
        """
        Construct the observation used later by SmolVLA.
        """

        front_image = self.render_front_camera()
        wrist_image = self.render_wrist_camera()

        state = self.get_robot_state()

        obs = {
            "observation.images.front": front_image,
            "observation.images.wrist": wrist_image,
            "observation.state": state,
            "task": self.TASK_INSTRUCTION,
        }

        return obs


    def get_robot_state(self):
        """
        Robot state:

            [q1, q2, ..., q7, gripper]

        Shape:
            (8,)
        """

        arm_qpos = self.data.qpos[
            self.arm_qpos_indices
        ].copy()

        finger_qpos = self.data.qpos[
            self.finger_qpos_indices
        ]

        # Represent the two finger joints using one scalar.
        gripper = np.mean(finger_qpos)

        state = np.concatenate(
            [
                arm_qpos,
                np.array([gripper]),
            ]
        )

        return state.astype(np.float32)


    def _render_camera(self, camera_name):
        """Render one named RGB camera."""

        self.renderer.update_scene(
            self.data,
            camera=camera_name,
        )

        image = self.renderer.render()

        return image.copy()


    def render_front_camera(self):
        """Render the fixed external front camera."""

        return self._render_camera(
            "front_camera"
        )


    def render_wrist_camera(self):
        """Render the camera attached to the Panda hand."""

        return self._render_camera(
            "wrist_camera"
        )


    def get_cube_position(self):
        """
        Ground-truth cube position.

        IMPORTANT:
        This is allowed for:
            - scripted expert
            - debugging
            - success evaluation

        It should NOT be provided to SmolVLA.
        """

        adr = self.cube_qpos_adr

        return self.data.qpos[
            adr : adr + 3
        ].copy()

    def get_cube_velocity(self):
        """
        Ground-truth cube linear and angular velocity.

        Returns:
            [vx, vy, vz, wx, wy, wz]
        """

        adr = self.cube_qvel_adr

        return self.data.qvel[
            adr : adr + 6
        ].copy()


    def get_tray_position(self):
        """Return the ground-truth tray geom center in world coordinates."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.data.geom_xpos[
            self.tray_geom_id
        ].copy()


    def is_cube_in_tray(
        self,
        position_tolerance=0.01,
    ):
        """Return whether the complete cube is resting inside the tray."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        cube_position = self.data.geom_xpos[
            self.cube_geom_id
        ]

        tray_position = self.data.geom_xpos[
            self.tray_geom_id
        ]

        cube_rotation = self.data.geom_xmat[
            self.cube_geom_id
        ].reshape(3, 3)

        tray_rotation = self.data.geom_xmat[
            self.tray_geom_id
        ].reshape(3, 3)

        relative_position = (
            tray_rotation.T
            @ (
                cube_position
                - tray_position
            )
        )

        relative_rotation = (
            tray_rotation.T
            @ cube_rotation
        )

        cube_half_extent = (
            np.abs(
                relative_rotation
            )
            @ self.model.geom_size[
                self.cube_geom_id
            ]
        )

        tray_half_size = self.model.geom_size[
            self.tray_geom_id
        ]

        inside_xy = np.all(
            np.abs(
                relative_position[:2]
            )
            + cube_half_extent[:2]
            <= tray_half_size[:2]
        )

        cube_bottom = (
            relative_position[2]
            - cube_half_extent[2]
        )

        on_tray_surface = (
            abs(
                cube_bottom
                - tray_half_size[2]
            )
            <= position_tolerance
        )

        return bool(
            inside_xy
            and on_tray_surface
        )


    def is_cube_stably_in_tray(
        self,
        position_tolerance=0.01,
        linear_velocity_tolerance=0.01,
        angular_velocity_tolerance=0.1,
    ):
        """Return whether the cube is inside the tray and nearly stationary."""

        if not self.is_cube_in_tray(
            position_tolerance=position_tolerance,
        ):
            return False

        cube_velocity = (
            self.get_cube_velocity()
        )

        linear_speed = np.linalg.norm(
            cube_velocity[:3]
        )

        angular_speed = np.linalg.norm(
            cube_velocity[3:]
        )

        return bool(
            linear_speed
            < linear_velocity_tolerance
            and angular_speed
            < angular_velocity_tolerance
        )


    def reset_policy_episode(
        self,
        seed=None,
        simulation_callback=None,
    ):
        """Reset and settle the simulation for a policy-controlled episode."""
        self.reset(seed=seed)
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[
            self.arm_qpos_indices
        ]
        self.data.ctrl[self.gripper_actuator_id] = 255.0

        for _ in range(self.RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
            if simulation_callback is not None:
                simulation_callback()

        self.policy_control_steps = 0
        self.policy_stable_steps = 0
        return self.get_observation()

    def step(self, action, simulation_callback=None):
        """Apply one 25 Hz dataset-format action and advance the simulation."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"Expected action shape ({self.ACTION_DIM},), got {action.shape}."
            )
        if not np.isfinite(action).all():
            raise ValueError("Policy action contains a non-finite value.")

        applied_action = action.copy()
        limited = self.model.actuator_ctrllimited[
            self.arm_actuator_ids
        ].astype(bool)
        arm_ranges = self.model.actuator_ctrlrange[self.arm_actuator_ids]
        applied_action[:7][limited] = np.clip(
            applied_action[:7][limited],
            arm_ranges[limited, 0],
            arm_ranges[limited, 1],
        )
        applied_action[7] = np.clip(applied_action[7], 0.0, 255.0)
        action_clipped = not np.array_equal(applied_action, action)

        self.data.ctrl[self.arm_actuator_ids] = applied_action[:7]
        self.data.ctrl[self.gripper_actuator_id] = applied_action[7]

        required_stable_steps = int(
            np.ceil(self.SUCCESS_HOLD_SECONDS / self.model.opt.timestep)
        )
        terminated = False
        for _ in range(self.SIM_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
            if simulation_callback is not None:
                simulation_callback()

            if self.is_cube_stably_in_tray():
                self.policy_stable_steps += 1
            else:
                self.policy_stable_steps = 0

            if self.policy_stable_steps >= required_stable_steps:
                terminated = True
                break

        self.policy_control_steps += 1
        control_period = (
            self.SIM_STEPS_PER_CONTROL * self.model.opt.timestep
        )
        max_control_steps = int(
            np.ceil(self.MAX_EPISODE_SECONDS / control_period)
        )
        truncated = (
            not terminated
            and self.policy_control_steps >= max_control_steps
        )

        cube_position = self.get_cube_position()
        tray_position = self.get_tray_position()
        info = {
            "is_success": terminated,
            "action_clipped": action_clipped,
            "applied_action": applied_action.astype(np.float32),
            "control_steps": self.policy_control_steps,
            "stable_duration": (
                self.policy_stable_steps * self.model.opt.timestep
            ),
            "cube_position": cube_position,
            "tray_position": tray_position,
            "tray_xy_error": float(
                np.linalg.norm(cube_position[:2] - tray_position[:2])
            ),
        }
        reward = 1.0 if terminated else 0.0
        return (
            self.get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.renderer.close()


if __name__ == "__main__":

    env = CubeTrayEnv()

    obs = env.reset(seed=42)

    print("\n=== Observation ===")

    for key, value in obs.items():

        if isinstance(value, np.ndarray):
            print(
                f"{key:30s}",
                value.shape,
                value.dtype,
            )
        else:
            print(
                f"{key:30s}",
                value,
            )

    print(
        "\nCube position:",
        env.get_cube_position(),
    )

    env.close()