from pathlib import Path

import mujoco
import numpy as np


class CubeTrayEnv:
    """
    Minimal MuJoCo environment for the SmolVLA cube-to-tray task.

    Observation:
        observation.images.front : uint8 RGB image, (H, W, 3)
        observation.state        : float32 robot state, (8,)
        task                     : language instruction

    Current implementation:
        - Load Panda + cube + tray scene
        - Reset Panda
        - Randomize cube position
        - Render front RGB image
        - Return robot state

    Not implemented yet:
        - step(action)
        - Cartesian controller / IK
        - wrist camera
        - success detection
    """

    TASK_INSTRUCTION = "Pick up the red cube and place it in the tray."

    def __init__(
        self,
        image_width=256,
        image_height=256,
        cube_x_range=(0.35, 0.50),
        cube_y_range=(-0.15, 0.15),
    ):
        # ------------------------------------------------------------
        # Paths
        # ------------------------------------------------------------

        project_root = Path(__file__).resolve().parents[1]

        self.scene_path = (
            project_root
            / "envs"
            / "assets"
            / "franka_emika_panda"
            / "task_scene.xml"
        )

        if not self.scene_path.exists():
            raise FileNotFoundError(
                f"MuJoCo scene not found:\n{self.scene_path}"
            )

        # ------------------------------------------------------------
        # Load MuJoCo
        # ------------------------------------------------------------

        self.model = mujoco.MjModel.from_xml_path(
            str(self.scene_path)
        )

        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------
        # Renderer
        # ------------------------------------------------------------

        self.image_width = image_width
        self.image_height = image_height

        self.renderer = mujoco.Renderer(
            self.model,
            height=image_height,
            width=image_width,
        )

        # ------------------------------------------------------------
        # Cube randomization range
        # ------------------------------------------------------------

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range

        # ------------------------------------------------------------
        # Find important MuJoCo objects
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Panda joint IDs
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Finger joints
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Initial Panda configuration
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # RNG
        # ------------------------------------------------------------

        self.rng = np.random.default_rng()

        # Do one initial reset
        self.reset()

    # ================================================================
    # Reset
    # ================================================================

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

        # ------------------------------------------------------------
        # Reset Panda arm
        # ------------------------------------------------------------

        self.data.qpos[
            self.arm_qpos_indices
        ] = self.home_qpos

        # ------------------------------------------------------------
        # Open gripper
        # ------------------------------------------------------------

        # Panda fingers usually have ~0.04 m travel each.
        self.data.qpos[
            self.finger_qpos_indices
        ] = 0.04

        # ------------------------------------------------------------
        # Randomize cube position
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Zero velocity
        # ------------------------------------------------------------

        self.data.qvel[:] = 0.0

        # Recompute kinematics
        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.get_observation()

    # ================================================================
    # Observation
    # ================================================================

    def get_observation(self):
        """
        Construct the observation used later by SmolVLA.
        """

        front_image = self.render_front_camera()

        state = self.get_robot_state()

        obs = {
            "observation.images.front": front_image,
            "observation.state": state,
            "task": self.TASK_INSTRUCTION,
        }

        return obs

    # ================================================================
    # Robot state
    # ================================================================

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

    # ================================================================
    # Rendering
    # ================================================================

    def render_front_camera(self):
        """
        Render RGB observation from the front camera.

        Returns:
            uint8 array of shape (H, W, 3)
        """

        self.renderer.update_scene(
            self.data,
            camera="front_camera",
        )

        image = self.renderer.render()

        return image.copy()

    # ================================================================
    # Utility
    # ================================================================

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

    # ================================================================
    # Close
    # ================================================================

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