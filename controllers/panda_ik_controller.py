import mujoco
import numpy as np


class PandaIKController:
    """Cartesian IK controller for the Franka Emika Panda."""

    def __init__(
        self,
        model,
        data,
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
    ):
        self.model = model
        self.data = data

        self.damping = damping
        self.step_size = step_size
        self.max_joint_step = max_joint_step

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.orientation_weight = orientation_weight

        self.gravity_compensation = gravity_compensation
        self.gravity_compensation_scale = gravity_compensation_scale
        self.max_gravity_offset = max_gravity_offset

        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            ee_site_name,
        )

        if self.ee_site_id == -1:
            raise RuntimeError(
                f"Could not find end-effector site '{ee_site_name}'."
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

        self.arm_joint_ids = []
        self.arm_qpos_indices = []
        self.arm_dof_indices = []

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

            self.arm_joint_ids.append(joint_id)

            self.arm_qpos_indices.append(
                self.model.jnt_qposadr[joint_id]
            )

            self.arm_dof_indices.append(
                self.model.jnt_dofadr[joint_id]
            )

        self.arm_joint_ids = np.asarray(
            self.arm_joint_ids,
            dtype=np.int32,
        )

        self.arm_qpos_indices = np.asarray(
            self.arm_qpos_indices,
            dtype=np.int32,
        )

        self.arm_dof_indices = np.asarray(
            self.arm_dof_indices,
            dtype=np.int32,
        )

        self.arm_actuator_names = [
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
        ]

        self.arm_actuator_ids = []

        for name in self.arm_actuator_names:

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )

            if actuator_id == -1:
                raise RuntimeError(
                    f"Could not find Panda actuator '{name}'."
                )

            self.arm_actuator_ids.append(
                actuator_id
            )

        self.arm_actuator_ids = np.asarray(
            self.arm_actuator_ids,
            dtype=np.int32,
        )

        # Gains from the Panda position-servo actuators.
        self.arm_kp = np.array(
            [
                4500.0,
                4500.0,
                3500.0,
                3500.0,
                2000.0,
                2000.0,
                2000.0,
            ],
            dtype=np.float64,
        )

        self.jacp = np.zeros(
            (3, self.model.nv),
            dtype=np.float64,
        )

        self.jacr = np.zeros(
            (3, self.model.nv),
            dtype=np.float64,
        )

    def get_ee_position(self):
        """Return EE Cartesian position."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.data.site_xpos[
            self.ee_site_id
        ].copy()

    def get_ee_rotation_matrix(self):
        """Return EE 3x3 rotation matrix."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.data.site_xmat[
            self.ee_site_id
        ].reshape(3, 3).copy()

    def get_ee_pose(self):
        """Return the end-effector position and rotation matrix."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        position = self.data.site_xpos[
            self.ee_site_id
        ].copy()

        rotation = self.data.site_xmat[
            self.ee_site_id
        ].reshape(3, 3).copy()

        return position, rotation

    def _compute_site_jacobian(self):
        """Compute translational and rotational site Jacobians."""

        self.jacp.fill(0.0)
        self.jacr.fill(0.0)

        mujoco.mj_jacSite(
            self.model,
            self.data,
            self.jacp,
            self.jacr,
            self.ee_site_id,
        )

    def get_position_jacobian(self):
        """Return the 3x7 translational Jacobian."""

        self._compute_site_jacobian()

        J = self.jacp[
            :,
            self.arm_dof_indices,
        ]

        return J.copy()

    def get_pose_jacobian(self):
        """Return the 6x7 geometric pose Jacobian."""

        self._compute_site_jacobian()

        J_position = self.jacp[
            :,
            self.arm_dof_indices,
        ]

        J_rotation = self.jacr[
            :,
            self.arm_dof_indices,
        ]

        J = np.vstack(
            (
                J_position,
                J_rotation,
            )
        )

        return J.copy()

    def compute_orientation_error(
        self,
        current_rotation,
        target_rotation,
    ):
        """Return the world-frame orientation error as a rotation vector."""

        current_rotation = np.asarray(
            current_rotation,
            dtype=np.float64,
        ).reshape(3, 3)

        target_rotation = np.asarray(
            target_rotation,
            dtype=np.float64,
        ).reshape(3, 3)

        R_error = (
            target_rotation
            @ current_rotation.T
        )

        error_quat = np.zeros(
            4,
            dtype=np.float64,
        )

        mujoco.mju_mat2Quat(
            error_quat,
            np.ascontiguousarray(
                R_error.reshape(-1)
            ),
        )

        # Select the quaternion sign for the shortest rotation.
        if error_quat[0] < 0.0:
            error_quat *= -1.0

        orientation_error = np.zeros(
            3,
            dtype=np.float64,
        )

        mujoco.mju_quat2Vel(
            orientation_error,
            error_quat,
            1.0,
        )

        return orientation_error

    def compute_joint_delta(
        self,
        position_error,
    ):
        """Compute a position-only Damped Least-Squares IK update."""

        position_error = np.asarray(
            position_error,
            dtype=np.float64,
        )

        J = self.get_position_jacobian()

        A = (
            J @ J.T
            + self.damping ** 2
            * np.eye(3)
        )

        dq = (
            J.T
            @ np.linalg.solve(
                A,
                position_error,
            )
        )

        dq *= self.step_size

        dq = np.clip(
            dq,
            -self.max_joint_step,
            self.max_joint_step,
        )

        return dq

    def compute_joint_delta_pose(
        self,
        position_error,
        orientation_error,
        orientation_weight=None,
    ):
        """Compute a 6D Damped Least-Squares IK update."""

        if orientation_weight is None:
            orientation_weight = (
                self.orientation_weight
            )

        position_error = np.asarray(
            position_error,
            dtype=np.float64,
        )

        orientation_error = np.asarray(
            orientation_error,
            dtype=np.float64,
        )

        J = self.get_pose_jacobian()

        J_weighted = J.copy()

        J_weighted[
            3:,
            :
        ] *= orientation_weight

        error = np.concatenate(
            (
                position_error,
                orientation_weight
                * orientation_error,
            )
        )

        A = (
            J_weighted
            @ J_weighted.T
            + self.damping ** 2
            * np.eye(6)
        )

        dq = (
            J_weighted.T
            @ np.linalg.solve(
                A,
                error,
            )
        )

        dq *= self.step_size

        dq = np.clip(
            dq,
            -self.max_joint_step,
            self.max_joint_step,
        )

        return dq

    def get_gravity_compensation_offset(self):
        """Convert MuJoCo bias forces into a position-servo offset."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        tau_bias = self.data.qfrc_bias[
            self.arm_dof_indices
        ].copy()

        dq_comp = (
            tau_bias
            / self.arm_kp
        )

        dq_comp *= (
            self.gravity_compensation_scale
        )

        dq_comp = np.clip(
            dq_comp,
            -self.max_gravity_offset,
            self.max_gravity_offset,
        )

        return dq_comp

    def clip_joint_limits(
        self,
        q,
    ):
        """Clamp joint command to Panda joint limits."""

        q_clipped = np.asarray(
            q,
            dtype=np.float64,
        ).copy()

        for i, joint_id in enumerate(
            self.arm_joint_ids
        ):

            if self.model.jnt_limited[
                joint_id
            ]:

                q_min = self.model.jnt_range[
                    joint_id,
                    0,
                ]

                q_max = self.model.jnt_range[
                    joint_id,
                    1,
                ]

                q_clipped[i] = np.clip(
                    q_clipped[i],
                    q_min,
                    q_max,
                )

        return q_clipped

    def set_joint_target(
        self,
        q_target,
    ):
        """Send joint-position command to Panda arm actuators."""

        q_target = np.asarray(
            q_target,
            dtype=np.float64,
        )

        self.data.ctrl[
            self.arm_actuator_ids
        ] = q_target

    def _build_joint_command(
        self,
        dq_ik,
    ):
        """Build and limit the final joint-position command."""

        current_q = self.data.qpos[
            self.arm_qpos_indices
        ].copy()

        q_command = (
            current_q
            + dq_ik
        )

        if self.gravity_compensation:

            dq_comp = (
                self.get_gravity_compensation_offset()
            )

            q_command += dq_comp

        q_command = self.clip_joint_limits(
            q_command
        )

        return q_command

    def step_towards(
        self,
        target_pos,
    ):
        """Perform one position-only IK update and return the error norm."""

        target_pos = np.asarray(
            target_pos,
            dtype=np.float64,
        )

        current_pos = (
            self.get_ee_position()
        )

        position_error = (
            target_pos
            - current_pos
        )

        error_norm = np.linalg.norm(
            position_error
        )

        if (
            error_norm
            < self.position_tolerance
        ):
            return error_norm

        dq_ik = self.compute_joint_delta(
            position_error
        )

        q_command = (
            self._build_joint_command(
                dq_ik
            )
        )

        self.set_joint_target(
            q_command
        )

        return error_norm

    def step_towards_pose(
        self,
        target_pos,
        target_rotation,
        orientation_tolerance=None,
        orientation_weight=None,
    ):
        """Perform one 6D pose IK update and return both error norms."""

        if orientation_tolerance is None:
            orientation_tolerance = (
                self.orientation_tolerance
            )

        if orientation_weight is None:
            orientation_weight = (
                self.orientation_weight
            )

        target_pos = np.asarray(
            target_pos,
            dtype=np.float64,
        )

        target_rotation = np.asarray(
            target_rotation,
            dtype=np.float64,
        ).reshape(3, 3)

        (
            current_pos,
            current_rotation,
        ) = self.get_ee_pose()

        position_error = (
            target_pos
            - current_pos
        )

        position_error_norm = (
            np.linalg.norm(
                position_error
            )
        )

        orientation_error = (
            self.compute_orientation_error(
                current_rotation,
                target_rotation,
            )
        )

        orientation_error_norm = (
            np.linalg.norm(
                orientation_error
            )
        )

        if (
            position_error_norm
            < self.position_tolerance
            and orientation_error_norm
            < orientation_tolerance
        ):

            return (
                position_error_norm,
                orientation_error_norm,
            )

        dq_ik = (
            self.compute_joint_delta_pose(
                position_error,
                orientation_error,
                orientation_weight,
            )
        )

        q_command = (
            self._build_joint_command(
                dq_ik
            )
        )

        self.set_joint_target(
            q_command
        )

        return (
            position_error_norm,
            orientation_error_norm,
        )

    def move_to(
        self,
        target_pos,
        max_steps=500,
        sim_steps_per_control=20,
        verbose=False,
    ):
        """Move the end effector to a Cartesian position."""

        target_pos = np.asarray(
            target_pos,
            dtype=np.float64,
        )

        for step in range(max_steps):

            error = self.step_towards(
                target_pos
            )

            if verbose and step % 20 == 0:

                current = (
                    self.get_ee_position()
                )

                print(
                    f"step={step:4d} | "
                    f"current={current} | "
                    f"error={error:.6f}"
                )

            if (
                error
                < self.position_tolerance
            ):

                if verbose:
                    print(
                        "Target reached! "
                        f"error={error:.6f}"
                    )

                return True

            for _ in range(
                sim_steps_per_control
            ):

                mujoco.mj_step(
                    self.model,
                    self.data,
                )

        if verbose:

            current = (
                self.get_ee_position()
            )

            final_error = np.linalg.norm(
                target_pos
                - current
            )

            print(
                "Target NOT reached. "
                f"Final error="
                f"{final_error:.6f}"
            )

        return False

    def move_to_pose(
        self,
        target_pos,
        target_rotation,
        max_steps=500,
        sim_steps_per_control=20,
        orientation_tolerance=None,
        orientation_weight=None,
        verbose=False,
    ):
        """Move the end effector to a full Cartesian pose."""

        if orientation_tolerance is None:
            orientation_tolerance = (
                self.orientation_tolerance
            )

        if orientation_weight is None:
            orientation_weight = (
                self.orientation_weight
            )

        target_pos = np.asarray(
            target_pos,
            dtype=np.float64,
        )

        target_rotation = np.asarray(
            target_rotation,
            dtype=np.float64,
        ).reshape(3, 3)

        for step in range(max_steps):

            (
                pos_error,
                rot_error,
            ) = self.step_towards_pose(
                target_pos,
                target_rotation,
                orientation_tolerance,
                orientation_weight,
            )

            if (
                verbose
                and step % 20 == 0
            ):

                current_pos, _ = (
                    self.get_ee_pose()
                )

                print(
                    f"step={step:4d} | "
                    f"position={current_pos} | "
                    f"pos_error="
                    f"{pos_error:.6f} m | "
                    f"rot_error="
                    f"{rot_error:.6f} rad"
                )

            if (
                pos_error
                < self.position_tolerance
                and rot_error
                < orientation_tolerance
            ):

                if verbose:

                    print(
                        "Target pose reached! "
                        f"pos_error="
                        f"{pos_error:.6f} m | "
                        f"rot_error="
                        f"{rot_error:.6f} rad"
                    )

                return True

            for _ in range(
                sim_steps_per_control
            ):

                mujoco.mj_step(
                    self.model,
                    self.data,
                )

        if verbose:

            (
                current_pos,
                current_rotation,
            ) = self.get_ee_pose()

            final_pos_error = (
                np.linalg.norm(
                    target_pos
                    - current_pos
                )
            )

            final_rot_vector = (
                self.compute_orientation_error(
                    current_rotation,
                    target_rotation,
                )
            )

            final_rot_error = (
                np.linalg.norm(
                    final_rot_vector
                )
            )

            print(
                "Target pose NOT reached. "
                f"pos_error="
                f"{final_pos_error:.6f} m | "
                f"rot_error="
                f"{final_rot_error:.6f} rad"
            )

        return False

    def get_diagnostics(self):
        """Return controller diagnostics."""

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        current_q = self.data.qpos[
            self.arm_qpos_indices
        ].copy()

        command_q = self.data.ctrl[
            self.arm_actuator_ids
        ].copy()

        tracking_error = (
            command_q
            - current_q
        )

        bias_force = self.data.qfrc_bias[
            self.arm_dof_indices
        ].copy()

        compensation_offset = (
            bias_force
            / self.arm_kp
        )

        compensation_offset *= (
            self.gravity_compensation_scale
        )

        compensation_offset = np.clip(
            compensation_offset,
            -self.max_gravity_offset,
            self.max_gravity_offset,
        )

        return {
            "current_q": current_q,
            "command_q": command_q,
            "tracking_error": tracking_error,
            "bias_force": bias_force,
            "compensation_offset": compensation_offset,
        }

    def set_gripper(self, value):
        """Set the gripper command, where 0 is closed and 255 is open."""

        value = float(
            np.clip(value, 0.0, 255.0)
        )

        gripper_actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator8",
        )

        if gripper_actuator_id == -1:
            raise RuntimeError(
                "Could not find gripper actuator 'actuator8'."
            )

        self.data.ctrl[
            gripper_actuator_id
        ] = value


    def open_gripper(self):
        """Fully open the Panda gripper."""

        self.set_gripper(255.0)


    def close_gripper(self):
        """Fully close the Panda gripper."""

        self.set_gripper(0.0)
