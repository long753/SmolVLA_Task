from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from envs.cube_tray_env import CubeTrayEnv
from controllers.panda_ik_controller import PandaIKController


env = CubeTrayEnv()

env.reset(
    seed=42
)


# ==============================================================
# Controller
# ==============================================================

controller = PandaIKController(
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


# ==============================================================
# Initial EE pose
# ==============================================================

(
    initial_pos,
    initial_rotation,
) = controller.get_ee_pose()


print(
    "\nInitial EE position:"
)

print(
    initial_pos
)


print(
    "\nInitial EE rotation:"
)

print(
    initial_rotation
)


# ==============================================================
# Target pose
# ==============================================================
#
# Move upward by 5 cm while maintaining the initial orientation.
#

target_pos = (
    initial_pos
    + np.array(
        [
            0.0,
            0.0,
            0.1,
        ],
        dtype=np.float64,
    )
)

target_rotation = (
    initial_rotation.copy()
)


print(
    "\nTarget EE position:"
)

print(
    target_pos
)


print(
    "\nTarget EE rotation:"
)

print(
    target_rotation
)


# ==============================================================
# Viewer
# ==============================================================

with mujoco.viewer.launch_passive(
    env.model,
    env.data,
) as viewer:

    success = False

    # ----------------------------------------------------------
    # Main control loop
    # ----------------------------------------------------------

    for step in range(500):

        (
            pos_error,
            rot_error,
        ) = controller.step_towards_pose(
            target_pos,
            target_rotation,
        )

        # ------------------------------------------------------
        # Let low-level Panda position servos track command
        # ------------------------------------------------------

        for _ in range(20):

            mujoco.mj_step(
                env.model,
                env.data,
            )

        # ------------------------------------------------------
        # Update viewer
        # ------------------------------------------------------

        viewer.sync()

        # ------------------------------------------------------
        # Print diagnostics
        # ------------------------------------------------------

        if step % 20 == 0:

            (
                current_pos,
                _,
            ) = controller.get_ee_pose()

            print(
                f"step={step:4d} | "
                f"current={current_pos} | "
                f"pos_error="
                f"{pos_error:.6f} m | "
                f"rot_error="
                f"{rot_error:.6f} rad"
            )

        # ------------------------------------------------------
        # Success condition
        # ------------------------------------------------------

        if (
            pos_error
            < controller.position_tolerance
            and rot_error
            < controller.orientation_tolerance
        ):

            print(
                "\nTarget pose reached!"
            )

            success = True

            break

        time.sleep(
            0.01
        )

    # ==========================================================
    # Final diagnostics
    # ==========================================================

    (
        final_pos,
        final_rotation,
    ) = controller.get_ee_pose()


    final_position_error = (
        np.linalg.norm(
            target_pos
            - final_pos
        )
    )


    final_orientation_vector = (
        controller.compute_orientation_error(
            final_rotation,
            target_rotation,
        )
    )


    final_orientation_error = (
        np.linalg.norm(
            final_orientation_vector
        )
    )


    print(
        "\n=============================="
    )

    print(
        "Final result"
    )

    print(
        "=============================="
    )


    print(
        "\nSuccess:",
        success,
    )


    print(
        "\nFinal EE position:"
    )

    print(
        final_pos
    )


    print(
        "\nFinal EE rotation:"
    )

    print(
        final_rotation
    )


    print(
        "\nFinal position error:",
        final_position_error,
        "m",
    )


    print(
        "Final orientation error:",
        final_orientation_error,
        "rad",
    )


    print(
        "Final orientation error:",
        np.degrees(
            final_orientation_error
        ),
        "deg",
    )


    # ==========================================================
    # Controller diagnostics
    # ==========================================================

    diagnostics = (
        controller.get_diagnostics()
    )


    print(
        "\n=============================="
    )

    print(
        "Controller diagnostics"
    )

    print(
        "=============================="
    )


    print(
        "\nActual q:"
    )

    print(
        diagnostics[
            "current_q"
        ]
    )


    print(
        "\nCommand q:"
    )

    print(
        diagnostics[
            "command_q"
        ]
    )


    print(
        "\nTracking error:"
    )

    print(
        diagnostics[
            "tracking_error"
        ]
    )


    print(
        "\nBias force:"
    )

    print(
        diagnostics[
            "bias_force"
        ]
    )


    print(
        "\nCompensation offset:"
    )

    print(
        diagnostics[
            "compensation_offset"
        ]
    )


    input(
        "\nPress Enter to close viewer..."
    )