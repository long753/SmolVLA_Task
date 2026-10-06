"""Simulation-step event metrics for policy evaluation."""

import mujoco

from smolvla_task.envs.cube_tray_env import CubeTrayEnv


class EpisodeEventMonitor:
    """Detect cube drops and new robot-environment contacts after reset."""

    LIFT_HEIGHT_METERS = 0.05

    def __init__(self, env: CubeTrayEnv, initial_cube_z: float):
        self.env = env
        self.initial_cube_z = float(initial_cube_z)
        self.max_cube_z = self.initial_cube_z
        self.lifted = False
        self.dropped = False
        self.collision = False
        self.collision_events = 0
        self.collision_sim_steps = 0
        self._collision_active = False

        named_geoms = {
            name: mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in ("floor", "table_top", "red_cube_geom", "tray_bottom")
        }
        missing = [name for name, geom_id in named_geoms.items() if geom_id == -1]
        if missing:
            raise RuntimeError(f"Missing evaluation geoms: {', '.join(missing)}")

        self.cube_geom_id = named_geoms["red_cube_geom"]
        self.support_geom_ids = {
            named_geoms["floor"],
            named_geoms["table_top"],
        }
        self.obstacle_geom_ids = self.support_geom_ids | {
            named_geoms["tray_bottom"]
        }
        task_geom_ids = self.obstacle_geom_ids | {self.cube_geom_id}
        self.robot_geom_ids = set(range(env.model.ngeom)) - task_geom_ids
        self.baseline_collision_pairs = set()
        for contact_index in range(env.data.ncon):
            contact = env.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._is_robot_obstacle_contact(geom1, geom2):
                self.baseline_collision_pairs.add(frozenset((geom1, geom2)))

    def _is_robot_obstacle_contact(self, geom1: int, geom2: int) -> bool:
        return (
            geom1 in self.robot_geom_ids and geom2 in self.obstacle_geom_ids
        ) or (
            geom2 in self.robot_geom_ids and geom1 in self.obstacle_geom_ids
        )

    def observe(self) -> None:
        """Inspect contacts and cube height after one MuJoCo simulation step."""
        cube_z = float(self.env.get_cube_position()[2])
        self.max_cube_z = max(self.max_cube_z, cube_z)
        if cube_z >= self.initial_cube_z + self.LIFT_HEIGHT_METERS:
            self.lifted = True

        collision_this_step = False
        for contact_index in range(self.env.data.ncon):
            contact = self.env.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if self.lifted and (
                (geom1 == self.cube_geom_id and geom2 in self.support_geom_ids)
                or (geom2 == self.cube_geom_id and geom1 in self.support_geom_ids)
            ):
                self.dropped = True

            collision_pair = frozenset((geom1, geom2))
            if (
                self._is_robot_obstacle_contact(geom1, geom2)
                and collision_pair not in self.baseline_collision_pairs
            ):
                collision_this_step = True

        if collision_this_step:
            self.collision = True
            self.collision_sim_steps += 1
            if not self._collision_active:
                self.collision_events += 1
        self._collision_active = collision_this_step
