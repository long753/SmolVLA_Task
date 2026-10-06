from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from smolvla_task.envs.cube_tray_env import CubeTrayEnv


class MuJoCoVideoRecorder:
    """Stream one MuJoCo camera to an H.264 MP4 at simulation-time FPS."""

    def __init__(
        self,
        env: CubeTrayEnv,
        output_path: str | Path,
        camera_name: str = "front",
        fps: int = 25,
    ):
        self.env = env
        self.output_path = Path(output_path)
        self.camera_name = camera_name
        self.fps = fps
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        steps_per_frame = 1.0 / (fps * env.model.opt.timestep)
        self.sim_steps_per_frame = int(round(steps_per_frame))
        if not np.isclose(
            steps_per_frame,
            self.sim_steps_per_frame,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Video FPS must map to an integer number of MuJoCo steps."
            )

        self.steps_since_frame = 0
        self.frame_count = 0
        self.writer = imageio.get_writer(
            str(self.output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

    def _render(self) -> np.ndarray:
        if self.camera_name == "front":
            return self.env.render_front_camera()
        if self.camera_name == "wrist":
            return self.env.render_wrist_camera()
        raise ValueError(f"Unknown camera: {self.camera_name}")

    def capture_frame(self) -> None:
        self.writer.append_data(self._render())
        self.frame_count += 1

    def on_simulation_step(self) -> None:
        self.steps_since_frame += 1
        if self.steps_since_frame >= self.sim_steps_per_frame:
            self.capture_frame()
            self.steps_since_frame -= self.sim_steps_per_frame

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
