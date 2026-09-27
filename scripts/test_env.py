from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENE_PATH = (
    PROJECT_ROOT
    / "envs"
    / "assets"
    / "franka_emika_panda"
    / "task_scene.xml"
)

print("Loading:", SCENE_PATH)

model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("Model loaded successfully")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

mujoco.viewer.launch(model, data)