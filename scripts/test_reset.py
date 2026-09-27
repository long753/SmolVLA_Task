from pathlib import Path
import sys

import imageio.v3 as iio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.cube_tray_env import CubeTrayEnv


env = CubeTrayEnv()

output_dir = PROJECT_ROOT / "outputs" / "reset_test"
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


for seed in range(10):

    obs = env.reset(seed=seed)

    image = obs["observation.images.front"]
    state = obs["observation.state"]

    cube_position = env.get_cube_position()

    print(
        f"seed={seed:2d} | "
        f"cube={cube_position} | "
        f"image={image.shape} | "
        f"state={state.shape}"
    )

    iio.imwrite(
        output_dir / f"seed_{seed:02d}.png",
        image,
    )


env.close()

print(
    f"\nImages saved to:\n{output_dir}"
)