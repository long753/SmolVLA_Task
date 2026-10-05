## Introduction

### Task Goal
基于 LeRobot 使用 smolVLA 抓取红色方块，并放入托盘。具体而言，在 MuJoCo 环境中采集 expert rollout，随后使用将 expert rollout 编码为 LeRobot Dataset，利用 LeRobot 来微调 smolVLA，随后对比原始 smolVLA 和微调后的 smolVLA 的任务成功率。

### Observation Space & Action Space

环境同时提供固定前置相机和随夹爪运动的腕部相机：

| LeRobot feature | 类型和形状 | 内容 |
| --- | --- | --- |
| `observation.images.front` | video, `(256, 256, 3)` | 固定前置 RGB |
| `observation.images.wrist` | video, `(256, 256, 3)` | 腕部 RGB |
| `observation.state` | float32, `(8,)` | 7 个关节位置和单侧夹指位置（米） |
| `action` | float32, `(8,)` | 7 个关节目标和范围为 `[0, 255]` 的夹爪命令 |
| `episode_seed` | int64, `(1,)` | 确定性 action replay 使用的环境 seed |

控制和采样频率均为 25 Hz。每个控制周期执行 20 个 MuJoCo step。任务文本为
`pick up the red cube and place it in the blue tray`。

脚本专家实现在 `controllers/scripted_expert.py`，环境、采集脚本和测试脚本共用
同一状态机。episode 只有在方块完整位于托盘内并稳定保持 2 秒后才会写入数据集；
失败 episode 会清空缓存并按 `--max-retries` 重试。

### Scripts Cdoe
- `collect_data.py:` 
- `evaluate.py:` 
- `test_scripted_pick.py:` 

## Start
**1. Create conda enviornment:**

```bash
conda activate lerobot
```

**2. Test environment：**

```bash
python scripts/test_scripted_pick.py 
  --headless false
  --seed 42 
  --video false
```

- `seed:` random seed num
- `headless:` use MuJoCo Viewer or not
- `video:` use video or not


**3. Collect LeRobotDataset：**

```bash
conda run -n lerobot python scripts/collect_data.py \
  --num-episodes 100 \
  --start-seed 0 \
  --repo-id local/smolvla_cube_tray \
  --root datasets/smolvla_cube_tray \
  --resume false
```

- `--num-episodes：` 表示数据集最终应包含的 episode 总数
- `start-seed：` 表示初始随机种子
- `repo-id：` 表示数据集标签
- `root：` 表示数据集存放位置
- `resume：` 表示
