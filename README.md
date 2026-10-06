# SmolVLA Cube-to-Tray

## 项目目标

在 MuJoCo 中采集 Panda 脚本专家轨迹，将轨迹编码为 LeRobotDataset，
微调 SmolVLA，并对比基础模型与微调模型的闭环任务成功率。

任务文本：

```text
Pick up the red cube and place it in the tray.
```

## 项目结构

```text
src/smolvla_task/
├── controllers/              # Panda IK 和脚本专家
├── envs/                     # MuJoCo 环境及模型资产
└── utils/                    # 数据集验证和视频录制
scripts/
├── collect_data.py           # 采集并验证 LeRobotDataset
├── evaluate.py               # 标准/微调 SmolVLA 闭环评估
└── test_scripted_pick.py     # 脚本专家 smoke
models/                       # 最终推理模型，本地 artifact
datasets/                     # 训练数据集，本地 artifact
outputs/                      # 训练、评估和视频输出
```

`models/`、`datasets/` 和 `outputs/` 不进入 Git。

## 数据接口

| LeRobot feature | 类型和形状 | 内容 |
| --- | --- | --- |
| `observation.images.front` | video, `(256, 256, 3)` | 固定前置 RGB |
| `observation.images.wrist` | video, `(256, 256, 3)` | 腕部 RGB |
| `observation.state` | float32, `(8,)` | 7 个关节位置和单侧夹指位置（米） |
| `action` | float32, `(8,)` | 7 个关节目标和 `[0, 255]` 夹爪命令 |
| `episode_seed` | int64, `(1,)` | 确定性 action replay 使用的环境 seed |

控制和采样频率均为 25 Hz，每个控制周期执行 20 个 MuJoCo step。
episode 只有在方块完整位于托盘内并稳定保持 2 秒后才会写入数据集；
失败 episode 会清空缓存并按 `--max-retries` 重试。

## 环境安装

所有命令在 `lerobot` conda 环境中执行：

```bash
conda activate lerobot
```

本项目不再保存 LeRobot 源码。训练时使用的版本为 LeRobot `0.6.2`，
对应上游 commit：

```text
2595896f8a5c70f06adc1bcdf446d3aaa4cc3f20
```

可安装预先构建的 wheel，或者从确切 commit 普通安装：

```bash
python -m pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@2595896f8a5c70f06adc1bcdf446d3aaa4cc3f20"
```

安装当前工程的开发包：

```bash
python -m pip install --no-deps -e .
```

## 验证脚本专家

```bash
python scripts/test_scripted_pick.py \
  --seed 42 \
  --headless \
  --video false
```

- `--seed`：环境随机种子。
- `--headless`：不启动 MuJoCo Viewer。
- `--video`：是否录制前置相机视频。

## 采集数据

```bash
python scripts/collect_data.py \
  --num-episodes 100 \
  --start-seed 0 \
  --repo-id local/smolvla_cube_tray \
  --root datasets/smolvla_cube_tray
```

- `--num-episodes`：数据集最终应包含的 episode 总数。
- `--start-seed`：初始随机种子。
- `--repo-id`：LeRobot 数据集逻辑标识。
- `--root`：数据集本地路径。
- `--resume`：从已有数据集继续补齐。

## 微调 SmolVLA

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=4000 \
  --policy.scheduler_decay_steps=120000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id=local/smolvla_cube_tray \
  --dataset.root=datasets/smolvla_cube_tray \
  --dataset.eval_split=0.2 \
  --batch_size=1 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=4 \
  --num_workers=2 \
  --steps=80000 \
  --eval_steps=2000 \
  --max_eval_samples=512 \
  --env_eval_freq=0 \
  --log_freq=40 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --output_dir=outputs/train/smolvla_cube_tray \
  --job_name=smolvla_cube_tray \
  --wandb.enable=false
```

最终推理 artifact 位于：

```text
models/smolvla_cube_tray_finetuned/
```

原始训练 checkpoint 仍保存在：

```text
outputs/train/smolvla_cube_tray/
```

## 评估微调模型

```bash
python scripts/evaluate.py \
  --policy-path models/smolvla_cube_tray_finetuned \
  --dataset-repo-id local/smolvla_cube_tray \
  --dataset-root datasets/smolvla_cube_tray \
  --seed 100 \
  --n-action-steps 10 \
  --max-episode-seconds 30 \
  --run-name smolvla_finetuned \
  --video true \
  --offline
```

JSON 指标和视频写入：

```text
outputs/evaluation/
```
