# SmolVLA Cube-to-Tray

在 MuJoCo 中采集 Panda 专家轨迹，微调 SmolVLA，并对比微调前后的闭环表现。

```text
Task: Pick up the red cube and place it in the tray.
Success: 方块完整位于托盘内并稳定保持 2 秒。
```

## 目录

```text
src/smolvla_task/   环境、控制器与公共工具
scripts/            数据采集、训练验证与模型对比入口
models/             最终推理模型
datasets/           LeRobotDataset
outputs/            训练和评估结果
```

`models/`、`datasets/` 和 `outputs/` 均不进入 Git。

## 环境配置

```bash
# 创建 conda 环境
conda create -n lerobot python=3.12 -y
conda activate lerobot

python -m pip install --upgrade \
  pip wheel "setuptools>=77,<82"

# 安装项目
python -m pip install -e .
```

## 实验步骤

### 1. 验证脚本专家

```bash
python scripts/test_scripted_pick.py \
  --seed 42 \
  --headless \
  --video false
```

### 2. 采集 100 个 episode

```bash
python scripts/collect_data.py \
  --num-episodes 100 \
  --start-seed 0 \
  --repo-id local/smolvla_cube_tray \
  --root datasets/smolvla_cube_tray
```

已有数据集时添加 `--resume`。

### 3. 微调 SmolVLA

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

训练 checkpoint 位于 `outputs/train/`；最终推理模型位于
`models/smolvla_cube_tray_finetuned/`。

### 4. 单模型评估

```bash
python scripts/evaluate.py \
  --policy-path models/smolvla_cube_tray_finetuned \
  --dataset-repo-id local/smolvla_cube_tray \
  --dataset-root datasets/smolvla_cube_tray \
  --seed 100 \
  --n-action-steps 10 \
  --max-episode-seconds 30 \
  --run-name smolvla_finetuned \
  --offline
```

基础模型使用 `--policy-path lerobot/smolvla_base`。结果写入：

```text
outputs/evaluation/<run_name>_seed_<seed>/
├── metrics.json
├── front.mp4
└── wrist.mp4
```

### 5. 50-seed 对比实验

```bash
python scripts/compare_models.py \
  --base-policy-path lerobot/smolvla_base \
  --finetuned-policy-path models/smolvla_cube_tray_finetuned \
  --dataset-repo-id local/smolvla_cube_tray \
  --dataset-root datasets/smolvla_cube_tray \
  --start-seed 100 \
  --num-seeds 50 \
  --n-action-steps 10 \
  --max-episode-seconds 30 \
  --run-name smolvla_base_vs_finetuned \
  --offline
```

输出：

```text
outputs/evaluation/smolvla_base_vs_finetuned_seeds_100_149/
├── summary.json    # 聚合结果
├── episodes.csv    # 逐 episode 结果
├── base/
└── finetuned/
```

统计指标：成功率、任务耗时、推理延迟、掉落率和碰撞率。对比实验默认不录制视频。
