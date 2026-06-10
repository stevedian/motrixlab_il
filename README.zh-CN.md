**语言**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.12-blue)

MotrixLab 是基于 [MotrixSim](https://github.com/Motphys/motrixsim-docs) 仿真引擎的强化学习框架，专为机器人仿真与训练设计，提供统一的 RL、IL 训练与评估接口。

## 项目概览

项目包含三个 workspace 成员，共享一个 Python 3.12 虚拟环境：

-   **motrix_envs**：基于 MotrixSim 的 RL 仿真环境，定义观测、动作和奖励，与框架无关
-   **motrix_rl**：RL 框架集成，支持 SKRL（JAX/PyTorch）和 RSLRL（PyTorch）PPO 算法
-   **motrix_il**：模仿学习子项目，基于 lerobot，集成 MotrixSim 后端，持续增加 benchmark 支持

## 支持的算法

### RL

| 算法 | 框架   | 后端          |
|------|--------|---------------|
| PPO  | SKRL   | JAX / PyTorch |
| PPO  | RSLRL  | PyTorch       |

### IL（LeRobot 策略）

| 策略            | 类型               | 说明                                |
|-----------------|--------------------|-------------------------------------|
| ACT             | `act`              | Action Chunking Transformer         |
| Diffusion       | `diffusion`        | Diffusion Policy                    |
| VQ-BeT          | `vqbet`            | Vector Quantized Behavior Transformer |
| TD-MPC          | `tdmpc`            | Task-oriented Model Predictive Control |
| Pi0 / Pi05      | `pi0` / `pi05`     | π₀ 系列                             |
| SmolVLA         | `smolvla`          | 轻量视觉-语言-动作模型              |
| Wall-X          | `wall_x`           | Wall-X 策略                         |
| Multi-Task DiT  | `multi_task_dit`   | 多任务 Diffusion Transformer        |

## 快速开始

> 前置依赖：[UV](https://docs.astral.sh/uv/) 包管理工具。

```bash
git clone https://github.com/Motphys/MotrixLab
cd MotrixLab
git lfs pull
```

### 安装

**RL 和 IL 所有环境：**

```bash
uv sync --all-packages --all-extras
```

**RL 基础环境：**

```bash
uv sync --all-packages
```

**RL 训练框架：**

```bash
uv sync --all-packages --extra skrl-jax      # SKRL + JAX（仅 Linux）
uv sync --all-packages --extra skrl-torch    # SKRL + PyTorch
uv sync --all-packages --extra rslrl         # RSLRL + PyTorch
```

**IL 环境：**

```bash
uv sync --all-packages --extra training                    # 基础训练管线
uv sync --all-packages --extra training --extra simulation # + 仿真环境 (PushT)
```

| extra        | 内容                                       |
|--------------|--------------------------------------------|
| `training`   | 训练管线（wandb、accelerate）              |
| `simulation` | 经典仿真环境（ALOHA、PushT、LIBERO、MuJoCo）|
| `aloha`      | ALOHA 仿真 + dm-control + MuJoCo           |
| `libero`     | LIBERO 仿真                                |
| `diffusion`  | Diffusion Policy                           |
| `pi`         | Pi0 策略                                   |
| `smolvla`    | SmolVLA 策略                               |

### RL 训练

```bash
uv run scripts/train.py --env cartpole               # SKRL（默认）
uv run scripts/train.py --env cartpole --rllib rslrl  # RSLRL
```

训练结果通过 TensorBoard 查看：

```bash
uv run tensorboard --logdir runs/{env-name}
```

### RL 环境可视化与推理

```bash
uv run scripts/view.py --env cartpole   # 仅查看环境
uv run scripts/play.py --env cartpole   # 策略推理
```

## lerobot 训练

### PushT

PushT 支持基于状态、基于图像、基于图像 + 状态模式：

| 模式          | `--env.obs_type`              | `--dataset.repo_id`        | 说明                  |
|---------------|-------------------------------|----------------------------|-----------------------|
| 纯图像        | `pixels`                      | `lerobot/pusht`            | 仅图像输入            |
| 图像 + 位姿   | `pixels_agent_pos`            | `lerobot/pusht`            | 图像 + 末端位姿（默认）|
| 关键点        | `environment_state_agent_pos` | `lerobot/pusht_keypoints`  | 纯状态/关键点         |

# 基于图像（默认 pixels_agent_pos）

```bash
uv run lerobot-train \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --dataset.video_backend=pyav \
  --env.type=pusht \
  --env.obs_type=pixels_agent_pos \
  --batch_size=4 \
  --steps=200000 \
  --save_freq=20000 \
  --eval_freq=0 \
  --output_dir=outputs/train/diffusion_pusht_image_state \
  --wandb.enable=true \
  --wandb.project=diffusion_pusht_image_state
```

更多 benchmark 持续接入中。

## 已注册的 MotrixSim 环境

| env.type            | 后端       | 说明                    |
|---------------------|------------|-------------------------|
| `pusht`             | MotrixSim  | 2D pusht任务           |
| `aloha_motrixsim`   | MotrixSim  | ALOHA 双臂操作任务      |

## lerobot 评估

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/diffusion_pusht_image_state/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=pusht \
  --env.obs_type=pixels_agent_pos \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/diffusion_pusht_image_state
```

`--env.discover_packages_path=motrix_il.lerobot` 会触发 `plugin.py` 中的 `register_envs()`，注册 `aloha_motrixsim` 和 `libero_motrixsim` 两个环境类型。

### 使用原生 MuJoCo 环境

原生的 `aloha`、`libero` 等环境类型不受影响，需要无头渲染时设置环境变量：

```bash
MUJOCO_GL=egl uv run lerobot-eval --env.type=aloha ...
```

> 子进程遇到 `NamespaceNotFound` 时，加 `--eval.use_async_envs=false` 使用同步模式。

## 联系方式

-   GitHub Issues：[提交 Issue](https://github.com/Motphys/MotrixLab/issues)
-   Discussions：[加入讨论](https://github.com/Motphys/MotrixLab/discussions)
