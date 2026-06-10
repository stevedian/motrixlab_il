**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.12-blue)

MotrixLab is a reinforcement learning framework built on the [MotrixSim](https://github.com/Motphys/motrixsim-docs) simulation engine, designed for robot simulation and training with a unified RL/IL interface.

## Overview

Three workspace members sharing a single Python 3.12 virtual environment:

-   **motrix_envs**: RL simulation environments built on MotrixSim, framework-agnostic
-   **motrix_rl**: RL framework integration, supporting SKRL (JAX/PyTorch) and RSLRL (PyTorch) PPO
-   **motrix_il**: Imitation learning subproject, based on lerobot with MotrixSim backends; more benchmarks coming

## Supported Algorithms

### RL

| Algorithm | Framework | Backend       |
|-----------|-----------|---------------|
| PPO       | SKRL      | JAX / PyTorch |
| PPO       | RSLRL     | PyTorch       |

### IL (LeRobot Policies)

| Policy          | Type             | Description                          |
|-----------------|------------------|--------------------------------------|
| ACT             | `act`            | Action Chunking Transformer          |
| Diffusion       | `diffusion`      | Diffusion Policy                     |
| VQ-BeT          | `vqbet`          | Vector Quantized Behavior Transformer |
| TD-MPC          | `tdmpc`          | Task-oriented Model Predictive Control |
| Pi0 / Pi05      | `pi0` / `pi05`   | π₀ family                            |
| SmolVLA         | `smolvla`        | Small Vision-Language-Action model   |
| Wall-X          | `wall_x`         | Wall-X policy                        |
| Multi-Task DiT  | `multi_task_dit` | Multi-task Diffusion Transformer     |

## Quick Start

> Prerequisite: [UV](https://docs.astral.sh/uv/) package manager.

```bash
git clone https://github.com/Motphys/MotrixLab
cd MotrixLab
git lfs pull
```

### Installation

**All RL and IL packages:**

```bash
uv sync --all-packages --all-extras
```

**RL base:**

```bash
uv sync --all-packages
```

**RL frameworks:**

```bash
uv sync --all-packages --extra skrl-jax      # SKRL + JAX (Linux only)
uv sync --all-packages --extra skrl-torch    # SKRL + PyTorch
uv sync --all-packages --extra rslrl         # RSLRL + PyTorch
```

**IL:**

```bash
uv sync --all-packages --extra training                    # Base training stack
uv sync --all-packages --extra training --extra simulation # + Simulation envs (PushT)
```

| extra        | Description                                |
|--------------|--------------------------------------------|
| `training`   | Training pipeline (wandb, accelerate)      |
| `simulation` | Simulation envs (ALOHA, PushT, LIBERO, MuJoCo) |
| `aloha`      | ALOHA simulation + dm-control + MuJoCo     |
| `libero`     | LIBERO simulation                          |
| `diffusion`  | Diffusion Policy                           |
| `pi`         | Pi0 policy                                 |
| `smolvla`    | SmolVLA policy                             |

### RL Training

```bash
uv run scripts/train.py --env cartpole               # SKRL (default)
uv run scripts/train.py --env cartpole --rllib rslrl  # RSLRL
```

View training metrics with TensorBoard:

```bash
uv run tensorboard --logdir runs/{env-name}
```

### RL Visualization & Inference

```bash
uv run scripts/view.py --env cartpole   # View environment
uv run scripts/play.py --env cartpole   # Run inference
```

## LeRobot Training

### PushT

PushT supports state-based, image-based, and image + state modes:

| Mode            | `--env.obs_type`              | `--dataset.repo_id`        | Description                |
|-----------------|-------------------------------|----------------------------|----------------------------|
| Image only      | `pixels`                      | `lerobot/pusht`            | Image input only           |
| Image + pose    | `pixels_agent_pos`            | `lerobot/pusht`            | Image + end-effector pose (default) |
| Keypoints       | `environment_state_agent_pos` | `lerobot/pusht_keypoints`  | State/keypoints only       |

```bash
# Image-based (default pixels_agent_pos)
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

More benchmarks coming.

## Registered MotrixSim Environments

| env.type            | Backend    | Description              |
|---------------------|------------|--------------------------|
| `pusht`             | MotrixSim  | 2D PushT task           |
| `aloha_motrixsim`   | MotrixSim  | ALOHA dual-arm task     |

## LeRobot Evaluation

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

`--env.discover_packages_path=motrix_il.lerobot` triggers `register_envs()` in `plugin.py`, registering both `aloha_motrixsim` and `libero_motrixsim`.

### Using Native MuJoCo Environments

The native `aloha`, `libero` etc. env types are unaffected. For headless rendering set:

```bash
MUJOCO_GL=egl uv run lerobot-eval --env.type=aloha ...
```

> Add `--eval.use_async_envs=false` if you encounter `NamespaceNotFound` errors from worker processes.

## Contact

-   GitHub Issues: [Submit an Issue](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [Join the Discussion](https://github.com/Motphys/MotrixLab/discussions)
