**Language**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.12-blue)

`MotrixLab` is a reinforcement learning framework based on the [MotrixSim](https://github.com/Motphys/motrixsim-docs) simulation engine, designed specifically for robot simulation and training. This project provides a complete reinforcement learning development platform that integrates multiple simulation environments and training frameworks.

## Project Overview

The project currently includes three workspace members that share one Python 3.12 environment:

-   **motrix_envs**: Various RL simulation environments built on MotrixSim, defining observation, action, and reward. Framework-agnostic and currently supports MotrixSim's CPU backend
-   **motrix_rl**: Integrates RL frameworks and uses various environment parameters from motrix_envs for training. Currently supports SKRL framework (JAX/PyTorch) and RSLRL framework (PyTorch) PPO algorithms
-   **motrix_il**: An imitation learning subproject backed by the vendored `motrix_il/src/lerobot-main` checkout, sharing the root `.venv` with the RL environment and training frameworks

> Documentation: https://motrixlab.readthedocs.io

## Key Features

-   **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
-   **Multi-framework Support**: Supports SKRL (JAX/PyTorch) and RSLRL (PyTorch) training frameworks with flexible selection based on hardware environment
-   **Rich Environments**: Includes various robot simulation environments such as basic control, locomotion, and manipulation tasks
-   **High-performance Simulation**: Built on MotrixSim's high-performance physics simulation engine
-   **Visual Training**: Supports real-time rendering and training process visualization

## 🚀 Quick Start

> The following examples use the Python project management tool: [UV](https://docs.astral.sh/uv/)
>
> Before starting, please [install](https://docs.astral.sh/uv/getting-started/installation/) this tool.

### Clone Repository

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### Install Dependencies

The root workspace now targets Python 3.12 and includes `motrix_il` in the same
`uv` workspace. RL and IL share the root `.venv`. Because `uv sync` converges the
environment to the full target set declared by the current command, do not install
RL and IL separately with `uv sync --package ...` in the shared environment; the
later sync will prune dependencies that belong only to the earlier one.

The safer pattern is to always use `uv sync --all-packages ...` and declare every
RL / IL extra you want to keep in that environment in the same command.

### Install the reinforcement learning environment

Install only the RL base packages:

```bash
uv sync --all-packages
```

Install SKRL + JAX(Flax) backend (Linux only):

```bash
uv sync --all-packages --extra skrl-jax
```

Install SKRL + PyTorch backend:

```bash
uv sync --all-packages --extra skrl-torch
```

Install the RSLRL backend (PyTorch):

```bash
uv sync --all-packages --extra rslrl
```

### Install the imitation learning environment

Install the base LeRobot training stack:

```bash
uv sync --all-packages --extra training
```

Add policy-specific extras only when needed:

```bash
uv sync --all-packages --extra training --extra diffusion
uv sync --all-packages --extra training --extra pi
uv sync --all-packages --extra training --extra smolvla
uv sync --all-packages --extra training --extra wallx
uv sync --all-packages --extra training --extra multi-task-dit
```

Install simulation environments such as PushT / ALOHA / LIBERO:

```bash
uv sync --all-packages --extra training --extra simulation
```

If you want RL and IL to coexist in one environment, declare both sides in the
same sync command. For example:

```bash
uv sync --all-packages --extra skrl-jax --extra training
uv sync --all-packages --extra skrl-torch --extra training --extra simulation
uv sync --all-packages --extra rslrl --extra training
```

### Verify the imitation learning environment

Verify LeRobot commands:

```bash
uv run python -c "import motrix_il, lerobot; print(motrix_il.describe())"
uv run lerobot-train --help
```

### Install And Train By Policy

All commands below are intended to be run from the repository root.

`ACT` / `VQ-BeT` / `TD-MPC`:
These policies work with the base training environment above and do not need extra installs.

```bash
uv run lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.video_backend=pyav \
  --num_workers=0 \
  --batch_size=4 \
  --steps=400000 \
  --save_freq=4000 \
  --output_dir=outputs/train/act_aloha_transfer_cube
  --wandb.enable=true \
  --wandb.project=motrix_il
```

```bash
uv run lerobot-train \
  --policy.type=vqbet \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/vqbet_pusht
```

```bash
uv run lerobot-train \
  --policy.type=tdmpc \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/tdmpc_pusht
```

`Diffusion`:
Install the diffusion extra first, then train.

```bash
uv sync --all-packages --extra diffusion
uv run lerobot-train \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/diffusion_pusht
```

`Multi-Task DiT`:
Install the extra first, then train.

```bash
uv sync --all-packages --extra multi-task-dit
uv run lerobot-train \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/multi_task_dit_pusht
```

`Pi0` / `Pi05`:
Install the `pi` extra first, then train.

```bash
uv sync --all-packages --extra pi
uv run lerobot-train \
  --policy.type=pi0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/pi0_pusht
```

```bash
uv run lerobot-train \
  --policy.type=pi05 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/pi05_pusht
```

`SmolVLA`:
Install the `smolvla` extra first, then train.

```bash
uv sync --all-packages --extra smolvla
uv run lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/smolvla_pusht
```

`Wall-X`:
Install the `wallx` extra first, then train.

```bash
uv sync --all-packages --extra wallx
uv run lerobot-train \
  --policy.type=wall_x \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/wall_x_pusht
```

### Train Other LeRobot Tasks

The three fields you usually change are:

-   `--dataset.repo_id`: the LeRobot dataset to train on
-   `--policy.type`: the policy algorithm, such as `act`, `diffusion`, `vqbet`, or `smolvla`
-   `--output_dir`: the run output directory; include the policy and task name to avoid overwriting older runs

For another dataset that already follows the LeRobot format, replace the dataset and output directory:

```bash
uv run lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=<user-or-org>/<dataset-name> \
  --output_dir=outputs/train/act_<task_name>
```

If CUDA is not available, do a short CPU smoke test first:

```bash
uv run lerobot-train \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --dataset.repo_id=<user-or-org>/<dataset-name> \
  --steps=100 \
  --save_freq=100 \
  --output_dir=outputs/train/act_<task_name>_smoke
```

For your own task, make sure the dataset can be loaded by LeRobot and contains the
`observation.*` and `action` fields required by the selected policy. Most policies infer
state, image, and action dimensions from dataset metadata. If your field names differ,
align them during dataset conversion or use the policy's feature/rename configuration.

Evaluation requires an interactive environment. Built-in simulation tasks such as PushT
can be evaluated with `lerobot-eval`; pure offline datasets need a Gym/robot environment
before rollout evaluation is possible.

Install simulation environment extras before evaluating PushT / ALOHA / LIBERO checkpoints:

```bash
uv sync --all-packages --extra simulation
```

Evaluate a checkpoint trained by this repo:
  
uv run lerobot-eval \
  --policy.path=outputs/train/act_aloha_transfer_cube/checkpoints/076000/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --env.episode_length=600 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --policy.use_amp=false \
  --output_dir=outputs/eval/act_aloha_transfer_cube_ep600 \
  --seed=1001

MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/train/act_aloha_transfer_cube/checkpoints/092000/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha_mujoco \
  --env.task=AlohaTransferCube-v0 \
  --env.episode_length=800 \
  --eval.n_episodes=5 \
  --eval.batch_size=2 \
  --policy.use_amp=false \
  --output_dir=outputs/eval/act_aloha_transfer_cube_mujoco \
  --seed=1000

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/act_<task_name>/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/act_<task_name>
```

To disable asynchronous environments during evaluation, add:

```bash
--eval.use_async_envs=false
```

## 🎯 Usage Guide

### Environment Visualization

View environments without executing training:

```bash
uv run scripts/view.py --env cartpole
```

### Model Training

Train with SKRL framework (default):

```bash
uv run scripts/train.py --env cartpole
```

Train with RSLRL framework:

```bash
uv run scripts/train.py --env cartpole --rllib rslrl
```

Training results are saved in the `runs/{env-name}/` directory.

View training data through TensorBoard:

```bash
uv run tensorboard --logdir runs/{env-name}
```

### Model Inference

```bash
uv run scripts/play.py --env cartpole
```

For more usage methods, please refer to the [User Documentation](https://motrixlab.readthedocs.io)

## 📬 Contact

Have questions or suggestions? Feel free to contact us through:

-   GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)
