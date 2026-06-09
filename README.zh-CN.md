**语言**: [English](README.md) | [简体中文](README.zh-CN.md)

# MotrixLab

![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)
![Python Version](https://img.shields.io/badge/python-3.12-blue)

`MotrixLab` 是一个基于 [MotrixSim](https://github.com/Motphys/motrixsim-docs) 仿真引擎构建的强化学习框架，专门面向机器人仿真与训练场景。该项目提供了一个集成多种仿真环境与训练框架的完整强化学习开发平台。

## 项目概述

The project currently includes three workspace members that share one Python 3.12 environment:

-   **motrix_envs**: Various RL simulation environments built on MotrixSim, defining observation, action, and reward. Framework-agnostic and currently supports MotrixSim's CPU backend
-   **motrix_rl**: Integrates RL frameworks and uses various environment parameters from motrix_envs for training. Currently supports SKRL framework (JAX/PyTorch) and RSLRL framework (PyTorch) PPO algorithms
-   **motrix_il**: An imitation learning subproject backed by the vendored `motrix_il/src/lerobot-main` checkout, sharing the root `.venv` with the RL environment and training frameworks

> 文档地址：https://motrixlab.readthedocs.io

## 主要特性

-   **Unified Interface**: Provides a concise and unified reinforcement learning training and evaluation interface
-   **Multi-framework Support**: Supports SKRL (JAX/PyTorch) and RSLRL (PyTorch) training frameworks with flexible selection based on hardware environment
-   **Rich Environments**: Includes various robot simulation environments such as basic control, locomotion, and manipulation tasks
-   **High-performance Simulation**: Built on MotrixSim's high-performance physics simulation engine
-   **Visual Training**: Supports real-time rendering and training process visualization

## 🚀 快速开始

> The following examples use the Python project management tool: [UV](https://docs.astral.sh/uv/)
>
> Before starting, please [install](https://docs.astral.sh/uv/getting-started/installation/) this tool.

### 克隆仓库

```bash
git clone https://github.com/Motphys/MotrixLab

cd MotrixLab

git lfs pull
```

### 安装依赖

The root workspace now targets Python 3.12 and includes `motrix_il` in the same
`uv` workspace. RL and IL share the root `.venv`. Because `uv sync` converges the
environment to the full target set declared by the current command, do not install
RL and IL separately with `uv sync --package ...` in the shared environment; the
later sync will prune dependencies that belong only to the earlier one.

The safer pattern is to always use `uv sync --all-packages ...` and declare every
RL / IL extra you want to keep in that environment in the same command.

### 安装强化学习环境

Install only the RL base packages:

```bash
uv sync --all-packages
```
注:以下命令不是增量安装
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

### 安装模仿学习环境

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

### 验证模仿学习环境

Verify LeRobot commands:

```bash
uv run python -c "import motrix_il, lerobot; print(motrix_il.describe())"
uv run lerobot-train --help
```

### 按策略安装与训练

All commands below are intended to be run from the repository root.

`ACT` / `VQ-BeT` / `TD-MPC`:
These policies work with the base training environment above and do not need extra installs.

```bash
基于状态的pusht
uv run lerobot-train \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/pusht_keypoints \
    --dataset.video_backend=pyav \
    --eval_freq=0 \
    --env.type=pusht \
    --env.obs_type=environment_state_agent_pos \
    --output_dir=outputs/train/diffusion_pusht_state \
    --eval.use_async_envs=false \
    --steps=100000 \
    --save_freq=10000

```bash
基于图像的pusht
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
续训
uv run lerobot-train \
  --config_path=outputs/train/act_pusht_state/checkpoints/100000/pretrained_model/train_config.json \
  --resume=true \
  --steps=400000 \
  --save_freq=10000

### 训练其他 LeRobot 任务

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

对于你自己的任务，请确保数据集能够被 LeRobot 正确加载，并且包含所选策略所需的 `observation.*` 和 `action` 字段。

大多数策略会根据数据集元数据自动推断状态（state）、图像（image）以及动作（action）的维度信息。如果你的字段名称与 LeRobot 约定的不一致，则需要在数据集转换阶段进行字段映射，或者使用策略提供的特征配置（feature configuration）/字段重命名配置（rename configuration）进行适配。

评估（Evaluation）需要一个可交互的环境。像 PushT 这样的内置仿真任务可以直接使用 `lerobot-eval` 进行评测；而对于纯离线数据集，由于没有可执行交互的环境，因此必须先提供对应的 Gym 环境或机器人环境，才能进行 Rollout 评估。

在评估 PushT、ALOHA 或 LIBERO 训练得到的模型检查点（checkpoint）之前，请先安装相应的仿真环境扩展包：

```bash
uv sync --all-packages --extra simulation
```

act算法评估：
act_aloha_transfer_cube任务

uv run lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht_state/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --env.type=pusht \
    --env.obs_type=environment_state_agent_pos \
    --eval.n_episodes=10 \
    --eval.use_async_envs=false \
    --output_dir=outputs/eval/diffusion_pusht_state
 
uv run lerobot-eval \
  --policy.path=outputs/train/act_aloha_transfer_cube/checkpoints/156000/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --env.episode_length=600 \
  --eval.n_episodes=1 \
  --eval.batch_size=2 \
  --policy.use_amp=false \
  --output_dir=outputs/eval/act_aloha_transfer_cube_ep600 \
  --seed=1000

MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/train/act_aloha_transfer_cube/checkpoints/156000/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha_mujoco \
  --env.task=AlohaTransferCube-v0 \
  --env.episode_length=800 \
  --eval.n_episodes=5 \
  --eval.batch_size=2 \
  --policy.use_amp=false \
  --output_dir=outputs/eval/act_aloha_transfer_cube_mujoco \
  --seed=1000

act_libero任务
env -u TORCH_LOGS \
TORCH_COMPILE_DISABLE=1 \
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_MAX_AUTOTUNE=0 \
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
TOKENIZERS_PARALLELISM=false \
uv run lerobot-eval \
  --policy.path=/home/jing/project/MotrixLab/models/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.n_action_steps=10 \
  --output_dir=outputs/eval/pi0_panda_libero

uv run lerobot-eval \
  --policy.path=/home/jing/project/MotrixLab/models/pi0_libero_finetuned \
  --env.type=libero_motrixsim \
  --env.task=libero_spatial \
  --env.episode_length=280 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.n_action_steps=10 \
  --output_dir=outputs/eval/pi0_libero_spatial_motrixsim1

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

## 🎯 使用指南

### 环境可视化

View environments without executing training:

```bash
uv run scripts/view.py --env cartpole
```

### 模型训练

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

### 模型推理

```bash
uv run scripts/play.py --env cartpole
```

For more usage methods, please refer to the [User Documentation](https://motrixlab.readthedocs.io)

## 📬 联系方式

Have questions or suggestions? Feel free to contact us through:

-   GitHub Issues: [Submit Issues](https://github.com/Motphys/MotrixLab/issues)
-   Discussions: [Join Discussion](https://github.com/Motphys/MotrixLab/discussions)
