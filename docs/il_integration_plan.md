# MotrixLab IL 项目结构：LeRobot 集成

## 1. 项目概述

MotrixLab 在 Python 3.12 workspace 中统一了强化学习（RL）与模仿学习（IL），三个 workspace 成员共享根目录 `.venv`：

| 组件 | Python | 定位 |
|---|---|---|
| motrix_envs | 3.12 | 仿真环境定义（MotrixSim 后端） |
| motrix_rl | 3.12 | RL 训练框架（SKRL / RSLRL） |
| motrix_il | 3.12 | 模仿学习（LeRobot vendored） |

## 2. 目录结构

```
MotrixLab/                          # 根 workspace（Python 3.12）
├── pyproject.toml                  # workspace 定义，members = [motrix_envs, motrix_rl, motrix_il]
├── scripts/                        # RL 入口脚本
│   ├── train.py                    # RL 训练
│   ├── view.py                     # 环境可视化
│   └── play.py                     # 策略评估
│
├── motrix_envs/                    # 仿真环境包
│   └── src/motrix_envs/
│       ├── registry.py             # 环境注册系统（@envcfg, @env 装饰器）
│       ├── base.py                 # EnvCfg, ABEnv 基类
│       ├── np/                     # MotrixSim NumPy 后端
│       │   ├── env.py              # NpEnv（所有环境的基类）
│       │   └── renderer.py         # 渲染器
│       ├── manipulation/           # 机械臂操作环境
│       │   ├── aloha_transfer_cube/    # 双臂 ALOHA 传方块（14-DOF）
│       │   ├── franka_lift_cube/       # Franka 抬方块
│       │   ├── franka_open_cabinet/    # Franka 开柜门
│       │   ├── libero/                 # LIBERO 空间任务（7-DOF Panda）
│       │   ├── rm65_open_cabinet/      # RM65 开柜门
│       │   ├── shadow_hand/            # Shadow Hand 重定位
│       │   ├── robomme/                # ★ RoboMME 桌面操作（7-DOF Franka，16 任务）
│       │   └── vlabench/               # ★ VLABench 复杂操作（7-DOF Franka，41 任务）
│       ├── locomotion/             # 四足运动环境
│       ├── basic/                  # 经典控制环境
│       └── imitation/              # ★ IL 桥接层 —— MotrixSim → gymnasium.Env
│           └── wrappers/
│               ├── __init__.py
│               ├── aloha_transfer_cube_gym.py   # MotrixAlohaTransferCubeGymEnv
│               ├── libero_gym.py                # MotrixLiberoGymEnv
│               ├── robomme_gym.py               # ★ MotrixRoboMMEGymEnv
│               └── vlabench_gym.py              # ★ MotrixVLABenchGymEnv
│
├── motrix_rl/                      # RL 训练框架包
│   └── src/motrix_rl/
│       ├── tasks/                  # 各环境的 RL 任务定义
│       ├── skrl/                   # SKRL 框架适配（JAX / PyTorch）
│       └── rslrl/                  # RSLRL 框架适配（PyTorch）
│
└── motrix_il/                      # 模仿学习包
    ├── pyproject.toml              # 依赖 lerobot（path = src/lerobot-main）
    └── src/
        ├── motrix_il/              # motrix_il 自身包
        │   └── __init__.py         # describe() 函数
        │
        └── lerobot-main/           # ★ vendored LeRobot 源码
            └── src/lerobot/
                ├── scripts/        # CLI：lerobot-train, lerobot-eval, lerobot-record 等
                ├── configs/        # draccus 配置系统
                ├── policies/       # 策略实现（ACT, Diffusion, VQ-BeT, Pi0, SmolVLA 等）
                ├── datasets/       # LeRobotDataset 数据加载
                ├── processor/      # 数据预处理管线
                └── envs/           # ★ LeRobot 环境系统
                    ├── configs.py                  # EnvConfig 注册表
                    ├── motrixsim.py                # MotrixSim 环境工厂（ALOHA）
                    ├── libero_motrixsim.py         # MotrixSim 环境工厂（LIBERO）
                    ├── robomme_motrixsim.py        # ★ MotrixSim 环境工厂（RoboMME）
                    └── vlabench_motrixsim.py       # ★ MotrixSim 环境工厂（VLABench）
```

## 3. 集成架构

### 3.1 数据流

```
LeRobot CLI (lerobot-train / lerobot-eval)
    │
    ▼
EnvConfig.create_envs()          ← configs.py 中注册的配置类
    │
    ├── env.type=aloha ────────────→ AlohaEnv.create_envs()
    │                                   │
    │                                   ▼
    │                              motrixsim.py:create_motrixsim_envs()
    │                                   │
    │                                   ▼
    │                              MotrixAlohaTransferCubeGymEnv (gym.Env)
    │
    ├── env.type=libero ───────────→ LiberoEnv.create_envs()
    │                                   │
    │                                   ▼
    │                              libero.py:create_libero_envs()
    │                              (原始 LIBERO MuJoCo 后端)
    │
    ├── env.type=libero_motrixsim ─→ LiberoMotrixSimEnv.create_envs()
    │                                   │
    │                                   ▼
    │                              libero_motrixsim.py
    │                                   │
    │                                   ▼
    │                              MotrixLiberoGymEnv (gym.Env)
    │
    ├── env.type=motrixsim ────────→ MotrixSimEnv.create_envs()
    │                                   │
    │                                   ├── task=libero*    → libero_motrixsim.py
    │                                   ├── task=robomme*   → robomme_motrixsim.py   ★
    │                                   ├── task=vlabench*  → vlabench_motrixsim.py  ★
    │                                   └── task=其他        → motrixsim.py
    │
    ├── env.type=robomme ──────────→ RoboMMEEnv.create_envs()
    │   (原始 SAPIEN/ManiSkill 后端)    │
    │                                   ▼
    │                              robomme.py:create_robomme_envs()
    │
    ├── env.type=robomme_motrixsim → RoboMMEMotrixSimEnv.create_envs()   ★
    │                                   │
    │                                   ▼
    │                              robomme_motrixsim.py
    │                                   │
    │                                   ▼
    │                              MotrixRoboMMEGymEnv (gym.Env)
    │
    ├── env.type=vlabench ─────────→ VLABenchEnv.create_envs()
    │   (原始 MuJoCo/dm_control 后端)   │
    │                                   ▼
    │                              vlabench.py:create_vlabench_envs()
    │
    └── env.type=vlabench_motrixsim → VLABenchMotrixSimEnv.create_envs() ★
                                        │
                                        ▼
                                   vlabench_motrixsim.py
                                        │
                                        ▼
                                   MotrixVLABenchGymEnv (gym.Env)
```

### 3.2 环境注册表

#### 已实现

| env.type | 后端 | 环境类 | 动作维度 | 相机 |
|---|---|---|---|---|
| `aloha` | MotrixSim | `MotrixAlohaTransferCubeGymEnv` | 14 | top ×1 |
| `aloha_mujoco` | MuJoCo | `gym_aloha.AlohaEnv` | 14 | top ×1 |
| `libero` | MuJoCo (LIBERO) | LIBERO 原生环境 | 7 | agentview + eye_in_hand |
| `libero_motrixsim` | MotrixSim | `MotrixLiberoGymEnv` | 7 | agentview + eye_in_hand |
| `motrixsim` | MotrixSim | 按 task 自动选择 wrapper | 7-14 | 按任务 |
| `pusht` | MuJoCo | `gym_pusht.PushTEnv` | 2 | top ×1 |

#### 待实现（RoboMME / VLABench MotrixSim 后端）

| env.type | 后端 | 环境类 | 动作维度 | 相机 | 任务数 |
|---|---|---|---|---|---|
| `robomme` | SAPIEN/ManiSkill | `RoboMMEGymEnv` (已有) | 8 (joint) / 7 (ee) | front + wrist | 16 |
| `robomme_motrixsim` | **MotrixSim** | `MotrixRoboMMEGymEnv` (新建) | 8 (joint) / 7 (ee) | front + wrist | 16 |
| `vlabench` | MuJoCo/dm_control | `VLABenchEnv` (已有) | 7 (eef) | front + second + wrist | 41 |
| `vlabench_motrixsim` | **MotrixSim** | `MotrixVLABenchGymEnv` (新建) | 7 (eef) | front + second + wrist | 41 |

### 3.3 桥接层 —— Gymnasium Wrappers

核心桥接代码在 `motrix_envs/imitation/wrappers/`，负责将 MotrixSim 环境封装为标准 `gymnasium.Env`：

**已实现：**

**MotrixAlohaTransferCubeGymEnv** (`aloha_transfer_cube_gym.py`):
- 通过 `registry.make("aloha-transfer-cube", "np")` 创建底层 MotrixSim 环境
- `observation_space`: `Dict{"pixels": Dict{"top": ...}, "agent_pos": ...}`
- `action_space`: `Box(-inf, inf, (14,))`
- 自带 `_MotrixTopCameraRenderer`，使用 MotrixSim `RenderApp` 渲染顶部相机 RGB

**MotrixLiberoGymEnv** (`libero_gym.py`):
- 通过 `registry.make("libero", "np")` 创建底层 MotrixSim 环境
- `observation_space`: `Dict{"pixels": Dict{"image", "image2"}, "robot_state": ...}`
- `action_space`: `Box(-1, 1, (7,))`
- 自带 `_MotrixLiberoCameraRenderer`，支持双相机渲染（agentview + eye_in_hand）
- 支持 LIBERO 任务套件：`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`

**待实现：**

**MotrixRoboMMEGymEnv** (`robomme_gym.py`):
- 通过 `registry.make("robomme", "np")` 创建底层 MotrixSim 环境
- `observation_space`: `Dict{"pixels": Dict{"image", "wrist_image"}, "agent_pos": (8,)}`
  - `image`: front camera, `Box(0, 255, (256, 256, 3), uint8)`
  - `wrist_image`: wrist camera, `Box(0, 255, (256, 256, 3), uint8)`
  - `agent_pos`: 7 joint angles + 1 gripper, `Box(-inf, inf, (8,), float32)`
- `action_space`: `Box(-1, 1, (8,))` (joint_angle) 或 `Box(-1, 1, (7,))` (ee_pose)
- 自带双相机渲染器（front + wrist）
- 支持 16 个任务，分 4 个套件：
  - Counting: `BinFill`, `PickXtimes`, `SwingXtimes`, `StopCube`
  - Permanence: `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap`
  - Reference: `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder`
  - Imitation: `MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick`
- Success 检测：通过 `info["status"] == "success"` 判断

**MotrixVLABenchGymEnv** (`vlabench_gym.py`):
- 通过 `registry.make("vlabench", "np")` 创建底层 MotrixSim 环境
- `observation_space`: `Dict{"pixels": Dict{"image", "second_image", "wrist_image"}, "agent_pos": (7,)}`
  - `image`: front camera, `Box(0, 255, (480, 480, 3), uint8)`
  - `second_image`: second camera, `Box(0, 255, (480, 480, 3), uint8)`
  - `wrist_image`: wrist camera, `Box(0, 255, (480, 480, 3), uint8)`
  - `agent_pos`: 3 pos_robot + 3 euler_xyz + 1 gripper, `Box(-inf, inf, (7,), float64)`
- `action_space`: `Box(low=[-1,-1,-1,-1,-1,-1,0], high=[1,1,1,1,1,1,1], float32)` (7-D eef)
  - Action 语义: `[x, y, z (robot frame), rx, ry, rz (extrinsic xyz euler), gripper(0=closed,1=open)]`
- 自带三相机渲染器（front + second + wrist）
- 支持 41 个任务，分 2 个套件：
  - Primitive (19): `select_fruit`, `select_toy`, `select_chemistry_tube`, `add_condiment`, `select_book`, `select_painting`, `select_drink`, `insert_flower`, `select_billiards`, `select_ingredient`, `select_mahjong`, `select_poker`, `density_qa`, `friction_qa`, `magnetism_qa`, `reflection_qa`, `simple_cuestick_usage`, `simple_seesaw_usage`, `sound_speed_qa`, `thermal_expansion_qa`, `weight_qa`
  - Composite (22): `cluster_billiards`, `cluster_book`, `cluster_drink`, `cluster_toy`, `cook_dishes`, `cool_drink`, `find_unseen_object`, `get_coffee`, `hammer_nail`, `heat_food`, `make_juice`, `play_mahjong`, `play_math_game`, `play_poker`, `play_snooker`, `rearrange_book`, `rearrange_chemistry_tube`, `set_dining_table`, `set_study_table`, `store_food`, `take_chemistry_experiment`, `use_seesaw_complex`
- Success 检测：通过 dm_control 任务 `should_terminate_episode()` 判断

### 3.4 LeRobot 环境工厂

在 vendored LeRobot 中新增/待新增的文件：

**已实现：**

**`lerobot/envs/motrixsim.py`** — ALOHA 任务环境工厂：
- `create_motrixsim_envs(task, n_envs, env_cls, episode_length)` → `{suite_name: {0: VectorEnv}}`
- 支持任务别名：`AlohaTransferCube-v0`, `aloha-transfer-cube`, `transfer_cube`

**`lerobot/envs/libero_motrixsim.py`** — LIBERO 任务环境工厂：
- `create_libero_motrixsim_envs(task, n_envs, env_cls, episode_length)` → `{suite_name: {0: VectorEnv}}`

**待实现：**

**`lerobot/envs/robomme_motrixsim.py`** — RoboMME 任务环境工厂：
- `create_robomme_motrixsim_envs(task, n_envs, env_cls, action_space, episode_length)` → `{suite_name: {0: VectorEnv}}`
- 参考 `robomme.py` 中 `RoboMMEGymEnv` 的接口，将 `MotrixRoboMMEGymEnv` 包装为 VectorEnv
- 需要支持 `action_space="joint_angle"|"ee_pose"` 参数

**`lerobot/envs/vlabench_motrixsim.py`** — VLABench 任务环境工厂：
- `create_vlabench_motrixsim_envs(task, n_envs, env_cls, gym_kwargs, episode_length)` → `{suite_name: {0: VectorEnv}}`
- 参考 `vlabench.py` 中 `VLABenchEnv` 的接口，将 `MotrixVLABenchGymEnv` 包装为 VectorEnv
- 需要支持 `robot="franka"` 和 `action_mode="eef"` 参数

### 3.5 LeRobot Config 类

在 `configs.py` 中待新增：

**`RoboMMEMotrixSimEnv`**（注册名 `robomme_motrixsim`）:
```python
@EnvConfig.register_subclass("robomme_motrixsim")
@dataclass
class RoboMMEMotrixSimEnv(EnvConfig):
    task: str = "PickXtimes"
    fps: int = 10
    episode_length: int = 300
    action_space: str = "joint_angle"  # or "ee_pose"
    features: dict  # ACTION(8), pixels/image, pixels/wrist_image, agent_pos(8)
    features_map: dict  # 标准映射
```

**`VLABenchMotrixSimEnv`**（注册名 `vlabench_motrixsim`）:
```python
@EnvConfig.register_subclass("vlabench_motrixsim")
@dataclass
class VLABenchMotrixSimEnv(EnvConfig):
    task: str = "select_fruit"
    fps: int = 10
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    robot: str = "franka"
    action_mode: str = "eef"
    features: dict  # ACTION(7), pixels/image, pixels/second_image, pixels/wrist_image, agent_pos(7)
    features_map: dict  # 标准映射
```

## 4. 使用方式

### 4.1 安装

```bash
# 基础 IL 环境
uv sync --all-packages --extra training

# 加仿真环境支持（PushT / ALOHA / LIBERO）
uv sync --all-packages --extra training --extra simulation

# 加特定策略
uv sync --all-packages --extra training --extra diffusion
uv sync --all-packages --extra training --extra pi
```

### 4.2 训练

使用 LeRobot 原生命令行，数据集来自 HuggingFace Hub：

```bash
# ACT 训练 ALOHA 任务
uv run lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --output_dir=outputs/train/act_aloha_transfer_cube

# SmolVLA 训练 RoboMME
uv run lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/robomme \
  --output_dir=outputs/train/smolvla_robomme

# SmolVLA 训练 VLABench
uv run lerobot-train \
  --policy.type=smolvla \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/vlabench \
  --output_dir=outputs/train/smolvla_vlabench
```

### 4.3 评估

**MotrixSim 后端评估：**

```bash
# ALOHA 任务（MotrixSim 后端）
uv run lerobot-eval \
  --policy.path=outputs/train/act_aloha_transfer_cube/checkpoints/156000/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --env.episode_length=600 \
  --eval.n_episodes=1 \
  --eval.batch_size=2 \
  --output_dir=outputs/eval/act_aloha_transfer_cube

# LIBERO 任务（MotrixSim 后端）
uv run lerobot-eval \
  --policy.path=models/pi0_libero_finetuned \
  --env.type=libero_motrixsim \
  --env.task=libero_spatial \
  --env.episode_length=280 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.n_action_steps=10 \
  --output_dir=outputs/eval/pi0_libero_spatial_motrixsim

# RoboMME 任务（MotrixSim 后端）★
uv run lerobot-eval \
  --policy.path=models/smolvla_robomme \
  --env.type=robomme_motrixsim \
  --env.task=PickXtimes,BinFill,StopCube,MoveCube,InsertPeg \
  --env.episode_length=300 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/smolvla_robomme_motrixsim

# VLABench 任务（MotrixSim 后端）★
uv run lerobot-eval \
  --policy.path=models/smolvla_vlabench \
  --env.type=vlabench_motrixsim \
  --env.task=select_fruit,select_toy,insert_flower,add_condiment \
  --env.episode_length=500 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval/smolvla_vlabench_motrixsim
```

**原始后端评估（对比用）：**

```bash
# RoboMME SAPIEN/ManiSkill 后端
uv run lerobot-eval \
  --policy.path=models/smolvla_robomme \
  --env.type=robomme \
  --env.task=PickXtimes,BinFill,StopCube,MoveCube,InsertPeg \
  --env.dataset_split=test \
  --eval.n_episodes=10 \
  --output_dir=outputs/eval/smolvla_robomme

# VLABench MuJoCo/dm_control 后端
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=models/smolvla_vlabench \
  --env.type=vlabench \
  --env.task=select_fruit,select_toy,insert_flower \
  --env.episode_length=50 \
  --eval.n_episodes=10 \
  --output_dir=outputs/eval/smolvla_vlabench
```

## 5. 实施路线

### Phase 1: 已完成

- [x] `motrix_il` 加入根 workspace，RL 与 IL 共享 Python 3.12 环境
- [x] LeRobot vendored 到 `motrix_il/src/lerobot-main/`
- [x] `uv sync --all-packages --extra training` 可正常解析和安装
- [x] MotrixSim 环境 wrapper：`MotrixAlohaTransferCubeGymEnv`、`MotrixLiberoGymEnv`
- [x] LeRobot 环境工厂：`motrixsim.py`、`libero_motrixsim.py`
- [x] LeRobot 配置注册：`MotrixSimEnv`（`motrixsim`）、`LiberoMotrixSimEnv`（`libero_motrixsim`）
- [x] `AlohaEnv` 默认使用 MotrixSim 后端
- [x] 端到端跑通：训练 → 评估（MotrixSim 和 MuJoCo 两种后端）
- [x] 多策略支持：ACT、Diffusion、VQ-BeT、TD-MPC、Pi0、SmolVLA 等

### Phase 2: RoboMME MotrixSim 适配（当前）

- [ ] 创建 `motrix_envs/manipulation/robomme/` — MotrixSim 场景定义
  - Franka 机械臂 + 桌面 + 操作对象（方块、按钮、骰子等）
  - 16 个任务的场景变体（不同物体组合和位置）
  - 注册环境名：`"robomme"`
- [ ] 创建 `motrix_envs/imitation/wrappers/robomme_gym.py`
  - `MotrixRoboMMEGymEnv(gym.Env)` 类
  - 双相机渲染（front 256x256 + wrist 256x256）
  - `action_space`: joint_angle (8-D) / ee_pose (7-D)
  - `observation_space`: pixels(image, wrist_image) + agent_pos(8)
- [ ] 创建 `motrix_il/src/lerobot-main/src/lerobot/envs/robomme_motrixsim.py`
  - `create_robomme_motrixsim_envs()` 工厂函数
  - `RoboMMEMotrixSimGymEnv` 适配层（如果需要标准化 LeRobot 期望的接口）
- [ ] 在 `configs.py` 中新增 `RoboMMEMotrixSimEnv` 配置类（注册名 `robomme_motrixsim`）
- [ ] 端到端验证：使用 `lerobot/smolvla_robomme` 策略在 MotrixSim 后端评估

### Phase 3: VLABench MotrixSim 适配（当前）

- [ ] 创建 `motrix_envs/manipulation/vlabench/` — MotrixSim 场景定义
  - Franka 机械臂 + 桌面 + 多样化操作对象（水果、玩具、书籍、台球等）
  - 41 个任务的场景变体（primitive 19 + composite 22）
  - 注册环境名：`"vlabench"`
- [ ] 创建 `motrix_envs/imitation/wrappers/vlabench_gym.py`
  - `MotrixVLABenchGymEnv(gym.Env)` 类
  - 三相机渲染（front 480x480 + second 480x480 + wrist 480x480）
  - `action_space`: 7-D eef control (pos3 + euler3 + gripper)
  - `observation_space`: pixels(image, second_image, wrist_image) + agent_pos(7)
- [ ] 创建 `motrix_il/src/lerobot-main/src/lerobot/envs/vlabench_motrixsim.py`
  - `create_vlabench_motrixsim_envs()` 工厂函数
- [ ] 在 `configs.py` 中新增 `VLABenchMotrixSimEnv` 配置类（注册名 `vlabench_motrixsim`）
- [ ] 端到端验证：使用 `lerobot/smolvla_vlabench` 策略在 MotrixSim 后端评估

### Phase 4: 后续计划

- [ ] **MotrixSim 演示数据采集**：实现从 MotrixSim 环境录制演示数据并转换为 LeRobot 格式（Parquet + 视频）
- [ ] **更多环境扩展**：franka_lift_cube、franka_open_cabinet 等环境的 gymnasium wrapper
- [ ] **robomimic 集成**：vendored robomimic 源码、适配 MotrixSim 环境、BC/BC-RNN 等经典算法

## 6. Benchmarks 详细规格

### 6.1 RoboMME

| 属性 | 原始后端 | MotrixSim 后端 |
|---|---|---|
| 仿真引擎 | SAPIEN / ManiSkill | MotrixSim |
| 机器人 | Franka (7-DOF + gripper) | Franka (7-DOF + gripper) |
| 任务数 | 16 (4 suites × 4 tasks) | 16 |
| 相机 | front 256×256 + wrist 256×256 | front 256×256 + wrist 256×256 |
| 动作模式 | joint_angle (8-D) / ee_pose (7-D) | joint_angle (8-D) / ee_pose (7-D) |
| 状态维度 | 8 (joints + gripper) | 8 (joints + gripper) |
| 最大步数 | 300 | 300 |
| FPS | 10 | 10 |
| Success 信号 | `info["status"] == "success"` | `info["status"] == "success"` |
| 数据集 | `lerobot/robomme` (1600 episodes) | 同上（兼容现有数据集） |

**4 个任务套件：**

| Suite | 任务 | 描述 |
|---|---|---|
| Counting | BinFill, PickXtimes, SwingXtimes, StopCube | 计数与序列操作 |
| Permanence | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap | 物体恒存性 |
| Reference | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder | 参照物理解 |
| Imitation | MoveCube, InsertPeg, PatternLock, RouteStick | 动作模仿 |

### 6.2 VLABench

| 属性 | 原始后端 | MotrixSim 后端 |
|---|---|---|
| 仿真引擎 | MuJoCo / dm_control | MotrixSim |
| 机器人 | Franka (7-DOF + gripper) | Franka (7-DOF + gripper) |
| 任务数 | 41 (19 primitive + 22 composite) | 41 |
| 相机 | front 480×480 + second 480×480 + wrist 480×480 | front 480×480 + second 480×480 + wrist 480×480 |
| 动作模式 | eef (7-D) | eef (7-D) |
| 状态维度 | 7 (pos3 + euler3 + gripper) | 7 (pos3 + euler3 + gripper) |
| 最大步数 | 500 | 500 |
| FPS | 10 | 10 |
| Success 信号 | `should_terminate_episode()` | `should_terminate_episode()` |
| 数据集 | `lerobot/vlabench` | 同上（兼容现有数据集） |

**Primitive 任务（19 个）：**
`select_fruit`, `select_toy`, `select_chemistry_tube`, `add_condiment`, `select_book`, `select_painting`, `select_drink`, `insert_flower`, `select_billiards`, `select_ingredient`, `select_mahjong`, `select_poker`, `density_qa`, `friction_qa`, `magnetism_qa`, `reflection_qa`, `simple_cuestick_usage`, `simple_seesaw_usage`, `sound_speed_qa`, `thermal_expansion_qa`, `weight_qa`

**Composite 任务（22 个）：**
`cluster_billiards`, `cluster_book`, `cluster_drink`, `cluster_toy`, `cook_dishes`, `cool_drink`, `find_unseen_object`, `get_coffee`, `hammer_nail`, `heat_food`, `make_juice`, `play_mahjong`, `play_math_game`, `play_poker`, `play_snooker`, `rearrange_book`, `rearrange_chemistry_tube`, `set_dining_table`, `set_study_table`, `store_food`, `take_chemistry_experiment`, `use_seesaw_complex`

## 7. 关键设计决策

1. **Gymnasium 作为桥梁**：MotrixSim 环境通过 `gymnasium.Env` 接口暴露给 LeRobot，不引入额外的抽象层
2. **Wrapper 在 motrix_envs 侧**：Gymnasium wrapper 放在 `motrix_envs/imitation/wrappers/`，因为它们是环境的一部分，不依赖 LeRobot
3. **环境工厂在 LeRobot 侧**：`create_*_motrixsim_envs()` 等工厂函数在 vendored LeRobot 的 `envs/` 目录中，负责将 gym 环境包装为 LeRobot 需要的 `VectorEnv` 格式
4. **统一 workspace**：RL 和 IL 共享同一个 `.venv`，避免依赖重复安装和版本冲突
5. **直接使用 LeRobot CLI**：不创建额外的封装脚本，直接使用 `lerobot-train` / `lerobot-eval`，减少维护成本
6. **兼容现有数据集**：MotrixSim 后端的 observation/action 空间与原始后端保持一致，确保可以直接使用 HuggingFace Hub 上已有的 LeRobot 格式数据集进行训练和评估
7. **独立 env.type 命名**：每个 MotrixSim 后端使用 `{name}_motrixsim` 命名（如 `robomme_motrixsim`），与原始后端（`robomme`）并存，方便对比测试

## 8. 依赖关系

```
motrix_il (Python 3.12)
├── lerobot (vendored, path = src/lerobot-main)
│   ├── torch, torchvision
│   ├── gymnasium
│   ├── diffusers (>=0.27.2)
│   ├── huggingface-hub (>=1.0.0)
│   └── ...
│
├── motrix_envs (workspace 成员，共享 .venv)
│   ├── motrixsim (>=0.7.0)
│   └── gymnasium (与 LeRobot 共用版本)
│
└── motrix_il 自身包
    └── __init__.py (describe 函数)
```

## 9. 新增/待新增文件清单

### 已实现

| 文件 | 说明 |
|---|---|
| `motrix_envs/src/motrix_envs/imitation/wrappers/aloha_transfer_cube_gym.py` | ALOHA gymnasium wrapper |
| `motrix_envs/src/motrix_envs/imitation/wrappers/libero_gym.py` | LIBERO gymnasium wrapper |
| `motrix_envs/src/motrix_envs/imitation/wrappers/__init__.py` | Wrapper 导出 |
| `motrix_envs/src/motrix_envs/manipulation/libero/` | LIBERO MotrixSim 环境定义 |
| `motrix_il/src/lerobot-main/src/lerobot/envs/motrixsim.py` | LeRobot 侧 MotrixSim 环境工厂 |
| `motrix_il/src/lerobot-main/src/lerobot/envs/libero_motrixsim.py` | LeRobot 侧 LIBERO MotrixSim 环境工厂 |
| `motrix_il/src/lerobot-main/src/lerobot/envs/configs.py` | 新增 `MotrixSimEnv`、`LiberoMotrixSimEnv`，修改 `AlohaEnv.create_envs()` |

### 待实现（RoboMME + VLABench）

| 文件 | 说明 |
|---|---|
| `motrix_envs/src/motrix_envs/manipulation/robomme/` | RoboMME MotrixSim 场景（16 任务，Franka + 桌面物体） |
| `motrix_envs/src/motrix_envs/manipulation/vlabench/` | VLABench MotrixSim 场景（41 任务，Franka + 复杂物体） |
| `motrix_envs/src/motrix_envs/imitation/wrappers/robomme_gym.py` | `MotrixRoboMMEGymEnv` — RoboMME gymnasium wrapper |
| `motrix_envs/src/motrix_envs/imitation/wrappers/vlabench_gym.py` | `MotrixVLABenchGymEnv` — VLABench gymnasium wrapper |
| `motrix_il/src/lerobot-main/src/lerobot/envs/robomme_motrixsim.py` | LeRobot 侧 RoboMME MotrixSim 环境工厂 |
| `motrix_il/src/lerobot-main/src/lerobot/envs/vlabench_motrixsim.py` | LeRobot 侧 VLABench MotrixSim 环境工厂 |
| `motrix_il/src/lerobot-main/src/lerobot/envs/configs.py` | 新增 `RoboMMEMotrixSimEnv`、`VLABenchMotrixSimEnv` 配置类 |
