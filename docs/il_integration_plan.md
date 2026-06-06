# MotrixLab IL Integration Plan: LeRobot + Robomimic

## 1. 背景与动机

### 1.1 现状

| 组件 | Python | 框架 | 定位 |
|---|---|---|---|
| motrix_envs + motrix_rl | 3.12 | SKRL / RSLRL | 强化学习（workspace 成员） |
| motrix_il | 3.12 | LeRobot (vendored) | 模仿学习（workspace 成员） |
| robomimic | 3.8+ | PyTorch | 模仿学习（待集成） |

### 1.2 核心冲突

- **Python 版本**：当前 workspace 已统一到 Python 3.12；后续需要持续确认 MotrixSim、SKRL、RSLRL 与 LeRobot 的依赖组合可解析
- **robomimic 依赖老旧**：`diffusers==0.11.1`、`transformers==4.41.2`、`huggingface_hub==0.23.4`，与 LeRobot 的现代依赖（diffusers>=0.27.2、huggingface-hub>=1.0.0）直接冲突

### 1.3 设计原则

1. **最小破坏**：不破坏现有 motrix_rl 工作流
2. **独立适配**：每个 IL 框架独立对接 MotrixSim，不引入不必要的抽象层
3. **优先 LeRobot**：LeRobot 更活跃、依赖更新、社区更大；robomimic 作为经典 IL 算法补充
4. **以 gymnasium 为桥梁**：MotrixSim 环境通过 gymnasium 接口暴露给两个框架

---

## 2. Python 版本策略

### 当前策略：统一 Python 3.12 workspace

```
Python 3.12
┌──────────────────────────────────────┐
│  motrix_envs                         │
│  motrix_rl                           │
│  motrix_il                           │
│  ├── lerobot (vendored)              │
│  └── robomimic (future vendor)       │
└──────────────────────────────────────┘
                   │
          MotrixSim / Gymnasium 环境接口
```

- 根目录 `.venv` 同时承载 RL 与 IL 依赖
- `motrix_il` 已加入根 workspace，不再使用独立 `motrix_il/.venv`
- LeRobot 和后续 robomimic 共享同一个 Python 3.12 环境，减少 torch / torchvision 等底层依赖重复安装

---

## 3. 目录结构

```
MotrixLab/
├── motrix_envs/          # RL 环境（Python 3.12 workspace 成员）
├── motrix_rl/            # RL 训练框架（Python 3.12 workspace 成员）
├── motrix_il/            # IL 集合（Python 3.12 workspace 成员）
│   ├── pyproject.toml
│   ├── src/
│   │   ├── lerobot-main/              # LeRobot vendored 源码（已有）
│   │   ├── robomimic/                 # robomimic vendored 源码（新增）
│   │   │   └── robomimic/             # robomimic Python 包本体
│   │   │       ├── algo/              # BC, BCQ, CQL, IQL 等
│   │   │       ├── config/            # 算法配置
│   │   │       ├── models/            # 网络架构
│   │   │       ├── envs/              # 环境 wrapper（已有 EnvBase 基类）
│   │   │       └── utils/
│   │   └── motrix_il/                 # motrix_il 自身包（已有）
│   │       ├── __init__.py
│   │       ├── lerobot_bridge/        # LeRobot 独立适配 MotrixSim（新增）
│   │       │   ├── __init__.py
│   │       │   ├── motrixsim_env.py   # MotrixSim 环境注册到 LeRobot env factory
│   │       │   └── collect_demo.py    # LeRobot 格式演示数据采集（Parquet + 视频）
│   │       └── robomimic_bridge/      # robomimic 独立适配 MotrixSim（新增）
│   │           ├── __init__.py
│   │           ├── motrixsim_env.py   # MotrixSim → robomimic EnvBase wrapper
│   │           ├── collect_demo.py    # robomimic 格式演示数据采集（HDF5）
│   │           └── config_template.py # robomimic config 模板（适配 MotrixSim 任务）
│   └── scripts/                       # 入口脚本（新增）
│       ├── train_lerobot.py           # LeRobot 训练入口
│       ├── eval_lerobot.py            # LeRobot 评估入口
│       ├── train_robomimic.py         # robomimic 训练入口
│       └── eval_robomimic.py          # robomimic 评估入口
```

两个框架各自完全独立的 bridge 和脚本，不共享抽象层。

---

## 4. robomimic 依赖冲突解决方案

### 4.1 问题

robomimic 在 `setup.py` 中硬编码了老旧版本的依赖：

```python
# robomimic 当前 pins
"huggingface_hub==0.23.4",   # → motrix_il 需要 >=1.0.0
"transformers==4.41.2",      # → 与 LeRobot 可选依赖冲突
"diffusers==0.11.1",         # → LeRobot 需要 >=0.27.2
```

### 4.2 方案：Vendor + 松绑

1. **不通过 pip 安装 robomimic**，直接将源码 vendored 到 `motrix_il/src/robomimic/`
2. **删除 `setup.py` 中的硬 pins**，改为合理的最低版本约束：

```python
# motrix_il/src/robomimic/pyproject.toml（新建）
[project]
name = "robomimic-vendored"
version = "0.3.0"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "numpy>=1.13.3",
    "h5py",
    "tqdm",
    "tensorboard",
    "imageio",
    "matplotlib",
    # 以下使用宽松约束，由 LeRobot 的锁文件统一解析
    "huggingface-hub>=0.23.0",
    "transformers>=4.41.0",
    "diffusers>=0.11.0",
]
```

3. **按需测试与修复**：

| robomimic 模块 | 是否使用冲突依赖 | 兼容性风险 | 处理方式 |
|---|---|---|---|
| `algo/bc.py` (BC, BC-RNN) | 否 | 低 | 直接可用 |
| `algo/bcq.py` | 否 | 低 | 直接可用 |
| `algo/cql.py` | 否 | 低 | 直接可用 |
| `algo/iql.py` | 否 | 低 | 直接可用 |
| `algo/td3_bc.py` | 否 | 低 | 直接可用 |
| `algo/hbc.py` | 否 | 低 | 直接可用 |
| `algo/diffusion_policy.py` | `diffusers` | 中 | 需要适配新版 diffusers API；**或者直接用 LeRobot 的 Diffusion Policy（更现代）** |
| `algo/bc.py` (BC-Transformer) | `transformers`, `huggingface_hub` | 中 | 需要验证新版 transformers 兼容性 |
| `models/transformers.py` | `transformers` | 中 | 同 BC-Transformer |
| `config/` | 无直接冲突 | 低 | config_template 生成逻辑直接可用 |

4. **关键决策**：Diffusion Policy 优先使用 LeRobot 的实现（更完善、社区维护更好），robomimic 的 `diffusion_policy.py` 仅作为参考。

---

## 5. 实施阶段

### Phase 1: motrix_il 修复与 LeRobot 跑通（1-2 天）

**目标**：让当前 motrix_il + LeRobot 能正常 `uv sync` 并运行。

- [ ] 修复 `pyproject.toml` 中 `lerobot` 的 path → `src/lerobot-main`
- [ ] 安装 Python 3.12（`uv python install 3.12`）
- [ ] `uv sync --all-packages --all-extras` 通过
- [ ] 验证 LeRobot CLI：`uv run lerobot-train --help`
- [ ] 验证策略导入：`uv run python -c "from lerobot.policies.diffusion import DiffusionPolicy"`

### Phase 2: LeRobot → MotrixSim 适配（2-3 天）

**目标**：能用 LeRobot 在 MotrixSim 环境中训练 Diffusion Policy。

- [ ] 在 `motrix_il/src/motrix_il/lerobot_bridge/` 创建 `MotrixSimEnvConfig`，注册到 LeRobot 的 env factory
- [ ] 实现 MotrixSim → gymnasium 环境 wrapper（实现 `gymnasium.Env` 接口，内部调用 motrixsim）
- [ ] 创建 LeRobot 格式的演示数据录制脚本 `scripts/collect_lerobot_demo.py`
- [ ] 端到端跑通：录数据 → 训练 Diffusion Policy → 评估
- [ ] 创建 `scripts/train_lerobot.py` 和 `scripts/eval_lerobot.py`

### Phase 3: robomimic Vendored + MotrixSim 适配（2-3 天）

**目标**：robomimic 核心算法（BC、BC-RNN）能在 MotrixSim 环境中训练。

- [ ] 将 robomimic 源码 vendored 到 `motrix_il/src/robomimic/`
- [ ] 新建 `motrix_il/src/robomimic/pyproject.toml`，松绑依赖版本
- [ ] 实现 `motrix_il/src/motrix_il/robomimic_bridge/motrixsim_env.py`：封装 MotrixSim 环境为 robomimic 的 `EnvBase` 子类
- [ ] 为 MotrixSim 任务编写 robomimic config JSON 模板
- [ ] 实现 `motrix_il/src/motrix_il/robomimic_bridge/collect_demo.py`：录制演示数据为 robomimic HDF5 格式
- [ ] 创建 `scripts/train_robomimic.py` 和 `scripts/eval_robomimic.py`
- [ ] 端到端跑通：录数据 → BC/BC-RNN 训练 → rollout

### Phase 4: Benchmark（持续）

- [ ] 为 MotrixSim 标准任务（cartpole、franka_lift_cube、go2 等）训练基线策略
- [ ] 分别在两个框架下评估，对比效果
- [ ] 文档记录每个任务的最佳配置和结果

---

## 6. 依赖关系总览

```
motrix_il (Python 3.12)
├── lerobot (vendored, path = src/lerobot-main)
│   ├── torch, torchvision
│   ├── gymnasium
│   ├── diffusers (>=0.27.2)
│   ├── huggingface-hub (>=1.0.0)
│   ├── datasets, pandas, pyarrow  [dataset extra]
│   ├── accelerate, wandb           [training extra]
│   └── ...
│
├── robomimic-vendored (path = src/robomimic)
│   ├── torch (共用 LeRobot 的版本)
│   ├── numpy
│   ├── huggingface-hub (共用 LeRobot 的版本)
│   ├── transformers (共用同一 venv 中的版本)
│   └── diffusers (共用 LeRobot 的版本)
│
└── motrix_il (自身包)
    ├── lerobot_bridge/     # LeRobot 独立适配 MotrixSim
    │   ├── motrixsim_env.py      # 环境注册到 LeRobot
    │   └── collect_demo.py       # LeRobot 格式数据采集
    └── robomimic_bridge/   # robomimic 独立适配 MotrixSim
        ├── motrixsim_env.py      # 环境 wrapper (EnvBase 子类)
        └── config_template.py    # JSON 配置模板
```

**关键**：robomimic 和 LeRobot 共享同一个 Python 3.12 venv。通过 vendored + 松绑 pins 的方式，让 UV 解析出一套兼容的依赖版本。两个框架各自独立适配 MotrixSim，bridge 层互不依赖。

---

## 7. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| robomimic 的旧代码不适配新版 diffusers API | Diffusion Policy 无法直接运行 | 优先使用 LeRobot 的 Diffusion Policy；robomimic 只取其 BC 等经典算法 |
| motrixsim 不支持 Python 3.12 | 无法在 motrix_il 中直接 `import motrix_envs` | 短期通过 gymnasium 接口（subprocess 或 socket 通信）绕过；中期推动 motrixsim 支持 3.12 |
| robomimic 长期不维护（最后 release 2023.07） | 安全漏洞、依赖过时 | 只 vendored 核心算法代码（algo/ + models/），按需修补；重点投资 LeRobot |
| robomimic 的 config 系统与 LeRobot 不同 | 用户需要学习两套配置方式 | 各自独立脚本，不强制统一；文档分别说明 |

---

## 8. 里程碑

| 版本 | 内容 |
|---|---|
| v0.4.0-alpha | Phase 1+2：LeRobot 在 MotrixSim 上跑通训练 |
| v0.4.0-beta | Phase 3：robomimic BC 系列跑通 |
| v0.4.0 | Phase 4：Benchmark |
| v0.5.0 | Python 3.12 全项目统一（待 motrixsim 支持） |

---

## 9. 未决问题

1. **motrixsim 是否有 Python 3.12 支持计划？** — 这决定了长期能否统一环境，以及短期内 bridge 层是否需要 subprocess 隔离
2. **MotrixSim 环境是否已有 gymnasium 接口？** — motrix_rl 依赖 `gymnasium===1.1.1`，需要确认 motrix_envs 中的环境是否实现了 `gymnasium.Env`
3. **robomimic 的哪些算法是必须的？** — 确定 vendored 后优先适配的范围。建议从 BC、BC-RNN 开始，后续按需添加 IQL、CQL 等
4. **motrix_il 是否需要单独的 robomimic venv？** — 如果 vendored + 松绑后依赖冲突仍然无法解决，可以将 robomimic 拆到 `motrix_il` 内的独立 venv（两个 pyproject.toml），但这增加了复杂度
