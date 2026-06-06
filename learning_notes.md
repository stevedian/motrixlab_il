# MotrixLab 强化学习框架学习笔记

## 项目架构概述

### 1. 整体架构
MotrixLab是一个基于MotrixSim仿真引擎的强化学习框架，采用模块化设计，主要包含两个核心包：

- **motrix_envs**: 仿真环境定义，基于MotrixSim物理引擎
- **motrix_rl**: RL框架集成，支持SKRL和RSLRL两种RL库

### 2. 核心设计模式
- **注册表模式**: 使用`registry`统一管理环境和配置
- **配置驱动**: 所有环境参数通过`@dataclass`配置类管理
- **抽象基类**: `ABEnv`定义标准环境接口
- **多后端支持**: 支持JAX和PyTorch两种深度学习后端

## GO1 Rough Terrain 环境分析

### 1. 环境配置 (cfg.py)
```python
@registry.envcfg("go1-rough-terrain-walk")
@dataclass
class Go1WalkNpRoughEnvCfg(Go1WalkNpEnvCfg):
    render_spacing: float = 0.0
    model_file: str = "scene_rough_terrain.xml"
```

**关键参数**:
- 最大 episode 时长: 20秒
- 仿真步长: 0.01秒
- 控制步长: 0.01秒
- 模型文件: 包含机器狗和崎岖地形的MJCF模型

### 2. 环境实现 (walk_rough_terrain.py)

#### 观察空间 (48维)
- 线性速度 (3维): 局部坐标系下的线速度
- 陀螺仪数据 (3维): 角速度
- 重力向量 (3维): 相对于机器人本体的重力方向
- 关节角度 (8维): 相对于默认位置的关节角度差
- 关节速度 (8维): 关节角速度
- 上一步动作 (8维): 上一时刻的动作
- 指令 (3维): 目标速度命令 [vx, vy, ωz]

#### 动作空间 (8维)
- 8个关节的扭矩控制
- 使用PD控制器: τ = Kp*(目标角度 - 当前角度) - Kd*当前速度

#### 奖励函数
包含多个子奖励项：
- **lin_vel_z**: 惩罚垂直方向运动 (-2.0权重)
- **ang_vel_xy**: 惩罚XY轴旋转 (-0.05权重)
- **orientation**: 惩罚非水平姿态
- **torques**: 惩罚过大扭矩 (-0.00001权重)
- **dof_vel**: 惩罚关节速度
- **dof_acc**: 惩罚关节加速度 (-2.5e-7权重)
- **action_rate**: 惩罚动作变化 (-0.001权重)
- **tracking_lin_vel**: 奖励跟踪线速度命令 (1.0权重)
- **tracking_ang_vel**: 奖励跟踪角速度命令 (0.5权重)
- **feet_air_time**: 奖励空中时间 (1.0权重)

#### 终止条件
- 躯干接触地面
- 速度过大 (>1e8)

#### 特殊功能
- **课程学习**: 当平均奖励>0.9时，切换到更难的训练模式
- **随机初始化**: 在崎岖地形上生成不同的起始位置
- **边界检查**: 防止机器人走出地形边界

## Franka Lift Cube 环境分析

### 1. 环境配置 (cfg.py)
```python
@registry.envcfg("franka-lift-cube")
@dataclass
class FrankaLiftCubeEnvCfg(EnvCfg):
    model_file: str = "mjx_scene.xml"
    max_episode_seconds: float = 2.5
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
```

**关键参数**:
- 最大 episode 时长: 2.5秒
- 7自由度Franka机械臂 + 2个手指
- 抓取对象: 立方体

### 2. 环境实现 (franka_lift_cube_np.py)

#### 观察空间 (36维)
- 关节位置相对值 (9维): 相对于默认位置的关节角度
- 关节速度相对值 (9维): 相对于初始速度的关节速度
- 物体抓取位姿 (7维): 立方体的位置和姿态
- 目标位置 (3维): 立方体需要到达的目标位置
- 上一步动作 (8维): 上一时刻的动作

#### 动作空间 (8维)
- 7个关节的位置增量控制
- 1个夹爪动作 (使用Sigmoid + Bernoulli采样)

#### 奖励函数
- **reach_reward**: 机械手接近立方体的奖励
- **lift_reward**: 立方体被抬起的奖励 (高度>4cm)
- **object_command_tracking_reward**: 立方体跟踪目标位置的奖励
- **action_penalty**: 动作变化惩罚 (早期1e-4，后期1e-1)
- **joint_vel_penalty**: 关节速度惩罚 (早期1e-4，后期1e-1)

#### 终止条件
- 立方体高度 < -0.05m
- 关节速度 > 10 rad/s
- 立方体速度 > 10 m/s

#### 特殊功能
- **域随机化**: 立方体初始位置在X/Y方向随机偏移
- **动作噪声**: 关节角度重置时添加噪声
- **ε-衰减**: 训练过程中ε从1.0衰减到0.05

## 训练流程分析 (train.py)

### 1. 后端选择
```python
def get_train_backend(supports, train_backend_arg, rllib):
    # RSLRL只支持PyTorch
    # 用户可手动指定，否则自动选择
    # 优先选择GPU可用的后端
```

### 2. 训练器初始化
- **SKRL**: 支持JAX和PyTorch后端
- **RSLRL**: 仅支持PyTorch后端

### 3. 配置覆盖
支持通过命令行参数覆盖配置：
- `--num-envs`: 并行环境数量
- `--seed`: 随机种子
- `--render`: 是否渲染

## 技术要点总结

### 1. 仿真引擎集成
- 使用MotrixSim (`mtx`) 作为物理仿真后端
- 支持批量仿真 (`num_envs` > 1)
- 通过`SceneModel`和`SceneData`管理仿真状态

### 2. RL框架集成
- **SKRL**: 支持JAX和PyTorch，提供PPO等算法
- **RSLRL**: 仅支持PyTorch，提供PPO实现
- 统一配置接口 (`SkrlCfg`, `RslrlCfg`)

### 3. 奖励设计原则
- **稀疏奖励问题**: 通过密集的子奖励项引导学习
- **多目标平衡**: 不同任务目标通过权重系数平衡
- **课程学习**: 根据训练进度动态调整难度

### 4. 观察空间设计
- **相对值表示**: 使用相对于默认状态的值，提高泛化性
- **多模态融合**: 融合本体感知、视觉、目标信息
- **归一化**: 对不同物理量进行归一化处理

### 5. 动作空间设计
- **连续控制**: 使用Box空间表示连续动作
- **PD控制**: GO1使用PD控制器将动作转换为扭矩
- **混合控制**: Franka使用位置增量 + 夹爪概率控制

### 6. 环境工程技巧
- **域随机化**: 提高策略鲁棒性
- **动作平滑**: 惩罚动作变化，提高稳定性
- **终止条件**: 合理设置终止条件，避免无效探索

## 最佳实践建议

### 1. 环境开发
- 继承`NpEnv`基类实现新环境
- 使用`@registry.env`注册环境
- 通过`@dataclass`管理配置

### 2. 训练配置
- 小网络适用于简单任务 (如GO1: 256-128-64)
- 大网络适用于复杂任务 (如GO1 Rough: 512-256-128)
- 学习率: 3e-4 (GO1) 到 1e-3 (Franka)

### 3. 奖励调优
- 从简单的奖励开始，逐步添加复杂项
- 使用课程学习提高训练效率
- 监控各子奖励项，确保平衡

### 4. 性能优化
- 使用批量仿真提高样本效率
- 合理设置仿真步长和控制步长
- 使用GPU加速训练

## 扩展方向

### 1. 新环境开发
- 继承`ABEnv`或`NpEnv`基类
- 实现`apply_action`, `update_state`, `reset`方法
- 注册到`registry`

### 2. 新算法集成
- 实现新的训练器类
- 统一配置接口
- 支持多后端

### 3. 多任务学习
- 扩展注册表支持多任务
- 实现任务调度器
- 设计多任务奖励函数