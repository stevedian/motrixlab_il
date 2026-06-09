import os
from dataclasses import dataclass, field

import numpy as np

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.join(os.path.dirname(__file__), "xmls/generated/libero_spatial_0_motrixsim.xml")


@dataclass
class InitState:
    default_joint_pos = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], np.float32)
    joint_pos_reset_noise_scale = 0.02


@dataclass
class ControlConfig:
    min_pos = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -np.pi / 2, 0], np.float32)
    max_pos = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, np.pi / 2, 0.04], np.float32)
    joint_delta_scale = 0.04


@dataclass
class ObjectResetConfig:
    # LIBERO spatial task 0: main_table_between_plate_ramekin_region
    # BDDL local xy range: (-0.06, 0.19, -0.04, 0.21), table x is shifted by +0.3 in this MotrixSim scene.
    x_range = [0.24, 0.26]
    y_range = [0.19, 0.21]
    z = 0.05


@registry.envcfg("libero")
@dataclass
class LiberoEnvCfg(EnvCfg):
    render_spacing: float = 2.0
    model_file: str = model_file
    max_episode_seconds: float = 6.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02
    init_state: InitState = field(default_factory=InitState)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    object_reset: ObjectResetConfig = field(default_factory=ObjectResetConfig)
