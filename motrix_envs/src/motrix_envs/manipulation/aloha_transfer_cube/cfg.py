import os
from dataclasses import dataclass

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/assets/bimanual_viperx_transfer_cube.xml"


@registry.envcfg("aloha-transfer-cube")
@dataclass
class AlohaTransferCubeEnvCfg(EnvCfg):
    render_spacing: float = 2.0
    model_file: str = model_file
    max_episode_seconds: float = 8.0
    sim_dt: float = 0.0025
    ctrl_dt: float = 0.02
