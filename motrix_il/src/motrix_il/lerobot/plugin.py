from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
from lerobot.utils.constants import (
    ACTION,
    LIBERO_KEY_EEF_MAT,
    LIBERO_KEY_EEF_POS,
    LIBERO_KEY_EEF_QUAT,
    LIBERO_KEY_GRIPPER_QPOS,
    LIBERO_KEY_GRIPPER_QVEL,
    LIBERO_KEY_JOINTS_POS,
    LIBERO_KEY_JOINTS_VEL,
    LIBERO_KEY_PIXELS_AGENTVIEW,
    LIBERO_KEY_PIXELS_EYE_IN_HAND,
    OBS_IMAGES,
    OBS_STATE,
)

from .envs.aloha_motrixsim import create_aloha_motrixsim_envs
from .envs.libero_motrixsim import create_libero_motrixsim_envs

_REGISTERED = False


def _make_vec_env_cls(use_async: bool, n_envs: int):
    if use_async and n_envs > 1:
        return gym.vector.AsyncVectorEnv
    return gym.vector.SyncVectorEnv


def _aloha_features() -> dict[str, PolicyFeature]:
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "pixels/top": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
    }


def _aloha_features_map() -> dict[str, str]:
    return {
        ACTION: ACTION,
        "agent_pos": OBS_STATE,
        "pixels/top": f"{OBS_IMAGES}.top",
    }


def _libero_features() -> dict[str, PolicyFeature]:
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        LIBERO_KEY_PIXELS_AGENTVIEW: PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        LIBERO_KEY_PIXELS_EYE_IN_HAND: PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        LIBERO_KEY_EEF_POS: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        LIBERO_KEY_EEF_QUAT: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        LIBERO_KEY_EEF_MAT: PolicyFeature(type=FeatureType.STATE, shape=(3, 3)),
        LIBERO_KEY_GRIPPER_QPOS: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        LIBERO_KEY_GRIPPER_QVEL: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        LIBERO_KEY_JOINTS_POS: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        LIBERO_KEY_JOINTS_VEL: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }


def _libero_features_map() -> dict[str, str]:
    return {
        ACTION: ACTION,
        LIBERO_KEY_EEF_POS: f"{OBS_STATE}.eef_pos",
        LIBERO_KEY_EEF_QUAT: f"{OBS_STATE}.eef_quat",
        LIBERO_KEY_EEF_MAT: f"{OBS_STATE}.eef_mat",
        LIBERO_KEY_GRIPPER_QPOS: f"{OBS_STATE}.gripper_qpos",
        LIBERO_KEY_GRIPPER_QVEL: f"{OBS_STATE}.gripper_qvel",
        LIBERO_KEY_JOINTS_POS: f"{OBS_STATE}.joint_pos",
        LIBERO_KEY_JOINTS_VEL: f"{OBS_STATE}.joint_vel",
        LIBERO_KEY_PIXELS_AGENTVIEW: f"{OBS_IMAGES}.image",
        LIBERO_KEY_PIXELS_EYE_IN_HAND: f"{OBS_IMAGES}.image2",
    }


@dataclass
class AlohaMotrixSimEnv(EnvConfig):
    task: str = "AlohaTransferCube-v0"
    fps: int = 50
    episode_length: int = 400
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.features = _aloha_features()
        self.features_map = _aloha_features_map()

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int, use_async_envs: bool = True):
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_aloha_motrixsim_envs(
            n_envs=n_envs,
            env_cls=env_cls,
            episode_length=self.episode_length,
        )

    def get_env_processors(self):
        return PolicyProcessorPipeline(steps=[]), PolicyProcessorPipeline(steps=[])


@dataclass
class LiberoMotrixSimEnv(EnvConfig):
    task: str = "libero_spatial"
    fps: int = 50
    episode_length: int = 280
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.features = _libero_features()
        self.features_map = _libero_features_map()

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int, use_async_envs: bool = True):
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_libero_motrixsim_envs(
            task=self.task,
            n_envs=n_envs,
            env_cls=env_cls,
            episode_length=self.episode_length,
        )

    def get_env_processors(self):
        return (
            PolicyProcessorPipeline(steps=[LiberoProcessorStep()]),
            PolicyProcessorPipeline(steps=[]),
        )


def register_envs() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    EnvConfig.register_subclass("aloha_motrixsim")(AlohaMotrixSimEnv)
    EnvConfig.register_subclass("libero_motrixsim")(LiberoMotrixSimEnv)
    _REGISTERED = True


register_envs()
