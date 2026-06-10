from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym

from motrix_envs.imitation.wrappers import MotrixAlohaTransferCubeGymEnv

from .vector import LazyAsyncVectorEnv

ALOHA_MOTRIXSIM_TASKS = {"AlohaTransferCube-v0", "aloha-transfer-cube", "transfer_cube"}


def _make_env_fns(n_envs: int, episode_length: int) -> list[Callable[[], gym.Env]]:
    return [
        partial(MotrixAlohaTransferCubeGymEnv, max_episode_steps=episode_length)
        for _ in range(n_envs)
    ]


def create_aloha_motrixsim_envs(
    n_envs: int = 1,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = 400,
    suite_name: str = "motrixsim",
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    env_fns = _make_env_fns(n_envs=n_envs, episode_length=episode_length)
    if env_cls is gym.vector.AsyncVectorEnv:
        vec_env = LazyAsyncVectorEnv(env_fns)
    else:
        vec_env = env_cls(env_fns)

    return {suite_name: {0: vec_env}}
