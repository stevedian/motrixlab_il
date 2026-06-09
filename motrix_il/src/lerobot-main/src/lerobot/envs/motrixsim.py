from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym

from motrix_envs.imitation.wrappers import MotrixAlohaTransferCubeGymEnv, MotrixLiberoGymEnv

from .utils import _LazyAsyncVectorEnv


LIBERO_MOTRIXSIM_TASKS = {
    "libero",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}


def _make_env_fns(task: str, n_envs: int, episode_length: int) -> list[Callable[[], gym.Env]]:
    task_aliases = {"AlohaTransferCube-v0", "aloha-transfer-cube", "transfer_cube"}
    if task in task_aliases:
        return [partial(MotrixAlohaTransferCubeGymEnv, max_episode_steps=episode_length) for _ in range(n_envs)]
    if task in LIBERO_MOTRIXSIM_TASKS:
        return [partial(MotrixLiberoGymEnv, max_episode_steps=episode_length, task=task) for _ in range(n_envs)]
    raise ValueError(f"Unsupported MotrixSim imitation task '{task}'.")


def create_motrixsim_envs(
    task: str,
    n_envs: int = 1,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = 250,
    suite_name: str = "motrixsim",
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create vectorized MotrixSim imitation environments for LeRobot."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    env_fns = _make_env_fns(task=task, n_envs=n_envs, episode_length=episode_length)
    is_async = env_cls is gym.vector.AsyncVectorEnv

    if is_async:
        vec_env = _LazyAsyncVectorEnv(env_fns)
    else:
        vec_env = env_cls(env_fns)

    return {suite_name: {0: vec_env}}
