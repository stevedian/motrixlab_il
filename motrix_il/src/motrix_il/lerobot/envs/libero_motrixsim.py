from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym

from motrix_envs.imitation.wrappers import MotrixLiberoGymEnv

from .vector import LazyAsyncVectorEnv

LIBERO_MOTRIXSIM_TASKS = {
    "libero",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
}


def _make_env_fns(task: str, n_envs: int, episode_length: int) -> list[Callable[[], gym.Env]]:
    if task not in LIBERO_MOTRIXSIM_TASKS:
        raise ValueError(
            f"Unsupported MotrixSim LIBERO task '{task}'. "
            f"Available: {', '.join(sorted(LIBERO_MOTRIXSIM_TASKS))}"
        )
    return [
        partial(MotrixLiberoGymEnv, max_episode_steps=episode_length, task=task)
        for _ in range(n_envs)
    ]


def create_libero_motrixsim_envs(
    task: str,
    n_envs: int = 1,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int = 280,
    suite_name: str = "libero_motrixsim",
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    env_fns = _make_env_fns(task=task, n_envs=n_envs, episode_length=episode_length)
    if env_cls is gym.vector.AsyncVectorEnv:
        vec_env = LazyAsyncVectorEnv(env_fns)
    else:
        vec_env = env_cls(env_fns)

    return {suite_name: {0: vec_env}}
