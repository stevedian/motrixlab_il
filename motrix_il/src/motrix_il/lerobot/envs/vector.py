from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym


class LazyAsyncVectorEnv:
    """Create AsyncVectorEnv on first use to avoid eager worker startup."""

    def __init__(
        self,
        env_fns: list[Callable],
        observation_space=None,
        action_space=None,
        metadata=None,
    ):
        self._env_fns = env_fns
        self._env: gym.vector.AsyncVectorEnv | None = None
        self.num_envs = len(env_fns)
        if observation_space is not None and action_space is not None and metadata is not None:
            self.observation_space = observation_space
            self.action_space = action_space
            self.metadata = metadata
        else:
            tmp = env_fns[0]()
            self.observation_space = tmp.observation_space
            self.action_space = tmp.action_space
            self.metadata = tmp.metadata
            tmp.close()
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def _ensure(self) -> None:
        if self._env is None:
            self._env = gym.vector.AsyncVectorEnv(self._env_fns, context="forkserver", shared_memory=True)

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self._ensure()
        return self._env.reset(**kwargs)

    def step(self, actions):
        self._ensure()
        return self._env.step(actions)

    def call(self, name, *args, **kwargs):
        self._ensure()
        return self._env.call(name, *args, **kwargs)

    def get_attr(self, name):
        self._ensure()
        return self._env.get_attr(name)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

