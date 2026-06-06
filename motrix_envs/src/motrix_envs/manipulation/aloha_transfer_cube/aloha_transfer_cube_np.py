import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AlohaTransferCubeEnvCfg

ACTION_DIM = 14
OBS_DIM = 14

START_ARM_POSE = np.array(
    [
        0.0,
        -0.96,
        1.16,
        0.0,
        -0.3,
        0.0,
        0.02239,
        -0.02239,
        0.0,
        -0.96,
        1.16,
        0.0,
        -0.3,
        0.0,
        0.02239,
        -0.02239,
    ],
    dtype=np.float32,
)

PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
BOX_X_RANGE = (-0.12, 0.12)
BOX_Y_RANGE = (0.42, 0.58)
BOX_Z_HEIGHT = 0.1


@registry.env("aloha-transfer-cube", "np")
class AlohaTransferCubeEnv(NpEnv):
    _cfg: AlohaTransferCubeEnvCfg

    def __init__(self, cfg: AlohaTransferCubeEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (ACTION_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self._ctrl_low = self._model.actuator_ctrl_limits[0].astype(np.float32)
        self._ctrl_high = self._model.actuator_ctrl_limits[1].astype(np.float32)

        self._red_box = self._model.get_geom("red_box")
        self._table = self._model.get_geom("table")
        self._left_finger = self._model.get_geom("vx300s_left/10_left_gripper_finger")
        self._right_finger = self._model.get_geom("vx300s_right/10_right_gripper_finger")
        self._contact_pairs = np.array(
            [
                [self._red_box.index, self._right_finger.index],
                [self._red_box.index, self._table.index],
                [self._red_box.index, self._left_finger.index],
            ],
            dtype=np.uint32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape != (state.data.shape[0], ACTION_DIM):
            raise ValueError(f"Expected action shape {(state.data.shape[0], ACTION_DIM)}, got {actions.shape}.")

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._action_to_ctrl(actions)
        return state

    def update_state(self, state: NpEnvState):
        obs = self._compute_observation(state.data)
        reward = self._compute_reward(state.data).astype(np.float32)
        success = reward >= 4.0

        state.obs = obs
        state.reward = reward
        state.terminated = success
        state.info["is_success"] = success
        return state

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]
        dof_pos = np.zeros((num_reset, self._model.num_dof_pos), dtype=np.float32)
        dof_vel = np.zeros((num_reset, self._model.num_dof_vel), dtype=np.float32)

        dof_pos[:, : START_ARM_POSE.shape[0]] = START_ARM_POSE
        dof_pos[:, -7:] = self._sample_box_pose(num_reset)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        data.actuator_ctrls = np.tile(START_ARM_POSE, (num_reset, 1)).astype(np.float32)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "last_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "is_success": np.zeros((num_reset,), dtype=bool),
        }
        return self._compute_observation(data), info

    def _compute_observation(self, data: mtx.SceneData):
        qpos = np.asarray(data.dof_pos, dtype=np.float32)
        left_qpos = qpos[:, :8]
        right_qpos = qpos[:, 8:16]
        obs = np.concatenate(
            [
                left_qpos[:, :6],
                self._normalize_gripper_position(left_qpos[:, 6:7]),
                right_qpos[:, :6],
                self._normalize_gripper_position(right_qpos[:, 6:7]),
            ],
            axis=-1,
        )
        return obs.astype(np.float32)

    def _action_to_ctrl(self, actions: np.ndarray):
        left_arm = actions[:, :6]
        left_gripper = self._unnormalize_gripper_position(actions[:, 6:7])
        right_arm = actions[:, 7:13]
        right_gripper = self._unnormalize_gripper_position(actions[:, 13:14])

        ctrl = np.concatenate(
            [
                left_arm,
                left_gripper,
                -left_gripper,
                right_arm,
                right_gripper,
                -right_gripper,
            ],
            axis=-1,
        ).astype(np.float32)
        return np.clip(ctrl, self._ctrl_low, self._ctrl_high)

    def _compute_reward(self, data: mtx.SceneData):
        reward = np.zeros((data.shape[0],), dtype=np.float32)
        try:
            contacts = self._model.get_contact_query(data).is_colliding(self._contact_pairs)
            contacts = np.asarray(contacts, dtype=bool).reshape(data.shape[0], 3)
        except Exception:
            return reward

        touch_right = contacts[:, 0]
        touch_table = contacts[:, 1]
        touch_left = contacts[:, 2]

        reward[touch_right] = 1.0
        reward[np.logical_and(touch_right, np.logical_not(touch_table))] = 2.0
        reward[touch_left] = 3.0
        reward[np.logical_and(touch_left, np.logical_not(touch_table))] = 4.0
        return reward

    @staticmethod
    def _sample_box_pose(num_envs: int):
        x = np.random.uniform(*BOX_X_RANGE, size=(num_envs, 1))
        y = np.random.uniform(*BOX_Y_RANGE, size=(num_envs, 1))
        z = np.full((num_envs, 1), BOX_Z_HEIGHT, dtype=np.float32)
        yaw = np.random.uniform(-np.pi, np.pi, size=(num_envs, 1)).astype(np.float32)
        half_yaw = yaw * 0.5
        quat = np.concatenate(
            [
                np.zeros((num_envs, 1), dtype=np.float32),
                np.zeros((num_envs, 1), dtype=np.float32),
                np.sin(half_yaw),
                np.cos(half_yaw),
            ],
            axis=-1,
        )
        return np.concatenate([x, y, z, quat], axis=-1).astype(np.float32)

    @staticmethod
    def _normalize_gripper_position(x: np.ndarray):
        return (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
        )

    @staticmethod
    def _unnormalize_gripper_position(x: np.ndarray):
        return x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
