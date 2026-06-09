import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import LiberoEnvCfg

ACTION_DIM = 7
OBS_DIM = 8
ROBOT_DOF_POS = 9
ROBOT_DOF_VEL = 9
ROBOT_ACTUATORS = 9
LIFT_Z_THRESHOLD = 0.08
GRIPPER_REACH_THRESHOLD = 0.08
MUJOCO_DEFAULT_FREE_BODY_POSES = {
    "akita_black_bowl_1_main": np.array([-0.063848, 0.185571, 0.97, 0.707107, 0.0, 0.0, 0.707107], dtype=np.float32),
    "akita_black_bowl_2_main": np.array([-0.181043, 0.320378, 0.97, 0.707107, 0.0, 0.0, 0.707107], dtype=np.float32),
    "cookies_1_main": np.array([0.06836, 0.030034, 0.97, 0.707107, 0.0, 0.0, 0.707107], dtype=np.float32),
    "glazed_rim_porcelain_ramekin_1_main": np.array(
        [-0.211551, 0.21184, 0.97, 0.707107, 0.0, 0.0, 0.707107], dtype=np.float32
    ),
    "plate_1_main": np.array([0.064058, 0.205965, 0.97, 0.707107, 0.0, 0.0, 0.707107], dtype=np.float32),
}
SPATIAL_OBJECT_POSES = {
    "plate": np.array([0.36, 0.20, 0.012], dtype=np.float32),
    "ramekin": np.array([0.10, 0.20, 0.03], dtype=np.float32),
    "cookies": np.array([0.37, 0.03, 0.025], dtype=np.float32),
    "distractor_bowl": np.array([0.12, 0.32, 0.04], dtype=np.float32),
    "wooden_cabinet": np.array([0.33, -0.27, 0.07], dtype=np.float32),
    "flat_stove": np.array([-0.11, -0.14, 0.015], dtype=np.float32),
}


@registry.env("libero", "np")
class LiberoEnv(NpEnv):
    _cfg: LiberoEnvCfg

    def __init__(self, cfg: LiberoEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.default_joint_pos = self._cfg.init_state.default_joint_pos.astype(np.float32)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self._init_dof_pos = self._model.compute_init_dof_pos().astype(np.float32)
        self._robot_body = self._model.get_body("robot0_base")
        self._target_object = self._model.get_site("cookies_1_default_site")
        self._eef_site = self._model.get_site("gripper0_grip_site")
        self._free_bodies = {name: self._model.get_body(name) for name in MUJOCO_DEFAULT_FREE_BODY_POSES}
        self._rng = np.random.default_rng()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed: int | None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape != (state.data.shape[0], ACTION_DIM):
            raise ValueError(f"Expected action shape {(state.data.shape[0], ACTION_DIM)}, got {actions.shape}.")

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        dof_pos = self.get_dof_pos(state.data)
        ctrl = np.zeros((state.data.shape[0], ROBOT_ACTUATORS), dtype=np.float32)
        ctrl[:, :7] = dof_pos[:, :7]
        ctrl[:, :6] += np.clip(actions[:, :6], -1.0, 1.0) * self._cfg.control_config.joint_delta_scale
        gripper = (np.clip(actions[:, 6], -1.0, 1.0) + 1.0) * 0.02
        ctrl[:, 7] = gripper
        ctrl[:, 8] = -gripper
        state.data.actuator_ctrls = ctrl
        return state

    def update_state(self, state: NpEnvState):
        obs = self._compute_observation(state.data)
        reward = self._compute_reward(state.data).astype(np.float32)
        success = self._compute_success(state.data)

        state.obs = obs
        state.reward = reward
        state.terminated = success
        state.info["is_success"] = success
        return state

    def reset(self, data: mtx.SceneData):
        num_reset = data.shape[0]
        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1)).astype(np.float32)
        dof_vel = np.zeros((num_reset, self._model.num_dof_vel), dtype=np.float32)
        robot_dof_pos = np.tile(self.default_joint_pos, (num_reset, 1))
        noise = self._rng.uniform(
            -self._cfg.init_state.joint_pos_reset_noise_scale,
            self._cfg.init_state.joint_pos_reset_noise_scale,
            size=(num_reset, ROBOT_DOF_POS),
        ).astype(np.float32)
        robot_dof_pos += noise
        dof_pos[:, :ROBOT_DOF_POS] = robot_dof_pos

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        for name, pose in MUJOCO_DEFAULT_FREE_BODY_POSES.items():
            self._free_bodies[name].set_dof_pos(data, np.tile(pose, (num_reset, 1)))
        data.actuator_ctrls = np.zeros((num_reset, self._model.num_actuators), dtype=np.float32)
        data.actuator_ctrls[:, :7] = robot_dof_pos[:, :7]
        data.actuator_ctrls[:, 7] = robot_dof_pos[:, 7]
        data.actuator_ctrls[:, 8] = -robot_dof_pos[:, 8]
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "last_actions": np.zeros((num_reset, ACTION_DIM), dtype=np.float32),
            "is_success": np.zeros((num_reset,), dtype=bool),
        }
        return self._compute_observation(data), info

    def get_dof_pos(self, data: mtx.SceneData):
        return self._robot_body.get_joint_dof_pos(data)[:, :ROBOT_DOF_POS]

    def get_dof_vel(self, data: mtx.SceneData):
        return self._robot_body.get_joint_dof_vel(data)[:, :ROBOT_DOF_VEL]

    def get_cube_pos(self, data: mtx.SceneData):
        return self.get_target_object_pos(data)

    def get_target_object_pos(self, data: mtx.SceneData):
        return self._target_object.get_pose(data)[:, :3].astype(np.float32)

    def get_scene_object_poses(self):
        return SPATIAL_OBJECT_POSES.copy()

    def get_eef_pos(self, data: mtx.SceneData):
        return self._eef_site.get_pose(data)[:, :3].astype(np.float32)

    def get_robot_state(self, data: mtx.SceneData):
        dof_pos = self.get_dof_pos(data).astype(np.float32)
        dof_vel = self.get_dof_vel(data).astype(np.float32)
        eef_pose = self._eef_site.get_pose(data).astype(np.float32)
        gripper = dof_pos[:, 7:8]
        return {
            "eef_pos": eef_pose[:, :3],
            "eef_quat": eef_pose[:, 3:7],
            "eef_mat": np.tile(np.eye(3, dtype=np.float32), (data.shape[0], 1, 1)),
            "gripper_qpos": np.repeat(gripper, 2, axis=-1),
            "gripper_qvel": np.zeros((data.shape[0], 2), dtype=np.float32),
            "joint_pos": np.concatenate([dof_pos[:, :7]], axis=-1),
            "joint_vel": dof_vel[:, :7],
        }

    def _compute_observation(self, data: mtx.SceneData):
        robot_state = self.get_robot_state(data)
        return np.concatenate(
            [
                robot_state["eef_pos"],
                np.zeros((data.shape[0], 3), dtype=np.float32),
                robot_state["gripper_qpos"],
            ],
            axis=-1,
        ).astype(np.float32)

    def _compute_reward(self, data: mtx.SceneData):
        cube_pos = self.get_target_object_pos(data)
        eef_pos = self.get_eef_pos(data)
        reach_dist = np.linalg.norm(cube_pos - eef_pos, axis=-1)
        reach = 1.0 - np.tanh(reach_dist / 0.12)
        lifted = cube_pos[:, 2] > LIFT_Z_THRESHOLD
        return reach + lifted.astype(np.float32) * 4.0

    def _compute_success(self, data: mtx.SceneData):
        cube_pos = self.get_target_object_pos(data)
        eef_pos = self.get_eef_pos(data)
        reach_dist = np.linalg.norm(cube_pos - eef_pos, axis=-1)
        return np.logical_and(cube_pos[:, 2] > LIFT_Z_THRESHOLD, reach_dist < GRIPPER_REACH_THRESHOLD)

    def _sample_target_object_pose(self, num_envs: int):
        reset = self._cfg.object_reset
        x = self._rng.uniform(*reset.x_range, size=(num_envs, 1))
        y = self._rng.uniform(*reset.y_range, size=(num_envs, 1))
        z = np.full((num_envs, 1), reset.z, dtype=np.float32)
        quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_envs, 1))
        return np.concatenate([x, y, z, quat], axis=-1).astype(np.float32)
