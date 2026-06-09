from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from motrix_envs import registry

LOGGER = logging.getLogger(__name__)
EYE_IN_HAND_CAMERA = "robot0_eye_in_hand"


class _MotrixLiberoCameraRenderer:
    def __init__(
        self,
        env,
        width: int,
        height: int,
        camera_names: tuple[str, str],
        required: bool = True,
    ):
        self._env = env
        self._width = width
        self._height = height
        self._camera_names = camera_names
        self._required = required
        self._app = None
        self._cameras = {}
        self._enabled = False
        self._warned = False
        self._dumped_first_frames = False
        self._init_renderer()

    @property
    def enabled(self):
        return self._enabled

    def render(self):
        if not self._enabled:
            return None

        try:
            self._app.sync(data=self._env.state.data, wait=True)
            frames = {}
            for name, camera in self._cameras.items():
                frame = self._capture_camera(camera)
                if frame is None:
                    return None
                frames[name] = frame
            return frames
        except Exception as exc:
            self._enabled = False
            if not self._warned:
                LOGGER.warning("MotrixSim LIBERO camera capture failed: %s", exc)
                self._warned = True
        return None

    def _init_renderer(self):
        if os.getenv("MOTRIX_DISABLE_RENDER") == "1":
            self._handle_unavailable("MOTRIX_DISABLE_RENDER=1")
            return
        if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            self._handle_unavailable("no DISPLAY or WAYLAND_DISPLAY is available")
            return
        if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
            os.environ.pop("WAYLAND_DISPLAY", None)
            os.environ.setdefault("WINIT_UNIX_BACKEND", "x11")

        try:
            from motrix_envs.np.renderer import _check_native_render
            from motrixsim.render import RenderApp, RenderSettings

            if not _check_native_render():
                self._handle_unavailable("MotrixSim RenderApp probe failed")
                return

            for name in self._camera_names:
                camera_cfg = self._env.model.cameras[name]
                camera_cfg.set_near_far(0.01, 10.0)
                camera_cfg.set_render_target("image", self._width, self._height)

            app = RenderApp()
            app.launch(self._env.model, batch=1, render_settings=RenderSettings.quality())
            if self._env.state is not None:
                for _ in range(5):
                    app.sync(data=self._env.state.data, wait=True)
                    time.sleep(0.02)

            self._app = app
            self._cameras = {
                name: app.get_camera(self._env.model.cameras[name].index) for name in self._camera_names
            }
            self._enabled = True
            LOGGER.warning(
                "MotrixSim LIBERO camera renderer initialized (cameras=%s, size=%dx%d)",
                ",".join(self._camera_names),
                self._width,
                self._height,
            )
        except Exception as exc:
            self._enabled = False
            self._handle_unavailable(str(exc))

    def _handle_unavailable(self, reason: str):
        if self._required:
            raise RuntimeError(
                "MotrixSim LIBERO camera rendering is required but unavailable: "
                f"{reason}. Start a graphical session or virtual X server. "
                "Set MOTRIX_ALLOW_RENDER_FALLBACK=1 only for smoke tests."
            )
        LOGGER.warning("MotrixSim LIBERO renderer unavailable; using fallback frames: %s", reason)

    def _capture_camera(self, camera):
        task = camera.capture()
        for _ in range(8):
            image = task.take_image()
            if image is not None:
                frame = self._coerce_frame(np.asarray(image.pixels))
                if frame.max() < 8 or int(frame.max()) - int(frame.min()) < 20:
                    self._app.sync(data=self._env.state.data, wait=True)
                    time.sleep(0.01)
                    continue
                return frame
            self._app.sync(data=self._env.state.data, wait=True)
            time.sleep(0.001)
        return None

    def dump_first_frames(self, frames: dict[str, np.ndarray]):
        if self._dumped_first_frames:
            return
        try:
            from PIL import Image

            debug_dir = Path("outputs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                out_path = debug_dir / f"motrixsim_libero_{name}_first_frame.png"
                Image.fromarray(frame).save(out_path)
                LOGGER.warning("MotrixSim LIBERO first real %s frame saved to %s", name, out_path)
            self._dumped_first_frames = True
        except Exception as exc:
            LOGGER.warning("Failed to save first MotrixSim LIBERO camera frames: %s", exc)

    def _coerce_frame(self, pixels: np.ndarray):
        if pixels.ndim == 4:
            pixels = pixels[0]
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        if pixels.shape[:2] != (self._height, self._width):
            pixels = np.resize(pixels, (self._height, self._width, 3))
        return np.ascontiguousarray(pixels, dtype=np.uint8)


class MotrixLiberoGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 400,
        observation_height: int = 256,
        observation_width: int = 256,
        task: str = "libero",
        task_description: str | None = None,
    ):
        super().__init__()
        if obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported MotrixSim LIBERO obs_type '{obs_type}'.")

        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.task = task
        self.task_description = task_description or "pick up the cube and move it to the target area"
        self._max_episode_steps = max_episode_steps
        self._env = registry.make(
            "libero",
            "np",
            env_cfg_override={"max_episode_seconds": max_episode_steps / self.metadata["render_fps"]},
            num_envs=1,
        )
        self._state = None
        self._seeded = False
        self._last_info: dict[str, Any] = {}
        self._renderer = None
        self._allow_render_fallback = os.getenv("MOTRIX_ALLOW_RENDER_FALLBACK") == "1"
        self._logged_fallback_frame = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = self._make_observation_space()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None and not self._seeded:
            self._env.seed(seed)
            self._seeded = True
        self._state = self._env.init_state()
        obs = self._format_observation()
        self._last_info = self._format_info(self._state)
        return obs, self._last_info

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float32).reshape(1, 7)
        self._state = self._env.step(action)
        obs = self._format_observation()
        reward = float(np.asarray(self._state.reward)[0])
        terminated = bool(np.asarray(self._state.terminated)[0])
        truncated = bool(np.asarray(self._state.truncated)[0])
        self._last_info = self._format_info(self._state)
        return obs, reward, terminated, truncated, self._last_info

    def render(self):
        if self._state is None:
            raise RuntimeError("Call reset() before render().")
        frames = self._render_camera_frames()
        return frames["agentview"]

    def close(self):
        return None

    def _make_observation_space(self):
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        if self.obs_type == "pixels":
            return spaces.Dict({"pixels": spaces.Dict({"image": image_space, "image2": image_space})})
        return spaces.Dict(
            {
                "pixels": spaces.Dict({"image": image_space, "image2": image_space}),
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict(
                            {
                                "pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                                "quat": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                                "mat": spaces.Box(-np.inf, np.inf, shape=(3, 3), dtype=np.float32),
                            }
                        ),
                        "gripper": spaces.Dict(
                            {
                                "qpos": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                                "qvel": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                            }
                        ),
                        "joints": spaces.Dict(
                            {
                                "pos": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                                "vel": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                            }
                        ),
                    }
                ),
            }
        )

    def _format_observation(self):
        frames = self._render_camera_frames()
        if frames is None:
            frame = self._synthetic_camera_frame()
            frames = {"agentview": frame, EYE_IN_HAND_CAMERA: self._make_wrist_like_frame(frame)}
        obs: dict[str, Any] = {"pixels": {"image": frames["agentview"], "image2": frames[EYE_IN_HAND_CAMERA]}}
        if self.obs_type == "pixels_agent_pos":
            obs["robot_state"] = self._format_robot_state()
        return obs

    def _format_info(self, state):
        success = state.info.get("is_success", np.asarray([False]))
        return {"is_success": bool(np.asarray(success)[0]), "task": self.task}

    def _render_camera_frames(self):
        if self._renderer is None:
            self._renderer = _MotrixLiberoCameraRenderer(
                self._env,
                width=self.observation_width,
                height=self.observation_height,
                camera_names=("agentview", EYE_IN_HAND_CAMERA),
                required=not self._allow_render_fallback,
            )
        frames = self._renderer.render()
        if frames is not None:
            self._renderer.dump_first_frames(frames)
            return frames
        if not self._allow_render_fallback:
            raise RuntimeError(
                "MotrixSim LIBERO camera did not return real frames. "
                "Check DISPLAY/WAYLAND_DISPLAY, Vulkan/OpenGL availability, and camera render targets. "
                "Set MOTRIX_ALLOW_RENDER_FALLBACK=1 only for smoke tests."
            )
        if not self._logged_fallback_frame:
            LOGGER.warning(
                "MotrixSim LIBERO camera source: fallback frames (size=%dx%d)",
                self.observation_width,
                self.observation_height,
            )
            self._logged_fallback_frame = True
        return None

    def _format_robot_state(self):
        state = self._env.get_robot_state(self._state.data)
        return {
            "eef": {
                "pos": state["eef_pos"][0].astype(np.float32),
                "quat": state["eef_quat"][0].astype(np.float32),
                "mat": state["eef_mat"][0].astype(np.float32),
            },
            "gripper": {
                "qpos": state["gripper_qpos"][0].astype(np.float32),
                "qvel": state["gripper_qvel"][0].astype(np.float32),
            },
            "joints": {
                "pos": state["joint_pos"][0].astype(np.float32),
                "vel": state["joint_vel"][0].astype(np.float32),
            },
        }

    def _synthetic_camera_frame(self):
        frame = np.full((self.observation_height, self.observation_width, 3), 228, dtype=np.uint8)
        table_y0 = int(self.observation_height * 0.08)
        table_y1 = int(self.observation_height * 0.94)
        frame[table_y0:table_y1, :, :] = np.array([178, 169, 150], dtype=np.uint8)

        for name, pos in self._env.get_scene_object_poses().items():
            self._draw_scene_object(frame, name, pos)

        target_pos = self._env.get_target_object_pos(self._state.data)[0]
        eef_pos = self._env.get_eef_pos(self._state.data)[0]
        self._draw_disk(frame, self._world_to_pixel(target_pos[:2]), radius=13, color=(8, 7, 6))
        self._draw_disk(frame, self._world_to_pixel(eef_pos[:2]), radius=6, color=(48, 88, 170))
        return frame

    def _world_to_pixel(self, xy: np.ndarray):
        x, y = float(xy[0]), float(xy[1])
        px = int(np.interp(x, [-0.2, 0.55], [0, self.observation_width - 1]))
        py = int(np.interp(y, [-0.38, 0.38], [self.observation_height - 1, 0]))
        return px, py

    def _make_wrist_like_frame(self, frame: np.ndarray):
        return np.ascontiguousarray(np.roll(frame, shift=max(1, frame.shape[1] // 12), axis=1))

    @staticmethod
    def _draw_square(frame: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]):
        cx, cy = center
        half = size // 2
        y0 = max(0, cy - half)
        y1 = min(frame.shape[0], cy + half)
        x0 = max(0, cx - half)
        x1 = min(frame.shape[1], cx + half)
        frame[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)

    def _draw_scene_object(self, frame: np.ndarray, name: str, pos: np.ndarray):
        px = self._world_to_pixel(pos[:2])
        if name == "plate":
            self._draw_disk(frame, px, radius=18, color=(232, 230, 210))
            self._draw_disk(frame, px, radius=12, color=(190, 184, 165))
        elif name == "ramekin":
            self._draw_disk(frame, px, radius=14, color=(78, 110, 190))
        elif name == "cookies":
            self._draw_rect(frame, px, size=(24, 16), color=(224, 164, 58))
        elif name == "distractor_bowl":
            self._draw_disk(frame, px, radius=12, color=(8, 7, 6))
        elif name == "wooden_cabinet":
            self._draw_rect(frame, px, size=(42, 28), color=(118, 70, 36))
        elif name == "flat_stove":
            self._draw_rect(frame, px, size=(38, 26), color=(36, 36, 34))

    @staticmethod
    def _draw_disk(frame: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]):
        try:
            import cv2

            cv2.circle(frame, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        except Exception:
            MotrixLiberoGymEnv._draw_square(frame, center, radius * 2, color)

    @staticmethod
    def _draw_rect(frame: np.ndarray, center: tuple[int, int], size: tuple[int, int], color: tuple[int, int, int]):
        cx, cy = center
        half_w = size[0] // 2
        half_h = size[1] // 2
        y0 = max(0, cy - half_h)
        y1 = min(frame.shape[0], cy + half_h)
        x0 = max(0, cx - half_w)
        x1 = min(frame.shape[1], cx + half_w)
        frame[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)
