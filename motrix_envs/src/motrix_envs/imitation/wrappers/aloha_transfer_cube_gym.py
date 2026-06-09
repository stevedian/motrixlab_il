from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from motrix_envs import registry

LOGGER = logging.getLogger(__name__)
_RENDER_LAUNCH_PROBE_CACHE: dict[tuple[str, str, int, int], bool] = {}


def _probe_render_launch(model_file: str, camera_name: str, width: int, height: int):
    key = (model_file, camera_name, width, height)
    if key in _RENDER_LAUNCH_PROBE_CACHE:
        return _RENDER_LAUNCH_PROBE_CACHE[key]

    script = f"""
import os
os.environ.setdefault("WINIT_UNIX_BACKEND", "x11")
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
import motrixsim as mtx
from motrixsim.render import RenderApp, RenderSettings
model = mtx.load_model({model_file!r})
camera = model.cameras[{camera_name!r}]
camera.set_render_target("image", {width}, {height})
app = RenderApp()
app.launch(model, batch=1, render_settings=RenderSettings.performance())
os._exit(0)
"""
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15)
        ok = result.returncode == 0
    except Exception:
        ok = False
    _RENDER_LAUNCH_PROBE_CACHE[key] = ok
    return ok


class _MotrixTopCameraRenderer:
    def __init__(self, env, width: int, height: int, camera_name: str = "top", required: bool = True):
        self._env = env
        self._width = width
        self._height = height
        self._camera_name = camera_name
        self._required = required
        self._app = None
        self._camera = None
        self._enabled = False
        self._warned = False
        self._logged_first_real_frame = False
        self._dumped_first_real_frame = False
        self._init_renderer()

    @property
    def enabled(self):
        return self._enabled

    def render(self):
        if not self._enabled:
            return None

        try:
            self._app.sync(data=self._env.state.data, wait=True)
            task = self._camera.capture()
            for _ in range(8):
                image = task.take_image()
                if image is not None:
                    pixels = np.asarray(image.pixels)
                    frame = self._coerce_frame(pixels)
                    if frame.max() < 8 or int(frame.max()) - int(frame.min()) < 20:
                        self._app.sync(data=self._env.state.data, wait=True)
                        time.sleep(0.01)
                        continue
                    frame = self._blacken_connected_background(frame)
                    if not self._logged_first_real_frame:
                        camera_cfg = self._env.model.cameras[self._camera_name]
                        LOGGER.warning(
                            (
                                "MotrixSim ALOHA camera source: real renderer "
                                "(camera=%s, index=%s, size=%dx%d, fovy=%.2f, target=%s, "
                                "near=%.3f, far=%.3f, pixel_mean=%.2f, pixel_min=%d, pixel_max=%d)"
                            ),
                            self._camera_name,
                            camera_cfg.index,
                            self._width,
                            self._height,
                            float(camera_cfg.fovy),
                            camera_cfg.render_target,
                            float(camera_cfg.near_plane),
                            float(camera_cfg.far_plane),
                            float(frame.mean()),
                            int(frame.min()),
                            int(frame.max()),
                        )
                        self._logged_first_real_frame = True
                    if not self._dumped_first_real_frame:
                        self._dump_debug_frame(frame)
                        self._dumped_first_real_frame = True
                    return frame
                self._app.sync(data=self._env.state.data, wait=True)
                time.sleep(0.001)
        except Exception as exc:
            self._enabled = False
            if not self._warned:
                LOGGER.warning("MotrixSim camera capture failed; using fallback frames: %s", exc)
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
            if not _probe_render_launch(
                self._env.cfg.model_file,
                self._camera_name,
                self._width,
                self._height,
            ):
                self._handle_unavailable("MotrixSim top camera render-target launch probe failed")
                return

            camera = self._env.model.cameras[self._camera_name]
            camera.set_near_far(0.01, 10.0)
            camera.set_render_target("image", self._width, self._height)
            app = RenderApp()
            settings = RenderSettings.quality()
            app.launch(self._env.model, batch=1, render_settings=settings)
            if self._env.state is not None:
                for _ in range(5):
                    app.sync(data=self._env.state.data, wait=True)
                    time.sleep(0.02)
            self._app = app
            self._camera = app.get_camera(camera.index)
            self._enabled = True
            LOGGER.warning(
                "MotrixSim ALOHA camera renderer initialized successfully (camera=%s, size=%dx%d)",
                self._camera_name,
                self._width,
                self._height,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            self._enabled = False
            self._handle_unavailable(str(exc))

    def _handle_unavailable(self, reason: str):
        if self._required:
            raise RuntimeError(
                "MotrixSim top camera rendering is required but unavailable: "
                f"{reason}. Start a graphical session or a virtual X server, or set "
                "MOTRIX_ALLOW_RENDER_FALLBACK=1 only for smoke tests."
            )
        LOGGER.warning("MotrixSim renderer unavailable; using fallback frames: %s", reason)

    def _coerce_frame(self, pixels: np.ndarray):
        pixels = np.asarray(pixels)
        if pixels.ndim == 4:
            pixels = pixels[0]
        if pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        if pixels.shape[:2] != (self._height, self._width):
            pixels = np.resize(pixels, (self._height, self._width, 3))
        return np.ascontiguousarray(pixels, dtype=np.uint8)

    @staticmethod
    def _blacken_connected_background(frame: np.ndarray):
        try:
            import cv2
        except Exception:
            return frame

        height, width = frame.shape[:2]
        border = np.concatenate(
            [
                frame[0, :, :],
                frame[-1, :, :],
                frame[:, 0, :],
                frame[:, -1, :],
            ],
            axis=0,
        )
        bg_color = np.median(border, axis=0)
        tolerance = 16
        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        work = frame.copy()
        flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo_diff = (tolerance, tolerance, tolerance)
        up_diff = (tolerance, tolerance, tolerance)

        seeds = []
        step = max(1, min(width, height) // 24)
        for x in range(0, width, step):
            seeds.append((x, 0))
            seeds.append((x, height - 1))
        for y in range(0, height, step):
            seeds.append((0, y))
            seeds.append((width - 1, y))

        for seed in seeds:
            x, y = seed
            if mask[y + 1, x + 1]:
                continue
            if np.max(np.abs(frame[y, x].astype(np.float32) - bg_color)) > tolerance:
                continue
            cv2.floodFill(work, mask, seed, (0, 0, 0), lo_diff, up_diff, flags)

        background = mask[1:-1, 1:-1] != 0
        if background.mean() > 0.8:
            return frame
        frame[background] = 0
        return frame

    def _dump_debug_frame(self, frame: np.ndarray):
        try:
            from PIL import Image

            debug_dir = Path("outputs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_path = debug_dir / "motrixsim_aloha_top_first_frame.png"
            Image.fromarray(frame).save(out_path)
            LOGGER.warning("MotrixSim ALOHA first real camera frame saved to %s", out_path)
        except Exception as exc:
            LOGGER.warning("Failed to save first real MotrixSim camera frame: %s", exc)


class MotrixAlohaTransferCubeGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 400,
        observation_height: int = 480,
        observation_width: int = 640,
    ):
        super().__init__()
        if obs_type not in {"pixels", "pixels_agent_pos"}:
            raise ValueError(f"Unsupported ALOHA obs_type '{obs_type}'.")

        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._max_episode_steps = max_episode_steps
        self._env = registry.make(
            "aloha-transfer-cube",
            "np",
            env_cfg_override={"max_episode_seconds": max_episode_steps / self.metadata["render_fps"]},
            num_envs=1,
        )
        self._state = None
        self._last_info: dict[str, Any] = {}
        self._seeded = False
        self._allow_render_fallback = os.getenv("MOTRIX_ALLOW_RENDER_FALLBACK") == "1"
        self._logged_fallback_frame = False
        self._renderer = None

        self.action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.observation_height, self.observation_width, 3),
            dtype=np.uint8,
        )
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict({"top": image_space})})
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
                    "pixels": spaces.Dict({"top": image_space}),
                }
            )

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
        action = np.asarray(action, dtype=np.float32).reshape(1, 14)
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
        self._ensure_renderer()
        frame = self._renderer.render()
        if frame is not None:
            return self._add_top_contact_shadow(frame)
        if not self._allow_render_fallback:
            raise RuntimeError(
                "MotrixSim top camera did not return a frame. Set MOTRIX_ALLOW_RENDER_FALLBACK=1 "
                "only if you intentionally want non-camera fallback frames."
            )
        if not self._logged_fallback_frame:
            LOGGER.warning(
                "MotrixSim ALOHA camera source: fallback frame (camera=%s, size=%dx%d)",
                "top",
                self.observation_width,
                self.observation_height,
            )
            self._logged_fallback_frame = True
        return self._add_top_contact_shadow(self._fallback_top_frame())

    def close(self):
        return None

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = _MotrixTopCameraRenderer(
                self._env,
                width=self.observation_width,
                height=self.observation_height,
                camera_name="top",
                required=not self._allow_render_fallback,
            )

    def _format_observation(self):
        if self._state is None:
            raise RuntimeError("Call reset() before requesting an observation.")
        obs: dict[str, Any] = {"pixels": {"top": self.render()}}
        if self.obs_type == "pixels_agent_pos":
            obs["agent_pos"] = np.asarray(self._state.obs[0], dtype=np.float32)
        return obs

    @staticmethod
    def _format_info(state):
        success = state.info.get("is_success", np.asarray([False]))
        return {"is_success": bool(np.asarray(success)[0])}

    def _fallback_top_frame(self):
        frame = np.full((self.observation_height, self.observation_width, 3), 235, dtype=np.uint8)
        table_y0 = int(self.observation_height * 0.18)
        table_y1 = int(self.observation_height * 0.86)
        frame[table_y0:table_y1, :, :] = np.array([200, 204, 198], dtype=np.uint8)

        qpos = np.asarray(self._state.data.dof_pos[0], dtype=np.float32)
        cube_xy = qpos[-7:-5]
        cube_px = self._world_to_top_pixel(cube_xy)
        self._draw_square(frame, cube_px, size=18, color=(196, 48, 42))

        left_hint = np.array([-0.25 + 0.08 * np.sin(qpos[0]), 0.52 + 0.08 * np.cos(qpos[1])])
        right_hint = np.array([0.25 + 0.08 * np.sin(qpos[8]), 0.52 + 0.08 * np.cos(qpos[9])])
        self._draw_square(frame, self._world_to_top_pixel(left_hint), size=12, color=(48, 88, 170))
        self._draw_square(frame, self._world_to_top_pixel(right_hint), size=12, color=(42, 132, 86))
        return frame

    def _add_top_contact_shadow(self, frame: np.ndarray):
        try:
            import cv2
        except Exception:
            return frame

        qpos = np.asarray(self._state.data.dof_pos[0], dtype=np.float32)
        cube_xy = qpos[-7:-5]
        cx, cy = self._world_to_top_pixel(cube_xy)
        offset_x = max(2, int(self.observation_width * 0.01))
        offset_y = max(2, int(self.observation_height * 0.012))
        half_w = max(5, int(self.observation_width * 0.018))
        half_h = max(3, int(self.observation_height * 0.011))
        center = np.array([cx - offset_x, cy + offset_y], dtype=np.float32)
        corners = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )
        angle = -0.35
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        pts = np.rint(corners @ rot.T + center).astype(np.int32)

        shadow = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(shadow, pts, 150)
        table_mask = np.logical_and.reduce(
            (
                frame[:, :, 0] > 45,
                frame[:, :, 0] < 180,
                frame[:, :, 1] > 45,
                frame[:, :, 1] < 180,
                frame[:, :, 2] > 45,
                frame[:, :, 2] < 180,
            )
        )
        alpha = (shadow.astype(np.float32) / 255.0) * table_mask.astype(np.float32) * 0.78
        shaded = frame.astype(np.float32) * (1.0 - alpha[:, :, None])
        return np.ascontiguousarray(np.clip(shaded, 0, 255).astype(np.uint8))

    def _world_to_top_pixel(self, xy: np.ndarray):
        x, y = float(xy[0]), float(xy[1])
        px = int(np.interp(x, [-0.35, 0.35], [0, self.observation_width - 1]))
        py = int(np.interp(y, [0.25, 0.75], [self.observation_height - 1, 0]))
        return px, py

    @staticmethod
    def _draw_square(frame: np.ndarray, center: tuple[int, int], size: int, color: tuple[int, int, int]):
        cx, cy = center
        half = size // 2
        y0 = max(0, cy - half)
        y1 = min(frame.shape[0], cy + half)
        x0 = max(0, cx - half)
        x1 = min(frame.shape[1], cx + half)
        frame[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)
