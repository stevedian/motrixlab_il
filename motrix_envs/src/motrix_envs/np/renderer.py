# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import logging
import numpy as np
from motrix_envs.np.env import NpEnv

# Force X11 backend on Wayland+XWayland systems: wgpu/Wayland has known
# compatibility issues with NVIDIA drivers that cause null-pointer dereference
# crashes in Vulkan surface creation. When XWayland is available (DISPLAY is
# set), we prefer it over native Wayland.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY")
    os.environ.setdefault("WINIT_UNIX_BACKEND", "x11")

try:
    from motrixsim.render import RenderApp, RenderSettings
except Exception:
    RenderApp = None
    RenderSettings = None
    logging.getLogger(__name__).warning("motrixsim.render is unavailable; rendering disabled")

_native_render_checked = False
_native_render_ok = False


def _check_native_render():
    """Test whether RenderApp can be created without crashing (SIGSEGV).

    Runs a subprocess to safely probe for native library crashes that Python
    cannot catch (e.g. null-pointer dereference in C/Rust extensions).
    The result is cached so the check only runs once.
    """
    global _native_render_checked, _native_render_ok
    if _native_render_checked:
        return _native_render_ok

    _native_render_checked = True

    if RenderApp is None or os.getenv("MOTRIX_DISABLE_RENDER") == "1":
        _native_render_ok = False
        return False

    import subprocess
    import sys

    check_script = (
        "import os; "
        "os.environ.pop('WAYLAND_DISPLAY', None); "
        "os.environ.setdefault('WINIT_UNIX_BACKEND', 'x11'); "
        "from motrixsim.render import RenderApp; "
        "RenderApp(); "
        "os._exit(0)"  # skip Rust destructor cleanup (known to crash on shutdown)
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True,
            timeout=15,
        )
        _native_render_ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        logging.getLogger(__name__).warning("Native renderer probe timed out; using headless dummy renderer")
        _native_render_ok = False
    except Exception as e:
        logging.getLogger(__name__).warning("Native renderer probe failed: %s; using headless dummy renderer", e)
        _native_render_ok = False

    return _native_render_ok


class _DummyRender:
    class _DummyInput:
        def is_key_just_pressed(self, _):
            return False

    class _DummyCamera:
        def __init__(self):
            self.active = False

    def __init__(self):
        self.input = _DummyRender._DummyInput()
        self.system_camera = _DummyRender._DummyCamera()

    def launch(self, *args, **kwargs):
        return None

    def sync(self, *args, **kwargs):
        return None


class NpRenderer:
    """
    The renderer for Np sim environments.
    """

    _env: NpEnv

    def __init__(self, env: NpEnv):
        num_envs = env.num_envs
        num_envs = 1 if num_envs is None else num_envs
        spacing = env.render_spacing
        cols = int(np.ceil(np.sqrt(num_envs)))
        offsets = []
        for i in range(num_envs):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])

        self._env = env

        disable_render = os.getenv("MOTRIX_DISABLE_RENDER", "0") == "1"

        if disable_render or RenderApp is None:
            logging.getLogger(__name__).warning("Native renderer disabled; using headless dummy renderer")
            self._render = _DummyRender()
            self._sync_render_data = False
            self._render.system_camera.active = self._sync_render_data
            return

        if not _check_native_render():
            logging.getLogger(__name__).warning("Native renderer unavailable; using headless dummy renderer")
            self._render = _DummyRender()
            self._sync_render_data = False
            self._render.system_camera.active = self._sync_render_data
            return

        try:
            for camera in env.model.cameras.tolist():
                camera.set_near_far(0.01, 10.0)
            self._render = RenderApp()
            settings = RenderSettings.quality()
            self._render.launch(
                env.model,
                batch=num_envs,
                render_offset=offsets,
                render_settings=settings,
            )
            self._sync_render_data = True
            self._render.system_camera.active = self._sync_render_data
        except Exception as e:
            logging.getLogger(__name__).exception("Failed to initialize native renderer, falling back to dummy: %s", e)
            self._render = _DummyRender()
            self._sync_render_data = False
            self._render.system_camera.active = self._sync_render_data

    def render(self) -> None:
        """
        render the env
        """

        self._render.sync(data=self._env.state.data if self._sync_render_data else None)
        if self._render.input.is_key_just_pressed("space"):
            self._sync_render_data = not self._sync_render_data
            self._render.system_camera.active = self._sync_render_data
