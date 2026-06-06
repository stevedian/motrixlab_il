import time
import pickle
import random
from pathlib import Path

import click
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


HELP_TEXT = """
Keyboard control (click the preview window first):
  w/s : +x / -x
  a/d : +y / -y
  e/q : +z / -z
  i/k : +rx / -rx
  j/l : +ry / -ry
  u/o : +rz / -rz
  m/n : open / close gripper

Episode control:
  c : save current episode and continue to next
  r : discard current episode and reset
  x : exit program
  h : print this help
"""

MOVE_KEYS = "wasdeqijkluomn"
CMD_KEYS = "crxh"


def _parse_csv_names(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_plate_layout(raw: str):
    """
    Parse layout string like: "0.20,0.76;0.50,0.76;0.80,0.76"
    to [(0.20, 0.76), (0.50, 0.76), (0.80, 0.76)].
    """
    points = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = [x.strip() for x in chunk.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Invalid plate layout chunk: {chunk}")
        x = float(xy[0])
        y = float(xy[1])
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            raise ValueError(f"Plate layout coordinates must be in [0,1], got {(x, y)}")
        points.append((x, y))
    return points


def _parse_world_xy_layout(raw: str):
    """
    Parse world-xy layout string like: "0.10,-0.20;0.10,0.00;0.10,0.20"
    to [(0.10, -0.20), (0.10, 0.00), (0.10, 0.20)].
    """
    points = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = [x.strip() for x in chunk.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Invalid world layout chunk: {chunk}")
        points.append((float(xy[0]), float(xy[1])))
    return points


def _parse_rgb3(raw: str):
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 3:
        raise ValueError(f"RGB must have 3 values, got: {raw}")
    vals = [max(0.0, min(1.0, v)) for v in vals]
    return tuple(vals)


def _make_env_robosuite(
    env_name: str,
    robot: str,
    camera_names,
    image_size: int,
    control_hz: int,
    max_steps: int,
    extra_env_kwargs=None,
):
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller_cfg = load_controller_config(default_controller="OSC_POSE")
    controller_cfg["control_delta"] = False

    make_kwargs = dict(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs=True,
        use_camera_obs=True,
        control_freq=control_hz,
        horizon=max_steps,
        ignore_done=True,
        camera_names=list(camera_names),
        camera_heights=image_size,
        camera_widths=image_size,
        reward_shaping=False,
    )
    if extra_env_kwargs:
        make_kwargs.update(extra_env_kwargs)
    env = suite.make(**make_kwargs)
    #env.sim.model.arena.set_camera("frontview", pos=[1.2, 0.0, 1.6], quat=[0.56, 0.43, 0.43, 0.56])
    return env


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise RuntimeError(f"Invalid image rank {arr.ndim}, expect 3.")
    # CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _center_zoom(img: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return img
    h, w = img.shape[:2]
    crop_h = max(1, int(round(h / zoom)))
    crop_w = max(1, int(round(w / zoom)))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    crop = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _draw_plate_markers(img: np.ndarray, points_norm, radius_px=22) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x_norm, y_norm in points_norm:
        cx = int(round(x_norm * (w - 1)))
        cy = int(round(y_norm * (h - 1)))
        # plate outer ring
        cv2.circle(out, (cx, cy), radius_px, (220, 220, 220), 2, lineType=cv2.LINE_AA)
        # plate inner ring
        cv2.circle(out, (cx, cy), int(radius_px * 0.62), (240, 240, 240), 1, lineType=cv2.LINE_AA)
    return out


def _find_table_geom_name(env):
    names = [str(x) for x in list(env.sim.model.geom_names)]
    for key in ("table_visual", "table_top", "table"):
        for n in names:
            if key in n.lower():
                return n
    return None


def _draw_circles_on_table_texture(
    env,
    circle_world_xy,
    ring_radius_m=0.055,
    ring_width_m=0.008,
    ring_rgb=(0.95, 0.95, 0.95),
):
    """
    Draw circles directly on table texture (world-attached), not image overlay.
    """
    try:
        from robosuite.utils.mjmod import TextureModder
    except Exception as e:
        print(f"[Warn] cannot import TextureModder, skip table circles: {e}")
        return False

    table_geom = _find_table_geom_name(env)
    if table_geom is None:
        print("[Warn] cannot find table geom name, skip table circles.")
        return False

    try:
        modder = TextureModder(env.sim)
        modder.whiten_materials()
        tex = modder.get_texture(table_geom)
        bitmap = tex.bitmap.copy()
    except Exception as e:
        print(f"[Warn] texture access failed for geom={table_geom}: {e}")
        return False

    if bitmap.dtype != np.uint8:
        bitmap = np.clip(bitmap, 0, 255).astype(np.uint8)
    h, w = bitmap.shape[:2]

    table_offset = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    table_full_size = np.array([0.8, 0.8, 0.05], dtype=np.float64)
    if hasattr(env, "table_offset"):
        table_offset = np.asarray(env.table_offset, dtype=np.float64)
    if hasattr(env, "table_full_size"):
        table_full_size = np.asarray(env.table_full_size, dtype=np.float64)
    cx, cy = float(table_offset[0]), float(table_offset[1])
    sx, sy = float(table_full_size[0]), float(table_full_size[1])

    color = tuple(int(round(c * 255)) for c in ring_rgb)
    for xw, yw in circle_world_xy:
        u = (xw - cx) / max(1e-6, sx) + 0.5
        v = (yw - cy) / max(1e-6, sy) + 0.5
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        px = int(round(u * (w - 1)))
        py = int(round((1.0 - v) * (h - 1)))
        r_px = max(3, int(round((ring_radius_m / max(1e-6, sx)) * w)))
        t_px = max(1, int(round((ring_width_m / max(1e-6, sx)) * w)))
        cv2.circle(bitmap, (px, py), r_px, color, thickness=t_px, lineType=cv2.LINE_AA)

    try:
        modder.set_texture(table_geom, bitmap, perturb=False)
        modder.upload_texture(table_geom)
        print(f"[Scene] drew circles on table texture geom={table_geom}")
        return True
    except Exception as e:
        print(f"[Warn] upload table texture failed: {e}")
        return False


def _set_cube_colors_from_geom_name(env, cube_name: str, rgb):
    rgba = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
    names = [str(x) for x in list(env.sim.model.geom_names)]
    changed = 0
    for i, name in enumerate(names):
        if cube_name.lower() in name.lower():
            env.sim.model.geom_rgba[i, :4] = rgba
            changed += 1
    return changed


def _apply_two_cube_colors(env, cube_a_rgb=(0.90, 0.25, 0.25), cube_b_rgb=(0.25, 0.55, 0.95)):
    c1 = _set_cube_colors_from_geom_name(env, "cubeA", cube_a_rgb)
    c2 = _set_cube_colors_from_geom_name(env, "cubeB", cube_b_rgb)
    _refresh_sim(env)
    print(f"[Scene] recolor cubes: cubeA_geoms={c1}, cubeB_geoms={c2}")


def _find_object_joint_name(env, obj_name: str):
    target = obj_name.lower()
    # Best path: read robosuite object metadata
    if hasattr(env, "objects"):
        try:
            for obj in env.objects:
                name = str(getattr(obj, "name", ""))
                if name.lower() == target:
                    joints = getattr(obj, "joints", None)
                    if joints and len(joints) > 0:
                        return joints[0]
        except Exception:
            pass

    # Fallback: search mujoco model joint names
    try:
        joint_names = list(env.sim.model.joint_names)
    except Exception:
        return None
    candidates = [jn for jn in joint_names if target in str(jn).lower()]
    if len(candidates) > 0:
        return candidates[0]
    return None


def _joint_qpos_slice(model, joint_name: str):
    # mujoco-py convenience API
    if hasattr(model, "get_joint_qpos_addr"):
        addr = model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            return slice(int(addr[0]), int(addr[1]))
        if isinstance(addr, list) and len(addr) > 0:
            return slice(int(addr[0]), int(addr[-1]) + 1)
        if isinstance(addr, np.ndarray) and addr.size > 0:
            return slice(int(addr[0]), int(addr[-1]) + 1)
        if isinstance(addr, (int, np.integer)):
            start = int(addr)
            jid = model.joint_name2id(joint_name)
            jtype = int(model.jnt_type[jid])
            qdim = {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)
            return slice(start, start + qdim)

    # generic fallback
    jid = model.joint_name2id(joint_name)
    start = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])
    qdim = {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)
    return slice(start, start + qdim)


def _refresh_sim(env):
    try:
        env.sim.forward()
    except Exception:
        pass


def _place_two_objects_in_world_circles(
    env,
    keep_names,
    circle_world_xy,
    xy_jitter=0.0,
    random_yaw=True,
):
    """
    Place the two kept objects onto first two world-circle centers.
    """
    if len(keep_names) < 2 or len(circle_world_xy) < 2:
        return

    model = env.sim.model
    data = env.sim.data
    rng = np.random.default_rng()

    for i, obj_name in enumerate(keep_names[:2]):
        joint_name = _find_object_joint_name(env, obj_name=obj_name)
        if joint_name is None:
            print(f"[Warn] cannot find joint for object {obj_name}, skip placement.")
            continue
        qslice = _joint_qpos_slice(model, joint_name)
        q = data.qpos[qslice].copy()
        if q.shape[0] < 7:
            print(f"[Warn] joint {joint_name} is not free-joint-like (qdim={q.shape[0]}), skip.")
            continue

        x, y = circle_world_xy[i]
        if xy_jitter > 0.0:
            x += rng.uniform(-xy_jitter, xy_jitter)
            y += rng.uniform(-xy_jitter, xy_jitter)
        q[0] = x
        q[1] = y

        if random_yaw:
            # MuJoCo qpos quaternion is wxyz
            q_wxyz = q[3:7].copy()
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
            yaw = rng.uniform(-np.pi, np.pi)
            q_delta = R.from_euler("z", yaw).as_quat()  # xyzw
            q_new_xyzw = (R.from_quat(q_delta) * R.from_quat(q_xyzw)).as_quat()
            q[3:7] = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=np.float64)

        data.qpos[qslice] = q

    _refresh_sim(env)


def _infer_action_dim(env) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "action_spec"):
        low, _ = env.action_spec
        return int(low.shape[0])
    raise RuntimeError("Cannot infer env action dimension.")


def _check_obs_keys(obs, keys):
    missing = [k for k in keys if k not in obs]
    if missing:
        available = sorted(list(obs.keys()))
        raise KeyError(f"Missing obs keys {missing}. Available keys: {available}")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _write_mp4(frames, out_path: Path, fps: int):
    if len(frames) == 0:
        raise RuntimeError(f"No frames to write: {out_path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _update_target_from_key(
    key: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    target_grasp: float,
    pos_step: float,
    rot_step_deg: float,
    gripper_step: float,
):
    dpos = np.zeros(3, dtype=np.float64)
    drot = np.zeros(3, dtype=np.float64)
    if key == ord("w"):
        dpos[0] += pos_step
    elif key == ord("s"):
        dpos[0] -= pos_step
    elif key == ord("a"):
        dpos[1] += pos_step
    elif key == ord("d"):
        dpos[1] -= pos_step
    elif key == ord("e"):
        dpos[2] += pos_step
    elif key == ord("q"):
        dpos[2] -= pos_step
    elif key == ord("i"):
        drot[0] += rot_step_deg
    elif key == ord("k"):
        drot[0] -= rot_step_deg
    elif key == ord("j"):
        drot[1] += rot_step_deg
    elif key == ord("l"):
        drot[1] -= rot_step_deg
    elif key == ord("u"):
        drot[2] += rot_step_deg
    elif key == ord("o"):
        drot[2] -= rot_step_deg
    elif key == ord("m"):
        target_grasp += gripper_step
    elif key == ord("n"):
        target_grasp -= gripper_step

    if np.any(dpos):
        target_pos = target_pos + dpos
    if np.any(drot):
        dq = R.from_euler("xyz", drot, degrees=True).as_quat()
        target_quat = (R.from_quat(dq) * R.from_quat(target_quat)).as_quat()
    target_grasp = float(np.clip(target_grasp, -1.0, 1.0))
    return target_pos, target_quat, target_grasp


def _init_viewer(mode: str, width: int, height: int, scale: int):
    disp_w = int(width * scale)
    disp_h = int(height * scale)
    viewer = {
        "mode": mode,
        "title": "collect_robot_swap_sim",
        "scale": scale,
        "disp_w": disp_w,
        "disp_h": disp_h,
    }
    if mode == "pygame":
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError(
                "viewer=pygame 需要安装 pygame。"
                "你可以先 `pip install pygame`，或者临时用 `--viewer none`。"
            ) from e
        pygame.init()
        screen = pygame.display.set_mode((disp_w, disp_h))
        pygame.display.set_caption(viewer["title"])
        viewer["pygame"] = pygame
        viewer["screen"] = screen
    elif mode not in ("opencv", "none"):
        raise ValueError(f"Unsupported viewer mode: {mode}")
    return viewer


def _render_viewer(viewer, vis_rgb: np.ndarray):
    mode = viewer["mode"]
    if viewer["scale"] > 1:
        vis_rgb = cv2.resize(
            vis_rgb,
            (viewer["disp_w"], viewer["disp_h"]),
            interpolation=cv2.INTER_NEAREST,
        )
    if mode == "pygame":
        pygame = viewer["pygame"]
        screen = viewer["screen"]
        # pygame expects array shape (W, H, 3)
        surf = pygame.surfarray.make_surface(np.transpose(vis_rgb, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
    elif mode == "opencv":
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(viewer["title"], vis_bgr)


def _poll_keys(viewer):
    commands = set()
    move_keys = set()
    mode = viewer["mode"]

    if mode == "pygame":
        pygame = viewer["pygame"]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                commands.add("x")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    commands.add("c")
                elif event.key == pygame.K_r:
                    commands.add("r")
                elif event.key == pygame.K_x:
                    commands.add("x")
                elif event.key == pygame.K_h:
                    commands.add("h")

        pressed = pygame.key.get_pressed()
        keymap = [
            (pygame.K_w, "w"),
            (pygame.K_a, "a"),
            (pygame.K_s, "s"),
            (pygame.K_d, "d"),
            (pygame.K_q, "q"),
            (pygame.K_e, "e"),
            (pygame.K_i, "i"),
            (pygame.K_j, "j"),
            (pygame.K_k, "k"),
            (pygame.K_l, "l"),
            (pygame.K_u, "u"),
            (pygame.K_o, "o"),
            (pygame.K_m, "m"),
            (pygame.K_n, "n"),
        ]
        for kc, name in keymap:
            if pressed[kc]:
                move_keys.add(name)
    elif mode == "opencv":
        key = cv2.pollKey()
        if key != -1:
            ch = chr(key & 0xFF).lower()
            if ch in MOVE_KEYS:
                move_keys.add(ch)
            if ch in CMD_KEYS:
                commands.add(ch)
    return commands, move_keys


def _close_viewer(viewer):
    mode = viewer["mode"]
    if mode == "pygame":
        viewer["pygame"].quit()
    elif mode == "opencv":
        cv2.destroyAllWindows()


def _refresh_obs_after_scene_edit(env, fallback_obs):
    # Robosuite internals vary a bit by version; try common getters.
    for method_name, kwargs in [
        ("_get_observations", {}),
        ("_get_observations", {"force_update": True}),
        ("get_observation", {}),
    ]:
        if hasattr(env, method_name):
            method = getattr(env, method_name)
            try:
                return method(**kwargs)
            except TypeError:
                continue
            except Exception:
                continue
    return fallback_obs


def _apply_pickplace_object_filter(env, keep_names):
    """
    Keep only selected PickPlace objects by clearing the rest.
    """
    if not hasattr(env, "clear_objects"):
        print("[Warn] env has no clear_objects(); cannot filter PickPlace objects.")
        return

    if hasattr(env, "obj_names"):
        all_names = [str(x) for x in list(env.obj_names)]
    else:
        all_names = ["Milk", "Bread", "Cereal", "Can"]

    keep_set = {x.lower() for x in keep_names}
    remove_names = [name for name in all_names if name.lower() not in keep_set]
    if len(remove_names) == 0:
        return
    try:
        env.clear_objects(remove_names)
        print(f"[Scene] Keep objects={keep_names}, removed={remove_names}")
    except Exception as e:
        print(f"[Warn] clear_objects failed: {e}")


@click.command()
@click.option("--output_dir", "-o", required=True, help="Output directory, e.g. data/robot_swap")
@click.option(
    "--scene_preset",
    type=click.Choice(
        ["lift_single", "pickplace_multi", "nutassembly_twoobj", "swap_3plate_2obj", "table3circle_2cube"]
    ),
    default="table3circle_2cube",
    help="场景预设：lift单物体 / pickplace多物体 / nutassembly双物体 / 3盘子2物体交换 / 桌面3圆圈2方块",
)
@click.option("--env_name", default="Lift", help="Robosuite task name")
@click.option("--robot", default="Panda", help="Robot name")
@click.option("--n_episodes", default=50, type=int, help="Number of episodes to save")
@click.option("--max_steps", default=3000, type=int, help="Max steps per episode")
@click.option("--record_start_step", default=15, type=int, help="每回合前多少步不记录（用于避开初始抖动）")
@click.option("--control_hz", default=10, type=int, help="Control frequency")
@click.option("--image_size", default=256, type=int, help="Camera image size (采集/预览清晰度)")
@click.option("--camera0_name", default="frontview", help="Robosuite camera name for cam0 (推荐全局)")
@click.option("--camera1_name", default="robot0_eye_in_hand", help="Robosuite camera name for cam1")
@click.option("--observer_camera_name", default="", help="仅预览用第三相机名（不写入数据），留空关闭")
@click.option("--camera0_key", default="frontview_image", help="Observation key for cam0 image")
@click.option("--camera1_key", default="robot0_eye_in_hand_image", help="Observation key for cam1 image")
@click.option("--observer_camera_key", default="", help="仅预览用第三相机观测键（不写入数据），留空关闭")
@click.option(
    "--flip_camera0/--no-flip_camera0",
    default=True,
    help="是否上下翻转第一个相机（cam0）。当 cam0 倒置时请保持开启。",
)
@click.option(
    "--camera0_zoom",
    default=1.6,
    type=float,
    help="cam0 数字变焦倍数（>1 更近，1 表示不变焦）",
)
@click.option("--eef_pos_key", default="robot0_eef_pos", help="Observation key for eef position")
@click.option("--eef_quat_key", default="robot0_eef_quat", help="Observation key for eef quaternion")
@click.option("--gripper_key", default="robot0_gripper_qpos", help="Observation key for gripper state")
@click.option("--pos_step", default=0.01, type=float, help="Cartesian step size in meter")
@click.option("--rot_step_deg", default=4.0, type=float, help="Rotation step in degree")
@click.option("--gripper_step", default=0.2, type=float, help="Gripper command step")
@click.option(
    "--viewer",
    type=click.Choice(["pygame", "opencv", "none"]),
    default="pygame",
    help="Preview / keyboard backend. 推荐 pygame（避免 Qt xcb 报错）",
)
@click.option("--viewer_scale", default=4, type=int, help="仅预览窗口放大倍数，不影响保存分辨率")
@click.option(
    "--print_obs_keys/--no-print_obs_keys",
    default=True,
    help="启动后打印一次可用观测键（含所有 *_image 相机键）",
)
@click.option(
    "--swap_keep_objects",
    default="Milk,Bread",
    help="swap_3plate_2obj 预设时保留的两个物体（逗号分隔）",
)
@click.option(
    "--swap_plate_layout",
    default="0.20,0.76;0.50,0.76;0.80,0.76",
    help="swap_3plate_2obj 盘子位置(归一化坐标): x1,y1;x2,y2;x3,y3",
)
@click.option(
    "--swap_plate_radius",
    default=44,
    type=int,
    help="swap_3plate_2obj 盘子半径(像素)",
)
@click.option(
    "--swap_draw_plates/--no-swap_draw_plates",
    default=False,
    help="是否在 cam0 叠加三个盘子标记（会写入保存视频）",
)
@click.option(
    "--swap_world_layout",
    default="-0.15,-0.12;0.0,0.00;-0.15,0.12",
    help="swap_3plate_2obj 三个圆圈在世界坐标系的中心(x,y)，用于真实摆位",
)
@click.option(
    "--swap_place_objects/--no-swap_place_objects",
    default=True,
    help="swap_3plate_2obj 时是否把两个物体强制摆到前两个圆圈中心",
)
@click.option(
    "--swap_xy_jitter",
    default=0.0,
    type=float,
    help="物体摆位时在 x/y 上的随机扰动（米）",
)
@click.option("--circle_radius_m", default=0.055, type=float, help="桌面圆圈半径（米）")
@click.option("--circle_width_m", default=0.008, type=float, help="桌面圆圈线宽（米）")
@click.option("--circle_rgb", default="0.2,0.2,0.2", help="桌面圆圈颜色 rgb(0~1)")
@click.option("--cubea_rgb", default="0.90,0.25,0.25", help="方块A颜色 rgb(0~1)")
@click.option("--cubeb_rgb", default="0.25,0.55,0.95", help="方块B颜色 rgb(0~1)")
def main(
    output_dir,
    scene_preset,
    env_name,
    robot,
    n_episodes,
    max_steps,
    record_start_step,
    control_hz,
    image_size,
    camera0_name,
    camera1_name,
    observer_camera_name,
    camera0_key,
    camera1_key,
    observer_camera_key,
    flip_camera0,
    camera0_zoom,
    eef_pos_key,
    eef_quat_key,
    gripper_key,
    pos_step,
    rot_step_deg,
    gripper_step,
    viewer,
    viewer_scale,
    print_obs_keys,
    swap_keep_objects,
    swap_plate_layout,
    swap_plate_radius,
    swap_draw_plates,
    swap_world_layout,
    swap_place_objects,
    swap_xy_jitter,
    circle_radius_m,
    circle_width_m,
    circle_rgb,
    cubea_rgb,
    cubeb_rgb,
):
    """
    Collect robot_swap-style demonstrations in simulation.

    Each saved episode contains:
      - cam0.mp4
      - cam1.mp4
      - traj.pkl with keys:
        ee_pos, ee_quat, ee_pos_target, ee_quat_target, grasp
    """
    save_root = Path(output_dir).expanduser()
    _ensure_dir(save_root)
    use_observer_cam = bool(observer_camera_name) and bool(observer_camera_key)

    preset_env_kwargs = {}
    resolved_env_name = env_name
    if scene_preset == "pickplace_multi":
        resolved_env_name = "PickPlace"
        preset_env_kwargs["single_object_mode"] = 0
    elif scene_preset == "nutassembly_twoobj":
        resolved_env_name = "NutAssembly"
        preset_env_kwargs["single_object_mode"] = 0
    elif scene_preset == "swap_3plate_2obj":
        resolved_env_name = "PickPlace"
        preset_env_kwargs["single_object_mode"] = 0
    elif scene_preset == "table3circle_2cube":
        resolved_env_name = "Stack"

    camera_names = [camera0_name, camera1_name]
    if use_observer_cam and observer_camera_name not in camera_names:
        camera_names.append(observer_camera_name)

    env = _make_env_robosuite(
        env_name=resolved_env_name,
        robot=robot,
        camera_names=camera_names,
        image_size=image_size,
        control_hz=control_hz,
        max_steps=max_steps,
        extra_env_kwargs=preset_env_kwargs,
    )
    action_dim = _infer_action_dim(env)
    if action_dim != 7:
        raise RuntimeError(
            "This collector currently supports single-arm 7D control only. "
            f"Expected action_dim=7, got {action_dim}."
        )

    print(HELP_TEXT)
    print(f"[Info] Saving to: {save_root}")
    print(f"[Info] ScenePreset={scene_preset}")
    print(f"[Info] Env={resolved_env_name}, Robot={robot}, ActionDim={action_dim}")
    print(f"[Info] record_start_step={record_start_step}")
    print(f"[Info] Viewer={viewer}, ViewerScale={viewer_scale}")
    print(f"[Info] camera_names={[camera0_name, camera1_name]}")
    print(f"[Info] observer_camera={'off' if not use_observer_cam else (observer_camera_name + '/' + observer_camera_key)}")
    print(f"[Info] flip_camera0={flip_camera0}")
    print(f"[Info] camera0_zoom={camera0_zoom}")
    if scene_preset == "swap_3plate_2obj":
        print(f"[Info] swap_keep_objects={swap_keep_objects}")
        print(f"[Info] swap_plate_layout={swap_plate_layout}")
        print(f"[Info] swap_plate_radius={swap_plate_radius}")
        print(f"[Info] swap_draw_plates={swap_draw_plates}")
        print(f"[Info] swap_world_layout={swap_world_layout}")
        print(f"[Info] swap_place_objects={swap_place_objects}")
        print(f"[Info] swap_xy_jitter={swap_xy_jitter}")
    if scene_preset == "table3circle_2cube":
        print(f"[Info] circle_radius_m={circle_radius_m}")
        print(f"[Info] circle_width_m={circle_width_m}")
        print(f"[Info] circle_rgb={circle_rgb}")
        print(f"[Info] cubea_rgb={cubea_rgb}")
        print(f"[Info] cubeb_rgb={cubeb_rgb}")
    if resolved_env_name.lower() == "lift":
        print("[Warn] Lift 环境只有 1 个物体。如果要多物体，请用 --scene_preset pickplace_multi")

    swap_keep_names = _parse_csv_names(swap_keep_objects)
    swap_plate_points = _parse_plate_layout(swap_plate_layout)
    swap_world_points = _parse_world_xy_layout(swap_world_layout)
    circle_rgb_v = _parse_rgb3(circle_rgb)
    cubea_rgb_v = _parse_rgb3(cubea_rgb)
    cubeb_rgb_v = _parse_rgb3(cubeb_rgb)
    if scene_preset == "swap_3plate_2obj" and len(swap_keep_names) != 2:
        raise ValueError("swap_3plate_2obj 预设要求 --swap_keep_objects 恰好 2 个物体名。")
    if scene_preset == "swap_3plate_2obj" and len(swap_plate_points) != 3:
        raise ValueError("swap_3plate_2obj 预设要求 --swap_plate_layout 恰好 3 个盘子坐标。")
    if scene_preset == "swap_3plate_2obj" and len(swap_world_points) != 3:
        raise ValueError("swap_3plate_2obj 预设要求 --swap_world_layout 恰好 3 个圆圈坐标。")
    if scene_preset == "table3circle_2cube" and len(swap_world_points) != 3:
        raise ValueError("table3circle_2cube 预设要求 --swap_world_layout 恰好 3 个圆圈坐标。")

    viewer_state = _init_viewer(
        viewer,
        width=image_size * (3 if use_observer_cam else 2),
        height=image_size,
        scale=max(1, int(viewer_scale)),
    )

    existing_nums = []
    for p in save_root.glob("*"):
        if p.is_dir() and p.name.isdigit():
            existing_nums.append(int(p.name))
    episode_idx = max(existing_nums) + 1 if existing_nums else 0
    dt = 1.0 / float(control_hz)
    has_printed_obs_keys = False

    try:
        while episode_idx < n_episodes:
            obs = env.reset()
            if scene_preset == "table3circle_2cube":
                # Draw circles once into table texture and recolor cube pair.
                _draw_circles_on_table_texture(
                    env,
                    circle_world_xy=swap_world_points,
                    ring_radius_m=circle_radius_m,
                    ring_width_m=circle_width_m,
                    ring_rgb=circle_rgb_v,
                )
                _apply_two_cube_colors(env, cube_a_rgb=cubea_rgb_v, cube_b_rgb=cubeb_rgb_v)

            if scene_preset == "swap_3plate_2obj":
                _apply_pickplace_object_filter(env, keep_names=swap_keep_names)
                if swap_place_objects:
                    _place_two_objects_in_world_circles(
                        env,
                        keep_names=swap_keep_names,
                        circle_world_xy=swap_world_points,
                        xy_jitter=swap_xy_jitter,
                        random_yaw=True,
                    )
                obs = _refresh_obs_after_scene_edit(env, obs)
            elif scene_preset == "table3circle_2cube":
                _place_two_objects_in_world_circles(
                    env,
                    keep_names=["cubeA", "cubeB"],
                    circle_world_xy=random.sample(swap_world_points, 2),
                    xy_jitter=0.02,  # 方块稍微远离圆心一点，增加难度
                    random_yaw=True,
                )
                obs = _refresh_obs_after_scene_edit(env, obs)
            if print_obs_keys and (not has_printed_obs_keys):
                all_keys = sorted(list(obs.keys()))
                image_keys = [k for k in all_keys if k.endswith("_image")]
                print("[Obs] available image keys:", image_keys)
                print("[Obs] all keys:", all_keys)
                has_printed_obs_keys = True
            required_keys = [camera0_key, camera1_key, eef_pos_key, eef_quat_key, gripper_key]
            if use_observer_cam:
                required_keys.append(observer_camera_key)
            _check_obs_keys(obs, required_keys)

            target_pos = np.asarray(obs[eef_pos_key], dtype=np.float64).copy()
            target_quat = np.asarray(obs[eef_quat_key], dtype=np.float64).copy()
            g0 = np.asarray(obs[gripper_key], dtype=np.float64).reshape(-1)
            target_grasp = float(np.clip(np.mean(g0), -1.0, 1.0))

            cam0_frames = []
            cam1_frames = []
            traj = {
                "ee_pos": [],
                "ee_quat": [],
                "ee_pos_target": [],
                "ee_quat_target": [],
                "grasp": [],
            }

            print(f"\n[Episode {episode_idx:06d}] start")
            should_save = False
            should_retry = False
            should_exit = False

            t_start = time.time()
            for step in range(max_steps):
                # Capture current state + current image first, then execute action.
                c0 = _to_hwc_uint8(obs[camera0_key])
                c1 = _to_hwc_uint8(obs[camera1_key])
                c_obs = None
                if use_observer_cam:
                    c_obs = _to_hwc_uint8(obs[observer_camera_key])
                if flip_camera0:
                    c0 = c0[::-1].copy()
                    if c_obs is not None:
                        c_obs = c_obs[::-1].copy()
                c0 = _center_zoom(c0, camera0_zoom)
                if scene_preset == "swap_3plate_2obj" and swap_draw_plates:
                    c0 = _draw_plate_markers(c0, points_norm=swap_plate_points, radius_px=swap_plate_radius)
                if step >= record_start_step:
                    cam0_frames.append(cv2.cvtColor(c0, cv2.COLOR_RGB2BGR))
                    cam1_frames.append(cv2.cvtColor(c1, cv2.COLOR_RGB2BGR))

                    ee_pos = np.asarray(obs[eef_pos_key], dtype=np.float64)
                    ee_quat = np.asarray(obs[eef_quat_key], dtype=np.float64)
                    traj["ee_pos"].append(ee_pos.astype(np.float32))
                    traj["ee_quat"].append(ee_quat.astype(np.float32))
                    traj["ee_pos_target"].append(target_pos.astype(np.float32))
                    traj["ee_quat_target"].append(target_quat.astype(np.float32))
                    traj["grasp"].append(np.float32(target_grasp))

                vis_parts = [c0, c1]
                if c_obs is not None:
                    if c_obs.shape[0] != c0.shape[0] or c_obs.shape[1] != c0.shape[1]:
                        c_obs = cv2.resize(c_obs, (c0.shape[1], c0.shape[0]), interpolation=cv2.INTER_AREA)
                    vis_parts.append(c_obs)
                vis = np.concatenate(vis_parts, axis=1)
                vis = cv2.putText(
                    vis,
                    f"ep={episode_idx:06d} step={step:04d}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                _render_viewer(viewer_state, vis)

                commands, move_keys = _poll_keys(viewer_state)
                if "h" in commands:
                    print(HELP_TEXT)
                if "c" in commands:
                    should_save = True
                    print("[Episode] save requested.")
                    break
                if "r" in commands:
                    should_retry = True
                    print("[Episode] retry requested (discard).")
                    break
                if "x" in commands:
                    should_exit = True
                    print("[Collector] exit requested.")
                    break

                for mk in sorted(move_keys):
                    target_pos, target_quat, target_grasp = _update_target_from_key(
                        key=ord(mk),
                        target_pos=target_pos,
                        target_quat=target_quat,
                        target_grasp=target_grasp,
                        pos_step=pos_step,
                        rot_step_deg=rot_step_deg,
                        gripper_step=gripper_step,
                    )

                action = np.zeros(action_dim, dtype=np.float64)
                action[:3] = target_pos
                action[3:6] = R.from_quat(target_quat).as_rotvec()
                action[6] = target_grasp

                obs, _, _, _ = env.step(action)

                elapsed = time.time() - t_start
                expected = (step + 1) * dt
                sleep_t = expected - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            if should_exit:
                break

            if should_save and len(cam0_frames) > 1:
                ep_dir = save_root / f"{episode_idx:06d}"
                _ensure_dir(ep_dir)
                _write_mp4(cam0_frames, ep_dir / "cam0.mp4", fps=control_hz)
                _write_mp4(cam1_frames, ep_dir / "cam1.mp4", fps=control_hz)
                with open(ep_dir / "traj.pkl", "wb") as f:
                    pickle.dump(traj, f)
                print(f"[Episode {episode_idx:06d}] saved. len={len(cam0_frames)}")
                episode_idx += 1
            elif should_save:
                print(
                    f"[Episode] not enough recorded frames (len={len(cam0_frames)}). "
                    "Increase episode length or reduce --record_start_step."
                )
            elif should_retry:
                print("[Episode] discarded.")
            else:
                print("[Episode] not saved. Press 'c' to save after a successful run.")
    finally:
        _close_viewer(viewer_state)

    print("Collection finished.")


if __name__ == "__main__":
    main()
