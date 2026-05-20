import argparse
import os
from pathlib import Path
import shutil
from copy import deepcopy


def _has_nvidia_device() -> bool:
    nvidia_dev = Path("/dev")
    if not nvidia_dev.exists():
        return False
    return any(nvidia_dev.glob("nvidia[0-9]*")) or (nvidia_dev / "nvidiactl").exists()


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "JAX_PLATFORMS" not in os.environ and not _has_nvidia_device():
    os.environ["JAX_PLATFORMS"] = "cpu"
    # report device info for confirmation
    print("No NVIDIA GPU detected. Using CPU backend for JAX.")

import numpy as np
import yaml

import irsim
from irsim.config.path_param import path_manager
from exact_mppi.mppi_jax.controller import MPPIController


def report_jax_device() -> None:
    import jax

    devices = jax.devices()
    summary = ", ".join(
        f"{device.platform}:{getattr(device, 'device_kind', str(device))}"
        for device in devices
    )
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {summary}")

ROBOT_SHAPES = {
    "t": {
        "env_shape": {
            "name": "mosaic",
            "vertices_list": [
                [[0.900, 0.300], [-0.900, 0.300], [-0.900, -0.300], [0.900, -0.300]],
                [[-0.600, 1.200], [-0.900, 1.200], [-0.900, -1.200], [-0.600, -1.200]],
            ],
            "wheelbase": 1.52,
        },
        "footprint_type": "rectangle",
        "planner_vertices": [
            [[0.900, 0.300], [-0.900, 0.300], [-0.900, -0.300], [0.900, -0.300]],
            [[-0.600, 1.200], [-0.900, 1.200], [-0.900, -1.200], [-0.600, -1.200]],
        ],
        "ani_suffix": "t",
        "description": "Exact T shape represented as a union of rectangles.",
    },
    "t_convex": {
        "env_shape": {
            "name": "polygon",
            "vertices": [
                [-0.600, 1.200],
                [-0.900, 1.200],
                [-0.900, -1.200],
                [-0.600, -1.200],
                [0.900, -0.300],
                [0.900, 0.300],
            ],
            "wheelbase": 1.52,
        },
        "footprint_type": "polygon",
        "planner_vertices": [
            [
                [-0.600, 1.200],
                [-0.900, 1.200],
                [-0.900, -1.200],
                [-0.600, -1.200],
                [0.900, -0.300],
                [0.900, 0.300],
            ]
        ],
        "ani_suffix": "t_convex",
        "description": "Convex polygon cover of the T shape using point-to-segment polygon SDF.",
    },
    "rect": {
        "env_shape": {
            "name": "rectangle",
            "length": 1.8,
            "width": 2.4,
            "wheelbase": 1.52,
        },
        "footprint_type": "rectangle",
        "planner_vertices": [
            [[0.900, 1.200], [-0.900, 1.200], [-0.900, -1.200], [0.900, -1.200]],
        ],
        "ani_suffix": "rect",
        "description": "Axis-aligned rectangle cover of the T shape.",
    },
}

CORRIDOR_BOUNDS = {
    "x_min": 0.0,
    "x_max": 50.0,
    "y_min": 13,
    "y_max": 27,
}

WALL_Y_MARGIN = 0.35
MIN_DYNAMIC_OBSTACLE_SPEED = 0.1
MAX_DYNAMIC_OBSTACLE_SPEED = 0.2
POLYGON_SHAPE_MODES = ("convex", "concave", "mixed")
IGNORED_OUTPUT_DIR = "ignored"


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def transfer_from_global_to_local_frame(
    points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    p = np.asarray(points)
    pose = np.asarray(pose).reshape(-1)
    x, y, yaw = pose[0], pose[1], pose[2]

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    trans = np.array([x, y])

    out = p.copy()
    xy_global = out[..., :2]
    xy_local = (xy_global - trans) @ rot
    out[..., :2] = xy_local

    if out.shape[-1] >= 3:
        out[..., 2] = _wrap_to_pi(out[..., 2] - yaw)

    return out


def transfer_from_local_to_global_frame(
    points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    p = np.asarray(points)
    pose = np.asarray(pose).reshape(-1)
    x, y, yaw = pose[0], pose[1], pose[2]

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    trans = np.array([x, y])

    out = p.copy()
    xy_local = out[..., :2]
    xy_global = xy_local @ rot.T + trans
    out[..., :2] = xy_global

    if out.shape[-1] >= 3:
        out[..., 2] = _wrap_to_pi(out[..., 2] + yaw)

    return out


def load_yaml(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _resolve_generated_env_dir(out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    if out_path.is_absolute():
        return out_path

    parts = out_path.parts
    if parts and parts[0] == IGNORED_OUTPUT_DIR:
        return Path(__file__).parent / out_path

    return Path(__file__).parent / IGNORED_OUTPUT_DIR / out_path


def _compute_furthest_reached_path_index(
    local_plan: np.ndarray, local_rollouts: np.ndarray
) -> int | None:
    if local_plan is None or local_rollouts is None:
        return None

    path = np.asarray(local_plan)
    rollouts = np.asarray(local_rollouts)
    if path.ndim != 2 or rollouts.ndim != 3:
        return None
    if path.shape[0] == 0 or rollouts.shape[0] == 0:
        return None

    last_xy = rollouts[:, -1, :2]
    dx = path[None, :, 0] - last_xy[:, None, 0]
    dy = path[None, :, 1] - last_xy[:, None, 1]
    dist2 = dx * dx + dy * dy
    nearest_idx = np.argmin(dist2, axis=1)
    return int(np.max(nearest_idx))


def _collect_cost_reference_points(
    local_plan: np.ndarray, local_rollouts: np.ndarray, planner_cfg: dict
) -> np.ndarray | None:
    furthest_idx = _compute_furthest_reached_path_index(local_plan, local_rollouts)
    if furthest_idx is None:
        return None

    path = np.asarray(local_plan)
    if path.ndim != 2 or path.shape[0] == 0:
        return None

    path_xy = path[:, :2]
    max_idx = path_xy.shape[0] - 1
    critics_cfg = planner_cfg.get("MPPI", {}).get("Critics", {})

    def critic_enabled(name: str, default: bool = True) -> bool:
        return bool(critics_cfg.get(name, {}).get("enabled", default))

    def offset_from_furthest(name: str) -> int:
        return int(critics_cfg.get(name, {}).get("offset_from_furthest", 0))

    idxs = [furthest_idx]
    if critic_enabled("PathFollowCritic", True):
        idxs.append(furthest_idx + offset_from_furthest("PathFollowCritic"))
    if critic_enabled("PathAngleCritic", True):
        idxs.append(furthest_idx + offset_from_furthest("PathAngleCritic"))
    if critic_enabled("GoalCritic", True) or critic_enabled("GoalAngleCritic", True):
        idxs.append(max_idx)

    clamped = []
    seen = set()
    for idx in idxs:
        idx = max(0, min(int(idx), max_idx))
        if idx not in seen:
            clamped.append(idx)
            seen.add(idx)

    return path_xy[clamped]


def _select_local_plan(
    global_path: np.ndarray, robot_pose: np.ndarray, plan_length: int
) -> np.ndarray | None:
    if global_path is None or global_path.shape[0] == 0:
        return None
    dx = global_path[:, 0] - robot_pose[0]
    dy = global_path[:, 1] - robot_pose[1]
    nearest_idx = int(np.argmin(dx * dx + dy * dy))
    end_idx = nearest_idx + plan_length
    if end_idx <= global_path.shape[0]:
        return global_path[nearest_idx:end_idx]

    last = global_path[-1]
    pad_count = end_idx - global_path.shape[0]
    pad = np.repeat(last[None, :], pad_count, axis=0)
    return np.vstack([global_path[nearest_idx:], pad])


def _summarize_cost_breakdown(cost_breakdown: dict, top_n: int = 6) -> str:
    if not cost_breakdown:
        return ""
    total = cost_breakdown.get("total")
    items = [(k, v) for k, v in cost_breakdown.items() if k != "total"]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:top_n]
    parts = [f"{k}={v:.3f}" for k, v in items]
    if total is not None:
        parts.append(f"total={total:.3f}")
    return ", ".join(parts)


def get_transform(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if state.shape == (2, 1):
        rot = np.array([[1, 0], [0, 1]])
        trans = state[0:2]
    else:
        rot = np.array(
            [
                [np.cos(state[2, 0]), -np.sin(state[2, 0])],
                [np.sin(state[2, 0]), np.cos(state[2, 0])],
            ]
        )
        trans = state[0:2]
    return trans, rot


def scan_to_points(
    state: np.ndarray,
    scan: dict,
    scan_offset: list[float] = [0, 0, 0],
    angle_range: list[float] = [-np.pi, np.pi],
):
    point_cloud = []
    ranges = np.array(scan["ranges"])
    angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))

    for index, scan_range in enumerate(ranges):
        angle = angles[index]
        if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
            if angle_range[0] < angle < angle_range[1]:
                point = np.array(
                    [[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]]
                )
                point_cloud.append(point)

    if len(point_cloud) == 0:
        return None, None

    point_array = np.hstack(point_cloud)
    sensor_trans, sensor_rot = get_transform(np.c_[scan_offset])
    local_points = sensor_rot @ point_array + sensor_trans

    trans, rot = get_transform(state)
    global_points = rot @ local_points + trans
    return global_points, local_points


def build_stack(planner_cfg: dict) -> MPPIController:
    return MPPIController(**planner_cfg.get("MPPI"))


def _render_env_frame(
    env: irsim.EnvBase,
    *,
    render: bool,
    save_animation: bool,
    save_frame: bool,
) -> bool:
    if not render:
        return False

    original_save_ani = env.save_ani
    try:
        env.save_ani = bool(save_animation and save_frame)
        env.render()
        return True
    finally:
        env.save_ani = original_save_ani


def _cleanup_animation_buffer_if_created(
    buffer_dir: Path, *, existed_before_run: bool, save_animation: bool
) -> None:
    if save_animation or existed_before_run or not buffer_dir.exists():
        return
    if buffer_dir.is_dir():
        shutil.rmtree(buffer_dir, ignore_errors=True)


def _planar_speed_from_velocity(velocity: np.ndarray) -> float:
    vel = np.asarray(velocity).reshape(-1)
    if vel.size == 0:
        return 0.0
    if vel.size == 1:
        return float(abs(vel[0]))
    if vel.size == 2:
        return float(abs(vel[0]))
    return float(np.linalg.norm(vel[:2]))


def _normalize_footprint_polygons(vertices: list | np.ndarray | None) -> list[np.ndarray]:
    if vertices is None:
        return []

    arr = np.asarray(vertices, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return [arr]
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return [arr[i] for i in range(arr.shape[0])]
    return []


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, direction) / denom)
    t = max(0.0, min(1.0, t))
    closest = start + t * direction
    return float(np.linalg.norm(point - closest))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    count = polygon.shape[0]
    if count < 3:
        return False

    for idx in range(count):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % count]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
        )
        if intersects:
            inside = not inside

    return inside


def _signed_distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float | None:
    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] != 2:
        return None

    min_edge_distance = min(
        _point_to_segment_distance(point, poly[idx], poly[(idx + 1) % poly.shape[0]])
        for idx in range(poly.shape[0])
    )
    return -min_edge_distance if _point_in_polygon(point, poly) else min_edge_distance


def _min_obstacle_distance_to_footprint(
    local_points: np.ndarray | None, footprint_polygons: list[np.ndarray]
) -> float | None:
    if local_points is None or len(footprint_polygons) == 0:
        return None

    points = np.asarray(local_points, dtype=np.float32)
    if points.ndim == 2 and points.shape[0] == 2:
        points = points.T
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        return None

    min_distance = np.inf
    for point in points:
        point_distance = min(
            _signed_distance_to_polygon(point, polygon)
            for polygon in footprint_polygons
        )
        min_distance = min(min_distance, point_distance)

    if not np.isfinite(min_distance):
        return None
    return float(min_distance)


def _format_runtime_title(
    robot_shape: str,
    step_idx: int,
    sim_time: float,
    min_obstacle_distance: float | None,
) -> str:
    if min_obstacle_distance is None:
        distance_text = "min obstacle dist: n/a"
    else:
        distance_text = f"min obstacle dist: {min_obstacle_distance:.2f} m"

    return (
        f"shape={robot_shape} | step={step_idx} | t={sim_time:.1f} s | "
        f"{distance_text}"
    )


def _obstacle_limits(shape_name: str, shape_cfg: dict) -> tuple[float, float]:
    if shape_name == "circle":
        radius = float(shape_cfg.get("radius", 0.5))
        return radius, radius
    if shape_name == "rectangle":
        return 0.5 * float(shape_cfg.get("length", 1.0)), 0.5 * float(
            shape_cfg.get("width", 1.0)
        )
    if shape_name == "polygon":
        vertices = np.asarray(shape_cfg.get("vertices", []), dtype=np.float32)
        if vertices.size == 0:
            return 0.5, 0.5
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        halfs = 0.5 * (maxs - mins)
        return float(halfs[0]), float(halfs[1])
    return 0.5, 0.5


def _assign_initial_bounce_velocity(
    rng: np.random.Generator, x_sign: float | None = None, y_sign: float | None = None
) -> np.ndarray:
    speed = float(rng.uniform(MIN_DYNAMIC_OBSTACLE_SPEED, MAX_DYNAMIC_OBSTACLE_SPEED))

    if x_sign is None and y_sign is None:
        angle = float(rng.uniform(-np.pi, np.pi))
        velocity_xy = np.array(
            [speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32
        )
    else:
        direction_xy = np.array(
            [
                rng.uniform(0.2, 1.0) * (1.0 if x_sign is None else np.sign(x_sign)),
                rng.uniform(0.2, 1.0) * (1.0 if y_sign is None else np.sign(y_sign)),
            ],
            dtype=np.float32,
        )
        if x_sign is None and rng.random() < 0.5:
            direction_xy[0] *= -1.0
        if y_sign is None and rng.random() < 0.5:
            direction_xy[1] *= -1.0
        norm = float(np.linalg.norm(direction_xy))
        if norm <= 1e-6:
            direction_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction_xy /= norm
        velocity_xy = direction_xy * speed

    speed = float(np.linalg.norm(velocity_xy))
    if speed > MAX_DYNAMIC_OBSTACLE_SPEED:
        velocity_xy *= MAX_DYNAMIC_OBSTACLE_SPEED / speed
    return velocity_xy


def _make_bounce_goal(position: np.ndarray, velocity_xy: np.ndarray) -> list[float]:
    vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
    x_target = CORRIDOR_BOUNDS["x_max"] if vx >= 0.0 else CORRIDOR_BOUNDS["x_min"]
    y_target = CORRIDOR_BOUNDS["y_max"] if vy >= 0.0 else CORRIDOR_BOUNDS["y_min"]
    return [x_target, y_target, float(np.arctan2(vy, vx))]


def initialize_dynamic_obstacles(env, obstacle_states: list[dict], static_offset: int = 0) -> None:
    start_idx = 2 + int(static_offset)
    end_idx = start_idx + len(obstacle_states)
    for obs, state in zip(env.obstacle_list[start_idx:end_idx], obstacle_states):
        obs.set_goal(_make_bounce_goal(obs.state[:2, 0], state["velocity_xy"]))
        obs.set_velocity(state["velocity_xy"])


def update_dynamic_obstacles(env, obstacle_states: list[dict], static_offset: int = 0) -> None:
    start_idx = 2 + int(static_offset)
    obs_list = list(env.obstacle_list[start_idx : start_idx + len(obstacle_states)])

    # wall bounces: reverse velocity components when hitting corridor bounds
    for obs, state in zip(obs_list, obstacle_states):
        pos_x = float(obs.state[0, 0])
        pos_y = float(obs.state[1, 0])
        vel = state["velocity_xy"]
        half_x = float(state["half_extents"][0])
        half_y = float(state["half_extents"][1])

        top_limit = CORRIDOR_BOUNDS["y_max"] - half_y - WALL_Y_MARGIN
        bottom_limit = CORRIDOR_BOUNDS["y_min"] + half_y + WALL_Y_MARGIN
        right_limit = CORRIDOR_BOUNDS["x_max"] - half_x
        left_limit = CORRIDOR_BOUNDS["x_min"] + half_x

        bounced = False
        if pos_y >= top_limit and vel[1] > 0.0:
            vel[1] *= -1.0
            bounced = True
        elif pos_y <= bottom_limit and vel[1] < 0.0:
            vel[1] *= -1.0
            bounced = True

        if pos_x >= right_limit and vel[0] > 0.0:
            vel[0] *= -1.0
            bounced = True
        elif pos_x <= left_limit and vel[0] < 0.0:
            vel[0] *= -1.0
            bounced = True

        if bounced:
            obs.set_goal(_make_bounce_goal(obs.state[:2, 0], vel))
            obs.set_velocity(vel)

    # inter-obstacle collisions: simple elastic-like response by swapping velocities
    n = len(obs_list)
    if n >= 2:
        margin = 1e-3
        for i in range(n):
            for j in range(i + 1, n):
                obs_i = obs_list[i]
                obs_j = obs_list[j]
                state_i = obstacle_states[i]
                state_j = obstacle_states[j]

                pos_i = np.array([float(obs_i.state[0, 0]), float(obs_i.state[1, 0])], dtype=np.float32)
                pos_j = np.array([float(obs_j.state[0, 0]), float(obs_j.state[1, 0])], dtype=np.float32)

                # approximate collision radius as the max half-extent
                ri = float(np.max(state_i["half_extents"]))
                rj = float(np.max(state_j["half_extents"]))
                dist = float(np.linalg.norm(pos_i - pos_j))

                if dist <= (ri + rj + margin):
                    # swap velocities to approximate an elastic bounce
                    vi = state_i["velocity_xy"].copy()
                    vj = state_j["velocity_xy"].copy()
                    state_i["velocity_xy"] = vj
                    state_j["velocity_xy"] = vi

                    obs_i.set_velocity(state_i["velocity_xy"])
                    obs_j.set_velocity(state_j["velocity_xy"])

                    obs_i.set_goal(_make_bounce_goal(obs_i.state[:2, 0], state_i["velocity_xy"]))
                    obs_j.set_goal(_make_bounce_goal(obs_j.state[:2, 0], state_j["velocity_xy"]))


def _robot_shape_config(robot_shape: str) -> dict:
    key = robot_shape.lower()
    if key not in ROBOT_SHAPES:
        raise ValueError(
            f"Unknown robot shape '{robot_shape}'. Choose from: {', '.join(ROBOT_SHAPES)}"
        )
    return deepcopy(ROBOT_SHAPES[key])


def _make_corridor_walls() -> list[dict]:
    return [
        {
            "number": 1,
            "distribution": {"name": "manual"},
            "shape": {"name": "rectangle", "length": 70, "width": 2},
            "state": [30, CORRIDOR_BOUNDS["y_min"], 0],
            "color": "k",
            "plot": {"show_goal": False, "show_arrow": False},
        },
        {
            "number": 1,
            "distribution": {"name": "manual"},
            "shape": {"name": "rectangle", "length": 70, "width": 2},
            "state": [30, CORRIDOR_BOUNDS["y_max"], 0],
            "color": "k",
            "plot": {"show_goal": False, "show_arrow": False},
        },
    ]


def _sample_obstacle_pose(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    occupied: list[np.ndarray],
    min_center_distance: float,
    max_attempts: int = 200,
) -> np.ndarray:
    for _ in range(max_attempts):
        pose = np.array(
            [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                rng.uniform(-np.pi, np.pi),
            ],
            dtype=np.float32,
        )
        if all(np.linalg.norm(pose[:2] - center) >= min_center_distance for center in occupied):
            occupied.append(pose[:2].copy())
            return pose
    occupied.append(pose[:2].copy())
    return pose


def _make_dynamic_obstacle(
    rng: np.random.Generator,
    shape_type: str,
    pose: np.ndarray,
    goal: np.ndarray,
    roam_low: list[float],
    roam_high: list[float],
    polygon_shape_mode: str = "convex",
) -> dict:
    base = {
        "number": 1,
        "distribution": {"name": "manual"},
        "kinematics": {"name": "omni"},
        "state": pose.round(4).tolist(),
        "goal": goal.round(4).tolist(),
        "behavior": {
            "name": "dash",
        },
        "vel_min": [-MAX_DYNAMIC_OBSTACLE_SPEED, -MAX_DYNAMIC_OBSTACLE_SPEED],
        "vel_max": [MAX_DYNAMIC_OBSTACLE_SPEED, MAX_DYNAMIC_OBSTACLE_SPEED],
        "arrive_mode": "position",
        "goal_threshold": 0.2,
        "plot": {"show_goal": False, "show_arrow": True},
    }

    if shape_type == "circle":
        base["shape"] = {
            "name": "circle",
            "radius": 0.45,
            "random_shape": True,
            "radius_range": [0.35, 0.8],
        }
    elif shape_type == "rectangle":
        base["shape"] = {
            "name": "rectangle",
            "length": float(rng.uniform(0.8, 1.8)),
            "width": float(rng.uniform(0.5, 1.2)),
        }
    elif shape_type == "polygon":
        if polygon_shape_mode not in POLYGON_SHAPE_MODES:
            raise ValueError(
                f"Unsupported polygon_shape_mode '{polygon_shape_mode}'. "
                f"Choose from: {', '.join(POLYGON_SHAPE_MODES)}"
            )

        use_convex = polygon_shape_mode == "convex"
        if polygon_shape_mode == "mixed":
            use_convex = bool(rng.integers(0, 2))

        # strengthen non-convex parameters when concave mode requested to
        # increase the chance the sampled polygon is actually concave.
        if use_convex:
            irregularity = [0.05, 0.30]
            spikeyness = [0.0, 0.0]
            num_vertices = [4, 7]
        else:
            irregularity = [0.30, 0.75]
            spikeyness = [0.25, 0.75]
            num_vertices = [5, 9]

        base["shape"] = {
            "name": "polygon",
            "random_shape": True,
            "is_convex": use_convex,
            "avg_radius_range": [0.45, 0.9],
            "irregularity_range": irregularity,
            "spikeyness_range": spikeyness,
            "num_vertices_range": num_vertices,
        }
    else:
        raise ValueError(f"Unsupported dynamic obstacle shape '{shape_type}'")

    return base


def build_dynamic_obstacles(
    seed: int,
    num_dynamic_obstacles: int,
    polygon_shape_mode: str = "concave",
    occupied: list[np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    if occupied is None:
        occupied = [np.array([0.0, 20.0]), np.array([60.0, 20.0])]
    roam_low = [8.0, 16.2, -np.pi]
    roam_high = [54.0, 23.8, np.pi]
    obstacle_entries = []
    obstacle_states = []
    # shape_types = ["circle", "rectangle", "polygon"]
    shape_types = ["polygon"]

    for _ in range(num_dynamic_obstacles):
        shape_type = str(rng.choice(shape_types))
        pose = _sample_obstacle_pose(
            rng,
            x_range=(roam_low[0], roam_high[0]),
            y_range=(roam_low[1], roam_high[1]),
            occupied=occupied,
            min_center_distance=2.6,
        )
        goal = np.array(
            [
                pose[0],
                pose[1],
                0.0,
            ],
            dtype=np.float32,
        )
        obstacle_entry = _make_dynamic_obstacle(
            rng,
            shape_type=shape_type,
            pose=pose,
            goal=goal,
            roam_low=roam_low,
            roam_high=roam_high,
            polygon_shape_mode=polygon_shape_mode,
        )
        obstacle_entries.append(obstacle_entry)

        half_extents = _obstacle_limits(shape_type, obstacle_entry["shape"])
        velocity_xy = _assign_initial_bounce_velocity(rng)
        obstacle_states.append(
            {
                "shape_type": shape_type,
                "half_extents": np.array(half_extents, dtype=np.float32),
                "velocity_xy": velocity_xy,
            }
        )

    return obstacle_entries, obstacle_states


def build_static_obstacles(
    seed: int,
    num_static_obstacles: int,
    polygon_shape_mode: str = "concave",
    occupied: list[np.ndarray] | None = None,
) -> list[dict]:
    """Create a list of static (non-moving) obstacle entries.

    Static obstacles are placed like dynamic ones but have no velocity
    or movement behavior. The function appends sampled centers to
    the provided `occupied` list to avoid overlap with other obstacles.
    """
    rng = np.random.default_rng(seed + 12345)
    if occupied is None:
        occupied = [np.array([0.0, 20.0]), np.array([60.0, 20.0])]

    roam_low = [8.0, 16.2, -np.pi]
    roam_high = [54.0, 23.8, np.pi]
    static_entries: list[dict] = []
    shape_types = ["polygon"]

    for _ in range(num_static_obstacles):
        shape_type = str(rng.choice(shape_types))
        pose = _sample_obstacle_pose(
            rng,
            x_range=(roam_low[0], roam_high[0]),
            y_range=(roam_low[1], roam_high[1]),
            occupied=occupied,
            min_center_distance=2.6,
        )

        # For a static obstacle, goal = state and no velocity fields
        goal = np.array([pose[0], pose[1], 0.0], dtype=np.float32)
        entry = _make_dynamic_obstacle(
            rng,
            shape_type=shape_type,
            pose=pose,
            goal=goal,
            roam_low=roam_low,
            roam_high=roam_high,
            polygon_shape_mode=polygon_shape_mode,
        )

        # Convert dynamic-style entry to static while keeping a valid
        # registered behavior so IR-SIM does not warn in auto mode.
        entry["behavior"] = {"name": "dash"}
        entry["vel_min"] = [0.0, 0.0]
        entry["vel_max"] = [0.0, 0.0]
        entry["distribution"] = {"name": "manual"}
        # keep goal same as state so obstacle won't move
        entry["goal"] = entry["state"]

        static_entries.append(entry)

    return static_entries


def prepare_env_config(
    base_env_cfg: dict,
    robot_shape: str,
    seed: int,
    num_dynamic_obstacles: int,
    polygon_shape_mode: str = "convex",
    num_static_obstacles: int = 0,
) -> tuple[dict, list[dict]]:
    env_cfg = deepcopy(base_env_cfg)
    shape_cfg = _robot_shape_config(robot_shape)
    env_cfg["robot"][0]["shape"] = deepcopy(shape_cfg["env_shape"])
    env_cfg["robot"][0]["description"] = shape_cfg["description"]
    # place static obstacles first (if requested), then dynamic obstacles avoiding occupied centers
    occupied: list[np.ndarray] = [np.array([0.0, 20.0]), np.array([60.0, 20.0])]
    static_obstacles = []
    if num_static_obstacles > 0:
        static_obstacles = build_static_obstacles(seed=seed, num_static_obstacles=num_static_obstacles, polygon_shape_mode=polygon_shape_mode, occupied=occupied)

    dynamic_obstacles, obstacle_states = build_dynamic_obstacles(
        seed=seed,
        num_dynamic_obstacles=num_dynamic_obstacles,
        polygon_shape_mode=polygon_shape_mode,
        occupied=occupied,
    )

    env_cfg["obstacle"] = _make_corridor_walls() + static_obstacles + dynamic_obstacles
    # env_cfg["obstacle"] = dynamic_obstacles
    return env_cfg, obstacle_states


def prepare_planner_config(base_planner_cfg: dict, robot_shape: str) -> dict:
    planner_cfg = deepcopy(base_planner_cfg)
    shape_cfg = _robot_shape_config(robot_shape)
    planner_cfg.setdefault("MPPI", {})
    planner_cfg["MPPI"]["footprint_type"] = shape_cfg["footprint_type"]
    planner_cfg["MPPI"]["vertices"] = deepcopy(shape_cfg["planner_vertices"])
    return planner_cfg


def write_generated_env(env_cfg: dict, robot_shape: str, seed: int, out_dir: str | Path = "generated_envs") -> Path:
    out_dir = _resolve_generated_env_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generated_env_{robot_shape}_{seed}.yaml"
    dump_yaml(out_path, env_cfg)
    return out_path


def main(
    robot_shape: str = "t",
    seed: int = 0,
    num_dynamic_obstacles: int = 8,
    polygon_shape_mode: str = "convex",
    time_limit: float = 50.0,
    save_animation: bool = False,
    num_static_obstacles: int = 0,
    ani_name: str = "corridor_dynamic_random",
    show_rollouts: bool = False,
    print_costs: bool = False,
    cost_interval: int = 20,
    display: bool = True,
    render: bool = True,
    verbose: bool = True,
    shutdown_wait: float = 3.0,
    gif_frame_stride: int = 1,
    gif_last_frame_duration: float = 1.0,
    generated_env_dir: str | Path = "generated_envs",
):
    base_dir = Path(__file__).parent
    base_env_cfg = load_yaml(base_dir / "env.yaml")
    base_planner_cfg = load_yaml(base_dir / "planner.yaml")

    env_cfg, obstacle_states = prepare_env_config(
        base_env_cfg,
        robot_shape,
        seed,
        num_dynamic_obstacles,
        polygon_shape_mode=polygon_shape_mode,
        num_static_obstacles=num_static_obstacles,
    )
    planner_cfg = prepare_planner_config(base_planner_cfg, robot_shape)
    generated_env_path = write_generated_env(env_cfg, robot_shape, seed, generated_env_dir)
    animation_buffer_dir = Path(path_manager.ani_buffer_path)
    animation_buffer_existed = animation_buffer_dir.exists()

    if save_animation:
        save_dir = Path("./")
        save_dir.mkdir(parents=True, exist_ok=True)
        path_manager.ani_path = str(save_dir)
        print(f"Animation will be saved to {save_dir}/{ani_name}_{robot_shape}.gif")
        buffer_dir = Path(path_manager.ani_buffer_path)
        if buffer_dir.exists():
            shutil.rmtree(buffer_dir)
        buffer_dir.mkdir()

    if verbose:
        print(
            f"Running robot_shape={robot_shape}, seed={seed}, "
            f"dynamic_obstacles={num_dynamic_obstacles}, "
            f"footprint_type={planner_cfg['MPPI']['footprint_type']}"
        )
        print(f"Generated environment: {generated_env_path}")

    env = irsim.make(
        str(generated_env_path),
        save_ani=save_animation,
        display=display,
        disable_all_plot=not display,
        seed=seed,
    )
    env.step(np.array([0, 0]))
    initialize_dynamic_obstacles(env, obstacle_states, static_offset=num_static_obstacles)

    start_state = np.array(env.robot.state, dtype=np.float32).reshape(-1)
    goal_state = np.array(env.robot.goal, dtype=np.float32).reshape(-1)

    world_cfg = env_cfg.get("world", {})
    sim_dt = float(world_cfg.get("step_time", 0.1))
    prev_robot_pose = start_state[:2].copy()
    path_length = 0.0
    accumulated_speed = 0.0
    speed_samples = 0
    executed_steps = 0
    time_limit_exceeded = False

    mppi = build_stack(planner_cfg)
    footprint_type = planner_cfg["MPPI"].get("footprint_type", "rectangle")
    vertices = planner_cfg["MPPI"].get("vertices")
    if footprint_type == "polygon":
        mppi.setPolygonFootprint(vertices)
    else:
        mppi.setRectangleFootprint(vertices)
    footprint_polygons = _normalize_footprint_polygons(vertices)

    global_reference_line = np.linspace(start_state[:3], goal_state[:3], 120)
    local_plan_length = 30
    gif_frame_stride = max(1, int(gif_frame_stride))
    frame_was_rendered = False
    last_min_obstacle_distance = None

    for step_idx in range(20000):
        if env.status == "Arrived":
            if verbose:
                print("Arrived at the goal")
            break
        if env.status == "Collision":
            if verbose:
                print("Robot collided before reaching the goal")
            break

        robot_state = env.get_robot_state()
        robot_vel = env.get_robot_velocity()
        lidar_scan = env.get_lidar_scan()

        global_lidar_points, local_lidar_points = scan_to_points(robot_state, lidar_scan)
        last_min_obstacle_distance = _min_obstacle_distance_to_footprint(
            local_lidar_points, footprint_polygons
        )

        if display:
            env.set_title(
                _format_runtime_title(
                    robot_shape=robot_shape,
                    step_idx=executed_steps,
                    sim_time=executed_steps * sim_dt,
                    min_obstacle_distance=last_min_obstacle_distance,
                )
            )

        robot_pose = robot_state[:3].reshape(-1)
        robot_speed = robot_vel.reshape(-1)
        robot_speed = np.array([robot_speed[0], 0.0, robot_speed[1]])

        step_distance = float(np.linalg.norm(robot_pose[:2] - prev_robot_pose))
        path_length += step_distance
        prev_robot_pose = robot_pose[:2].copy()
        accumulated_speed += _planar_speed_from_velocity(robot_vel)
        speed_samples += 1

        global_plan = _select_local_plan(global_reference_line, robot_pose, local_plan_length)
        if global_plan is None:
            global_plan = np.linspace(robot_pose[:3], goal_state[:3], local_plan_length)
        local_plan = transfer_from_global_to_local_frame(global_plan, robot_pose)
        local_goal = transfer_from_global_to_local_frame(
            goal_state[:3].reshape(-1), robot_pose
        )
        lidar_points = None if local_lidar_points is None else local_lidar_points.T

        action = mppi.computeVelocityCommands(
            robot_pose=np.array([0.0, 0.0, 0.0]),
            robot_speed=robot_speed,
            plan=local_plan,
            goal=local_goal,
            lidar_points=lidar_points,
        )

        if print_costs and (step_idx % max(cost_interval, 1) == 0):
            breakdown = mppi.getCostBreakdown()
            if breakdown:
                summary = _summarize_cost_breakdown(breakdown)
                if summary:
                    print(f"Cost breakdown @ step {step_idx}: {summary}")

        opt_trajectory = mppi.getOptimalTrajectory()
        all_traj_rollouts = mppi.getGeneratedTrajectories()

        if global_lidar_points is not None and global_lidar_points.shape[1] > 0:
            env.draw_points(global_lidar_points, s=20, c="r", refresh=True)

        env.draw_trajectory(
            global_reference_line.T, "-k", linewidth=1.0, alpha=0.4, refresh=True
        )
        if global_plan is not None:
            env.draw_trajectory(global_plan.T, "b", linewidth=1.5, alpha=0.7, refresh=True)

        cost_points_local = _collect_cost_reference_points(
            local_plan, all_traj_rollouts, planner_cfg
        )
        if cost_points_local is not None and cost_points_local.size > 0:
            cost_points_global = transfer_from_local_to_global_frame(
                cost_points_local, robot_pose
            )
            env.draw_points(cost_points_global.T, s=45, c="m", refresh=True)

        if show_rollouts and all_traj_rollouts is not None:
            all_traj_rollouts = transfer_from_local_to_global_frame(
                all_traj_rollouts, robot_pose
            )
            for rollout_index in range(all_traj_rollouts.shape[0]):
                env.draw_trajectory(
                    all_traj_rollouts[rollout_index].T,
                    "blue",
                    linewidth=0.5,
                    alpha=0.3,
                    refresh=True,
                )

        if opt_trajectory is not None:
            opt_trajectory = transfer_from_local_to_global_frame(opt_trajectory, robot_pose)
            # env.draw_trajectory(
            #     opt_trajectory.T,
            #     "g",
            #     linewidth=2.2,
            #     alpha=0.95,
            #     refresh=True,
            # )

        cmd = action.reshape(-1).astype(np.float32)
        env.step(cmd)
        update_dynamic_obstacles(env, obstacle_states, static_offset=num_static_obstacles)
        should_save_gif_frame = (not save_animation) or (executed_steps % gif_frame_stride == 0)
        frame_was_rendered = _render_env_frame(
            env,
            render=render,
            save_animation=save_animation,
            save_frame=should_save_gif_frame,
        )
        executed_steps += 1

        # Stop early when simulated navigation time exceeds the requested time limit.
        # navigation_time is measured as executed_steps * sim_dt (simulated seconds).
        if (executed_steps * sim_dt) > float(time_limit) and env.status != "Arrived":
            time_limit_exceeded = True
            if verbose:
                print(
                    f"Time limit exceeded ({executed_steps * sim_dt:.3f}s > {float(time_limit):.3f}s), stopping simulation"
                )
            break

    if render and save_animation and executed_steps > 0 and not frame_was_rendered:
        _render_env_frame(
            env,
            render=render,
            save_animation=save_animation,
            save_frame=True,
        )

    if display:
        env.set_title(
            _format_runtime_title(
                robot_shape=robot_shape,
                step_idx=executed_steps,
                sim_time=executed_steps * sim_dt,
                min_obstacle_distance=last_min_obstacle_distance,
            )
            + f" | status={env.status}"
        )

    env.end(
        shutdown_wait,
        ani_name=f"{ani_name}_{robot_shape}",
        last_frame_duration=max(0.0, float(gif_last_frame_duration)),
    )
    _cleanup_animation_buffer_if_created(
        animation_buffer_dir,
        existed_before_run=animation_buffer_existed,
        save_animation=save_animation,
    )

    navigation_time = executed_steps * sim_dt
    if navigation_time > time_limit and env.status != "Arrived":
        time_limit_exceeded = True

    average_robot_speed = accumulated_speed / max(speed_samples, 1)
    terminal_pose = np.array(env.robot.state, dtype=np.float32).reshape(-1)
    path_length += float(np.linalg.norm(terminal_pose[:2] - prev_robot_pose))

    status = env.status
    if status == "Arrived" and navigation_time > time_limit:
        status = "ArrivedLate"
        time_limit_exceeded = True
    elif time_limit_exceeded:
        status = "Timeout"

    result = {
        "robot_shape": robot_shape,
        "seed": seed,
        "num_dynamic_obstacles": num_dynamic_obstacles,
        "polygon_shape_mode": polygon_shape_mode,
        "status": status,
        "success": status == "Arrived" and navigation_time <= time_limit,
        "navigation_time": float(navigation_time),
        "path_length": float(path_length),
        "average_robot_speed": float(average_robot_speed),
        "time_limit": float(time_limit),
        "time_limit_exceeded": bool(time_limit_exceeded),
    }

    if verbose:
        print("Run summary:")
        print(f"  status: {result['status']}")
        print(f"  navigation_time: {result['navigation_time']:.2f} s")
        print(f"  path_length: {result['path_length']:.2f} m")
        print(f"  average_robot_speed: {result['average_robot_speed']:.2f} m/s")

    return result



if __name__ == "__main__":
    report_jax_device()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot_shape",
        type=str,
        default="t",
        choices=sorted(ROBOT_SHAPES.keys()),
        help="Robot footprint to evaluate: exact T (t), convex T cover (t_convex), or rectangle cover (rect).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generated dynamic obstacles.")
    parser.add_argument(
        "--num_dynamic_obstacles",
        type=int,
        default=8,
        help="Number of roaming dynamic obstacles placed inside the corridor.",
    )
    parser.add_argument(
        "--num_static_obstacles",
        type=int,
        default=0,
        help="Number of static (non-moving) obstacles placed inside the corridor.",
    )
    parser.add_argument(
        "--polygon_shape_mode",
        type=str,
        default="convex",
        choices=POLYGON_SHAPE_MODES,
        help="Polygon obstacle shape type: convex, concave, or mixed.",
    )
    parser.add_argument("--time_limit", type=float, default=50.0, help="Treat runs taking longer than this many seconds as failures.")
    parser.add_argument("--generated-env-dir", type=str, default="generated_envs", help="Directory to store generated environment YAMLs relative to `ignored/` unless an absolute path is provided")
    parser.add_argument("--result-file", type=str, default=None, help="Write run result JSON to this path")
    parser.add_argument("-a", "--save_animation", action="store_true", help="Save animation as a GIF.")
    parser.add_argument(
        "--gif-frame-stride",
        type=int,
        default=1,
        help="Keep one GIF frame every N simulation steps when saving animation.",
    )
    parser.add_argument(
        "--gif-last-frame-duration",
        type=float,
        default=1.0,
        help="Final GIF frame hold time in seconds.",
    )
    parser.add_argument("--show_rollouts", action="store_true", help="Visualize rollout trajectories.")
    parser.add_argument("--print_costs", action="store_true", help="Print cost breakdown periodically.")
    parser.add_argument("--cost_interval", type=int, default=20, help="Print cost breakdown every N steps.")
    args = parser.parse_args()

    result = main(
        robot_shape=args.robot_shape,
        seed=args.seed,
        num_dynamic_obstacles=args.num_dynamic_obstacles,
        polygon_shape_mode=args.polygon_shape_mode,
        time_limit=args.time_limit,
        save_animation=args.save_animation,
        gif_frame_stride=args.gif_frame_stride,
        gif_last_frame_duration=args.gif_last_frame_duration,
        show_rollouts=args.show_rollouts,
        print_costs=args.print_costs,
        cost_interval=args.cost_interval,
        generated_env_dir=args.generated_env_dir,
        num_static_obstacles=args.num_static_obstacles,
    )
    # optional: write result JSON when requested
    # _maybe_write_result_file(result, args.result_file)
