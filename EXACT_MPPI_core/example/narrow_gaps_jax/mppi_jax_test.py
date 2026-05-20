import argparse
from pathlib import Path
import shutil
import yaml
import numpy as np
import time
import os
import irsim
from irsim.config.path_param import path_manager
from exact_mppi.mppi_jax.controller import MPPIController
from exact_mppi.path.path_search import PathSearch
from exact_mppi.utils import env_config_to_grid

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def _report_jax_device() -> None:
    import jax

    devices = jax.devices()
    summary = ", ".join(
        f"{device.platform}:{getattr(device, 'device_kind', str(device))}"
        for device in devices
    )
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {summary}")


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def transfer_from_global_to_local_frame(
    points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    p = np.asarray(points)
    pose = np.asarray(pose).reshape(-1)
    x, y, yaw = pose[0], pose[1], pose[2]

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    trans = np.array([x, y])

    out = p.copy()
    xy_g = out[..., :2]
    xy_l = (xy_g - trans) @ R
    out[..., :2] = xy_l

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
    R = np.array([[c, -s], [s, c]])
    trans = np.array([x, y])

    out = p.copy()
    xy_l = out[..., :2]
    xy_g = xy_l @ R.T + trans
    out[..., :2] = xy_g

    if out.shape[-1] >= 3:
        out[..., 2] = _wrap_to_pi(out[..., 2] + yaw)

    return out


def load_env_config(env_path):
    with open(env_path, "r") as f:
        return yaml.safe_load(f)


def load_planner_config():
    """Load planner.yaml to pick up mosaic_unit_vertices (matches original setup)."""
    planner_path = Path(__file__).parent / "planner.yaml"
    if planner_path.exists():
        with open(planner_path, "r") as f:
            return yaml.safe_load(f)
    return {}


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

def _path_list_to_array(path_list: list[np.ndarray]) -> np.ndarray | None:
    if not path_list:
        return None
    path = np.array([[p[0, 0], p[1, 0], p[2, 0]] for p in path_list], dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0:
        return None
    return path


def _build_reference_path(
    env_cfg: dict,
    planner_cfg: dict,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
) -> np.ndarray | None:
    ref_cfg = planner_cfg.get("reference_path", {})
    if not ref_cfg.get("enabled", False):
        return None

    path_type = str(ref_cfg.get("path_type", "line"))
    if path_type not in ["line", "astar", "asymmetric_astar"]:
        path_type = "line"

    if path_type == "line":
        return None

    grid_res = float(ref_cfg.get("grid_resolution", 0.5))
    grid_infl = float(ref_cfg.get("grid_inflation", 0.5))
    grid_origin = env_cfg.get("world", {}).get("offset", [0.0, 0.0])
    grid_map = env_config_to_grid(
        env_cfg, resolution=grid_res, inflation_radius=grid_infl
    )

    vehicle_polygons = planner_cfg.get("MPPI", {}).get("vertices")
    path_searcher = PathSearch(
        grid_map,
        resolution=grid_res,
        origin=grid_origin,
        curve_style=path_type,
        vehicle_polygons=vehicle_polygons,
    )
    grid_path, _ = path_searcher.find_initial_path(start_pose, goal_pose)
    if not grid_path:
        print("Warning: A* path search failed; falling back to line path.")
        return None

    world_path = path_searcher.path_to_world_coords(grid_path, interval=grid_res)
    if world_path:
        world_path[-1][0, 0] = goal_pose[0]
        world_path[-1][1, 0] = goal_pose[1]
        world_path[-1][2, 0] = goal_pose[2]
    return _path_list_to_array(world_path)


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
    if top_n is not None:
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

    for i in range(len(ranges)):
        scan_range = ranges[i]
        angle = angles[i]

        if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
            if angle > angle_range[0] and angle < angle_range[1]:
                point = np.array(
                    [[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]]
                )
                point_cloud.append(point)

    if len(point_cloud) == 0:
        return None, None

    point_array = np.hstack(point_cloud)
    s_trans, s_R = get_transform(np.c_[scan_offset])
    local_points = s_R @ point_array + s_trans

    trans, R = get_transform(state)
    global_points = (R @ local_points + trans)

    return global_points, local_points

def build_stack(planner_cfg):
    # MPPI controller
    mppi_optimizer = MPPIController(
        **planner_cfg.get("MPPI"),
    )
    return mppi_optimizer

def main(
    env_file="env.yaml",
    save_animation=False,
    ani_name="mppi_animation",
    show_rollouts=False,
    print_costs=False,
    cost_interval=20,
):
    env_path = Path(__file__).parent / env_file
    cfg = load_env_config(env_path)

    if save_animation:
        save_dir = Path("./")
        save_dir.mkdir(parents=True, exist_ok=True)
        path_manager.ani_path = str(save_dir)
        print(f"Animation will be saved to {save_dir}/{ani_name}.gif")
        
        # Clean animation buffer
        buffer_dir = Path(path_manager.ani_buffer_path)
        if buffer_dir.exists():
            shutil.rmtree(buffer_dir)
        buffer_dir.mkdir()

    env = irsim.make(str(env_path), save_ani=save_animation)
    env.step(np.array([0, 0, 0]))

    robot_cfg = cfg["robot"][0]
    
    start_state = np.array(env.robot.state, dtype=np.float32).reshape(-1)
    goal_state = np.array(env.robot.goal, dtype=np.float32).reshape(-1)

    # Load planner configuration
    planner_cfg = load_planner_config()

    mppi = build_stack(planner_cfg)

    footprint_type = planner_cfg["MPPI"].get("footprint_type", "rectangle")
    vertices = planner_cfg["MPPI"].get("vertices")
    
    if footprint_type == "polygon":
        mppi.setPolygonFootprint(vertices)
    else:
        mppi.setRectangleFootprint(vertices)

    reference_path = _build_reference_path(cfg, planner_cfg, start_state[:3], goal_state[:3])
    global_reference_line = (
        reference_path
        if reference_path is not None
        else np.linspace(start_state[:3], goal_state[:3], 120)
    )
    local_plan_length = 30

    # Tracking metrics
    nav_start_time = time.perf_counter()
    position_history = [start_state[:2].copy()]
    velocity_history = []

    for step_idx in range(20000):
        if env.status == "Arrived":
            print("Arrived at the goal")
            break

        robot_state = env.get_robot_state()
        robot_vel = env.get_robot_velocity()
        lidar_scan = env.get_lidar_scan()

        global_lidar_points, local_lidar_points = scan_to_points(robot_state, lidar_scan)
        
        robot_pose = robot_state[:3].reshape(-1)
        robot_speed = robot_vel.reshape(-1) # TODO: only works for diff now due to ir_sim implementation
        if robot_speed.shape[0] == 2:
            robot_speed = np.array([robot_speed[0], 0.0, robot_speed[1]])   

        global_plan = _select_local_plan(reference_path, robot_pose, local_plan_length)
        if global_plan is None:
            global_plan = np.linspace(robot_pose[:3], goal_state[:3], local_plan_length)
        local_plan = transfer_from_global_to_local_frame(global_plan, robot_pose)
        local_goal = transfer_from_global_to_local_frame(goal_state[:3].reshape(-1), robot_pose)
        lidar_points = None if local_lidar_points is None else local_lidar_points.T

        t_start = time.perf_counter()
        action = mppi.computeVelocityCommands(
            robot_pose=np.array([0.0, 0.0, 0.0]),
            robot_speed=robot_speed,
            plan=local_plan,
            goal=local_goal,
            lidar_points=lidar_points,
            )
        t_elapsed = time.perf_counter() - t_start
        print(f"MPPI execution time: {t_elapsed*1000:.2f} ms")

        if print_costs and (step_idx % max(cost_interval, 1) == 0):
            breakdown = mppi.getCostBreakdown()
            if breakdown:
                summary = _summarize_cost_breakdown(breakdown)
                if summary:
                    print(f"Cost breakdown @ step {step_idx}: {summary}")

        # Get trajectories for visualization
        opt_trajectory = mppi.getOptimalTrajectory()  # Optimal trajectory (best rollout)
        all_traj_rollouts = mppi.getGeneratedTrajectories()  # All sampled rollouts
        
        # Draw visualization (same style as neupan)
        if global_lidar_points is not None and global_lidar_points.shape[1] > 0:
            env.draw_points(global_lidar_points, s=20, c="r", refresh=True)

        # Draw global and local reference lines
        if global_reference_line is not None:
            env.draw_trajectory(
                global_reference_line.T, "-k", linewidth=1.0, alpha=0.4, refresh=True
            )
        if global_plan is not None:
            env.draw_trajectory(
                global_plan.T, "b", linewidth=1.5, alpha=0.7, refresh=True
            )

        # Draw cost reference points (from active critics) in magenta
        cost_points_local = _collect_cost_reference_points(
            local_plan, all_traj_rollouts, planner_cfg
        )
        if cost_points_local is not None and cost_points_local.size > 0:
            cost_points_global = transfer_from_local_to_global_frame(
                cost_points_local, robot_pose
            )
            env.draw_points(cost_points_global.T, s=45, c="m", refresh=True)
        
        # Draw all rollouts in light cyan (background)
        if show_rollouts and all_traj_rollouts is not None:
            all_traj_rollouts = transfer_from_local_to_global_frame(all_traj_rollouts, robot_pose)
            for j in range(all_traj_rollouts.shape[0]):
                env.draw_trajectory(all_traj_rollouts[j].T, "c", linewidth=0.5, alpha=0.15, refresh=True)
        
        # Draw optimal trajectory in red (like neupan)
        if opt_trajectory is not None:
            opt_trajectory = transfer_from_local_to_global_frame(opt_trajectory, robot_pose)
            env.draw_trajectory(opt_trajectory.T, "r", linewidth=2, refresh=True)

        cmd = action.reshape(-1).astype(np.float32)
        env.step(cmd)
        
        # Track position and velocity
        current_pos = robot_pose[:2].copy()
        position_history.append(current_pos)
        current_speed = np.linalg.norm(robot_speed[:2])  # Speed magnitude
        velocity_history.append(current_speed)

        env.render()

    env.end(3, ani_name=ani_name)
    
    # Calculate and report navigation metrics
    nav_end_time = time.perf_counter()
    navigation_time = nav_end_time - nav_start_time
    
    # Calculate path length
    position_history = np.array(position_history)
    if len(position_history) > 1:
        path_segments = np.diff(position_history, axis=0)
        path_length = np.sum(np.linalg.norm(path_segments, axis=1))
    else:
        path_length = 0.0
    
    # Calculate mean speed
    if len(velocity_history) > 0:
        mean_speed = np.mean(velocity_history)
        max_speed = np.max(velocity_history)
    else:
        mean_speed = 0.0
        max_speed = 0.0
    
    # Report metrics
    print("\n" + "="*60)
    print("NAVIGATION TASK COMPLETED")
    print("="*60)
    print(f"Navigation Time:     {navigation_time:.2f} seconds")
    print(f"Path Length:         {path_length:.2f} meters")
    print(f"Mean Speed:          {mean_speed:.2f} m/s")
    print(f"Max Speed:           {max_speed:.2f} m/s")
    print(f"Number of Steps:     {step_idx + 1}")
    print(f"Average Step Time:   {navigation_time / (step_idx + 1):.4f} seconds")
    print("="*60)


if __name__ == "__main__":
    _report_jax_device()
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    parser.add_argument("--env", type=str, default="env.yaml")
    parser.add_argument(
        "--show_rollouts", action="store_true", help="visualize rollout trajectories"
    )
    parser.add_argument(
        "--print_costs", action="store_true", help="print cost breakdown"
    )
    parser.add_argument(
        "--cost_interval", type=int, default=20, help="print cost breakdown every N steps"
    )
    args = parser.parse_args()
    main(
        env_file=args.env,
        save_animation=args.save_animation,
        show_rollouts=args.show_rollouts,
        print_costs=args.print_costs,
        cost_interval=args.cost_interval,
    )
