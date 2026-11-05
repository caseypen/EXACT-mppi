"""Shared runner for the six reconstructed T-shaped robot map examples."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

EXAMPLE_ROOT = Path(__file__).resolve().parent
CORRIDOR_EXAMPLE = EXAMPLE_ROOT / "corridor_jax_cluttered"
sys.path.insert(0, str(CORRIDOR_EXAMPLE))
import mppi_jax_test as corridor  # noqa: E402
from irsim.config.path_param import path_manager  # noqa: E402


class StopRun(Exception):
    pass


def main(case_dir: Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--save-animation", action="store_true")
    parser.add_argument("--live", action="store_true", help="show the IR-SIM window while running")
    parser.add_argument("--show-rollouts", action="store_true")
    parser.add_argument("--print-costs", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true", help="save robot poses to CSV")
    parser.add_argument("--diagnostics", action="store_true", help="print Ackermann commands and state while running")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--ani-name", help="output animation name (without .gif)")
    args = parser.parse_args()

    planner_path = case_dir / "planner.yaml"
    planner_config = yaml.safe_load(planner_path.read_text(encoding="utf-8"))
    guide_cfg = planner_config.get("ipath", {})
    guide_waypoints = np.asarray(
        [np.asarray(point, dtype=np.float32)[:2] for point in guide_cfg.get("waypoints", [])],
        dtype=np.float32,
    )
    waypoint_labels = guide_cfg.get("waypoint_ids", range(1, len(guide_waypoints) + 1))
    corridor.load_planner_config = lambda: yaml.safe_load(
        planner_path.read_text(encoding="utf-8")
    )

    def build_guide(_env_cfg, planner_cfg, start_pose, goal_pose):
        points = [np.asarray(start_pose, dtype=np.float32)[:3]]
        for candidate in [
            *(np.asarray(p, dtype=np.float32)[:3] for p in planner_cfg["ipath"]["waypoints"]),
            np.asarray(goal_pose, dtype=np.float32)[:3],
        ]:
            previous = points[-1]
            yaw_delta = (float(candidate[2] - previous[2]) + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(candidate[:2] - previous[:2]) < 1e-4 and abs(yaw_delta) < 1e-4:
                continue
            points.append(candidate)
        ref_cfg = planner_cfg.get("reference_path", {})
        spacing = float(ref_cfg.get("path_spacing", ref_cfg.get("grid_resolution", 0.5)))
        curve_style = str(planner_cfg.get("ipath", {}).get("curve_style", "line"))
        if curve_style == "dubins":
            from gctl import curve_generator

            dubins_points = curve_generator().generate_curve(
                curve_style="dubins",
                way_points=[point.reshape(3, 1) for point in points],
                step_size=spacing,
                min_radius=float(planner_cfg["ipath"].get("min_radius", 4.0)),
                include_gear=False,
            )
            return np.asarray(
                [np.asarray(point, dtype=np.float32).reshape(-1)[:3] for point in dubins_points],
                dtype=np.float32,
            )

        sampled = []
        for start, end in zip(points[:-1], points[1:]):
            delta = end[:2] - start[:2]
            count = max(1, int(np.ceil(float(np.linalg.norm(delta)) / spacing)))
            yaw_delta = (float(end[2] - start[2]) + np.pi) % (2 * np.pi) - np.pi
            for i in range(count):
                fraction = i / count
                pose = start + fraction * (end - start)
                pose[2] = start[2] + fraction * yaw_delta
                sampled.append(pose)
        sampled.append(points[-1])
        return np.asarray(sampled, dtype=np.float32)

    corridor._build_reference_path = build_guide
    cursor = {"index": 0}

    def select_local(path, robot_pose, plan_length):
        if path is None or len(path) == 0:
            return None
        path = np.asarray(path)
        cursor["path"] = path
        index = min(cursor["index"], len(path) - 1)
        ref_cfg = planner_config.get("reference_path", {})
        spacing = float(ref_cfg.get("path_spacing", ref_cfg.get("grid_resolution", 0.5)))
        if guide_cfg.get("sequential_cursor", False):
            # Search only forward along a bounded route arc. This keeps the
            # cursor monotonic while letting it catch up when obstacle
            # avoidance or a shortcut puts the robot beside a later sample.
            capture = max(
                float(guide_cfg.get("close_threshold", 0.1)),
                float(guide_cfg.get("arrive_threshold", 0.1)),
                spacing * 0.5,
                3.5,
            )
            search_distance = float(guide_cfg.get("cursor_search_distance", 10.0))
            if search_distance <= 0.0:
                raise ValueError("cursor_search_distance must be positive")
            search_ahead = max(1, int(np.ceil(search_distance / spacing)))
            max_advance_distance = guide_cfg.get("cursor_max_advance_distance")
            max_advance_samples = (
                max(1, int(np.floor(float(max_advance_distance) / spacing)))
                if max_advance_distance is not None
                else None
            )
            starting_index = index
            candidates = path[index:min(len(path), index + search_ahead + 1)]
            distances = np.linalg.norm(
                candidates[:, :2] - np.asarray(robot_pose[:2])[None, :], axis=1
            )
            if len(distances) and float(distances.min()) <= capture:
                advance = int(np.argmin(distances))
                if max_advance_samples is not None:
                    advance = min(advance, max_advance_samples)
                cursor["index"] = index + advance
                index += advance

            # If the robot is not close to a forward sample, still advance
            # when it has passed the current segment endpoint.
            if index < len(path) - 1:
                segment = path[index + 1, :2] - path[index, :2]
                segment_length_sq = float(np.dot(segment, segment))
                progress = (
                    float(np.dot(np.asarray(robot_pose[:2]) - path[index, :2], segment))
                    / segment_length_sq
                    if segment_length_sq > 1e-8
                    else 0.0
                )
                within_advance_limit = (
                    max_advance_samples is None
                    or index - starting_index < max_advance_samples
                )
                if progress >= 1.0 and within_advance_limit:
                    cursor["index"] = index + 1
                    index += 1
            lookahead_distance = float(
                ref_cfg.get("local_plan_lookahead_distance", 40.0)
            )
            if lookahead_distance <= 0.0:
                raise ValueError("local_plan_lookahead_distance must be positive")
            local_length = max(
                plan_length, int(np.ceil(lookahead_distance / spacing))
            )
            local = path[index:min(len(path), index + local_length)]
            if len(local) < local_length:
                local = np.vstack(
                    (local, np.repeat(path[-1][None, :], local_length - len(local), axis=0))
                )
            return local

        capture = max(
            float(guide_cfg.get("close_threshold", 0.1)),
            float(guide_cfg.get("arrive_threshold", 0.1)),
            spacing * 0.5,
            3.5,
        )
        max_advance = max(1, int(np.floor(0.5 / spacing)))
        search_ahead = max(max_advance, int(np.ceil(10.0 / spacing)))
        candidates = path[index:min(len(path), index + search_ahead + 1)]
        distances = np.linalg.norm(
            candidates[:, :2] - np.asarray(robot_pose[:2])[None, :], axis=1
        )
        if len(distances) and float(distances.min()) <= capture:
            index += min(int(np.argmin(distances)), max_advance)
            cursor["index"] = index
        lookahead_distance = float(
            ref_cfg.get("local_plan_lookahead_distance", 40.0)
        )
        if lookahead_distance <= 0.0:
            raise ValueError("local_plan_lookahead_distance must be positive")
        local_length = max(
            plan_length, int(np.ceil(lookahead_distance / spacing))
        )
        local = path[index:min(len(path), index + local_length)]
        if len(local) < local_length:
            local = np.vstack((local, np.repeat(path[-1][None, :], local_length - len(local), axis=0)))
        return local

    corridor._select_local_plan = select_local
    corridor._report_jax_device()
    animation_buffer = case_dir / ".animation_buffer"
    path_manager.ani_buffer_path = str(animation_buffer)
    if args.save_animation and animation_buffer.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(animation_buffer)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    animation_buffer.mkdir(exist_ok=True)
    original_make = corridor.irsim.make
    run = {"env": None, "steps": 0, "trajectory": []}
    animation_name = args.ani_name or f"{case_dir.name}_t_mppi"

    def monitored_make(*make_args, **make_kwargs):
        make_kwargs.setdefault("display", args.live)
        env = original_make(*make_args, **make_kwargs)
        run["env"] = env
        run["trajectory"].append(
            np.asarray(env.get_robot_state(), dtype=float).reshape(-1)[:3].copy()
        )
        for obstacle in env.obstacle_list:
            behavior = getattr(obstacle, "obj_behavior", None)
            behavior_cfg = getattr(behavior, "behavior_dict", {}) or {}
            if not behavior_cfg.get("bounce", False):
                continue
            start = np.asarray(obstacle.state, dtype=float).reshape(-1)
            goal = np.asarray(obstacle.goal, dtype=float).reshape(-1)
            obstacle._tshape_bounce_points = (start.copy(), goal.copy())
            obstacle._tshape_bounce_target = 1

        original_render = env.render

        def render_with_guides(*render_args, **render_kwargs):
            if args.save_animation:
                animation_buffer.mkdir(exist_ok=True)
            if guide_waypoints.size and guide_cfg.get("show_waypoints", True):
                env.draw_points(
                    guide_waypoints.T, s=95, c="#ffd54f", marker="o",
                    edgecolors="#202020", linewidths=1.2, refresh=True, zorder=20,
                )
                for index, (x, y) in zip(waypoint_labels, guide_waypoints):
                    label = env._env_plot.ax.annotate(
                        f"W{index}", (float(x), float(y)), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, fontweight="bold",
                        color="#202020",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "#fff3b0", "ec": "#202020", "alpha": 0.95},
                        zorder=21,
                    )
                    env._env_plot.dyna_point_list.append(label)
            path = cursor.get("path")
            index = cursor.get("index", 0)
            if guide_cfg.get("show_cursor", True) and path is not None and len(path):
                cursor_pose = np.asarray(path[min(index, len(path) - 1), :2])
                env.draw_points(
                    cursor_pose.reshape(1, 2).T,
                    s=240,
                    c="#f500ff",
                    marker="X",
                    edgecolors="#202020",
                    linewidths=1.5,
                    refresh=True,
                    zorder=25,
                )
                label = env._env_plot.ax.annotate(
                    f"CURSOR {index}",
                    (float(cursor_pose[0]), float(cursor_pose[1])),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#f500ff",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#202020", "alpha": 0.95},
                    zorder=26,
                )
                env._env_plot.dyna_point_list.append(label)
            return original_render(*render_args, **render_kwargs)

        env.render = render_with_guides
        original_step = env.step

        def monitored_step(action):
            # MPPI's Ackermann output is [speed, steering angle]. The local
            # IR-SIM angular mode accepts [speed, yaw rate], so convert before
            # applying simulator velocity and acceleration limits.
            if (
                env.robot.kinematics == "acker"
                and getattr(env.robot.kf, "mode", "steer") == "angular"
                and planner_config["MPPI"].get("motion_model", "diff") in ("acker", "ackermann")
            ):
                speed, steer = np.asarray(action, dtype=float).reshape(-1)
                action = np.array([speed, speed * np.tan(steer) / env.robot.wheelbase])
            result = original_step(action)
            run["trajectory"].append(
                np.asarray(env.get_robot_state(), dtype=float).reshape(-1)[:3].copy()
            )
            for obstacle in env.obstacle_list:
                bounce_points = getattr(obstacle, "_tshape_bounce_points", None)
                if bounce_points is None or not obstacle.arrive_flag:
                    continue
                obstacle._tshape_bounce_target = 1 - obstacle._tshape_bounce_target
                obstacle.set_goal(
                    bounce_points[obstacle._tshape_bounce_target].tolist()
                )
                obstacle.arrive_flag = False
            run["steps"] += 1
            if args.diagnostics and run["steps"] % 10 == 0:
                pose = np.asarray(env.get_robot_state()).reshape(-1)
                robot = env.robot
                velocity = np.asarray(robot._velocity).reshape(-1)
                measured_twist = np.asarray(env.get_robot_velocity()).reshape(-1)
                mppi_speed = corridor._simulator_velocity_to_mppi_speed(
                    robot, measured_twist
                )
                action_array = np.asarray(action).reshape(-1)
                state = np.asarray(robot._state).reshape(-1)
                steering_info = (
                    f", state_steer={state[3]:.3f}"
                    if state.size > 3
                    else ""
                )
                print(
                    f"Progress: step={run['steps']}, guide_index={cursor['index']}, "
                    f"pose=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}), "
                    f"robot_velocity={velocity.tolist()}, action={action_array.tolist()}"
                    f", simulator_twist={measured_twist.tolist()}, "
                    f"mppi_initial_speed={mppi_speed.tolist()}{steering_info}, "
                    f"stop={robot.stop_flag}, collision={robot.collision_flag}"
                )
            elif run["steps"] % 100 == 0:
                pose = np.asarray(env.get_robot_state()).reshape(-1)
                print(f"Progress: step={run['steps']}, guide_index={cursor['index']}, pose=({pose[0]:.2f}, {pose[1]:.2f})")
            if env.status == "Collision" or env.robot.stop_flag or run["steps"] >= args.max_steps:
                if args.save_animation:
                    env.render()
                raise StopRun
            return result

        env.step = monitored_step
        return env

    corridor.irsim.make = monitored_make
    try:
        corridor.main(
            env_file=str(case_dir / "env.yaml"),
            save_animation=args.save_animation,
            ani_name=animation_name,
            show_rollouts=args.show_rollouts,
            print_costs=args.print_costs,
            cost_interval=20,
        )
    except StopRun:
        env = run["env"]
        pose = np.asarray(env.get_robot_state()).reshape(-1)
        colliding_with = [getattr(obj, "name", str(obj)) for obj in env.robot.collision_obj]
        print(
            f"Run stopped with status={env.status}, stop={env.robot.stop_flag}, "
            f"collision={env.robot.collision_flag}, colliding_with={colliding_with} "
            f"after {run['steps']} steps at ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})."
        )
        env.end(3, ani_name=animation_name)
    except OSError as error:
        # ir-sim can finish writing a GIF and then race while removing its
        # temporary PNG buffer. Preserve the successful artifact and report it.
        if not (args.save_animation and "Directory not empty" in str(error)
                and (case_dir / f"{animation_name}.gif").is_file()):
            raise
        print(f"Animation saved to {animation_name}.gif; ignored a temporary-frame cleanup race.")
    finally:
        corridor.irsim.make = original_make

    if args.save_trajectory and run["trajectory"]:
        trajectory_path = case_dir / f"{case_dir.name}_trajectory.csv"
        trajectory = np.asarray(run["trajectory"], dtype=float)
        np.savetxt(
            trajectory_path,
            trajectory,
            delimiter=",",
            header="x,y,yaw",
            comments="",
        )
        print(f"Trajectory saved to {trajectory_path}")
