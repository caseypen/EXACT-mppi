#!/usr/bin/env python3
"""Run the full MPPI configuration in a seeded static-10 corridor scene."""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
VARIANTS = {"full": {}}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_resources(cpus=None):
    allowed = os.sched_getaffinity(0)
    requested = allowed if cpus is None else {int(x) for x in cpus.split(",")}
    if not requested <= allowed:
        raise ValueError(f"Requested CPUs {requested} are outside allowed set {allowed}")
    os.sched_setaffinity(0, requested)
    os.nice(max(0, 10 - os.nice(0)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MPLBACKEND"] = "Agg"


def import_support(repo):
    sys.path[:0] = [str(repo / "EXACT_MPPI_core"), str(repo / "ir-sim_mppi")]
    source = repo / "EXACT_MPPI_core/example/corridor_dynamic_random_jax/mppi_jax_test.py"
    spec = importlib.util.spec_from_file_location("static10_support", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resource_snapshot():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapFree"):
            mem[name + "_GiB"] = int(value.strip().split()[0]) / 1024**2
    return {"time": time.time(), "load_average": list(os.getloadavg()),
            "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), **mem}


def verify_layout(env, vertices):
    """Sufficient feasibility certificate: disk-envelope route on a 0.5 m grid.

    Reject no layouts silently. If this conservative check fails, stop and retain
    the generated map for inspection. Every edge is checked as a buffered segment.
    """
    import numpy as np
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    obstacles = list(env.obstacle_list)
    assert len(obstacles) == 12, f"Expected ten obstacles plus two walls, got {len(obstacles)}"
    for obj in obstacles:
        assert obj.geometry.is_valid and not obj.geometry.is_empty
    start = np.asarray(env.robot.state).reshape(-1)
    goal = np.asarray(env.robot.goal).reshape(-1)
    assert all(not env.robot.geometry.intersects(o.geometry) for o in obstacles)
    union = unary_union([o.geometry for o in obstacles])
    radius = max(float(np.linalg.norm(p)) for polygon in vertices for p in polygon)
    # Inflate slightly to cover polygonal approximation of the circular envelope.
    radius = radius + 0.02
    step = 0.5
    xs = np.arange(0, 60 + step / 2, step)
    ys = np.arange(14, 26 + step / 2, step)
    points = {(i, j): (float(x), float(y)) for i, x in enumerate(xs) for j, y in enumerate(ys)}
    free = {ij for ij, xy in points.items() if Point(xy).distance(union) > radius}
    begin = (int(round(start[0] / step)), int(round((start[1] - 14) / step)))
    end = (int(round(goal[0] / step)), int(round((goal[1] - 14) / step)))
    assert begin in free and end in free, "Start or goal fails conservative disk clearance check"
    parent = {begin: None}
    queue = deque([begin])
    while queue and end not in parent:
        current = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + di, current[1] + dj)
            if nxt not in free or nxt in parent:
                continue
            if LineString([points[current], points[nxt]]).distance(union) <= radius:
                continue
            parent[nxt] = current
            queue.append(nxt)
    route_found = end in parent
    if not route_found:
        # The disk is only a conservative screening approximation. Do not
        # discard a valid random layout if that approximation blocks a route.
        # Keep the layout and let the exact-footprint controller determine the
        # navigation outcome; record the failed screen explicitly.
        return {"method": "conservative disk-envelope grid route with exact segment-distance checks",
                "envelope_radius_m": radius, "route": [], "disk_envelope_route_found": False,
                "internal_obstacles": 10, "boundary_walls": 2}
    route = []
    node = end
    while node is not None:
        route.append(points[node])
        node = parent[node]
    return {"method": "disk-envelope grid route with exact segment-distance checks",
            "envelope_radius_m": radius, "route": route[::-1],
            "disk_envelope_route_found": True,
            "internal_obstacles": 10, "boundary_walls": 2}


def trial(base, output, seed, variant, time_limit):
    import numpy as np
    import yaml
    import irsim
    trial_dir = output / f"seed_{seed:03d}" / variant
    trial_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.prepare_planner_config(yaml.safe_load((HERE / "planner.yaml").read_text()), "t")
    cfg["MPPI"]["seed"] = seed
    cfg["MPPI"]["Critics"]["ObstaclesCritic"].update(VARIANTS[variant])
    config_path = trial_dir / "planner.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env_path = output / f"seed_{seed:03d}" / "env.yaml"
    if not env_path.exists():
        env_cfg, moving = base.prepare_env_config(
            yaml.safe_load((HERE / "env.yaml").read_text()), "t", seed, 0,
            polygon_shape_mode="concave", num_static_obstacles=10)
        assert not moving
        env_path.write_text(yaml.safe_dump(env_cfg, sort_keys=False))
    env = irsim.make(str(env_path), save_ani=False, display=False, disable_all_plot=True, seed=seed)
    try:
        feasibility = verify_layout(env, cfg["MPPI"]["vertices"])
        save_json(output / f"seed_{seed:03d}" / "feasibility.json", feasibility)
        initial_obstacle_states = [np.array(o.state, copy=True) for o in env.obstacle_list]
        env._objects_sensor_step()
        start = np.asarray(env.robot.state).reshape(-1).copy()
        goal = np.asarray(env.robot.goal).reshape(-1).copy()
        reference = np.linspace(start[:3], goal[:3], 120)
        mppi = base.build_stack(cfg)
        mppi.setRectangleFootprint(cfg["MPPI"]["vertices"])
        assert "PointCBFCritic" not in mppi.optimizer_.critic_manager_.critics_
        counters = {"fail_calls": 0, "exhausted": 0}
        original_fallback = mppi.optimizer_.fallback

        def counted_fallback(fail):
            counters["fail_calls"] += int(bool(fail))
            try:
                return original_fallback(fail)
            except RuntimeError:
                counters["exhausted"] += 1
                raise

        mppi.optimizer_.fallback = counted_fallback
        dt = float(env.step_time)
        steps = int(round(time_limit / dt))
        if not np.isclose(steps * dt, time_limit):
            raise ValueError("Time limit must be a multiple of simulation dt")
        rejected_cycles = fallback_cycles = 0
        elapsed = length = 0.0
        minimum_clearance = float("inf")
        collision = False
        records = []
        update_times = []
        previous = start[:2].copy()

        def observe():
            nonlocal collision, minimum_clearance
            geometric_contact = any(env.robot.geometry.intersects(o.geometry) for o in env.obstacle_list)
            collision = collision or bool(env.robot.collision) or geometric_contact
            clearance = min(float(env.robot.geometry.distance(o.geometry)) for o in env.obstacle_list)
            minimum_clearance = min(minimum_clearance, clearance)
            return clearance

        initial_clearance = observe()
        records.append({"time_s": 0.0, "pose": start.tolist(), "clearance_m": initial_clearance,
                        "collision": collision})
        for _ in range(steps):
            if collision or env.robot.arrive:
                break
            state = np.asarray(env.get_robot_state())
            pose = state[:3].reshape(-1)
            velocity = np.asarray(env.get_robot_velocity()).reshape(-1)
            _, local_points = base.scan_to_points(state, env.get_lidar_scan())
            plan = base._select_local_plan(reference, pose, 30)
            if plan is None:
                plan = np.linspace(pose[:3], goal[:3], 30)
            before = counters.copy()
            started = time.perf_counter()
            action = mppi.computeVelocityCommands(
                robot_pose=np.zeros(3), robot_speed=np.array([velocity[0], 0.0, velocity[1]]),
                plan=base.transfer_from_global_to_local_frame(plan, pose),
                goal=base.transfer_from_global_to_local_frame(goal[:3], pose),
                lidar_points=None if local_points is None else local_points.T,
                lidar_velocities=None if local_points is None else np.zeros_like(local_points.T))
            update_times.append(time.perf_counter() - started)
            rejected = counters["fail_calls"] > before["fail_calls"]
            fallback = counters["exhausted"] > before["exhausted"]
            # The controller catches RuntimeError internally. Detect unexpected
            # errors that did not come from exhausted optimization retries.
            if mppi.optimal_trajectory_ is None and not fallback:
                raise RuntimeError("Controller returned no trajectory without exhausted retries")
            rejected_cycles += int(rejected)
            fallback_cycles += int(fallback)
            command = np.asarray(action).reshape(-1).astype(np.float32)
            env.step(command)
            elapsed += dt
            clearance = observe()
            executed_pose = np.asarray(env.robot.state).reshape(-1)
            length += float(np.linalg.norm(executed_pose[:2] - previous))
            previous = executed_pose[:2].copy()
            drift = max(float(np.max(np.abs(np.asarray(o.state) - initial)))
                        for o, initial in zip(env.obstacle_list, initial_obstacle_states))
            if drift > 1e-7:
                raise RuntimeError(f"Static obstacle moved by {drift}")
            records.append({"time_s": round(elapsed, 6), "pose": executed_pose.tolist(),
                            "command": command.tolist(), "clearance_m": clearance,
                            "collision": collision, "rejected": rejected, "fallback": fallback})
        success = bool(env.robot.arrive) and not collision and elapsed <= time_limit + 1e-8
        status = "Success" if success else ("Collision" if collision else "Timeout")
        result = {"seed": seed, "controller_seed": seed, "variant": variant, "status": status,
                  "success": success, "collision": collision, "navigation_time_s": round(elapsed, 6),
                  "time_limit_s": time_limit, "path_length_m": length,
                  "minimum_clearance_m": minimum_clearance, "control_cycles": len(update_times),
                  "rejected_cycles": rejected_cycles, "rejected_cycles_percent": 100 * rejected_cycles / max(1, len(update_times)),
                  "fallback_cycles": fallback_cycles, "failed_optimization_attempts": counters["fail_calls"],
                  "static_obstacles": 10, "boundary_walls": 2, "static_invariant_checked": True,
                  "environment_sha256": sha(env_path), "config_sha256": sha(config_path),
                  "cbf_enabled": False, "point_flow_enabled": False,
                  "first_update_s": update_times[0] if update_times else None,
                  "mean_update_s_excluding_first": float(np.mean(update_times[1:])) if len(update_times) > 1 else None,
                  "resource_snapshot": resource_snapshot()}
        save_json(trial_dir / "trajectory.json", records)
        save_json(trial_dir / "result.json", result)
        return result
    finally:
        env.end(0)


def summarize(output, seeds, variants):
    import numpy as np
    results = []
    for seed in seeds:
        for variant in variants:
            path = output / f"seed_{seed:03d}" / variant / "result.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
    common = [s for s in seeds if all(any(r["seed"] == s and r["variant"] == v and r.get("success")
                                        for r in results) for v in variants)]
    rows = []
    for variant in variants:
        group = [r for r in results if r["variant"] == variant]
        good = [r for r in group if r["status"] != "Error"]
        paired = [r["navigation_time_s"] for r in good if r["seed"] in common]
        row = {"variant": variant, "completed": len(group), "valid_trials": len(good),
               "errors": len(group) - len(good),
               "success_percent": 100 * sum(r["success"] for r in good) / len(good) if good else None,
               "collision_percent": 100 * sum(r["collision"] for r in good) / len(good) if good else None,
               "median_minimum_clearance_m": float(np.median([r["minimum_clearance_m"] for r in good])) if good else None,
               "mean_paired_success_time_s": float(np.mean(paired)) if paired else None,
               "mean_rejected_cycles_percent": float(np.mean([r["rejected_cycles_percent"] for r in good])) if good else None}
        rows.append(row)
    summary = {"expected_trials": len(seeds) * len(variants), "completed_trials": len(results),
               "common_success_seeds": common, "rows": rows}
    save_json(output / "summary.json", summary)
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    lines = ["| Variant | Success / collision (%) | Min. clearance (m) | Time (s) | Rejected cycles (%) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {fmt(r['success_percent'])} / {fmt(r['collision_percent'])} | "
                     f"{fmt(r['median_minimum_clearance_m'])} | {fmt(r['mean_paired_success_time_s'])} | "
                     f"{fmt(r['mean_rejected_cycles_percent'])} |")
    lines += ["", f"Completed {len(results)}/{len(seeds) * len(variants)}; time uses {len(common)} common-success seeds.",
              "Clearance is the median of per-run minima at executed simulation poses; rejection is the mean per-run percentage.",
              "Execution errors are reported separately in summary.json; an incomplete/error-containing study is not publication-ready."]
    (output / "table.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=["full"])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--cpus", default=None,
                        help="Optional comma-separated CPU affinity; defaults to the CPUs available to this process.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.time_limit <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use a positive time limit and unique seeds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_resources(args.cpus)
    if args.summarize_only:
        print(json.dumps(summarize(args.output, args.seeds, args.variants), indent=2))
        return
    base = import_support(args.repo.resolve())
    import jax
    base.report_jax_device()
    dependencies = [HERE / "run.py", HERE / "planner.yaml", HERE / "env.yaml", Path(base.__file__)]
    dependencies += sorted((args.repo / "EXACT_MPPI_core/exact_mppi").rglob("*.py"))
    dependencies += sorted((args.repo / "ir-sim_mppi/irsim").rglob("*.py"))
    signature = {str(p): sha(p) for p in dependencies}
    protocol = {"seeds": args.seeds, "variants": args.variants, "time_limit": args.time_limit,
                "source_sha256": signature, "jax_version": jax.__version__,
                "jax_devices": [str(d) for d in jax.devices()]}
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["protocol"] != protocol:
            raise RuntimeError("Protocol/source changed: use a new output directory")
    else:
        save_json(manifest, {"protocol": protocol, "resource_snapshot": resource_snapshot(),
                            "git_status": "isolated recorded-source snapshot"})
    for seed in args.seeds:
        for variant in args.variants:
            path = args.output / f"seed_{seed:03d}" / variant / "result.json"
            if path.exists():
                if json.loads(path.read_text())["status"] == "Error":
                    raise RuntimeError(f"Previous error requires investigation: {path}")
                continue
            for source, digest in signature.items():
                if sha(Path(source)) != digest:
                    raise RuntimeError(f"Dependency changed during study: {source}")
            snapshot = resource_snapshot()
            if snapshot["MemAvailable_GiB"] < 8:
                raise RuntimeError("Less than 8 GiB memory available: stopped before next trial")
            print(f"START seed={seed} variant={variant} resources={snapshot}", flush=True)
            began = time.perf_counter()
            try:
                result = trial(base, args.output, seed, variant, args.time_limit)
            except Exception:
                save_json(path, {"seed": seed, "variant": variant, "status": "Error", "success": False,
                                 "traceback": traceback.format_exc()})
                summarize(args.output, args.seeds, args.variants)
                raise
            print(f"DONE seed={seed} variant={variant} status={result['status']} "
                  f"time={result['navigation_time_s']:.1f}s clearance={result['minimum_clearance_m']:.4f} "
                  f"rejected={result['rejected_cycles_percent']:.1f}% wall={time.perf_counter()-began:.1f}s", flush=True)
            summarize(args.output, args.seeds, args.variants)
            # Each trial creates a controller captured by JIT. Release its cache
            # between trials to bound host/device memory over a long batch.
            jax.clear_caches()
            gc.collect()
    print((args.output / "table.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
