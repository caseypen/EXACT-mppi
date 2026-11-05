# Map 3 with L-shaped robot and T-MPPI

This example uses the reconstructed `maps/map3` environment and a digitized continuous guide route, with an L-shaped footprint assembled from two rectangles and a differential-drive T-MPPI planner. The robot keeps the same overall 1.8 m by 2.4 m bounds as the previous T shape. Its MPPI settings match Map 1, with the PathFollow offset adjusted for Map 3's 0.5 m path spacing so the target remains 10 m ahead. The guide uses Dubins curves with a 2 m radius. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

The simulator uses differential-drive kinematics with linear velocity bounds of ±6 m/s and angular velocity bounds of ±12 rad/s. The path cursor advances by at most 0.5 m per control step.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py -a
```

Use `--max-steps N` to change the safety limit (default: 5000). The [L-shaped PNG preview](./map3_l_figure.png) and local GIF (`map3_l_mppi.gif`) show the latest run. It arrived in 705 steps with a path length of 106.73 m and final goal distance of 0.323 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
