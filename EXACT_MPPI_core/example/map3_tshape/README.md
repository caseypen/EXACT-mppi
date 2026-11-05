# Map 3 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map3` environment and a digitized continuous guide route, with the T-shaped mosaic footprint and differential-drive T-MPPI planner. Its MPPI settings match Map 1, with the PathFollow offset adjusted for Map 3's 0.5 m path spacing so the target remains 10 m ahead. The guide uses Dubins curves with a 2 m radius. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

The simulator uses differential-drive kinematics with linear velocity bounds of ±6 m/s and angular velocity bounds of ±12 rad/s. The path cursor advances by at most 0.5 m per control step.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py -a
```

Use `--max-steps N` to change the safety limit (default: 5000). The [PNG preview](./map3_tshape_preview.png) shows the final frame. The latest run arrived in 811 steps with a path length of 105.74 m and final goal distance of 0.313 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
