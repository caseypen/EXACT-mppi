# Map 4 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map4` environment and selected guide poses 1, 3, 5, 9, 12, 15, 18, 21, and 23, with the T-shaped mosaic footprint and Ackermann T-MPPI planner. The robot starts aligned with the first pose, and the goal matches the last. The guide uses Dubins connections with a 2 m turning radius; intermediate guide poses are not arrival goals. Differently shaped polygon obstacles move back and forth between their configured endpoints.

The MPPI parameters and critic weights match Map3, including the 5 m forward limit used when projecting rollout endpoints onto the path. The local path cursor advances sequentially when the robot passes the current path segment endpoint; it does not search for the nearest path sample or use a distance threshold. The visualization shows the active PathFollow track point. Randomized polygon obstacles bounce between their configured endpoints.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py --live --diagnostics
```

Use `--max-steps N` to change the safety limit (default: 5000). The [PNG preview](./map4_tshape_figure.png) shows the final frame of the latest run, which arrived in 862 steps with a path length of 77.39 m and final goal distance of 0.301 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
