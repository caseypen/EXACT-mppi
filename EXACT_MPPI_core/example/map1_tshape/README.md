# Map 1 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map1` environment and digitized guide route, with the T-shaped mosaic footprint and differential-drive T-MPPI planner.

The guide is interpolated at 0.2 m spacing, and the reference cursor advances continuously in order. Intermediate poses are guide samples rather than arrival goals; only the final goal activates the goal critics. The lower section routes around the dense obstacle cluster.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py --live
```

Use `--max-steps N` to change the safety limit (default: 5000). The [PNG preview](./map1_tshape_figure.png) shows the final rendered frame from the current run.

In `--live` and saved-animation runs, guide poses are labeled `W1`, `W2`, and so on in `ipath.waypoints` order.

The robot arrived in 789 steps with a path length of 117.17 m and final goal distance of 0.359 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
