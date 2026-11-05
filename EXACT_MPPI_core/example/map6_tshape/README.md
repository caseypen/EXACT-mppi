# Map 6 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map6` environment and guide poses 1, 5, 7, 9, and 12, with the T-shaped mosaic footprint and Ackermann T-MPPI planner. It uses Map 4's MPPI parameters and Dubins reference connections. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py -a
```

Use `--max-steps N` to change the safety limit (default: 5000). Robot trail alpha is now 1.0; the [refreshed PNG](./map6_alpha_figure.png) and local GIF (`map6_alpha_mppi.gif`) show the rerun. The robot arrived in 293 steps with a path length of 51.00 m and final goal distance of 0.300 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
