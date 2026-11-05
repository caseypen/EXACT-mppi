# Map 1 with F-shaped robot and T-MPPI

This example uses the reconstructed `maps/map1` environment and digitized guide route, with an F-shaped robot footprint assembled from three rectangles and a differential-drive T-MPPI planner. The footprint has the same overall 1.8 m by 2.4 m bounds as the previous robot shape.

The guide is interpolated at 0.2 m spacing, and the reference cursor advances continuously in order. Intermediate poses are guide samples rather than arrival goals; only the final goal activates the goal critics. The lower section routes around the dense obstacle cluster.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py --live
```

Use `--max-steps N` to change the safety limit (default: 5000). The [F-shaped robot PNG](./map1_fshape_figure.png) shows the final rendered frame from the latest run. The GIF is saved locally as `map1_fshape_mppi.gif` when run with `--save-animation`.

In `--live` and saved-animation runs, guide poses are labeled `W1`, `W2`, and so on in `ipath.waypoints` order.

The robot arrived in 849 steps with a path length of 116.90 m and final goal distance of 0.308 m.
