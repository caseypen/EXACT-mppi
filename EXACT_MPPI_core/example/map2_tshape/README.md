# Map 2 with star-shaped robot and T-MPPI

This example uses the reconstructed `maps/map2` environment and a digitized continuous guide route, with a star-shaped polygon footprint and differential-drive T-MPPI planner. The footprint keeps the previous robot's overall 1.8 m by 2.4 m dimensions. The Dubins guide uses a 2 m turn radius. The lower route detours around the tight polygon cluster. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py --live
```

Guide poses are marked with their retained W labels. Add `-a` to save a GIF, and use `--max-steps N` to change the safety limit (default: 5000). The [star-shaped PNG preview](./map2_star_figure.png) and local GIF (`map2_star_mppi.gif`) show the latest run. It arrived in 1008 steps with a path length of 118.51 m and final goal distance of 0.352 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
