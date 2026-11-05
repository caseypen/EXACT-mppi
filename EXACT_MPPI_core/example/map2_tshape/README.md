# Map 2 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map2` environment and a digitized continuous guide route, with the T-shaped mosaic footprint and differential-drive T-MPPI planner. The Dubins guide uses a 2 m turn radius. The lower route detours around the tight polygon cluster. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py --live
```

Guide poses are marked with their retained W labels. Use `--max-steps N` to change the safety limit (default: 5000). The [PNG preview](./map2_tshape_figure.png) shows the final rendered frame. The latest run arrived in 694 steps with a path length of 109.83 m and final goal distance of 0.407 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
