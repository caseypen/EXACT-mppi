# Map 5 with T-shaped robot and T-MPPI

This example uses the reconstructed `maps/map5` environment and guide poses 1, 5, 7, 11, 14, 17, 18, and 20, with the T-shaped mosaic footprint and Ackermann T-MPPI planner. It uses Map 4's MPPI parameters, sequential reference-cursor advancement, and active PathFollow track-point visualization. Its obstacles have randomized polygon shapes. Intermediate guide poses are not arrival goals; only the final goal activates the goal critics.

Run from this directory:

```bash
../../../.exact_mppi/bin/python run_case.py -a
```

Use `--max-steps N` to change the safety limit (default: 5000). The [PNG preview](./map5_tshape_preview.png) shows the final frame of the Ackermann run. It arrived in 512 steps with a path length of 88.73 m and final goal distance of 0.305 m. Run metrics for all maps are indexed in `../T-SHAPE_MAPS.md`.
