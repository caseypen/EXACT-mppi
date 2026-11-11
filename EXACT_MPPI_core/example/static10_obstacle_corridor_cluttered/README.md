# Static 10-obstacle corridor case

This case evaluates the T-shaped differential-drive robot between two corridor walls with ten seeded, randomly placed static polygon obstacles. It uses the MPPI settings in `planner.yaml`, with PointCBF disabled and the normal obstacle repulsion, collision cost, collision margins, and trajectory validator enabled. The MPPI horizon is 8 seconds (80 steps at 0.1 seconds per step). Each trial has a 60-second simulation timeout.

From the repository root, run all 50 seeds (0–49) with:

```bash
python EXACT_MPPI_core/example/static10_obstacle_corridor_cluttered/run.py \
  --output EXACT_MPPI_core/example/static10_obstacle_corridor_cluttered/results/seeds_0_49 \
  --seeds $(seq 0 49)
```

The run creates a reproducibility manifest, per-seed generated environment and planner configs, feasibility checks, trajectories, and aggregate `summary.json` and `table.md`. Use a Python environment with the dependencies in `EXACT_MPPI_core/requirements.txt` and JAX installed. A GPU is optional; the script uses the available JAX backend. CPU affinity can be set with `--cpus 0,1,2,3` when needed.

The environment is generated from each seed using the same seed for obstacle placement and MPPI sampling. The runner checks that each map has ten internal obstacles and two corridor walls, then records the exact layout and controller outcome.
