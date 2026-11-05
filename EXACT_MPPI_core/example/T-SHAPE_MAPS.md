# T-MPPI examples for Maps 1–6

Each `mapN_tshape` directory contains its environment, planner configuration, and case runner. Maps 1–3 use differential drive; Maps 4–6 use Ackermann steering.

## Run a map

From the repository root, choose a map and pass any run options in one command:

```bash
.exact_mppi/bin/python EXACT_MPPI_core/example/run_tshape_map.py --map 3 --live --max-steps 5000
```

`--map` accepts `1` through `6`. Available options include `--live`, `--show-rollouts`, `--print-costs`, `--diagnostics`, and `--save-trajectory`. Add `--save-animation` to create a GIF locally; GIFs and animation buffers are ignored by Git. The case PNGs below are the shareable previews.

## Cases

| Map | Example | PNG preview |
|---|---|---|
| 1 | [map1_tshape](./map1_tshape/) | [preview](./map1_tshape/map1_tshape_figure.png) |
| 2 | [map2_tshape](./map2_tshape/) | [preview](./map2_tshape/map2_tshape_figure.png) |
| 3 | [map3_tshape](./map3_tshape/) | [preview](./map3_tshape/map3_tshape_preview.png) |
| 4 | [map4_tshape](./map4_tshape/) | [preview](./map4_tshape/map4_tshape_figure.png) |
| 5 | [map5_tshape](./map5_tshape/) | [preview](./map5_tshape/map5_tshape_preview.png) |
| 6 | [map6_tshape](./map6_tshape/) | [preview](./map6_tshape/map6_tshape_preview.png) |

Map routes are approximate reconstructions from the supplied figures. Intermediate guide poses are path samples; only the final goal activates the goal critics.
