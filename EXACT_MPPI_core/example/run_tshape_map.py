#!/usr/bin/env python3
"""Run one of the six reconstructed T-shaped robot map examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXAMPLE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", "-m", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--live", action="store_true", help="show the simulator window")
    parser.add_argument("--show-rollouts", action="store_true")
    parser.add_argument("--print-costs", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--save-animation", action="store_true", help="save a GIF locally")
    parser.add_argument("--save-trajectory", action="store_true", help="save robot poses to CSV")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--ani-name", help="GIF output name without .gif")
    args = parser.parse_args()

    case_dir = EXAMPLE_DIR / f"map{args.map}_tshape"
    command = [sys.executable, str(case_dir / "run_case.py")]
    for option, flag in (
        (args.live, "--live"),
        (args.show_rollouts, "--show-rollouts"),
        (args.print_costs, "--print-costs"),
        (args.diagnostics, "--diagnostics"),
        (args.save_animation, "--save-animation"),
        (args.save_trajectory, "--save-trajectory"),
    ):
        if option:
            command.append(flag)
    command.extend(["--max-steps", str(args.max_steps)])
    if args.ani_name:
        command.extend(["--ani-name", args.ani_name])

    return subprocess.run(command, cwd=case_dir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
