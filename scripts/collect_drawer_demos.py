#!/usr/bin/env python3
"""Collect scripted FrankaDrawerOpen demonstrations.

Hardcoded trajectory (top-down grasp of the added handle, then pull toward the
robot along -x):

    1. move above the handle (gripper open)
    2. descend onto the handle bar (gripper open)
    3. close the gripper around the handle
    4. pull along -x (the drawer opens toward the robot) until success

Saves a robomimic-style HDF5. Run from the project root:

    python scripts/collect_drawer_demos.py --out data/drawer.hdf5 --n 10
    python scripts/collect_drawer_demos.py --out data/drawer.hdf5 --n 1 --render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import franka_drawer_bottle  # noqa: F401  (registers FrankaDrawerOpen)
from scripts.demo_common import Recorder, collect

ENV_NAME = "FrankaDrawerOpen"
HORIZON = 1500
CONTROL_FREQ = 20


def generate_episode(env, render=False, noise_scale=0.0):
    rec = Recorder(env, render=render)

    handle_id = env.handle_site_id
    handle_pos = lambda: np.array(env.sim.data.site_xpos[handle_id])
    # Small fixed lateral offset per episode (np.random already seeded by caller).
    jitter = np.random.normal(scale=noise_scale, size=3)

    # 1) above the handle, gripper open
    rec.reach(handle_pos, jitter + [0.0, 0.0, 0.10], gripper=-1.0, n_steps=70)
    # 2) descend onto the handle bar (a touch above so fingers straddle it)
    rec.reach(handle_pos, jitter + [0.0, 0.0, 0.012], gripper=-1.0, n_steps=70)
    # 3) close around the handle
    rec.reach(handle_pos, [0.0, 0.0, 0.012], gripper=1.0, n_steps=35)
    # 4) pull toward the robot (-x) until the drawer is open
    rec.reach_until(
        pos_fn=lambda: rec.eef_pos,           # target trails the eef, biased -x
        offset=[-0.05, 0.0, 0.0],
        gripper=1.0,
        max_steps=200,
        success_fn=env._check_success,
    )

    return rec.episode(success=env._check_success())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output HDF5 path, e.g. data/drawer.hdf5")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--noise-scale", type=float, default=0.0, help="Std of per-episode approach jitter (m).")
    p.add_argument("--keep-failures", action="store_true")
    args = p.parse_args()

    collect(
        env_name=ENV_NAME,
        generate_episode_fn=lambda env, render: generate_episode(env, render=render, noise_scale=args.noise_scale),
        out=args.out,
        n=args.n,
        seed=args.seed,
        render=args.render,
        keep_failures=args.keep_failures,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
    )


if __name__ == "__main__":
    main()
