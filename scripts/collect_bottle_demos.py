#!/usr/bin/env python3
"""Collect scripted FrankaBottleUntwist demonstrations.

Hardcoded trajectory (top-down press on the cap, then rotate the wrist about the
world z-axis to spin the cap on its hinge):

    1. move above the cap (gripper open)
    2. descend onto the cap (gripper open)
    3. close the gripper onto the cap brim
    4. press down + rotate the wrist (drz) until success
       -- if the wrist runs out of travel before the cap is turned far enough,
          release, counter-rotate the wrist, regrasp, and twist again (a ratchet)

Saves a robomimic-style HDF5. Run from the project root:

    python scripts/collect_bottle_demos.py --out data/bottle.hdf5 --n 10
    python scripts/collect_bottle_demos.py --out data/bottle.hdf5 --n 1 --render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import franka_drawer_bottle  # noqa: F401  (registers FrankaBottleUntwist)
from scripts.demo_common import Recorder, collect

ENV_NAME = "FrankaBottleUntwist"
HORIZON = 2500
CONTROL_FREQ = 20

TWIST_RATE = 0.6          # per-step wrist rotation command about world z
TWIST_STEPS = 60          # steps per twist before re-grasping
MAX_RATCHETS = 6          # max twist/regrasp cycles


def generate_episode(env, render=False, noise_scale=0.0):
    rec = Recorder(env, render=render)

    cap_id = env.cap_site_id
    cap_pos = lambda: np.array(env.sim.data.site_xpos[cap_id])
    jitter = np.random.normal(scale=noise_scale, size=3)

    # 1) above the cap, gripper open
    rec.reach(cap_pos, jitter + [0.0, 0.0, 0.10], gripper=-1.0, n_steps=60)
    # 2) descend onto the cap
    rec.reach(cap_pos, jitter + [0.0, 0.0, 0.015], gripper=-1.0, n_steps=80)
    # 3) close onto the cap
    rec.reach(cap_pos, [0.0, 0.0, 0.015], gripper=1.0, n_steps=30)

    # 4) press + twist, ratcheting (release / counter-rotate / regrasp) if needed
    for _ in range(MAX_RATCHETS):
        done = rec.reach_until(
            pos_fn=cap_pos,
            offset=[0.0, 0.0, 0.012],          # press down slightly for friction
            gripper=1.0,
            max_steps=TWIST_STEPS,
            success_fn=env._check_success,
            rot=[0.0, 0.0, TWIST_RATE],        # rotate wrist about world z
        )
        if done:
            break
        # Release and lift.
        rec.reach(cap_pos, [0.0, 0.0, 0.06], gripper=-1.0, n_steps=25)
        # Counter-rotate the wrist back while lifted and open.
        rec.reach(cap_pos, [0.0, 0.0, 0.06], gripper=-1.0, n_steps=40, rot=[0.0, 0.0, -TWIST_RATE])
        # Descend and regrasp.
        rec.reach(cap_pos, [0.0, 0.0, 0.015], gripper=-1.0, n_steps=40)
        rec.reach(cap_pos, [0.0, 0.0, 0.015], gripper=1.0, n_steps=25)

    return rec.episode(success=env._check_success())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output HDF5 path, e.g. data/bottle.hdf5")
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
