#!/usr/bin/env python3
"""Collect scripted FrankaBottleUntwist demonstrations.

Non-prehensile trajectory (no grasping): the gripper is kept CLOSED the whole
time and used as a finger. It is placed in a valley just behind ONE of the cap's
tabs and pushes that single tab around, spinning the cap on its hinge:

    1. (gripper closed) move above the valley just behind the target tab
    2. descend hard into the valley, beside the tab (resting near the body top)
    3. push the tab around an arc about the cap center until the cap turns past
       the success angle

Key detail (avoids slipping to the next tab): the hand's commanded angle is
locked to the MEASURED cap angle -- `hand_angle = tab_angle0 + cap_angle + DRIVE`.
Because `cap_angle` only advances when the cap actually turns, the hand can never
run ahead of the tab it is pushing; `DRIVE` (< the 45 deg tab spacing) keeps it
pressed against that one tab. (An open-loop sweep at a fixed rate outruns the cap
and skips to the next tab -- that was the bug.)

The cap mesh has 8 flat radial tabs at its brim, modelled in collision (a small
`cap_core` + 8 `cap_petal_*`), so the closed hand drops into a valley and pushes a
single tab. `tab_site` marks that tab; `cap_site` is the cap center (arc pivot).
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

PUSH_RADIUS = 0.059       # arc radius (the cap tab ring radius)
PUSH_Z = -0.03            # eef z offset vs cap_site: drive the closed hand down into the valley
LEAD = 0.32               # start this many radians behind the target tab (in the valley)
DRIVE = 0.33              # hold the hand this many radians ahead of the (moving) tab to push it
DESCEND_STEPS = 120       # steps to drive the hand down into the valley beside the tab
MAX_PUSH_STEPS = 340      # cap on the push


def generate_episode(env, render=False, noise_scale=0.0, frame_cb=None):
    rec = Recorder(env, render=render, frame_cb=frame_cb)
    sim = env.sim

    c = np.array(sim.data.site_xpos[env.cap_site_id])   # cap center (fixed; only the cap spins)
    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
    push_z = cz + PUSH_Z
    jit = np.random.normal(scale=noise_scale, size=2)

    t = np.array(sim.data.site_xpos[env.tab_site_id])
    tab_ang0 = np.arctan2(t[1] - cy, t[0] - cx)         # initial angle of the target tab
    cap_a0 = float(env._cap_angle)

    def arc(angle, dz=0.0):
        return [cx + jit[0] + PUSH_RADIUS * np.cos(angle),
                cy + jit[1] + PUSH_RADIUS * np.sin(angle),
                push_z + dz]

    # 1) above the valley just behind the target tab, gripper CLOSED from the start
    for _ in range(40):
        rec.servo(arc(tab_ang0 - LEAD, dz=0.12), gripper=1.0)
    # 2) drive down hard into the valley, beside the tab
    for _ in range(DESCEND_STEPS):
        rec.servo(arc(tab_ang0 - LEAD), gripper=1.0)
    # 3) push that ONE tab: keep the hand DRIVE rad ahead of the *measured* cap angle, so it
    #    tracks the same tab (never outruns it to the next one), until the cap turns enough
    for _ in range(MAX_PUSH_STEPS):
        cap_a = float(env._cap_angle) - cap_a0
        rec.servo(arc(tab_ang0 + cap_a + DRIVE), gripper=1.0)
        if env._check_success():
            break

    return rec.episode(success=env._check_success())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output HDF5 path, e.g. data/bottle.hdf5")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--noise-scale", type=float, default=0.0, help="Std of per-episode push-position jitter (m).")
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
