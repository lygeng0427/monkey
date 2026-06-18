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
from scripts.demo_common import Recorder, collect, collect_grid, grid_positions

ENV_NAME = "FrankaBottleUntwist"
HORIZON = 2500
CONTROL_FREQ = 20
NOMINAL_XY = (0.10, 0.0)  # FrankaBottleUntwist default placement

# Arc radius (the cap tab-ring radius) is read LIVE per episode from
# ||tab_site - cap_site|| so it tracks object_scale automatically (baseline ~0.059 m).
PUSH_Z = -0.03            # eef z offset vs cap_site (baseline; scaled by object_scale): dip into the valley
LEAD = 0.32               # start this many radians behind the target tab (in the valley)
DRIVE = 0.33              # hold the hand this many radians ahead of the (moving) tab to push it
DESCEND_STEPS = 120       # steps to drive the hand down into the valley beside the tab
MAX_PUSH_STEPS = 800      # cap on the push (the loop breaks on success, so this only
                          # affects slow configs -- the SMALL cap at x-extremes turns
                          # ~2x slower per step and needs ~760 steps to reach pi/2; 340
                          # cut it off ~0.27 rad short. Push budget 40+120+800 < HORIZON.)


def generate_episode(env, render=False, noise_scale=0.0, frame_cb=None):
    rec = Recorder(env, render=render, frame_cb=frame_cb)
    sim = env.sim

    c = np.array(sim.data.site_xpos[env.cap_site_id])   # cap center (fixed; only the cap spins)
    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
    # PUSH_Z (the dip into the valley) is a vertical clearance, so it scales with the
    # object; the descent below the cap_site goes deeper for a taller cap.
    s = float(getattr(env, "object_scale", 1.0))
    push_z = cz + PUSH_Z * s
    jit = np.random.normal(scale=noise_scale, size=2)

    t = np.array(sim.data.site_xpos[env.tab_site_id])
    tab_ang0 = np.arctan2(t[1] - cy, t[0] - cx)         # initial angle of the target tab
    cap_a0 = float(env._cap_angle)
    # Read the tab-ring radius LIVE (cap center -> tab, in xy) so the arc tracks the
    # cap size automatically -- no hardcoded PUSH_RADIUS to retune per size.
    push_radius = float(np.linalg.norm((t - c)[:2]))

    def arc(angle, dz=0.0):
        return [cx + jit[0] + push_radius * np.cos(angle),
                cy + jit[1] + push_radius * np.sin(angle),
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
    p.add_argument("--object-scale", type=float, default=1.0, help="Bottle size multiplier (1.0 = baseline).")
    p.add_argument("--px", type=float, default=None, help="Object placement x (default: env default).")
    p.add_argument("--py", type=float, default=None, help="Object placement y (default: env default).")
    # Grid sweep over the SEEN configs (size x position); per-demo attrs record each.
    p.add_argument("--grid", action="store_true", help="Sweep --sizes x a position grid instead of a single config.")
    p.add_argument("--sizes", type=float, nargs="+", default=[0.85, 1.0, 1.15], help="object_scale values for --grid.")
    p.add_argument("--per-config", type=int, default=5, help="Demos per (size,position) config for --grid.")
    p.add_argument("--extent", type=float, default=0.05, help="Half-width of the xy position grid (m).")
    p.add_argument("--grid-n", type=int, default=3, help="Grid points per axis (grid-n^2 positions).")
    args = p.parse_args()

    episode_fn = lambda env, render: generate_episode(env, render=render, noise_scale=args.noise_scale)

    if args.grid:
        collect_grid(
            env_name=ENV_NAME,
            generate_episode_fn=episode_fn,
            out=args.out,
            sizes=args.sizes,
            positions=grid_positions(NOMINAL_XY, extent=args.extent, n=args.grid_n),
            per_config=args.per_config,
            seed=args.seed,
            keep_failures=args.keep_failures,
            control_freq=CONTROL_FREQ,
            horizon=HORIZON,
        )
        return

    placement_xy = None if (args.px is None and args.py is None) else (args.px or 0.0, args.py or 0.0)
    collect(
        env_name=ENV_NAME,
        generate_episode_fn=episode_fn,
        out=args.out,
        n=args.n,
        seed=args.seed,
        render=args.render,
        keep_failures=args.keep_failures,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        object_scale=args.object_scale,
        placement_xy=placement_xy,
    )


if __name__ == "__main__":
    main()
