#!/usr/bin/env python3
"""Collect scripted FrankaDrawerOpen demonstrations.

We grasp the drawer's REAL CAD handle -- a bag-style loop on the front face whose
graspable front bar runs along world y (the env yaws the object 90 deg). The
trajectory is a top-down cross-bar grasp, then a pull toward the robot along -x:

    1. move above the handle bar (gripper open) while yawing the gripper so its
       finger-closing axis aligns with world +x (ACROSS the bar) -- one finger
       ends in front of the bar, one drops into the loop hole behind it
    2. descend so the bar seats DEEP between the pads
    3. close across the bar
    4. pull along -x (the drawer opens toward the robot) until success

The gripper yaw is held by a closed-loop orientation servo every step: the
finger-closing direction is measured from the two finger pads and a rotation
delta about the base/vertical z-axis (action[3:6]) is commanded to drive that
direction onto world +x. The grasp is centered slightly FORWARD of the bar
(GRASP_XOFF) so the rear finger drops into the loop hole rather than onto the
drawer body / cabinet; the descend seats below the bar center (GRASP_DZ) -- a
shallow grasp catches only the bar top and slips. (The cabinet top is pulled
back off the handle in the XML so it doesn't block this top-down approach.)

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

GRASP_XOFF = -0.035   # grasp centered this far in front (-x) of the bar, so the rear
                      # finger drops into the loop hole, not onto the drawer/cabinet
GRASP_DZ = -0.020     # descend eef this far below the bar center (deep seat)
PULL_DZ = -0.005      # pull slightly downward too, wedging the bar against the lower
                      # finger so the round bar can't pop out of the (pull-axis) pinch


def _find_pad_geom_ids(sim):
    """Resolve the two gripper finger-pad geom ids (names are robosuite-prefixed)."""
    ids = {}
    for gn in sim.model.geom_names:
        low = gn.lower()
        if "finger1_pad" in low:
            ids[1] = sim.model.geom_name2id(gn)
        elif "finger2_pad" in low:
            ids[2] = sim.model.geom_name2id(gn)
    if 1 not in ids or 2 not in ids:
        raise RuntimeError(f"Could not find finger pad geoms in {sim.model.geom_names}")
    return ids[1], ids[2]


def generate_episode(env, render=False, noise_scale=0.0, frame_cb=None):
    rec = Recorder(env, render=render, frame_cb=frame_cb)
    sim = env.sim

    handle_id = env.handle_site_id
    handle_pos = lambda: np.array(sim.data.site_xpos[handle_id])
    # Small fixed lateral offset per episode (np.random already seeded by caller).
    jitter = np.random.normal(scale=noise_scale, size=3)

    pad1, pad2 = _find_pad_geom_ids(sim)

    def yaw_rot():
        """Rotation delta (axis-angle about base z) to align the finger-closing
        axis with world +x. The handle bar runs along world y, so closing across
        it (along x) is the proper drawer grip. The closing axis is bidirectional,
        so we drive its angle to the nearest of 0 / pi."""
        cdir = np.array(sim.data.geom_xpos[pad2]) - np.array(sim.data.geom_xpos[pad1])
        ang = np.arctan2(cdir[1], cdir[0])              # current closing-axis angle (xy)
        err = np.arctan2(np.sin(-2.0 * ang), np.cos(-2.0 * ang)) / 2.0  # -> 0 or pi
        return np.array([0.0, 0.0, float(np.clip(3.0 * err, -1.0, 1.0))])

    def step_to(target, gripper):
        rec.servo(target, gripper=gripper, rot=yaw_rot())

    # 1) above the handle bar, gripper open, rotating to grasp across the bar
    for _ in range(70):
        step_to(handle_pos() + jitter + [GRASP_XOFF, 0.0, 0.10], gripper=-1.0)
    # 2) descend so the bar seats deep between the pads (front finger ahead of the
    #    bar, rear finger into the loop hole).
    for _ in range(100):
        step_to(handle_pos() + jitter + [GRASP_XOFF, 0.0, GRASP_DZ], gripper=-1.0)
    # 3) close across the bar
    for _ in range(45):
        step_to(handle_pos() + [GRASP_XOFF, 0.0, GRASP_DZ], gripper=1.0)
    # 4) pull toward the robot (-x), slightly downward, until the drawer is open
    for _ in range(320):
        if env._check_success():
            break
        step_to(rec.eef_pos + np.array([-0.04, 0.0, PULL_DZ]), gripper=1.0)

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
