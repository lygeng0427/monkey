#!/usr/bin/env python3
"""Collect scripted FrankaDrawerOpenTable demonstrations (drawer ON the table).

Sibling of collect_drawer_demos.py for the table-standing drawer
(franka_drawer_bottle/drawer_table_env.py): the cabinet rests on the tabletop, so
the loop handle sits at the usual table-pick height (world z ~= 0.882 at scale 1),
NOT the raised z~1.15 the floating task uses.

At that height the original ~45 deg grasp is unreachable (the Panda wrist cannot
hold the 45 deg pose -- pre-grasp pos error ~9 cm; verified) and a pure top-down
pinch slips off the round handle bar during the pull. The reachable+secure
compromise is a TILTED ~30 deg grasp (top-down-ward): the gripper points mostly
down with a 30 deg lean toward horizontal, yawed 90 deg so the fingers close
ACROSS the world-y handle bar, then pulls the drawer toward the robot along -x.
This is the most-horizontal grasp reachable at the table-height handle that still
holds the bar through the pull (max slide ~0.27 m vs the 0.18 success threshold).

Validated 8/8 at noise 0/0.01/0.02 and at sizes {0.85, 1.0, 1.15} (see the probe
sweep that set PITCH/SEAT_FWD). Same robomimic-style HDF5 layout as the other
collectors. Run from the project root:

    python scripts/collect_drawer_table_demos.py --out data/drawer_table.hdf5 --n 10
    python scripts/collect_drawer_table_demos.py --out data/drawer_table.hdf5 --n 1 --render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robosuite.utils.transform_utils as T

import franka_drawer_bottle  # noqa: F401  (registers the floating envs)
# Importing the table env module registers FrankaDrawerOpenTable by class name
# (no change to franka_drawer_bottle/__init__.py needed).
from franka_drawer_bottle.drawer_table_env import FrankaDrawerOpenTable  # noqa: F401
from scripts.demo_common import Recorder, collect

ENV_NAME = "FrankaDrawerOpenTable"
HORIZON = 1500
CONTROL_FREQ = 20
NOMINAL_XY = (0.05, 0.0)  # FrankaDrawerOpenTable default placement

# Grasp tuned for the table-height handle (see module docstring + the probe sweep).
PITCH = -30.0         # grasp tilt (deg) from top-down toward horizontal, about world y
BACKOFF = 0.13        # pre-grasp standoff along -approach (m)
SEAT_FWD = 0.022      # advance past the bar center along +approach to seat deep (m)
GRASP_XBIAS = -0.016  # seat this far toward the robot (-x) of the bar center so the
                      # lower (forward) finger descends in the clear corridor in
                      # FRONT of the drawer body instead of catching on it
PULL_DX = -0.05       # per-step pull toward the robot (drawer slides world -x)


def _rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def generate_episode(env, render=False, noise_scale=0.0, frame_cb=None):
    rec = Recorder(env, render=render, frame_cb=frame_cb)
    sim = env.sim

    # Offsets scale with the object (handle/body geometry grows with object_scale).
    s = float(getattr(env, "object_scale", 1.0))
    backoff, seat_fwd, grasp_xbias = BACKOFF * s, SEAT_FWD * s, GRASP_XBIAS * s

    handle_id = env.handle_site_id
    handle_pos = lambda: np.array(sim.data.site_xpos[handle_id])
    jitter = np.random.normal(scale=noise_scale, size=3)

    # Fixed tilted target orientation: pitch toward horizontal (world y) on top of
    # the across-the-bar yaw (world z), relative to the reset (top-down) pose.
    base_mat = T.quat2mat(np.asarray(rec.obs["robot0_eef_quat"]))
    target_mat = _rot_y(np.deg2rad(PITCH)) @ _rot_z(np.deg2rad(90.0)) @ base_mat
    target_quat = T.mat2quat(target_mat)
    approach = target_mat[:, 2]  # gripper approach axis in world frame (forward-down)
    xbias = np.array([grasp_xbias, 0.0, 0.0])  # forward (-x) seat bias, clear of the body

    def ori_err():
        cq = np.asarray(rec.obs["robot0_eef_quat"])
        return np.clip(T.get_orientation_error(target_quat, cq) * 3.0, -1.0, 1.0)

    def step_to(target, gripper):
        rec.servo(target, gripper=gripper, rot=ori_err())

    # 1) pre-grasp: settle the tilted pose, backed off along -approach from the bar.
    for _ in range(110):
        step_to(handle_pos() + jitter - backoff * approach + xbias, gripper=-1.0)
    # 2) seat: advance along +approach onto the true bar so it sits between the pads
    for _ in range(90):
        step_to(handle_pos() + seat_fwd * approach + xbias, gripper=-1.0)
    # 3) close across the bar
    for _ in range(45):
        step_to(handle_pos() + seat_fwd * approach + xbias, gripper=1.0)
    # 4) pull toward the robot (-x) until the drawer is open, holding the tilt
    for _ in range(320):
        if env._check_success():
            break
        step_to(rec.eef_pos + np.array([PULL_DX, 0.0, 0.0]), gripper=1.0)

    return rec.episode(success=env._check_success())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output HDF5 path, e.g. data/drawer_table.hdf5")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--noise-scale", type=float, default=0.0, help="Std of per-episode approach jitter (m).")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--object-scale", type=float, default=1.0, help="Drawer size multiplier (1.0 = baseline).")
    p.add_argument("--px", type=float, default=None, help="Object placement x (default: env default).")
    p.add_argument("--py", type=float, default=None, help="Object placement y (default: env default).")
    args = p.parse_args()

    episode_fn = lambda env, render: generate_episode(env, render=render, noise_scale=args.noise_scale)
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
