#!/usr/bin/env python3
"""Collect scripted FrankaDrawerOpen demonstrations.

We grasp the drawer's REAL CAD handle -- a bag-style loop on the front face whose
graspable front bar runs along world y (the env yaws the object 90 deg). The grasp
is TILTED ~45 deg between top-down and horizontal (the "in-between" grasp): the
gripper points forward-and-down (approach axis ~[0.7, 0, -0.7]) and pulls toward
the robot along -x.

Why tilted and not pure top-down or pure horizontal: the cabinet sits on a 0.27 m
pedestal (see drawer_articulated.xml) so the handle is at z~1.15. At that height
the Panda wrist can hold a clean 45 deg grasp (pos err <1 cm, orientation exact);
a *fully* horizontal grasp is still infeasible (the wrist can only point horizontal
with the arm fully extended upward, ~0.6 m from the handle), and a top-down grasp
grazes the drawer body. 45 deg is the most-horizontal pose reachable at the handle.

Trajectory (orientation held by a full-quaternion servo to a fixed tilted target
every step):

    1. pre-grasp: hold the tilted pose, backed off along -approach from the bar
    2. seat: advance along +approach so the bar sits between the finger pads
    3. close across the bar (fingers close in the x-z plane => across the y bar)
    4. pull along -x (the drawer opens toward the robot) until success

The target orientation is pitch -45 deg (toward horizontal, about world y) applied
to the top-down across-the-bar pose (yaw 90 deg about world z); see
scripts/probe_tilt_grasp.py for the reachability/grip-axis check.

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

import robosuite.utils.transform_utils as T

import franka_drawer_bottle  # noqa: F401  (registers FrankaDrawerOpen)
from scripts.demo_common import Recorder, collect, collect_grid, grid_positions

ENV_NAME = "FrankaDrawerOpen"
HORIZON = 1500
CONTROL_FREQ = 20
NOMINAL_XY = (0.05, 0.0)  # FrankaDrawerOpen default placement

PITCH = -45.0         # grasp tilt (deg) from top-down toward horizontal, about world y
BACKOFF = 0.13        # pre-grasp standoff along -approach (m)
SEAT_FWD = 0.012      # advance past the bar center along +approach to seat deep (m)
GRASP_XBIAS = -0.016  # seat this far toward the robot (world -x) of the bar center, so
                      # the lower (forward) finger descends in the clear corridor in
                      # FRONT of the bar instead of catching on the drawer body top
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

    # The standoff/seat/clearance offsets are tuned to the handle+body geometry, so
    # they scale with the object. The handle WORLD height is held fixed across sizes
    # (see drawer_env._load_model), so the orientation/pull are size-independent.
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
    #    The jitter perturbs only the APPROACH start (so the demo isn't a single
    #    canned path); the seat below converges onto the true bar.
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
    p.add_argument("--out", required=True, help="Output HDF5 path, e.g. data/drawer.hdf5")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--noise-scale", type=float, default=0.0, help="Std of per-episode approach jitter (m).")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--object-scale", type=float, default=1.0, help="Drawer size multiplier (1.0 = baseline).")
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
