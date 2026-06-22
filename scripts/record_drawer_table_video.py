#!/usr/bin/env python3
"""Render a collected FrankaDrawerOpenTable demo to mp4(s) by replaying its states.

Robomimic convention: the collector stores sim STATES + the scaled model_file, so
we rebuild the table-drawer env from the demo's own model_file, replay the states
(set_state_from_flattened), and grab camera frames -- the video reproduces the
demo trajectory exactly, rendered at any resolution/camera offline.

Two cameras by default:
  - agentview : the env's top-down-on-handle view (what the policy sees), matching
    the earlier demo agentview clips. A top-down view has no vertical reference, so
    it does NOT reveal that the drawer is on the table -- it's for comparability.
  - sideview  : a scene view that DOES show the cabinet resting on the tabletop and
    the drawer sliding out (the actual "on the table" change).

640x480 @ 20 fps by default, matching the bottle expert-demo clips
(videos/demo_bottle_*.mp4). Written under videos/. Needs an EGL offscreen GL
context:

    MUJOCO_GL=egl python scripts/record_drawer_table_video.py
    MUJOCO_GL=egl python scripts/record_drawer_table_video.py --demo demo_3 --cameras agentview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import imageio.v2 as imageio
import robosuite as suite

import franka_drawer_bottle  # noqa: F401
from franka_drawer_bottle.drawer_table_env import FrankaDrawerOpenTable  # noqa: F401  (registers)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "FrankaDrawerOpenTable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/drawer_table.hdf5")
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--cameras", nargs="+", default=["agentview", "sideview"])
    # Default resolution/fps match the bottle expert-demo clips (640x480 @ 20 fps,
    # real-time since control_freq=20).
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    with h5py.File(args.inp, "r") as f:
        d = f["data"][args.demo]
        model_file = d.attrs["model_file"]
        states = d["states"][:]

    env = suite.make(
        ENV_NAME, robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        render_visual_mesh=True, render_collision_mesh=False,
        use_camera_obs=False, camera_names=list(args.cameras),
        camera_heights=args.height, camera_widths=args.width,
        control_freq=20,
    )
    env.reset_from_xml_string(model_file)

    # Aim the agentview ONCE at the initial (closed-drawer) handle pose, matching the
    # env's reset-time behavior. Re-aiming every frame would make the camera track the
    # moving handle, so the static cabinet would appear to slide backward while the
    # handle stayed centered -- a tracking artifact, not real motion. With a fixed aim
    # the drawer correctly slides out within the frame (consistent with the sideview).
    env.sim.set_state_from_flattened(np.asarray(states[0]))
    env.sim.forward()
    env._aim_agentview_topdown()

    frames = {cam: [] for cam in args.cameras}
    for s in states:
        env.sim.set_state_from_flattened(np.asarray(s))
        env.sim.forward()
        for cam in args.cameras:
            img = np.flipud(env.sim.render(width=args.width, height=args.height, camera_name=cam))
            frames[cam].append(img.copy())
    env.close()

    n = len(states)
    outdir = REPO_ROOT / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    for cam in args.cameras:
        out = outdir / f"drawer_table_demo_{cam}.mp4"
        imageio.mimwrite(out, frames[cam], fps=args.fps, codec="libx264", quality=8)
        print(f"[wrote] {out}  ({n} frames @ {args.fps} fps = {n/args.fps:.2f}s, "
              f"{args.width}x{args.height})")


if __name__ == "__main__":
    main()
