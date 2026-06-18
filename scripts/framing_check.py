#!/usr/bin/env python3
"""Render agentview + sideview across size/position corners to check framing.

Before the offline image pass (render_obs.py) we must confirm the object stays in
frame for BOTH policy cameras at every grid extreme -- the drawer floats at world
z~1.15 and the position grid shifts it +/-0.05 m, so a default camera may clip it.

Writes a montage PNG per camera (rows = configs, one tile each) to /tmp so the
frames can be eyeballed. Needs an offscreen GL context:

    MUJOCO_GL=egl python scripts/framing_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import robosuite as suite
from PIL import Image

import franka_drawer_bottle  # noqa: F401

CAMERAS = ("agentview", "sideview")
TILE = 256


def make_env(env_name, object_scale, placement_xy):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,
        camera_names=list(CAMERAS),
        camera_heights=TILE,
        camera_widths=TILE,
        control_freq=20,
        horizon=100,
        ignore_done=True,
        hard_reset=True,
        object_scale=object_scale,
        placement_xy=placement_xy,
    )


def render(env, cam):
    img = env.sim.render(width=TILE, height=TILE, camera_name=cam)
    return np.flipud(img).copy()


def montage(env_name, configs):
    rows = {cam: [] for cam in CAMERAS}
    labels = []
    for s, xy in configs:
        env = make_env(env_name, s, xy)
        env.reset()
        for cam in CAMERAS:
            rows[cam].append(render(env, cam))
        labels.append(f"s={s} xy={xy}")
        env.close()
    for cam in CAMERAS:
        strip = np.concatenate(rows[cam], axis=0)  # stack configs vertically
        out = Path(f"/tmp/framing_{env_name}_{cam}.png")
        Image.fromarray(strip).save(out)
        print(f"  wrote {out}  ({len(configs)} configs: " + " | ".join(labels) + ")")


def main():
    # nominal + the 4 xy corners at the LARGEST size (worst case for clipping),
    # plus the smallest size at nominal.
    drawer_cfgs = [
        (1.0, (0.05, 0.0)),
        (1.15, (0.0, -0.05)), (1.15, (0.10, -0.05)),
        (1.15, (0.0, 0.05)), (1.15, (0.10, 0.05)),
        (0.85, (0.05, 0.0)),
    ]
    bottle_cfgs = [
        (1.0, (0.10, 0.0)),
        (1.15, (0.05, -0.05)), (1.15, (0.15, -0.05)),
        (1.15, (0.05, 0.05)), (1.15, (0.15, 0.05)),
        (0.85, (0.10, 0.0)),
    ]
    print("=== FrankaDrawerOpen ===")
    montage("FrankaDrawerOpen", drawer_cfgs)
    print("=== FrankaBottleUntwist ===")
    montage("FrankaBottleUntwist", bottle_cfgs)


if __name__ == "__main__":
    main()
