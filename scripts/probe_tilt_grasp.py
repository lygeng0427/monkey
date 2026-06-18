#!/usr/bin/env python3
"""Verify the raised-handle tilted grasp pose is reachable AND grips across the bar.

Target orientation = pitch -45 deg (toward horizontal) applied to the top-down
across-the-bar pose (yaw 90 deg about world z). Servo the eef to the handle while
holding that orientation; report pos/orn error and the finger-closing axis (must
be in the x-z plane = across the world-y bar, not along it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robosuite as suite
import robosuite.utils.transform_utils as T

import franka_drawer_bottle  # noqa: F401
from scripts.demo_common import load_controller_config


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pad_ids(sim):
    a = b = None
    for gn in sim.model.geom_names:
        lo = gn.lower()
        if "finger1_pad" in lo:
            a = sim.model.geom_name2id(gn)
        elif "finger2_pad" in lo:
            b = sim.model.geom_name2id(gn)
    return a, b


env = suite.make(
    env_name="FrankaDrawerOpen", robots="Panda", gripper_types="default",
    controller_configs=load_controller_config(), has_renderer=False,
    has_offscreen_renderer=False, use_camera_obs=False, control_freq=20,
    horizon=2000, ignore_done=True, hard_reset=False, initialization_noise=None,
)
obs = env.reset()
base_mat = T.quat2mat(np.asarray(obs["robot0_eef_quat"]))
handle = np.asarray(env.sim.data.site_xpos[env.handle_site_id])
print(f"handle world pos = {handle.round(3)}  (z should be ~1.15)")

p1, p2 = pad_ids(env.sim)
for pitch in [-35, -45, -55]:
    target_mat = rot_y(np.deg2rad(pitch)) @ rot_z(np.deg2rad(90)) @ base_mat
    tq = T.mat2quat(target_mat)
    obs = env.reset()
    # pre-grasp slightly back along -approach so we settle the pose, then to handle
    approach = target_mat[:, 2]
    for phase, tgt in [("pose", handle - 0.12 * approach), ("seat", handle)]:
        for _ in range(120):
            cp = np.asarray(obs["robot0_eef_pos"]); cq = np.asarray(obs["robot0_eef_quat"])
            a = np.zeros(env.action_dim, dtype=np.float32)
            a[:3] = np.clip((tgt - cp) * 20.0, -1, 1)
            a[3:6] = np.clip(T.get_orientation_error(tq, cq) * 3.0, -1, 1)
            a[-1] = -1.0
            obs = env.step(a)[0]
    cp = np.asarray(obs["robot0_eef_pos"]); cq = np.asarray(obs["robot0_eef_quat"])
    perr = np.linalg.norm(handle - cp)
    oerr = np.linalg.norm(T.get_orientation_error(tq, cq))
    close_axis = np.asarray(env.sim.data.geom_xpos[p2]) - np.asarray(env.sim.data.geom_xpos[p1])
    close_axis = close_axis / (np.linalg.norm(close_axis) + 1e-9)
    app = T.quat2mat(cq)[:, 2]
    print(f"pitch={pitch:>4} deg: pos_err={perr:.3f} orn_err={oerr:.2f} "
          f"approach={app.round(2)} close_axis={close_axis.round(2)} eef={cp.round(3)}")
env.close()
