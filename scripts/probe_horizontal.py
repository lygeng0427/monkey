#!/usr/bin/env python3
"""Discriminating test: is there a HEIGHT at which the Panda wrist can hold a
horizontal forward-pointing pose in front of the robot?

The earlier horizontal-grasp attempt failed (orientation servo stuck ~1.66 rad)
at the drawer handle height (z~0.88). Before paying for a drawer rescale to raise
the handle, we check whether raising the eef target z alone lets the wrist achieve
a horizontal (approach axis = world +x) orientation. No XML / rescale needed.

For each candidate height we servo the eef to (handle_x, handle_y, z) with a fixed
horizontal target orientation, then report the residual orientation error. A small
residual = the wrist CAN hold horizontal there (crossover exists -> raising the
handle is justified). If every height stays large, the limit is distance/azimuth,
not height, and scaling the drawer won't enable horizontal.
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


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make():
    return suite.make(
        env_name="FrankaDrawerOpen",
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=2000,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
    )


def servo_to(env, target_pos, target_quat, n=120, gain=20.0):
    """Servo eef to a world pos + orientation; return (pos_err, orn_err_norm)."""
    obs = env._get_observations()
    for _ in range(n):
        cur_pos = np.asarray(obs["robot0_eef_pos"])
        cur_quat = np.asarray(obs["robot0_eef_quat"])  # xyzw
        act = np.zeros(env.action_dim, dtype=np.float32)
        act[:3] = np.clip((target_pos - cur_pos) * gain, -1, 1)
        act[3:6] = np.clip(T.get_orientation_error(target_quat, cur_quat) * 3.0, -1, 1)
        act[-1] = -1.0
        obs = env.step(act)[0]
    cur_pos = np.asarray(obs["robot0_eef_pos"])
    cur_quat = np.asarray(obs["robot0_eef_quat"])
    orn_err = np.linalg.norm(T.get_orientation_error(target_quat, cur_quat))
    return np.linalg.norm(target_pos - cur_pos), orn_err, cur_pos


def main():
    env = make()
    obs = env.reset()
    base_quat = np.asarray(obs["robot0_eef_quat"])
    base_mat = T.quat2mat(base_quat)
    eef0 = np.asarray(obs["robot0_eef_pos"])
    handle = np.asarray(env.sim.data.site_xpos[env.handle_site_id])
    print(f"reset eef={eef0.round(3)}  handle={handle.round(3)}  base_approach(-z col)={base_mat[:,2].round(2)}")

    # Targets: tilt the top-down pose toward horizontal by rotating about world y.
    # 0 deg = top-down (baseline sanity), 45 deg = in-between, 90 deg = horizontal.
    # Try both tilt signs (approach -> +x vs -x) since the face orientation matters.
    tilts = {
        "topdown(0)":   rot_y(0.0),
        "tilt45+":      rot_y(np.deg2rad(45)),
        "tilt45-":      rot_y(np.deg2rad(-45)),
        "horiz90+":     rot_y(np.deg2rad(90)),
        "horiz90-":     rot_y(np.deg2rad(-90)),
    }
    # Sweep target height; xy held at the handle (in front of robot).
    heights = [0.88, 1.00, 1.10, 1.20, 1.30, 1.45]
    xy = handle[:2]

    print("\norn_err (rad) by [target] x [height]; small => wrist holds that pose there")
    header = "target      " + "".join(f"{z:>8.2f}" for z in heights)
    print(header)
    for name, R in tilts.items():
        target_mat = R @ base_mat
        target_quat = T.mat2quat(target_mat)
        row = f"{name:<12}"
        for z in heights:
            env.reset()
            tp = np.array([xy[0], xy[1], z])
            _, oerr, _ = servo_to(env, tp, target_quat, n=140)
            row += f"{oerr:>8.2f}"
        print(row)

    # Detail on the two PROMISING poses: report pos error + achieved approach axis
    # (col 2 of eef rot = gripper approach direction) so we know it really reaches
    # the handle while holding the pose, not just that orn_err is small.
    print("\nDetail (pos_err m, orn_err rad, achieved approach axis):")
    for name, deg in [("horiz90-", -90), ("tilt45-", -45)]:
        target_mat = rot_y(np.deg2rad(deg)) @ base_mat
        tq = T.mat2quat(target_mat)
        for z in [0.88, 0.95, 1.05, 1.15]:
            env.reset()
            perr, oerr, cur = servo_to(env, np.array([xy[0], xy[1], z]), tq, n=160)
            cq = np.asarray(env._get_observations()["robot0_eef_quat"])
            app = T.quat2mat(cq)[:, 2]
            print(f"  {name:<9} z={z:.2f}  pos_err={perr:.3f}  orn_err={oerr:.2f}  approach={app.round(2)}  eef={cur.round(3)}")
        print(f"   (target approach axis = {(rot_y(np.deg2rad(deg)) @ base_mat)[:,2].round(2)})")
    env.close()


if __name__ == "__main__":
    main()
