#!/usr/bin/env python3
"""Pilot check for the object SIZE (and position) knob.

Before mass-collecting demos across sizes/positions, this validates the
parameterization itself at the grid extremes for both tasks:

  1. Builds each env at object_scale in {0.85, 1.0, 1.15} x a few placement_xy.
  2. Confirms the scaled object is physically sane: drive the articulation joint
     directly and assert qpos moves past the (scaled) threshold and _check_success
     flips False->True. A collision jam between the scaled moving part and its
     housing would stick the joint -- this catches it.
  3. Reports the handle/cap WORLD z and the gripper reset pose, so we can confirm
     the drawer handle stays at the reachable ~1.15 m across sizes (the placement-z
     is chosen to hold it fixed) and the gripper is not trapped.

This is the gate the size knob must pass before the collectors are run over the
grid. Run from the project root:

    python scripts/pilot_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import robosuite as suite

import franka_drawer_bottle  # noqa: F401  (registers the envs)

SIZES = (0.85, 1.0, 1.15)


def make_env(env_name, object_scale, placement_xy):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        object_scale=object_scale,
        placement_xy=placement_xy,
    )


def _step(env):
    out = env.step(np.zeros(env.action_dim))
    return out[0]


def drive_joint(env, addr, target, n=60):
    """Ramp the joint 0 -> target over n steps, holding it each step."""
    for i in range(1, n + 1):
        env.sim.data.qpos[addr] = target * i / n
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        _step(env)
    return float(env.sim.data.qpos[addr])


def eef_pos(env):
    arm = env.robots[0].arms[0]
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]])


def check_drawer(object_scale, placement_xy):
    env = make_env("FrankaDrawerOpen", object_scale, placement_xy)
    env.reset()
    addr = env.slide_qpos_addr
    handle = env._handle_xpos
    eef = eef_pos(env)
    thresh = env.success_thresh

    # closed -> not success
    env.sim.data.qpos[addr] = 0.0
    env.sim.forward()
    _step(env)
    assert not env._check_success(), "drawer reports success while closed"

    above = min(thresh * 1.4, 0.30)  # stay below the unscaled slide range (0.32)
    reached = drive_joint(env, addr, above)
    assert reached > thresh, (
        f"drawer slide stuck at {reached:.4f} (thresh {thresh:.4f}) -- possible "
        f"collision jam in the scaled geometry"
    )
    assert env._check_success(), "drawer did not report success past threshold"

    print(
        f"  scale={object_scale:<4} xy={tuple(placement_xy)} | "
        f"handle_z={handle[2]:.3f} eef_z={eef[2]:.3f} "
        f"thresh={thresh:.3f} drove->{reached:.3f}  OK"
    )
    env.close()
    return handle[2]


def check_bottle(object_scale, placement_xy):
    env = make_env("FrankaBottleUntwist", object_scale, placement_xy)
    env.reset()
    addr = env.cap_qpos_addr
    cap = env._cap_xpos
    tab = np.array(env.sim.data.site_xpos[env.tab_site_id])
    push_radius = float(np.linalg.norm((tab - cap)[:2]))  # live tab-ring radius
    eef = eef_pos(env)
    thresh = env.success_thresh

    env.sim.data.qpos[addr] = 0.0
    env.sim.forward()
    _step(env)
    assert not env._check_success(), "bottle reports success while unturned"

    reached = drive_joint(env, addr, 2.5)
    assert reached > thresh, f"cap stuck at {reached:.4f} (thresh {thresh:.4f})"
    assert env._check_success(), "bottle did not report success past threshold"

    print(
        f"  scale={object_scale:<4} xy={tuple(placement_xy)} | "
        f"cap_z={cap[2]:.3f} push_radius={push_radius:.4f} eef_z={eef[2]:.3f} "
        f"thresh={thresh:.3f} drove->{reached:.3f}  OK"
    )
    env.close()
    return cap[2], push_radius


def main():
    print("=== FrankaDrawerOpen: size x position ===")
    drawer_handle_z = []
    for s in SIZES:
        for xy in [(0.05, 0.0), (0.05, 0.05), (0.0, -0.05)]:
            drawer_handle_z.append(check_drawer(s, xy))
    spread = max(drawer_handle_z) - min(drawer_handle_z)
    print(f"  handle world-z spread across sizes: {spread*1000:.1f} mm (want ~0)")
    assert spread < 0.005, "drawer handle height drifts with size -- reach not preserved"

    print("\n=== FrankaBottleUntwist: size x position ===")
    radii = {}
    for s in SIZES:
        for xy in [(0.10, 0.0), (0.10, 0.05), (0.05, -0.05)]:
            _, r = check_bottle(s, xy)
            radii.setdefault(s, r)
    print(f"  tab-ring radius by size: " + ", ".join(f"{s}->{radii[s]:.4f}" for s in SIZES))
    # The push radius must track size (the collector reads it live, no hardcode).
    assert radii[1.15] > radii[1.0] > radii[0.85], "tab-ring radius did not scale with size"

    print("\nPilot passed: size knob is physically sane at all grid extremes.")


if __name__ == "__main__":
    main()
