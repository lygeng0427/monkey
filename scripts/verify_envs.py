#!/usr/bin/env python3
"""Headless correctness checks for the articulated task environments.

For each task env this:
  1. builds the env (no renderer) and resets it,
  2. confirms the articulation joint qpos address and the task observables resolve,
  3. *drives the joint* by writing increasing values into sim.data.qpos and stepping
     the simulation, asserting the observable tracks it and that _check_success()
     flips False -> True at the configured threshold.

This catches a mis-resolved joint, a mis-anchored root, or a collision jam between
the moving part and its housing (the joint would refuse to move).

Run from the project root:

    python scripts/verify_envs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import robosuite as suite

import franka_drawer_bottle  # noqa: F401  (registers the envs)


def make_env(env_name):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )


def drive_joint(env, qpos_addr, obs_key, target, n=60):
    """Ramp the joint from 0 to `target` over n steps, holding it each step.

    We overwrite qpos directly (the gripper isn't actually grasping in this check),
    step the sim so contacts/forward-kinematics resolve, and read back the value.
    """
    last_obs = None
    for i in range(1, n + 1):
        val = target * i / n
        env.sim.data.qpos[qpos_addr] = val
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        obs, reward, done, info = _step(env)
        last_obs = obs
    return last_obs


def _step(env):
    out = env.step(np.zeros(env.action_dim))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out


def check(env_name, obs_key, addr_attr, threshold, below, above):
    print(f"\n=== {env_name} ===")
    env = make_env(env_name)
    obs = env.reset()

    addr = getattr(env, addr_attr)
    print(f"  joint qpos addr resolved: {addr_attr} = {addr}")
    assert obs_key in obs, f"observable {obs_key!r} missing; have {sorted(obs)}"
    print(f"  observable present: {obs_key} = {np.round(obs[obs_key], 5)}")

    # Closed: not successful.
    env.sim.data.qpos[addr] = below
    env.sim.forward()
    _step(env)
    assert not env._check_success(), "should NOT be success when closed/unturned"
    print(f"  success @ qpos={below}: {env._check_success()} (expected False)")

    # Drive the joint open/turned past the threshold.
    drive_joint(env, addr, obs_key, above)
    reached = float(env.sim.data.qpos[addr])
    print(f"  drove joint to qpos={reached:.4f} (target {above}, threshold {threshold:.4f})")
    assert reached > threshold, (
        f"joint did not reach past threshold (stuck at {reached:.4f}); "
        f"possible collision jam between moving part and housing"
    )
    assert env._check_success(), "should be success after driving past threshold"
    print(f"  success @ qpos={reached:.4f}: {env._check_success()} (expected True)")

    env.close()
    print(f"  OK: {env_name}")


def main():
    check(
        "FrankaDrawerOpen",
        obs_key="drawer_slide",
        addr_attr="slide_qpos_addr",
        threshold=0.18,
        below=0.0,
        above=0.28,
    )
    check(
        "FrankaBottleUntwist",
        obs_key="cap_angle",
        addr_attr="cap_qpos_addr",
        threshold=np.pi / 2.0,
        below=0.0,
        above=2.5,
    )
    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()
