#!/usr/bin/env python3
"""Pilot the SCRIPTED collectors at the size extremes -- the real gate.

pilot_size.py only drives the joint directly (gripper never touches the object),
so it proves the static scaled geometry isn't jammed but NOT that the closed-loop
grasp/push still works at a new size. This runs the actual collector trajectories
at object_scale in {0.85, 1.15} for both tasks and asserts:

  - the scripted demo SUCCEEDS (the size-coupled offsets are right), and
  - (drawer) gripper<->drawer-body contacts stay 0 over the episode -- the tilted
    grasp must seat on the handle bar in the clear corridor, never catching on the
    drawer block top (the documented noisy-failure mode).

Run from the project root (a couple minutes):

    python scripts/pilot_collect.py
    python scripts/pilot_collect.py --n 2 --sizes 0.85 1.0 1.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import franka_drawer_bottle  # noqa: F401  (registers the envs)
from scripts.demo_common import make_env
import scripts.collect_drawer_demos as drawer
import scripts.collect_bottle_demos as bottle


def _geom_name(env, gid):
    try:
        return env.sim.model.geom_id2name(gid) or ""
    except Exception:
        return ""


def _gripper_geom_ids(env):
    ids = set()
    for gid in range(env.sim.model.ngeom):
        nm = _geom_name(env, gid).lower()
        if "gripper" in nm or "finger" in nm:
            ids.add(gid)
    return ids


def _body_collision_geom_ids(env):
    ids = set()
    for gid in range(env.sim.model.ngeom):
        if "tray_collision" in _geom_name(env, gid).lower():
            ids.add(gid)
    return ids


def run_drawer(scale, n, render):
    env = make_env("FrankaDrawerOpen", render=render, control_freq=drawer.CONTROL_FREQ,
                   horizon=drawer.HORIZON, object_scale=scale)
    grip = _gripper_geom_ids(env)
    body = _body_collision_geom_ids(env)
    assert grip and body, f"could not resolve gripper/body geoms (grip={len(grip)} body={len(body)})"

    results = []
    for i in range(n):
        np.random.seed(i)
        contacts = {"max": 0}

        def frame_cb():
            d = env.sim.data
            c = 0
            for k in range(d.ncon):
                g1, g2 = d.contact[k].geom1, d.contact[k].geom2
                if (g1 in grip and g2 in body) or (g2 in grip and g1 in body):
                    c += 1
            contacts["max"] = max(contacts["max"], c)

        ep = drawer.generate_episode(env, render=render, frame_cb=frame_cb)
        results.append((ep["success"], contacts["max"]))
        print(f"  drawer scale={scale} demo[{i}] success={ep['success']} "
              f"max gripper<->body contacts={contacts['max']}")
    env.close()
    return results


def run_bottle(scale, n, render):
    env = make_env("FrankaBottleUntwist", render=render, control_freq=bottle.CONTROL_FREQ,
                   horizon=bottle.HORIZON, object_scale=scale)
    results = []
    for i in range(n):
        np.random.seed(i)
        ep = bottle.generate_episode(env, render=render)
        results.append((ep["success"], 0))
        print(f"  bottle scale={scale} demo[{i}] success={ep['success']}")
    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--sizes", type=float, nargs="+", default=[0.85, 1.15])
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    print("=== drawer (scripted grasp at size extremes) ===")
    drawer_ok = True
    for s in args.sizes:
        for success, max_contacts in run_drawer(s, args.n, args.render):
            drawer_ok &= success and (max_contacts == 0)

    print("\n=== bottle (scripted push at size extremes) ===")
    bottle_ok = True
    for s in args.sizes:
        for success, _ in run_bottle(s, args.n, args.render):
            bottle_ok &= success

    print(f"\ndrawer: {'PASS' if drawer_ok else 'FAIL'} (success AND zero gripper<->body contacts)")
    print(f"bottle: {'PASS' if bottle_ok else 'FAIL'} (success)")
    if not (drawer_ok and bottle_ok):
        sys.exit(1)
    print("Pilot collect PASSED: scripted demos succeed at all size extremes.")


if __name__ == "__main__":
    main()
