#!/usr/bin/env python3
"""Open a robosuite viewer with Panda + uploaded drawer/bottle meshes.

Run from the project root:

    python scripts/run_random_robosuite_env.py

Optional:
    python scripts/run_random_robosuite_env.py --steps 5000 --seed 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import robosuite as suite

# Important: importing this package registers FrankaDrawerBottleScene with robosuite.
import franka_drawer_bottle  # noqa: F401


def load_basic_osc_controller():
    """Robosuite 1.5 uses composite controllers; older versions use controller configs.

    This function supports both styles so the starter script is less brittle.
    """
    try:
        from robosuite import load_composite_controller_config

        return load_composite_controller_config(controller="BASIC")
    except Exception:
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller="OSC_POSE")


def step_env(env, action):
    """Compatibility for robosuite versions returning 4-tuple or Gymnasium-style 5-tuple."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="FrankaDrawerBottleScene",
        choices=["FrankaDrawerBottleScene", "FrankaDrawerOpen", "FrankaBottleUntwist"],
        help="Which registered environment to launch.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"])
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--show-collision", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    controller_config = load_basic_osc_controller()

    env = suite.make(
        env_name=args.env,
        robots="Panda",
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=args.camera,
        render_collision_mesh=True,
        render_visual_mesh=True,
        control_freq=20,
        horizon=args.steps + 1,
        ignore_done=True,
        renderer=args.renderer,
    )

    obs = env.reset()
    print(f"Loaded env: {env.__class__.__name__}")
    print(f"Action dim: {env.action_dim}")
    print("Close the MuJoCo viewer window or press Ctrl+C in the terminal to stop.")

    # Smooth random action target. For OSC_POSE this is normalized input, internally
    # scaled by the controller limits. Small random values are easier to watch.
    action = np.zeros(env.action_dim, dtype=np.float32)
    target = np.zeros_like(action)

    try:
        for t in range(args.steps):
            if t % 50 == 0:
                target = rng.uniform(low=-0.35, high=0.35, size=env.action_dim).astype(np.float32)
                # Treat the last dimension as gripper open/close for the default Panda setup.
                target[-1] = rng.choice([-1.0, 1.0])
                # Make rotations gentler than translations.
                if env.action_dim >= 7:
                    target[3:6] *= 0.35

            action = 0.92 * action + 0.08 * target
            obs, reward, done, info = step_env(env, action)
            env.render()

            # Sleep is optional, but makes viewer motion easier to inspect on fast machines.
            time.sleep(0.002)

            if done:
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
