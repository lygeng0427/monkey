#!/usr/bin/env python3
"""Shared infrastructure for the scripted demo collectors.

Used by collect_drawer_demos.py and collect_bottle_demos.py. Provides:
  - make_env: build a task env with the Panda BASIC/OSC_POSE controller,
  - Recorder: step the env while recording sim states + actions (a scripted
    "policy" just calls recorder.reach(...) / recorder.step(...)),
  - save_hdf5: write episodes in a robomimic-style layout (data/demo_i with
    states, actions, rewards, dones, plus model_file + env_args for replay).

The OSC_POSE action layout is [dx, dy, dz, drx, dry, drz, gripper]; gripper -1
opens, +1 closes.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- env
def load_controller_config():
    """Panda controller config, robust across robosuite versions (1.5 vs older)."""
    try:
        from robosuite import load_composite_controller_config

        return load_composite_controller_config(controller="BASIC")
    except Exception:
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller="OSC_POSE")


def make_env(env_name, render=False, control_freq=20, horizon=1000,
             object_scale=1.0, placement_xy=None):
    import robosuite as suite

    kwargs = dict(
        env_name=env_name,
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        # No arm start-pose randomization: scripted demos must be reproducible for a
        # given seed (otherwise tuned grasp offsets drift run-to-run).
        initialization_noise=None,
        has_renderer=render,
        has_offscreen_renderer=False,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        hard_reset=False,
        object_scale=object_scale,
    )
    # placement_xy left None -> keep each env's own default (drawer/bottle differ).
    if placement_xy is not None:
        kwargs["placement_xy"] = tuple(placement_xy)
    return suite.make(**kwargs)


def get_model_xml(env):
    if hasattr(env.sim.model, "get_xml"):
        return env.sim.model.get_xml()
    if hasattr(env.model, "get_xml"):
        return env.model.get_xml()
    raise RuntimeError("Could not get MuJoCo XML string from env.")


def get_env_args(env, env_name, control_freq=20, horizon=1000,
                 object_scale=1.0, placement_xy=None):
    """Minimal robomimic-style env metadata (enough to reconstruct the env)."""
    env_kwargs = {
        "robots": "Panda",
        "gripper_types": "default",
        "controller_configs": load_controller_config(),
        "use_camera_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "control_freq": control_freq,
        "horizon": horizon,
        "ignore_done": True,
        "object_scale": object_scale,
    }
    if placement_xy is not None:
        env_kwargs["placement_xy"] = tuple(placement_xy)
    return {
        "env_name": env_name,
        "type": 1,  # robosuite-type env in robomimic conventions
        "env_kwargs": env_kwargs,
    }


def state_flatten(env):
    return np.array(env.sim.get_state().flatten(), dtype=np.float64)


# ------------------------------------------------------------------ recorder
class Recorder:
    """Steps the env while accumulating sim states and actions for one episode.

    A scripted policy drives the arm purely through ``reach`` / ``step``; the
    state recorded for a transition is the state *before* the action is applied
    (robomimic convention), so states[i] + actions[i] -> states[i+1].
    """

    def __init__(self, env, render=False, frame_cb=None):
        self.env = env
        self.render = render
        # Optional callable invoked after every env.step (and once at reset) with
        # no args -- used by scripts/record_bottle_video.py to grab an offscreen
        # frame per step, so the recorded video matches the demo trajectory exactly.
        self.frame_cb = frame_cb
        self.states = []
        self.actions = []
        self.obs = env.reset()
        if self.frame_cb is not None:
            self.frame_cb()

    @property
    def eef_pos(self):
        return np.asarray(self.obs["robot0_eef_pos"])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.states.append(state_flatten(self.env))
        self.actions.append(action)
        out = self.env.step(action)
        self.obs = out[0]
        if self.render:
            self.env.render()
        if self.frame_cb is not None:
            self.frame_cb()
        return self.obs

    def servo(self, target, gripper, rot=None, gain=20.0, max_step=1.0):
        """Take ONE step servoing the eef toward an absolute world ``target`` xyz.

        Useful for trajectories whose target is computed each step (e.g. sweeping
        the hand along an arc), where ``reach``'s ``pos_fn() + offset`` form does
        not fit. Records the transition like ``step``.
        """
        action = np.zeros(self.env.action_dim, dtype=np.float32)
        action[:3] = np.clip((np.asarray(target) - self.eef_pos) * gain, -max_step, max_step)
        if rot is not None:
            action[3:6] = rot
        action[-1] = gripper
        return self.step(action)

    def reach(self, pos_fn, offset, gripper, n_steps, rot=None, gain=20.0, max_step=1.0):
        """Servo the eef toward ``pos_fn() + offset`` for ``n_steps`` steps.

        ``pos_fn`` is a callable returning the (possibly moving) target site
        position each step. ``rot`` is an optional 3-vector orientation delta
        (axis-angle) applied every step, e.g. for twisting about world z. For the
        success-gated variant used while pulling/twisting, see ``reach_until``.
        """
        offset = np.asarray(offset, dtype=float)
        for _ in range(n_steps):
            target = np.asarray(pos_fn()) + offset
            action = np.zeros(self.env.action_dim, dtype=np.float32)
            action[:3] = np.clip((target - self.eef_pos) * gain, -max_step, max_step)
            if rot is not None:
                action[3:6] = rot
            action[-1] = gripper
            self.step(action)

    def reach_until(self, pos_fn, offset, gripper, max_steps, success_fn, rot=None, gain=20.0, max_step=1.0):
        """Like ``reach`` but stops as soon as ``success_fn()`` is True.

        Returns True if success was reached within ``max_steps``.
        """
        offset = np.asarray(offset, dtype=float)
        for _ in range(max_steps):
            if success_fn():
                return True
            target = np.asarray(pos_fn()) + offset
            action = np.zeros(self.env.action_dim, dtype=np.float32)
            action[:3] = np.clip((target - self.eef_pos) * gain, -max_step, max_step)
            if rot is not None:
                action[3:6] = rot
            action[-1] = gripper
            self.step(action)
        return success_fn()

    def episode(self, success):
        states = list(self.states)
        actions = list(self.actions)
        final_state = state_flatten(self.env)

        # Append the terminal (post-last-action) state to `states` so the success
        # condition is observable by iterating the array -- robomimic's
        # dataset_states_to_obs.py reconstructs reward/success by reset_to(states[i]),
        # so the success state must appear in `states`, not only in `final_state`.
        # Pair it with a hold action (zero deltas, gripper unchanged) to keep
        # states/actions the same length.
        if success and actions:
            hold = np.zeros(self.env.action_dim, dtype=np.float32)
            hold[-1] = actions[-1][-1]
            states.append(final_state)
            actions.append(hold)

        return {
            "states": np.asarray(states),
            "actions": np.asarray(actions),
            "final_state": final_state,
            "success": bool(success),
            "model_file": get_model_xml(self.env),
        }


# -------------------------------------------------------------------- saving
def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return str(obj)


def save_hdf5(path, episodes, env_args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(sum(ep["actions"].shape[0] for ep in episodes))

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = total
        data_grp.attrs["env_args"] = json.dumps(env_args, default=_json_default)

        for i, ep in enumerate(episodes):
            n = int(ep["actions"].shape[0])
            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = n
            demo_grp.attrs["model_file"] = ep["model_file"]
            # Per-demo config so training/eval can filter by size & position. The
            # model_file already encodes the *scaled* geometry for offline rendering.
            if "object_scale" in ep:
                demo_grp.attrs["object_scale"] = float(ep["object_scale"])
            if "placement_xy" in ep:
                demo_grp.attrs["placement_xy"] = np.asarray(ep["placement_xy"], dtype=np.float64)
            demo_grp.create_dataset("states", data=ep["states"], compression="gzip")
            demo_grp.create_dataset("actions", data=ep["actions"], compression="gzip")
            demo_grp.create_dataset("final_state", data=ep["final_state"])

            rewards = np.zeros((n,), dtype=np.float32)
            dones = np.zeros((n,), dtype=np.float32)
            if ep["success"] and n > 0:
                rewards[-1] = 1.0
                dones[-1] = 1.0
            demo_grp.create_dataset("rewards", data=rewards, compression="gzip")
            demo_grp.create_dataset("dones", data=dones, compression="gzip")

    print(f"Saved {len(episodes)} demos, {total} samples to {path}")


def grid_positions(nominal, extent=0.05, n=3):
    """A square grid of n*n placements: nominal +/- extent in x and y (n=1 -> nominal)."""
    offs = np.array([0.0]) if n == 1 else np.linspace(-extent, extent, n)
    return [(float(nominal[0] + dx), float(nominal[1] + dy)) for dx in offs for dy in offs]


def collect_grid(env_name, generate_episode_fn, out, sizes, positions, per_config,
                 seed, keep_failures, control_freq=20, horizon=1000):
    """Sweep a (size x position) grid, building a FRESH env per config.

    A fresh env per config sidesteps the hard_reset=False re-placement pitfall (the
    object only moves/re-scales when the model is rebuilt). Every demo carries its
    own object_scale/placement_xy (per-demo HDF5 attrs) and its scaled model_file,
    so the combined dataset is self-describing for training/eval and offline render.
    """
    episodes = []
    env_args = None
    n_total = n_success = 0
    for s in sizes:
        for xy in positions:
            env = make_env(env_name, render=False, control_freq=control_freq, horizon=horizon,
                           object_scale=s, placement_xy=xy)
            if env_args is None:  # representative; per-demo attrs/model_file are authoritative
                env_args = get_env_args(env, env_name, control_freq=control_freq, horizon=horizon,
                                        object_scale=s, placement_xy=xy)
            actual_xy = tuple(float(v) for v in env.placement_xy)
            cfg_succ = 0
            for i in range(per_config):
                np.random.seed(seed + i)
                ep = generate_episode_fn(env, render=False)
                ep["object_scale"] = float(s)
                ep["placement_xy"] = actual_xy
                n_total += 1
                if ep["success"]:
                    n_success += 1
                    cfg_succ += 1
                    episodes.append(ep)
                elif keep_failures:
                    episodes.append(ep)
            print(f"  scale={s:<4} xy={actual_xy} : {cfg_succ}/{per_config}")
            env.close()

    print(f"Grid: {n_success}/{n_total} successes across {len(sizes)*len(positions)} configs")
    if not episodes:
        raise RuntimeError("No episodes to save. Re-run a single config with --render to debug.")
    save_hdf5(out, episodes, env_args)


def collect(env_name, generate_episode_fn, out, n, seed, render, keep_failures,
            control_freq=20, horizon=1000, object_scale=1.0, placement_xy=None):
    """Shared CLI driver: run generate_episode_fn n times and save successes."""
    env = make_env(env_name, render=render, control_freq=control_freq, horizon=horizon,
                   object_scale=object_scale, placement_xy=placement_xy)
    env_args = get_env_args(env, env_name, control_freq=control_freq, horizon=horizon,
                            object_scale=object_scale, placement_xy=placement_xy)
    # The actual placement may be the env's own default when placement_xy was None.
    actual_xy = tuple(float(v) for v in env.placement_xy)

    episodes = []
    n_success = 0
    for i in range(n):
        np.random.seed(seed + i)
        ep = generate_episode_fn(env, render=render)
        ep["object_scale"] = float(object_scale)
        ep["placement_xy"] = actual_xy
        if ep["success"]:
            n_success += 1
            episodes.append(ep)
            print(f"[{i}] success, len={ep['actions'].shape[0]}")
        else:
            print(f"[{i}] FAILURE, len={ep['actions'].shape[0]}")
            if keep_failures:
                episodes.append(ep)

    print(f"Successes: {n_success}/{n}")
    if len(episodes) == 0:
        raise RuntimeError("No episodes to save. Re-run with --render to debug the scripted policy.")
    save_hdf5(out, episodes, env_args)
    env.close()
