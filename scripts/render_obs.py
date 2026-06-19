#!/usr/bin/env python3
"""Offline state->image pass: add camera observations to a states demo dataset.

The collectors store sim STATES + the (scaled) model_file per demo (robomimic
convention), NOT images. For an image-based policy we render the images offline:
for each demo we rebuild the env from the demo's own model_file (so the object is
the right SIZE and POSITION) and replay the stored states, grabbing agentview +
sideview frames. This keeps the states HDF5 small and lets us re-render at any
resolution / camera set later.

Writes a new HDF5 mirroring the input layout, adding per demo:
    data/demo_i/obs/agentview_image     uint8   (T, H, W, 3)
    data/demo_i/obs/sideview_image      uint8   (T, H, W, 3)
    data/demo_i/obs/robot0_eef_pos      float32 (T, 3)
    data/demo_i/obs/robot0_eef_quat     float32 (T, 4)   xyzw (robosuite convention)
    data/demo_i/obs/robot0_gripper_qpos float32 (T, 2)
and carrying over states/actions/rewards/dones + attrs (object_scale, placement_xy,
model_file). The proprio keys are the standard robomimic low-dim set; a diffusion
policy conditions on image(s) + this proprioception. They are *positional* (read
from the same sim state that produced the image), so they reproduce exactly from
the stored states. Needs an offscreen GL context:

    MUJOCO_GL=egl python scripts/render_obs.py --in data/drawer.hdf5 --out data/drawer_img.hdf5
    MUJOCO_GL=egl python scripts/render_obs.py --in data/bottle.hdf5 --out data/bottle_img.hdf5 --size 84
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import robosuite as suite

import franka_drawer_bottle  # noqa: F401  (registers the envs)
from scripts.demo_common import load_controller_config

CAMERAS = ("agentview", "sideview")
# Standard robomimic low-dim proprio set; a diffusion policy conditions on these
# alongside the camera image(s). All positional, so they reproduce from the state.
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def make_render_env(env_name, cameras, size):
    return suite.make(
        env_name=env_name,
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,            # we call sim.render directly
        camera_names=list(cameras),
        camera_heights=size,
        camera_widths=size,
        control_freq=20,
        horizon=10,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
    )


def render_demo(env, model_file, states, cameras, size, proprio_keys):
    # Rebuild the model so the object matches this demo's scaled geometry, then
    # replay states (model_file already encodes the size/position + drawer camera).
    env.reset_from_xml_string(model_file)
    frames = {cam: np.empty((len(states), size, size, 3), dtype=np.uint8) for cam in cameras}
    proprio = {k: [] for k in proprio_keys}
    for t, s in enumerate(states):
        env.sim.set_state_from_flattened(np.asarray(s))
        env.sim.forward()
        for cam in cameras:
            img = env.sim.render(width=size, height=size, camera_name=cam)
            frames[cam][t] = np.flipud(img)  # MuJoCo renders bottom-up
        if proprio_keys:
            # Recompute observables from the current (just-set) sim state. These are
            # positional proprio, so they match the state that produced the image.
            obs = env._get_observations(force_update=True)
            for k in proprio_keys:
                proprio[k].append(np.asarray(obs[k], dtype=np.float32))
    proprio = {k: np.stack(v) for k, v in proprio.items()}
    return frames, proprio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Input states HDF5 (from a collector).")
    p.add_argument("--out", required=True, help="Output HDF5 with image observations added.")
    p.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    p.add_argument("--proprio", nargs="*", default=list(PROPRIO_KEYS),
                   help="Proprio observable keys to store (empty list = none).")
    p.add_argument("--size", type=int, default=84, help="Square image side (px).")
    p.add_argument("--max-demos", type=int, default=None, help="Render only the first N demos (smoke test).")
    args = p.parse_args()

    fin = h5py.File(args.inp, "r")
    env_args = json.loads(fin["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    env = make_render_env(env_name, args.cameras, args.size)

    demo_keys = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[1]))
    if args.max_demos is not None:
        demo_keys = demo_keys[: args.max_demos]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as fout:
        dgrp = fout.create_group("data")
        for k, v in fin["data"].attrs.items():
            dgrp.attrs[k] = v
        total = 0
        for di, key in enumerate(demo_keys):
            din = fin["data"][key]
            states = din["states"][:]
            frames, proprio = render_demo(
                env, din.attrs["model_file"], states, args.cameras, args.size, args.proprio
            )

            dout = dgrp.create_group(key)
            for ak, av in din.attrs.items():
                dout.attrs[ak] = av
            for ds in ("states", "actions", "rewards", "dones"):
                if ds in din:
                    dout.create_dataset(ds, data=din[ds][:], compression="gzip")
            if "final_state" in din:
                dout.create_dataset("final_state", data=din["final_state"][:])
            ogrp = dout.create_group("obs")
            for cam in args.cameras:
                # NO compression on the image datasets: gzip decompression per batch was
                # the training data-loading bottleneck (~97% of epoch time). Uncompressed,
                # the images load far faster (and fit in RAM with hdf5_cache_mode="all").
                ogrp.create_dataset(f"{cam}_image", data=frames[cam])
            for k, arr in proprio.items():
                ogrp.create_dataset(k, data=arr, compression="gzip")
            total += states.shape[0]
            print(f"  [{di+1}/{len(demo_keys)}] {key}: {states.shape[0]} frames x {len(args.cameras)} cams")
        dgrp.attrs["total"] = total
    fin.close()
    env.close()
    print(f"Wrote image obs for {len(demo_keys)} demos -> {out}")


if __name__ == "__main__":
    main()
