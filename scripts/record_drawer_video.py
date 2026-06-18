#!/usr/bin/env python3
"""Render a FrankaDrawerOpen scripted demonstration to an mp4 video.

Runs the exact same top-down-grasp-and-pull trajectory used by
collect_drawer_demos.py (imported, not duplicated), but with an offscreen
renderer attached. A frame is grabbed after every env step via the Recorder's
frame_cb hook, so the video matches the demo step-for-step. Frames are piped to
ffmpeg and encoded as H.264 + yuv420p -- the universally playable combo.

The drawer slides out toward the robot (world -x), so a side view shows the
pull-out in profile while the (now visible) cabinet body stays put. Output mp4s
are git-ignored (see .gitignore). Run from the project root:

    python scripts/record_drawer_video.py                       # -> videos/drawer_open_sideview.mp4
    python scripts/record_drawer_video.py --camera agentview    # -> videos/drawer_open_agentview.mp4
    python scripts/record_drawer_video.py --out videos/foo.mp4 --camera frontview
    python scripts/record_drawer_video.py --seed 3 --fps 30 --width 640 --height 480
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robosuite as suite

import franka_drawer_bottle  # noqa: F401  (registers FrankaDrawerOpen)
from scripts.collect_drawer_demos import CONTROL_FREQ, ENV_NAME, HORIZON, generate_episode
from scripts.demo_common import load_controller_config


def make_render_env(camera, width, height, control_freq, horizon, object_scale=1.0, placement_xy=None):
    """Drawer env with an offscreen renderer (camera obs off; we call sim.render)."""
    kwargs = dict(
        env_name=ENV_NAME,
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,
        camera_names=camera,
        camera_heights=height,
        camera_widths=width,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,  # reproducible: match the collector's start pose
        object_scale=object_scale,
    )
    if placement_xy is not None:
        kwargs["placement_xy"] = tuple(placement_xy)
    return suite.make(**kwargs)


def write_mp4_ffmpeg(path, frames, fps):
    """Encode RGB uint8 frames to an H.264 + yuv420p mp4 via ffmpeg (stdin pipe)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode mp4.")
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err[-2000:]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="Output mp4 path (default: videos/drawer_open_<camera>.mp4)")
    p.add_argument("--camera", default="sideview", help="MuJoCo camera name (e.g. sideview, frontview, agentview)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-scale", type=float, default=0.0)
    p.add_argument("--object-scale", type=float, default=1.0, help="Drawer size multiplier.")
    p.add_argument("--px", type=float, default=None, help="Object placement x (default: env default).")
    p.add_argument("--py", type=float, default=None, help="Object placement y (default: env default).")
    args = p.parse_args()
    if args.out is None:
        args.out = f"videos/drawer_open_{args.camera}.mp4"

    placement_xy = None if (args.px is None and args.py is None) else (args.px or 0.0, args.py or 0.0)
    env = make_render_env(args.camera, args.width, args.height, CONTROL_FREQ, HORIZON,
                          object_scale=args.object_scale, placement_xy=placement_xy)
    np.random.seed(args.seed)

    frames = []

    def grab():
        # MuJoCo renders bottom-up; flip to image (top-down) orientation.
        img = env.sim.render(width=args.width, height=args.height, camera_name=args.camera)
        frames.append(np.flipud(img).copy())

    ep = generate_episode(env, render=False, noise_scale=args.noise_scale, frame_cb=grab)
    env.close()

    print(f"success={ep['success']}, demo_len={ep['actions'].shape[0]}, frames={len(frames)}")
    if not frames:
        raise RuntimeError("No frames captured.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_mp4_ffmpeg(out, frames, args.fps)
    h, w = frames[0].shape[:2]
    print(f"Wrote {len(frames)} frames ({w}x{h} @ {args.fps}fps, H.264) to {out}")


if __name__ == "__main__":
    main()
