#!/usr/bin/env python3
"""Render a scripted demo to an mp4 showing BOTH stored camera frames side by side.

The image policy is trained on two fixed views per step -- `agentview` and
`sideview` (the exact pair `render_obs.py` writes into the *_img.hdf5 obs). This
script renders that pair per step and stitches them horizontally into one video,
so you can watch what the policy actually sees. It reuses each task's collector
trajectory (imported, not duplicated) via the Recorder `frame_cb` hook, so the
video matches the demo step-for-step.

The drawer env repoints its `agentview` straight down at the (floating) handle on
every reset, so rendering through the env -- rather than a raw camera -- yields
the same top-down view that is baked into the dataset. Each panel is labelled with
its camera name; a header strip shows the task + config. H.264 + yuv420p mp4
(git-ignored). Needs an EGL offscreen context (MUJOCO_GL=egl) on a headless box.

    MUJOCO_GL=egl python scripts/record_dualview_video.py --task drawer
    MUJOCO_GL=egl python scripts/record_dualview_video.py --task bottle
    MUJOCO_GL=egl python scripts/record_dualview_video.py --task drawer \
        --object-scale 0.85 --px 0.0 --py 0.05 --out videos/drawer_dual_s085.mp4
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

import franka_drawer_bottle  # noqa: F401  (registers both task envs)
from scripts.demo_common import load_controller_config

# Per-task wiring: env name, default placement, collector entry point, defaults.
TASKS = {
    "drawer": dict(
        env_name="FrankaDrawerOpen",
        label="FrankaDrawerOpen",
        init_noise_none=True,  # collector relies on a reproducible reset pose
    ),
    "bottle": dict(
        env_name="FrankaBottleUntwist",
        label="FrankaBottleUntwist",
        init_noise_none=False,
    ),
}

CAMERAS = ("agentview", "sideview")


def _import_collector(task):
    if task == "drawer":
        from scripts.collect_drawer_demos import CONTROL_FREQ, HORIZON, generate_episode
    else:
        from scripts.collect_bottle_demos import CONTROL_FREQ, HORIZON, generate_episode
    return CONTROL_FREQ, HORIZON, generate_episode


def make_render_env(env_name, cameras, width, height, control_freq, horizon,
                    init_noise_none, object_scale=1.0, placement_xy=None):
    """Task env with an offscreen renderer for multiple cameras (we call sim.render)."""
    kwargs = dict(
        env_name=env_name,
        robots="Panda",
        gripper_types="default",
        controller_configs=load_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=True,
        render_visual_mesh=True,
        render_collision_mesh=False,
        use_camera_obs=False,
        camera_names=list(cameras),
        camera_heights=height,
        camera_widths=width,
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
        hard_reset=False,
        object_scale=object_scale,
    )
    if init_noise_none:
        kwargs["initialization_noise"] = None
    if placement_xy is not None:
        kwargs["placement_xy"] = tuple(placement_xy)
    return suite.make(**kwargs)


def _label_strip(width, text, height=22, bg=(20, 20, 20), fg=(235, 235, 235)):
    """A tiny rasterized-text banner (5x7 block font) so panels are self-describing
    without a font dependency."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = bg
    _draw_text(strip, text.upper(), 4, (height - 7) // 2, fg)
    return strip


# Minimal 5x7 uppercase block font (only the glyphs we emit). Each glyph is 5 rows
# of a 5-wide bit pattern read MSB-first; rows are 7 tall (top/bottom padded).
_FONT = {
    " ": ["00000"] * 7,
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10011", "10101", "10101", "10101", "11001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00000", "00100"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    ",": ["00000", "00000", "00000", "00000", "00000", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ":": ["00000", "00100", "00000", "00000", "00000", "00100", "00000"],
}


def _draw_text(img, text, x, y, color):
    """Blit `text` at (x,y) top-left using the 5x7 block font (1px gaps)."""
    cx = x
    for ch in text:
        glyph = _FONT.get(ch, _FONT[" "])
        for ry, row in enumerate(glyph):
            for rxi, bit in enumerate(row):
                if bit == "1":
                    py, px = y + ry, cx + rxi
                    if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                        img[py, px] = color
        cx += 6
        if cx >= img.shape[1]:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS), help="Which task to record.")
    p.add_argument("--out", default=None, help="Output mp4 (default: videos/<task>_dualview.mp4)")
    p.add_argument("--width", type=int, default=320, help="Per-panel render width.")
    p.add_argument("--height", type=int, default=320, help="Per-panel render height.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--px", type=float, default=None)
    p.add_argument("--py", type=float, default=None)
    args = p.parse_args()

    cfg = TASKS[args.task]
    control_freq, horizon, generate_episode = _import_collector(args.task)
    if args.out is None:
        args.out = f"videos/{args.task}_dualview.mp4"

    placement_xy = None if (args.px is None and args.py is None) else (args.px or 0.0, args.py or 0.0)
    env = make_render_env(
        cfg["env_name"], CAMERAS, args.width, args.height, control_freq, horizon,
        cfg["init_noise_none"], object_scale=args.object_scale, placement_xy=placement_xy,
    )
    np.random.seed(args.seed)

    px = placement_xy[0] if placement_xy else 0.0
    py = placement_xy[1] if placement_xy else 0.0
    header_txt = f"{cfg['label']}  SCALE={args.object_scale:.2f} XY=({px:.2f},{py:.2f})"

    frames = []

    def grab():
        # MuJoCo renders bottom-up; flip each panel to image orientation, then label
        # and concatenate the two camera views horizontally.
        panels = []
        for cam in CAMERAS:
            img = np.flipud(env.sim.render(width=args.width, height=args.height, camera_name=cam)).copy()
            strip = _label_strip(args.width, cam)
            panels.append(np.vstack([strip, img]))
        sep = np.full((panels[0].shape[0], 4, 3), 60, dtype=np.uint8)  # thin divider
        row = np.hstack([panels[0], sep, panels[1]])
        header = _label_strip(row.shape[1], header_txt, height=24)
        frames.append(np.vstack([header, row]))

    ep = generate_episode(env, render=False, noise_scale=0.0, frame_cb=grab)
    env.close()

    print(f"success={ep['success']}, demo_len={ep['actions'].shape[0]}, frames={len(frames)}")
    if not frames:
        raise RuntimeError("No frames captured.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4(out, frames, args.fps)
    h, w = frames[0].shape[:2]
    print(f"Wrote {len(frames)} frames ({w}x{h} @ {args.fps}fps, H.264) to {out}")


def _write_mp4(path, frames, fps):
    """Encode RGB uint8 frames to an H.264 + yuv420p mp4 via ffmpeg (stdin pipe).
    Pads odd dimensions to even (yuv420p requirement)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode mp4.")
    h, w = frames[0].shape[:2]
    pad_h, pad_w = h + (h % 2), w + (w % 2)
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an",
        "-vf", f"pad={pad_w}:{pad_h}:0:0",
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


if __name__ == "__main__":
    main()
