#!/usr/bin/env python3
"""Render the drawer + bottle at every evaluated object size into one montage.

For each task we build the env at the *nominal* position and at each distinct
tested object_scale, reset, and grab a single fixed-camera frame. Same camera,
resolution, and crop for every tile, so the only thing that changes across a row
is the object size (the size comparison would be meaningless otherwise).

Sizes are tagged SEEN / UNSEEN (in-range, held-out) / OOB (just outside the
trained [0.85, 1.15] hull) with a colored tile border.

Run headless with an EGL context, preferably on GPU0 so it doesn't slow a
concurrent eval on GPU1:

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
        python scripts/render_size_montage.py            # -> figures/size_montage.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import franka_drawer_bottle  # noqa: F401  (registers the envs)
from eval_diffusion_policy import TASKS, make_eval_env  # reuse env construction

OUT = REPO_ROOT / "figures" / "size_montage_oob.png"
RES = 512                      # render resolution per tile (square)
CAMERA = "sideview"            # fixed scene view that frames the whole object

# Distinct sizes evaluated, with their generalization role (color-coded). OOB sizes
# match the expanded eval set (0.78 below, 1.20/1.25 above the trained [0.85,1.15]).
SIZES = [
    (0.78, "OOB"),
    (0.85, "SEEN"),
    (0.925, "UNSEEN"),
    (1.00, "SEEN"),
    (1.05, "UNSEEN"),
    (1.10, "UNSEEN"),
    (1.15, "SEEN"),
    (1.20, "OOB"),
    (1.25, "OOB"),
]
TAG_COLOR = {"SEEN": "#2e7d32", "UNSEEN": "#1565c0", "OOB": "#c62828"}

# Fixed crop box per task (fractions of the RES x RES frame: top, bottom, left, right).
# IDENTICAL across every size in a row -> only the object size changes, so the
# comparison is honest; we just zoom each row onto its object (which lives in a
# different part of the sideview for the floating drawer vs the table-top bottle).
CROP = {
    "drawer": (0.10, 0.66, 0.04, 0.56),
    "bottle": (0.44, 0.92, 0.12, 0.60),
}


def render_one(env_name, nominal_xy, scale, crop):
    env = make_eval_env(env_name, scale, nominal_xy, max_horizon=100, init_noise=None)
    env.reset()
    # render at our montage resolution (env offscreen buffer was built >= this)
    img = np.flipud(env.sim.render(width=RES, height=RES, camera_name=CAMERA)).copy()
    env.close()
    t, b, l, r = crop
    return img[int(t * RES):int(b * RES), int(l * RES):int(r * RES)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tasks = ["drawer", "bottle"]
    ncol = len(SIZES)
    fig, axes = plt.subplots(len(tasks), ncol, figsize=(2.2 * ncol, 2.2 * len(tasks) + 0.4))

    for r, task in enumerate(tasks):
        spec = TASKS[task]
        print(f"[render] {task}: ", end="", flush=True)
        for c, (scale, tag) in enumerate(SIZES):
            print(f"{scale}", end=" ", flush=True)
            img = render_one(spec["env_name"], spec["nominal_xy"], scale, CROP[task])
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            color = TAG_COLOR[tag]
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(3)
            if r == 0:
                ax.set_title(f"{scale:.3f}×\n{tag}", color=color,
                             fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(task.capitalize(), fontsize=14, fontweight="bold")
        print()

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=TAG_COLOR[k], lw=4,
                      label={"SEEN": "Seen (trained)",
                             "UNSEEN": "Unseen (in-range)",
                             "OOB": "OOB (extrapolation)"}[k])
               for k in ["SEEN", "UNSEEN", "OOB"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Object size variation",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
