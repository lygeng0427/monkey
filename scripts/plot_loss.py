#!/usr/bin/env python3
"""Plot diffusion-policy training/validation loss curves from robomimic TensorBoard logs.

robomimic logs `Train/Loss` and `Valid/Loss` scalars (one point per epoch) under
runs/diffusion_<task>/.../logs/tb. This reads those event files and saves a PNG. By
default it plots BOTH tasks side by side; pass --task to plot just one.

    python scripts/plot_loss.py                       # both tasks -> runs/loss_curves.png
    python scripts/plot_loss.py --task drawer         # one task
    python scripts/plot_loss.py --out /tmp/loss.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["drawer", "bottle"]


def latest_tb_dir(task):
    """Newest logs/tb dir for a task (handles multiple timestamped runs)."""
    hits = glob.glob(str(REPO_ROOT / "runs" / f"diffusion_{task}" / "*" / "*" / "logs" / "tb"))
    if not hits:
        return None
    return max(hits, key=lambda p: Path(p).stat().st_mtime)


def load_scalars(tb_dir, tag):
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    pts = ea.Scalars(tag)
    return [p.step for p in pts], [p.value for p in pts]


def plot_task(task, ax):
    tb = latest_tb_dir(task)
    if tb is None:
        ax.set_title(f"{task}: no logs yet")
        return
    tr_x, tr_y = load_scalars(tb, "Train/Loss")
    va_x, va_y = load_scalars(tb, "Valid/Loss")
    ax.plot(tr_x, tr_y, "-", color="tab:blue", label="train")
    if va_x:
        ax.plot(va_x, va_y, "-", color="tab:orange", label="valid")
    ax.set_title(f"diffusion_{task}  (final train={tr_y[-1]:.4f})" if tr_y else task)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None, help="One task (default: both).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        plot_task(task, ax)
    fig.suptitle("Diffusion Policy training loss")
    fig.tight_layout()

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / (
        f"loss_curve_{args.task}.png" if args.task else "loss_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
