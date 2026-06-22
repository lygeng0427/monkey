#!/usr/bin/env python3
"""Flatten the diffusion-policy eval JSONs into tidy CSVs for plotting.

Reads every ``eval/*_full.json`` (written by scripts/eval_diffusion_policy.py),
keys each by (task, checkpoint), and emits two stdlib-csv files under figures/:

  * eval_numbers.csv  -- one row per evaluated config:
        task, checkpoint, scale, x, y, dx, dy, category, n, n_success, rate
  * eval_summary.csv  -- one row per (task, checkpoint, category) from the JSON
        summary block: task, checkpoint, category, success, total, rate

`checkpoint` is parsed from the filename: ``<task>_full.json`` -> To2 (the main,
To=2 runs); ``<task>_h4_full.json`` -> To4 (the obs-horizon-4 experiment).

No pandas (not installed) and no heavy imports: this only touches JSON, so it
runs in any interpreter. Read-only on eval/ -- writes only under figures/.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
OUT_DIR = REPO_ROOT / "figures"

# Nominal object placement per task (mirrors eval_diffusion_policy.TASKS;
# hard-coded here so we don't import that module's torch/robosuite top-level deps
# just for two constants). dx/dy below are offsets from these centers.
NOMINAL_XY = {"drawer": (0.05, 0.0), "bottle": (0.10, 0.0)}

CATEGORY_ORDER = ["SEEN", "UNSEEN_SIZE", "UNSEEN_POS", "UNSEEN_BOTH",
                  "UNSEEN_OOB", "OVERALL"]


def parse_name(stem: str):
    """'<task>_full' -> (task, 'To2'); '<task>_h4_full' -> (task, 'To4')."""
    m = re.match(r"^(drawer|bottle)(_h4)?_full$", stem)
    if not m:
        return None
    task, h4 = m.group(1), m.group(2)
    return task, ("To4" if h4 else "To2")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(EVAL_DIR.glob("*_full.json"))
    if not files:
        raise SystemExit(f"No eval/*_full.json found under {EVAL_DIR}")

    per_config_rows = []
    summary_rows = []
    for f in files:
        parsed = parse_name(f.stem)
        if parsed is None:
            print(f"[skip] {f.name} (unrecognized name)")
            continue
        task, ckpt = parsed
        data = json.loads(f.read_text())
        nx, ny = NOMINAL_XY[task]

        for c in data["per_config"]:
            x, y = c["xy"]
            per_config_rows.append(dict(
                task=task, checkpoint=ckpt, scale=c["scale"],
                x=round(x, 4), y=round(y, 4),
                dx=round(x - nx, 4), dy=round(y - ny, 4),
                category=c["category"], n=c["n"],
                n_success=c["n_success"], rate=c["rate"],
            ))

        summ = data.get("summary", {})
        for cat in CATEGORY_ORDER:
            if cat in summ:
                s = summ[cat]
                summary_rows.append(dict(
                    task=task, checkpoint=ckpt, category=cat,
                    success=s["success"], total=s["total"], rate=s["rate"],
                ))
        print(f"[read] {f.name:24s} -> task={task} ckpt={ckpt} "
              f"({len(data['per_config'])} configs)")

    pc_path = OUT_DIR / "eval_numbers.csv"
    with pc_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "task", "checkpoint", "scale", "x", "y", "dx", "dy",
            "category", "n", "n_success", "rate"])
        w.writeheader()
        w.writerows(per_config_rows)

    sm_path = OUT_DIR / "eval_summary.csv"
    with sm_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "task", "checkpoint", "category", "success", "total", "rate"])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[wrote] {pc_path}  ({len(per_config_rows)} rows)")
    print(f"[wrote] {sm_path}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
