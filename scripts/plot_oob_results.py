#!/usr/bin/env python3
"""Visualize the expanded-OOB re-evaluation (To=2 checkpoints, DDIM-16).

Reads eval/<task>_oobfull.json (the fresh full eval with the EXPANDED 27-config
UNSEEN_OOB set, written by eval_diffusion_policy.py) and reports three unseen
regimes side by side:

  * Unseen (in-boundary)  = UNSEEN_SIZE + UNSEEN_POS + UNSEEN_BOTH (inside the hull)
  * Unseen (OOB)          = UNSEEN_OOB (27 configs, outside the hull)
  * Unseen (all)          = in-boundary + OOB pooled

plus the SEEN baseline. Writes NEW files only (never touches the previous figures):
  figures/oob_bars.png          grouped bars per task, binomial 95% CI + k/N
  figures/oob_vs_size.png       success vs object size at nominal pos (seen->OOB)
  figures/results_table_oob.md  + results_table_oob.png

No pandas; stdlib json/csv + matplotlib only.
"""
from __future__ import annotations
import json, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval"
FIG = REPO / "figures"
TASKS = ["drawer", "bottle"]
NOMINAL_XY = {"drawer": (0.05, 0.0), "bottle": (0.10, 0.0)}  # cf. eval_diffusion_policy.TASKS

INBOUND = {"UNSEEN_SIZE", "UNSEEN_POS", "UNSEEN_BOTH"}
# Regimes to report (label -> set of source categories).
REGIMES = [
    ("Seen (trained)", {"SEEN"}),
    ("Unseen in-boundary", INBOUND),
    ("Unseen OOB", {"UNSEEN_OOB"}),
    ("Unseen (all)", INBOUND | {"UNSEEN_OOB"}),
]
BAR_COLOR = {"Seen (trained)": "#2e7d32", "Unseen in-boundary": "#1565c0",
             "Unseen OOB": "#c62828", "Unseen (all)": "#6a1b9a"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def load(task):
    f = EVAL / f"{task}_oobfull.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["per_config"]


def agg(per_config, cats):
    rows = [c for c in per_config if c["category"] in cats]
    k = sum(c["n_success"] for c in rows)
    n = sum(c["n"] for c in rows)
    return k, n


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {t: load(t) for t in TASKS}
    present = [t for t in TASKS if data[t]]
    if not present:
        raise SystemExit("No eval/<task>_oobfull.json found yet.")

    # ---------- (1) grouped bar chart ----------
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.8), squeeze=False)
    for ax, task in zip(axes[0], present):
        pc = data[task]
        labels = [r[0] for r in REGIMES]
        xs = np.arange(len(labels))
        for x, (lab, cats) in zip(xs, REGIMES):
            k, n = agg(pc, cats)
            p, lo, hi = wilson(k, n)
            ax.bar(x, p, color=BAR_COLOR[lab], width=0.7,
                   yerr=[[p - lo], [hi - p]], capsize=4, ecolor="#333")
            ax.text(x, min(hi + 0.03, 1.06), f"{p:.0%}\n{k}/{n}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("success rate")
        ax.set_title(task.capitalize(), fontsize=13, fontweight="bold")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", ls=":", alpha=0.5)
    fig.suptitle("In-boundary vs out-of-boundary generalization\n"
                 "error bars = binomial 95% CI;  n labels = successes / rollouts",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "oob_bars.png", dpi=140)
    print("[wrote]", FIG / "oob_bars.png")

    # ---------- (2) success vs size at nominal position ----------
    SEEN_S = {0.85, 1.0, 1.15}
    fig, axes = plt.subplots(1, len(present), figsize=(6.0 * len(present), 4.4), squeeze=False)
    for ax, task in zip(axes[0], present):
        pc = data[task]
        nx, ny = NOMINAL_XY[task]
        # configs at the nominal position, any size
        at_nom = [c for c in pc
                  if abs(c["xy"][0] - nx) < 1e-9 and abs(c["xy"][1] - ny) < 1e-9]
        bysize = {}
        for c in at_nom:
            bysize.setdefault(round(c["scale"], 3), [0, 0])
            bysize[round(c["scale"], 3)][0] += c["n_success"]
            bysize[round(c["scale"], 3)][1] += c["n"]
        sizes = sorted(bysize)
        ps, los, his, cols = [], [], [], []
        for s in sizes:
            k, n = bysize[s]
            p, lo, hi = wilson(k, n)
            ps.append(p); los.append(p - lo); his.append(hi - p)
            if s in SEEN_S:
                cols.append("#2e7d32")
            elif 0.85 <= s <= 1.15:
                cols.append("#1565c0")
            else:
                cols.append("#c62828")
        ax.axvspan(0.85, 1.15, color="#2e7d32", alpha=0.08, label="trained size range")
        ax.errorbar(sizes, ps, yerr=[los, his], fmt="-", color="#888", zorder=1, capsize=3)
        ax.scatter(sizes, ps, c=cols, s=70, zorder=2, edgecolor="k", linewidth=0.5)
        for s, p in zip(sizes, ps):
            ax.annotate(f"{p:.0%}", (s, p), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_xlabel("object size (scale x)")
        ax.set_ylabel("success rate")
        ax.set_title(task.capitalize(), fontsize=13, fontweight="bold")
        ax.grid(ls=":", alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", label=l)
               for c, l in [("#2e7d32", "trained size"), ("#1565c0", "unseen in-range"),
                            ("#c62828", "OOB (extrapolation)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Success rate vs object size at nominal position", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIG / "oob_vs_size.png", dpi=140)
    print("[wrote]", FIG / "oob_vs_size.png")

    # ---------- (2b) success rate over position at nominal size (incl. OOB) ----------
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, len(present), figsize=(6.6 * len(present), 5.4), squeeze=False)
    for ax, task in zip(axes[0], present):
        pc = data[task]
        nx, ny = NOMINAL_XY[task]
        # nominal SIZE (1.0) configs whose generalization axis is POSITION
        rows = [c for c in pc if abs(c["scale"] - 1.0) < 1e-9
                and c["category"] in ("SEEN", "UNSEEN_POS", "UNSEEN_OOB")]
        dxs = sorted({round(c["xy"][0] - nx, 3) for c in rows})
        dys = sorted({round(c["xy"][1] - ny, 3) for c in rows})
        grid = np.full((len(dys), len(dxs)), np.nan)
        for c in rows:
            i = dys.index(round(c["xy"][1] - ny, 3))
            j = dxs.index(round(c["xy"][0] - nx, 3))
            grid[i, j] = c["n_success"] / c["n"]
        im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(dxs))); ax.set_xticklabels([f"{v:+.03f}" for v in dxs], fontsize=8)
        ax.set_yticks(range(len(dys))); ax.set_yticklabels([f"{v:+.03f}" for v in dys], fontsize=8)
        for i in range(len(dys)):
            for j in range(len(dxs)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                            fontsize=8, fontweight="bold")
        # outline the trained position region (offsets within +/-0.05)
        inx = [j for j, v in enumerate(dxs) if abs(v) <= 0.05 + 1e-9]
        iny = [i for i, v in enumerate(dys) if abs(v) <= 0.05 + 1e-9]
        if inx and iny:
            ax.add_patch(Rectangle((min(inx) - 0.5, min(iny) - 0.5),
                                   len(inx), len(iny), fill=False,
                                   edgecolor="#1565c0", lw=2.5))
            ax.text(min(inx) - 0.4, max(iny) + 0.55, "trained region (+/-0.05)",
                    color="#1565c0", fontsize=8, fontweight="bold", va="bottom")
        ax.set_xlabel("x offset from nominal (m)")
        ax.set_ylabel("y offset from nominal (m)")
        ax.set_title(task.capitalize(), fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.suptitle("Success rate over object position at nominal size\n"
                 "cells outside the blue box are OOB position extrapolation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "perf_vs_position_oob.png", dpi=140)
    print("[wrote]", FIG / "perf_vs_position_oob.png")

    # ---------- (3) table (md + png) ----------
    def cell(task, cats):
        k, n = agg(data[task], cats)
        p, _, _ = wilson(k, n)
        return f"{p:.0%} ({k}/{n})" if n else "-"

    rows = [(lab, [cell(t, cats) for t in present]) for lab, cats in REGIMES]
    rows.append(("Overall (all configs)",
                 [cell(t, {"SEEN"} | INBOUND | {"UNSEEN_OOB"}) for t in present]))

    md = ["# Expanded-OOB generalization results",
          "",
          "_Fresh full re-evaluation of the To=2 checkpoints with the expanded 27-config "
          "UNSEEN_OOB set (= the in-boundary unseen count). Sampler: DDIM-16. "
          "Previous results files are untouched._",
          "",
          "## Design",
          "",
          "| Axis | Seen (trained) | Unseen in-boundary | Unseen OOB |",
          "|---|---|---|---|",
          "| Object size (scale x) | 0.85, 1.00, 1.15 | 0.925, 1.05, 1.10 | 0.78, 1.20, 1.25 |",
          "| Position offset (m) | {-0.05,0,0.05}^2 grid | within +/-0.05 (off-grid) | "
          "+/-0.075, +/-0.10 |",
          "| # configs | 27 | 27 | 27 |",
          "",
          "## Success rate by regime",
          "",
          "| Regime | " + " | ".join(t.capitalize() for t in present) + " |",
          "|---|" + "---|" * len(present)]
    for lab, vals in rows:
        md.append(f"| {lab} | " + " | ".join(vals) + " |")
    (FIG / "results_table_oob.md").write_text("\n".join(md) + "\n")
    print("[wrote]", FIG / "results_table_oob.md")

    # png table
    col_labels = ["Regime"] + [t.capitalize() for t in present]
    table_rows = [[lab] + vals for lab, vals in rows]
    fig, ax = plt.subplots(figsize=(2.4 + 2.4 * len(present), 0.5 + 0.5 * len(table_rows)))
    ax.axis("off")
    tb = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.6)
    for j in range(len(col_labels)):
        tb[0, j].set_facecolor("#263238"); tb[0, j].set_text_props(color="w", fontweight="bold")
    rowcol = {"Seen (trained)": "#e8f5e9", "Unseen in-boundary": "#e3f2fd",
              "Unseen OOB": "#ffebee", "Unseen (all)": "#f3e5f5"}
    for i, (lab, _) in enumerate(rows, start=1):
        for j in range(len(col_labels)):
            tb[i, j].set_facecolor(rowcol.get(lab, "#fafafa"))
    ax.set_title("Success rate by regime (expanded OOB)", fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "results_table_oob.png", dpi=150, bbox_inches="tight")
    print("[wrote]", FIG / "results_table_oob.png")


if __name__ == "__main__":
    main()
