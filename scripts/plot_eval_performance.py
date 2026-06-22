#!/usr/bin/env python3
"""Plot diffusion-policy generalization performance from the eval CSVs.

Reads figures/eval_numbers.csv + eval_summary.csv (written by
scripts/collect_eval_numbers.py) and produces, for the two task checkpoints:

  figures/perf_by_category.png  -- grouped bars over SEEN / UNSEEN_* / OVERALL,
                                   annotated k/N with binomial (Wilson) 95% CI.
  figures/perf_vs_size.png      -- success rate vs object size (mean over tested
                                   positions), seen size range [0.85,1.15] shaded.
  figures/perf_vs_position.png  -- success rate over the x/y offset grid at the
                                   nominal size (seen 3x3 + held-out positions).
  figures/results_table.md      -- design (which sizes/positions are seen vs
                                   held-out vs OOB) + per-category performance.
  figures/results_table.png     -- the same performance table, styled.

Default checkpoint = To2 (the main runs); pass --checkpoint To4 to switch. No
pandas (stdlib csv only). Read-only on eval/.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG = REPO_ROOT / "figures"

TASKS = ["drawer", "bottle"]
# The standalone UNSEEN_OOB category is folded into the size / pos / size+pos
# axes (see merge_category): "unseen" now spans in-range interpolation AND the
# mild OOB extrapolation, per request. OVERALL is unchanged (same rollouts,
# only relabeled).
CATS = ["SEEN", "UNSEEN_SIZE", "UNSEEN_POS", "UNSEEN_BOTH", "OVERALL"]
CAT_LABEL = {
    "SEEN": "Seen\n(trained)", "UNSEEN_SIZE": "Unseen\nsize",
    "UNSEEN_POS": "Unseen\npos", "UNSEEN_BOTH": "Unseen\nsize+pos",
    "UNSEEN_OOB": "OOB\nextrap.", "OVERALL": "Overall",
}
# interpolation (in-hull) categories vs the out-of-bound extrapolation one
CAT_FILL = {
    "SEEN": "#2e7d32", "UNSEEN_SIZE": "#1565c0", "UNSEEN_POS": "#1976d2",
    "UNSEEN_BOTH": "#0d47a1", "UNSEEN_OOB": "#c62828", "OVERALL": "#555555",
}

SIZE_ROLE = {0.85: "SEEN", 1.0: "SEEN", 1.15: "SEEN",
             0.925: "UNSEEN", 1.05: "UNSEEN", 1.10: "UNSEEN",
             0.80: "OOB", 1.20: "OOB"}
ROLE_COLOR = {"SEEN": "#2e7d32", "UNSEEN": "#1565c0", "OOB": "#c62828"}


def load_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def wilson(k, n, z=1.96):
    """Return (rate, lo, hi) Wilson 95% interval; (0,0,0) if n==0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def fmt_pct(rate):
    return f"{100 * rate:.0f}%"


def merge_category(row):
    """Fold UNSEEN_OOB into the unseen size/pos/both axis it actually varies.

    The 8 OOB configs are: 2 size-OOB at nominal pos (-> UNSEEN_SIZE), 4 pos-OOB
    at nominal size (-> UNSEEN_POS), 2 size+pos-OOB corners (-> UNSEEN_BOTH).
    So "unseen size/pos" now include the mild extrapolation past the trained hull.
    Other categories pass through unchanged.
    """
    cat = row["category"]
    if cat != "UNSEEN_OOB":
        return cat
    size_oob = abs(float(row["scale"]) - 1.0) > 1e-9
    pos_oob = abs(float(row["dx"])) > 1e-9 or abs(float(row["dy"])) > 1e-9
    if size_oob and pos_oob:
        return "UNSEEN_BOTH"
    return "UNSEEN_SIZE" if size_oob else "UNSEEN_POS"


def build_summary(per_config):
    """Recompute per-(task,checkpoint,category) summaries from the per-config rows
    using the merged categories (so OOB is counted inside size/pos/both), plus an
    OVERALL bucket. Returns rows shaped like the eval_summary.csv rows."""
    agg = defaultdict(lambda: [0, 0])  # (task, ckpt, cat) -> [success, total]
    for r in per_config:
        k, n = int(r["n_success"]), int(r["n"])
        for cat in (merge_category(r), "OVERALL"):
            key = (r["task"], r["checkpoint"], cat)
            agg[key][0] += k
            agg[key][1] += n
    rows = []
    for (task, ckpt, cat), (s, t) in agg.items():
        rows.append(dict(task=task, checkpoint=ckpt, category=cat,
                         success=s, total=t, rate=(s / t if t else 0.0)))
    return rows


# --------------------------------------------------------------------------- #
def plot_by_category(summary, ckpt, out):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        rows = {r["category"]: r for r in summary
                if r["task"] == task and r["checkpoint"] == ckpt}
        if not rows:
            ax.set_visible(False)
            continue
        xs = np.arange(len(CATS))
        rates, los, his, labels, colors = [], [], [], [], []
        for cat in CATS:
            r = rows.get(cat)
            k, n = (int(r["success"]), int(r["total"])) if r else (0, 0)
            p, lo, hi = wilson(k, n)
            rates.append(p); los.append(p - lo); his.append(hi - p)
            labels.append(f"{k}/{n}")
            colors.append(CAT_FILL[cat])
        bars = ax.bar(xs, rates, color=colors, width=0.7,
                      yerr=[los, his], capsize=4, ecolor="#333333",
                      error_kw=dict(lw=1.2))
        for x, p, hi, lab in zip(xs, rates, his, labels):
            ax.text(x, min(p + hi + 0.03, 1.07), lab, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.axhline(1.0, color="#cccccc", lw=0.8, zorder=0)
        ax.set_xticks(xs); ax.set_xticklabels([CAT_LABEL[c] for c in CATS], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_title(f"{task.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Success rate")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Success by generalization regime — unseen size/pos include OOB extrapolation\n"
                 "error bars = binomial 95% CI;  n labels = successes / rollouts",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140)
    print(f"[wrote] {out}")


def plot_vs_size(per_config, ckpt, out):
    """Isolate the SIZE effect: hold position fixed.

    Every in-range size {0.85,0.925,1.0,1.05,1.10,1.15} was tested on the SAME
    position row -- y-offset 0, x-offset in {-0.05, 0, +0.05} (3 configs = 15
    rollouts each) -- so restricting to that row makes the curve a clean function
    of size (aggregating over *all* tested positions confounds it, because size
    1.0 alone also carries the +/-0.075 OOB positions). The two OOB sizes
    {0.80,1.20} were only run at the nominal position, so they're shown as
    separate nominal-only points (n=5, hollow marker).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ROW_DX = {-0.05, 0.0, 0.05}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, task in zip(axes, TASKS):
        rows = [r for r in per_config
                if r["task"] == task and r["checkpoint"] == ckpt]
        inrange = defaultdict(lambda: [0, 0])   # fixed position row
        oob = defaultdict(lambda: [0, 0])        # nominal only
        for r in rows:
            s = round(float(r["scale"]), 3)
            dx, dy = round(float(r["dx"]), 3), round(float(r["dy"]), 3)
            if SIZE_ROLE.get(s) == "OOB":
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    oob[s][0] += int(r["n_success"]); oob[s][1] += int(r["n"])
            elif abs(dy) < 1e-6 and dx in ROW_DX:
                inrange[s][0] += int(r["n_success"]); inrange[s][1] += int(r["n"])

        ax.axvspan(0.85, 1.15, color="#2e7d32", alpha=0.08)
        # in-range curve (size isolated)
        sizes = sorted(inrange)
        rates, los, his, colors = [], [], [], []
        for s in sizes:
            k, n = inrange[s]
            p, lo, hi = wilson(k, n)
            rates.append(p); los.append(p - lo); his.append(hi - p)
            colors.append(ROLE_COLOR[SIZE_ROLE[s]])
        ax.plot(sizes, rates, "-", color="#888888", lw=1.5, zorder=1)
        ax.errorbar(sizes, rates, yerr=[los, his], fmt="none",
                    ecolor="#999999", capsize=3, lw=1, zorder=2)
        ax.scatter(sizes, rates, c=colors, s=95, zorder=3, edgecolor="white", lw=1)
        for s, p in zip(sizes, rates):
            ax.annotate(fmt_pct(p), (s, p), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8)
        # OOB points (nominal position only) -- hollow red squares
        for s in sorted(oob):
            k, n = oob[s]
            p, lo, hi = wilson(k, n)
            ax.errorbar([s], [p], yerr=[[p - lo], [hi - p]], fmt="none",
                        ecolor="#cc9999", capsize=3, lw=1, zorder=2)
            ax.scatter([s], [p], facecolors="none", edgecolors=ROLE_COLOR["OOB"],
                       s=110, lw=2, marker="s", zorder=3)
            ax.annotate(fmt_pct(p), (s, p), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=ROLE_COLOR["OOB"])
        ax.set_xlabel("Object size (scale ×)")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{task.capitalize()}", fontsize=13, fontweight="bold")
        ax.grid(ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLOR["SEEN"],
               markersize=10, label="Seen size (fixed-pos row, n=15)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLOR["UNSEEN"],
               markersize=10, label="Unseen size, in-range (n=15)"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor=ROLE_COLOR["OOB"],
               markerfacecolor="none", markeredgewidth=2, markersize=10,
               label="OOB size, nominal pos only (n=5)"),
        plt.Rectangle((0, 0), 1, 1, color="#2e7d32", alpha=0.15, label="trained size range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Success rate vs object size with position held fixed",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=140)
    print(f"[wrote] {out}")


def plot_vs_position(per_config, ckpt, out):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, task in zip(axes, TASKS):
        # nominal-size configs whose generalization axis is POSITION
        rows = [r for r in per_config
                if r["task"] == task and r["checkpoint"] == ckpt
                and abs(float(r["scale"]) - 1.0) < 1e-6
                and r["category"] in ("SEEN", "UNSEEN_POS", "UNSEEN_OOB")]
        dxs = sorted({round(float(r["dx"]), 3) for r in rows})
        dys = sorted({round(float(r["dy"]), 3) for r in rows})
        grid = np.full((len(dys), len(dxs)), np.nan)
        for r in rows:
            i = dys.index(round(float(r["dy"]), 3))
            j = dxs.index(round(float(r["dx"]), 3))
            grid[i, j] = float(r["rate"])
        im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(dxs))); ax.set_xticklabels([f"{v:+.02f}" for v in dxs], fontsize=8)
        ax.set_yticks(range(len(dys))); ax.set_yticklabels([f"{v:+.02f}" for v in dys], fontsize=8)
        for i in range(len(dys)):
            for j in range(len(dxs)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, fmt_pct(grid[i, j]), ha="center", va="center",
                            fontsize=8, fontweight="bold")
        # mark the trained 3x3 box (offsets within +/-0.05)
        ax.set_xlabel("x offset from nominal (m)")
        ax.set_ylabel("y offset from nominal (m)")
        ax.set_title(f"{task.capitalize()}", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.suptitle("Success rate over object position at nominal size",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140)
    print(f"[wrote] {out}")


def write_table(summary, ckpt, md_out, png_out):
    import matplotlib.pyplot as plt

    # ---- design block: seen vs held-out (held-out = in-range interp + OOB extrap) ----
    design = [
        ("Object size (scale×)", "0.85, 1.00, 1.15",
         "0.925, 1.05, 1.10  +  0.80, 1.20 (OOB)"),
        ("Position offset (m)", "{−0.05, 0, +0.05}² grid",
         "{−0.04, +0.01, +0.04}  +  ±0.075 (OOB)"),
    ]
    # ---- performance block ----
    by = {(r["task"], r["category"]): r for r in summary if r["checkpoint"] == ckpt}
    perf_rows = []
    for cat in CATS:
        cells = [CAT_LABEL[cat].replace("\n", " ")]
        for task in TASKS:
            r = by.get((task, cat))
            if r:
                cells.append(f"{fmt_pct(float(r['rate']))} ({r['success']}/{r['total']})")
            else:
                cells.append("—")
        perf_rows.append(cells)

    # markdown
    lines = [f"# Diffusion-policy generalization results", "",
             "_Unseen size/pos include both held-out in-range configs and OOB "
             "configs that extrapolate past the trained hull._", "",
             "## Experiment design", "",
             "| Axis | Seen (trained) | Unseen (held-out, incl. OOB extrap.) |",
             "|---|---|---|"]
    for name, seen, uns in design:
        lines.append(f"| {name} | {seen} | {uns} |")
    lines += ["", "## Success rate by regime", "",
              "| Regime | Drawer | Bottle |", "|---|---|---|"]
    for cells in perf_rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    md_out.write_text("\n".join(lines))
    print(f"[wrote] {md_out}")

    # styled PNG of the performance table
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axis("off")
    col_labels = ["Regime", "Drawer", "Bottle"]
    table = ax.table(cellText=perf_rows, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1, 1.6)
    for j in range(len(col_labels)):
        c = table[0, j]; c.set_facecolor("#37474f"); c.set_text_props(color="white", fontweight="bold")
    for i, cat in enumerate(CATS, start=1):
        table[i, 0].set_text_props(color=CAT_FILL[cat], fontweight="bold")
        if cat == "OVERALL":
            for j in range(len(col_labels)):
                table[i, j].set_facecolor("#eceff1")
    ax.set_title("Success rate by regime", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    print(f"[wrote] {png_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="To2", choices=["To2", "To4"])
    args = p.parse_args()
    ckpt = args.checkpoint

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 13,
                         "figure.facecolor": "white"})

    per_config = load_csv(FIG / "eval_numbers.csv")
    # Recompute summaries with the OOB-merged categories (eval_summary.csv on disk
    # still has the original 5-way split; we don't touch it).
    summary = build_summary(per_config)

    # Suffix non-default checkpoints so the To4 figures DON'T overwrite the To2
    # ones (To2 keeps the bare filenames it was first written with).
    sfx = "" if ckpt == "To2" else f"_{ckpt}"
    plot_by_category(summary, ckpt, FIG / f"perf_by_category{sfx}.png")
    plot_vs_size(per_config, ckpt, FIG / f"perf_vs_size{sfx}.png")
    plot_vs_position(per_config, ckpt, FIG / f"perf_vs_position{sfx}.png")
    write_table(summary, ckpt, FIG / f"results_table{sfx}.md",
                FIG / f"results_table{sfx}.png")


if __name__ == "__main__":
    main()
