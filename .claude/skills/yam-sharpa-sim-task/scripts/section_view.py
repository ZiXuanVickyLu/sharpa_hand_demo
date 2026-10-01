#!/usr/bin/env python3
"""Section views of a run: the bodies' and colliders' vertices within a slab around a plane, at several steps.

  python3 .claude/skills/yam-sharpa-sim-task/scripts/section_view.py output/output/<run> app/config/<cfg>.json out.png 1100 1500 2500 \
      [--plane y --at 0.30 --band 0.008] [--xlim -0.15 0.15 --zlim -0.02 0.15] [--floor 0.005]

Default: an x-z section through y = 0.30 m (the hands' working depth), 8 mm thick. Bodies are coloured per
body, colliders grey; the floor line is drawn at --floor. This is the view that answers "where does the
object touch the hand, at what height, on which side" - the questions a clip cannot answer.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def layout(cfg):
    c = json.load(open(cfg)); ranges = []; o = 0
    for kind in ("bodies", "colliders"):
        for e in c.get(kind, []):
            f = os.path.join(ROOT, "asset", e["file"])
            if not f.endswith(".obj"):
                return ranges   # unknown count: stop the split here
            n = sum(1 for line in open(f) if line.startswith("v ")); ranges.append((kind, e["name"], o, o + n)); o += n
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("config"); ap.add_argument("out"); ap.add_argument("steps", type=int, nargs="+")
    ap.add_argument("--plane", choices=("x", "y"), default="y", help="axis normal to the section plane")
    ap.add_argument("--at", type=float, default=0.30, help="plane position (m)")
    ap.add_argument("--band", type=float, default=0.008, help="slab thickness (m)")
    ap.add_argument("--xlim", type=float, nargs=2, default=None); ap.add_argument("--zlim", type=float, nargs=2, default=None)
    ap.add_argument("--floor", type=float, default=0.005)
    a = ap.parse_args()
    ranges = layout(a.config); n = len(a.steps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5)); axes = np.atleast_1d(axes)
    h = 1 if a.plane == "y" else 0; v = 0 if a.plane == "y" else 1    # horizontal axis of the plot
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, sum(1 for k, *_ in ranges if k == "bodies"))))
    for ax, s in zip(axes, a.steps):
        X = np.load(os.path.join(a.run, "x_%d.npy" % s)) * 1e3
        bi = 0
        for kind, name, i0, i1 in ranges:
            P = X[i0:i1]; sel = np.abs(P[:, h] - a.at * 1e3) < a.band * 1e3
            if not sel.any():
                if kind == "bodies": bi += 1
                continue
            if kind == "bodies":
                Q = P[sel]; order = np.argsort(Q[:, v]); ax.plot(Q[order, v], Q[order, 2], "-", lw=0.8, color=colors[bi % len(colors)], alpha=0.9, label=name if bi < 12 else None); bi += 1
            else:
                ax.scatter(P[sel, v], P[sel, 2], s=3, c="0.4", alpha=0.6, label=name)
        ax.axhline(a.floor * 1e3, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_title("step %d, section %s = %.3f m" % (s, a.plane, a.at)); ax.set_xlabel("%s (mm)" % ("x" if a.plane == "y" else "y")); ax.set_ylabel("z (mm)")
        if a.xlim: ax.set_xlim(a.xlim[0] * 1e3, a.xlim[1] * 1e3)
        if a.zlim: ax.set_ylim(a.zlim[0] * 1e3, a.zlim[1] * 1e3)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    plt.tight_layout(); plt.savefig(a.out, dpi=80); print("wrote", a.out)


if __name__ == "__main__":
    main()
