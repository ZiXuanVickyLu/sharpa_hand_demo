#!/usr/bin/env python3
"""Per-body state of a run at given steps, from x_<step>.npy and the config (any scene, not only cards).

  python3 .claude/skills/yam-sharpa-sim-task/scripts/body_state.py output/output/<run> app/config/<cfg>.json 500 1000 1500 \
      [--floor 0.005] [--moved 0.002] [--ref 0]

For every body: centroid, z range, x/y extents (mm) and how far its vertices moved since the previous
listed step; for every collider: its lowest point and extents. Then a one-line summary: bodies resting on
the floor (top below --floor + a small margin), bodies that moved more than --moved since --ref.

The vertex layout of x_<step>.npy is the bodies in config order, then the colliders in config order; the
split comes from counting each file's vertices (OBJ: 'v ' lines; bgeo/msh bodies are reported as one block
with an unknown count, so put OBJ meshes first when mixing formats).
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def count_vertices(path):
    if path.endswith(".obj"):
        return sum(1 for line in open(path) if line.startswith("v "))
    return None


def layout(cfg):
    c = json.load(open(cfg)); ranges = []; o = 0; unknown = False
    for kind in ("bodies", "colliders"):
        for e in c.get(kind, []):
            f = e.get("file")
            n = count_vertices(os.path.join(ROOT, "asset", f)) if f else None
            if n is None:
                ranges.append((kind, e["name"], o, None)); unknown = True; break
            ranges.append((kind, e["name"], o, o + n)); o += n
        if unknown: break
    return c, ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("config"); ap.add_argument("steps", type=int, nargs="+")
    ap.add_argument("--floor", type=float, default=0.005, help="floor height (m), e.g. the mat plane")
    ap.add_argument("--moved", type=float, default=0.002, help="m; a body counts as moved beyond this")
    ap.add_argument("--ref", type=int, default=None, help="reference step for 'moved' (default: the first listed)")
    ap.add_argument("--quiet", action="store_true", help="summary lines only")
    a = ap.parse_args()
    c, ranges = layout(a.config)
    ref_step = a.ref if a.ref is not None else a.steps[0]
    Xref = np.load(os.path.join(a.run, "x_%d.npy" % ref_step))
    prev = None
    for s in a.steps:
        X = np.load(os.path.join(a.run, "x_%d.npy" % s))
        print("== step %d (%d vertices)" % (s, len(X)))
        on_floor, moved = [], []
        for kind, name, i0, i1 in ranges:
            P = X[i0:i1] * 1e3; Pr = Xref[i0:i1] * 1e3
            if len(P) == 0: continue
            disp = np.linalg.norm(P - Pr, axis=1).max() if len(Pr) == len(P) else float("nan")
            dprev = np.linalg.norm(P - prev[i0:i1] * 1e3, axis=1).max() if prev is not None and len(prev) == len(X) else float("nan")
            c0 = P.mean(0)
            if kind == "bodies":
                if P[:, 2].max() < a.floor * 1e3 + 3.0: on_floor.append(name)
                if disp > a.moved * 1e3: moved.append(name)
                if not a.quiet:
                    print("  %-12s centre (%6.1f %6.1f %5.1f)  z %5.1f..%5.1f  x %6.1f..%6.1f  y %6.1f..%6.1f  moved since %d: %5.1f mm, since prev: %5.1f mm" % (
                        name, *c0, P[:, 2].min(), P[:, 2].max(), P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), ref_step, disp, dprev))
            elif not a.quiet:
                print("  %-12s (collider) lowest z %5.1f  x %6.1f..%6.1f  y %6.1f..%6.1f  moved since prev: %5.1f mm" % (
                    name, P[:, 2].min(), P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), dprev))
        nb = sum(1 for k, *_ in ranges if k == "bodies")
        print("  summary: %d/%d bodies on the floor, %d moved > %.0f mm since step %d: %s" % (
            len(on_floor), nb, len(moved), a.moved * 1e3, ref_step, ", ".join(moved[:12]) + (" ..." if len(moved) > 12 else "")))
        prev = X


if __name__ == "__main__":
    main()
