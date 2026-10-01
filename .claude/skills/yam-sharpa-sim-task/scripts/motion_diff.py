#!/usr/bin/env python3
"""Compare two IK recordings (shuffle_motion.npz): where they first differ, how much, and their phases.

  uv run python .claude/skills/yam-sharpa-sim-task/scripts/motion_diff.py new.npz base.npz [--start 145 --preroll 15] [--tol 1e-6]

The step mapping (recording frame f -> simulation step substeps * (f - (start - preroll))) is the one the
scene generator uses; pass the values of the asset's <name>_sim.json when they differ from the defaults.
A variant made with an additive option must differ first at the frame where that option starts; anything
earlier means a frame the user may have verified has changed.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--start", type=int, default=145, help="recording frame of the simulation's start phase (hold)")
    ap.add_argument("--preroll", type=int, default=15, help="synthetic pre-roll frames before it")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-6, help="rad; below this two frames count as equal")
    a = ap.parse_args()
    A = np.load(a.a, allow_pickle=True); B = np.load(a.b, allow_pickle=True)
    qa, qb = A["joint_q"], B["joint_q"]; m = min(len(qa), len(qb))
    d = np.abs(qa[:m] - qb[:m]).max(axis=1)
    diff = np.flatnonzero(d > a.tol)
    step = lambda f: a.substeps * (f - (a.start - a.preroll))
    print("frames: %d vs %d (compared %d); max |dq| %.3e rad = %.4f deg" % (len(qa), len(qb), m, d.max(), np.degrees(d.max())))
    if len(diff):
        f = int(diff[0]); print("first differing frame: %d (simulation step %d); differing frames: %d of %d" % (f, step(f), len(diff), m))
        worst = int(np.argmax(d)); j = int(np.argmax(np.abs(qa[worst] - qb[worst])))
        names = [str(x) for x in A["joint_names"]] if "joint_names" in A.files else None
        print("largest difference at frame %d, joint %s" % (worst, names[j] if names else j))
    else:
        print("identical within the tolerance")
    for name, Z in (("a", A), ("b", B)):
        ph = list(zip([str(x) for x in Z["phase_names"]], Z["phase_start"].tolist(), Z["phase_end"].tolist()))
        print("%s phases: %s" % (name, ", ".join("%s %d-%d (step %d-%d)" % (n, s, e, step(s), step(e)) for n, s, e in ph)))
    if "scene" in A.files and "scene" in B.files:
        sa, sb = json.loads(str(A["scene"])), json.loads(str(B["scene"]))
        keys = sorted(set(sa) | set(sb)); changed = [k for k in keys if sa.get(k) != sb.get(k)]
        if changed: print("scene record differs in: %s" % ", ".join("%s (%s vs %s)" % (k, str(sa.get(k))[:30], str(sb.get(k))[:30]) for k in changed[:12]))


if __name__ == "__main__":
    main()
