#!/usr/bin/env python3
"""Where are the last cards of each packet? (moving-boundary hand shuffle)
   python3 tool/analyze_hand_shuffle_late.py output/<run> [first_late_card=21] [--every 2]
Per saved frame: for the late cards of each side the outer end's |x| (mm), the inner end's height
(mm) and the overlap of the two sides' late cards at the centre (mm); plus the lowest hand vertex."""
import os, re, sys, numpy as np
run = sys.argv[1]; k0 = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 21
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 2
first = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
N, nv = 27, 105
steps = sorted(int(m.group(1)) for m in (re.match(r"x_(\d+)\.npy$", f) for f in os.listdir(run)) if m)
print("frame | L late: outer|x| min..max, inner z max | R late: outer|x| min..max, inner z max | late overlap | all cards: max |x|, max z | hand min z")
for s in steps:
    f = s // 10
    if s % 10 or f < first or (f - first) % every: continue
    X = np.load(os.path.join(run, "x_%d.npy" % s)); C = X[:2 * N * nv].reshape(2, N, nv, 3); H = X[2 * N * nv:]
    L, R = C[0, k0:], C[1, k0:]
    lo, ro = -L[:, :, 0].min(1) * 1e3, R[:, :, 0].max(1) * 1e3
    li = np.array([c[np.argmax(c[:, 0]), [0, 2]] for c in L]) * 1e3; ri = np.array([c[np.argmin(c[:, 0]), [0, 2]] for c in R]) * 1e3
    ov = li[:, 0].min() - ri[:, 0].max()
    print("%5d | %6.1f..%6.1f  %5.1f | %6.1f..%6.1f  %5.1f | %6.1f | %6.1f %6.1f | %5.2f" % (
        f, lo.min(), lo.max(), li[:, 1].max(), ro.min(), ro.max(), ri[:, 1].max(), ov,
        np.abs(C[..., 0]).max() * 1e3, C[..., 2].max() * 1e3, H[:, 2].min() * 1e3))
