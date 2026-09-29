#!/usr/bin/env python3
"""State of the two piles at one sim frame of a hand-shuffle run:
   python3 tool/analyze_hand_shuffle_pile.py output/<run> <frame> [<frame> ...]
Per frame: cards still held up (any vertex above 50 mm), cards lying flat (whole card below
flat_z), cards leaning (neither); the order of the cards through the centre strip (|x| < 4 mm)
from the bottom up with the number of L/R alternations; the shared length of the two piles; the
outer ends' |x| range per side; the highest point of any card."""
import sys, os, json, numpy as np
run = sys.argv[1]; frames = [int(f) for f in sys.argv[2:] if f.lstrip("-").isdigit()]
name = os.path.basename(run.rstrip("/"))
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfgp = os.path.join(root, "app", "config", name + ".json")
n_sub = json.load(open(cfgp))["simulation"]["save_every"] if os.path.exists(cfgp) else 10
mat = json.load(open(cfgp))["planes"][0]["point"][2] if os.path.exists(cfgp) else 0.002
N, nv = 27, 105
for f in frames:
    X = np.load(os.path.join(run, "x_%d.npy" % (f * n_sub))); C = X[:2 * N * nv].reshape(2, N, nv, 3)
    zmax = C[..., 2].max(2); flat_z = mat + 0.030
    up = zmax > 0.050; flat = zmax < flat_z; lean = ~up & ~flat
    items = []
    for k, side in enumerate("LR"):
        for i in range(N):
            m = np.abs(C[k, i, :, 0]) < 0.004
            if m.any() and not up[k, i]: items.append((C[k, i][m, 2].mean(), side))
    items.sort(); order = "".join(t for _, t in items)
    alt = sum(1 for a, b in zip(order, order[1:]) if a != b)
    runs = max((len(r) for r in "".join(c if c == "L" else " " for c in order).split() + "".join(c if c == "R" else " " for c in order).split()), default=0)
    inL = np.array([C[0, i, :, 0].max() for i in range(N) if flat[0, i]]); inR = np.array([C[1, i, :, 0].min() for i in range(N) if flat[1, i]])
    outL = np.array([-C[0, i, :, 0].min() for i in range(N) if flat[0, i]]); outR = np.array([C[1, i, :, 0].max() for i in range(N) if flat[1, i]])
    print("frame %d: held up L %d R %d | flat L %d R %d | leaning L %s R %s" % (f, up[0].sum(), up[1].sum(), flat[0].sum(), flat[1].sum(),
          [int(v) for v in np.flatnonzero(lean[0])], [int(v) for v in np.flatnonzero(lean[1])]))
    print("   centre strip, bottom up: %s  (%d cards, %d alternations, longest run %d)" % (order, len(order), alt, runs))
    cx = C[..., 0].mean(2) * 1e3   # card centres along x
    print("   deck: centres L %.1f +- %.1f mm, R %.1f +- %.1f mm (offset %.1f), extent in x %.1f mm (card 111.1), in y %.1f mm (card 70.0), top %.1f mm above the mat" % (
        cx[0].mean(), cx[0].std(), cx[1].mean(), cx[1].std(), cx[1].mean() - cx[0].mean(), np.ptp(C[..., 0]) * 1e3, np.ptp(C[..., 1]) * 1e3, (C[..., 2].max() - mat) * 1e3))
    if len(inL) and len(inR):
        print("   shared length (median inner ends): %.1f mm | outer ends |x|: L %.1f..%.1f  R %.1f..%.1f mm | highest card point %.1f mm above the mat" % (
            (np.median(inL) - np.median(inR)) * 1e3, outL.min() * 1e3, outL.max() * 1e3, outR.min() * 1e3, outR.max() * 1e3, (C[..., 2].max() - mat) * 1e3))
