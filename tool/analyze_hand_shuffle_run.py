#!/usr/bin/env python3
"""Card state per sim frame of a hand-shuffle run (needs simulation.save_npy):
   python3 tool/analyze_hand_shuffle_run.py output/hand_shuffle [--every 10]"""
import json, os, re, sys, numpy as np
run = sys.argv[1]; every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 10
name = os.path.basename(run.rstrip("/"))
cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "config", name + ".json")))
n_sub = cfg["simulation"]["save_every"]; N = len(cfg["bodies"]) // 2; nv = 105
lv = np.array([[float(t) for t in l.split()[1:4]] for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset", "card_shuffle", "card15x7.obj")) if l.startswith("v ")])
xs = np.unique(np.round(lv[:, 0], 9)); g_top = np.abs(lv[:, 0] - xs[0]) < 1e-9; g_bot = np.abs(lv[:, 0] - xs[-1]) < 1e-9; g_mid = np.abs(lv[:, 0] - xs[len(xs) // 2]) < 1e-9
steps = sorted(int(m.group(1)) for m in (re.match(r"x_(\d+)\.npy$", f) for f in os.listdir(run)) if m)
print("frame  side  in-hand on-table  lowest-z(mm) highest-z(mm)  mid-card: tilt(deg) sagitta(mm) yaw(deg)   flat cards x-range (mm)")
for s in steps:
    j = s // n_sub
    if s % n_sub or j % every: continue
    X = np.load(os.path.join(run, "x_%d.npy" % s))[:2 * N * nv].reshape(2, N, nv, 3)
    for k, side in enumerate(("L", "R")):
        C = X[k]; zmax = C[:, :, 2].max(1); zmin = C[:, :, 2].min(1)
        flat = zmax < 0.012; held = zmax > 0.05
        mid = C[N // 2]
        top, bot, mdl = mid[g_top].mean(0), mid[g_bot].mean(0), mid[g_mid].mean(0)
        chord = bot - top; tilt = np.degrees(np.arctan2(-chord[2], abs(chord[0])))
        sag = np.linalg.norm(np.cross(mdl - top, chord)) / np.linalg.norm(chord)
        yaw = np.degrees(np.arctan2(chord[1], abs(chord[0])))
        xr = "[%6.1f, %6.1f]" % (C[flat][:, :, 0].min() * 1e3, C[flat][:, :, 0].max() * 1e3) if flat.any() else "-"
        print("%5d   %s    %4d    %4d      %8.1f     %8.1f        %6.1f     %6.1f    %6.1f     %s" % (j, side, held.sum(), flat.sum(), zmin.min() * 1e3, zmax.max() * 1e3, tilt, sag * 1e3, yaw, xr))
