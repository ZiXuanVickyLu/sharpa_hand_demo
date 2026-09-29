#!/usr/bin/env python3
"""Side sections (x-z) of a hand-shuffle run: every card's centre line, left red / right blue, the hand
collider vertices near the cards' depth as grey dots.
   python3 tool/plot_hand_shuffle_section.py output/<run> 425,455,500 [--out file.png] [--zmax 60] [--label 19]"""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
run = sys.argv[1]; frames = [int(f) for f in sys.argv[2].split(",")]
def opt(k, d): return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
out = opt("--out", os.path.join(run, "diag", "section_%s.png" % "_".join(map(str, frames[:2] + frames[-1:]))))
zmax, lab, xmax = opt("--zmax", 60.0), opt("--label", 99), opt("--xmax", 140.0)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lv = np.array([[float(t) for t in l.split()[1:4]] for l in open(os.path.join(root, "asset", "card_shuffle", "card15x7.obj")) if l.startswith("v ")])
zs = np.unique(np.round(lv[:, 2], 9)); row = np.abs(lv[:, 2] - zs[len(zs) // 2]) < 1e-9; idx = np.flatnonzero(row)[np.argsort(lv[row, 0])]
N, nv = 27, 105
os.makedirs(os.path.dirname(out), exist_ok=True)
fig, axs = plt.subplots(len(frames), 1, figsize=(16, (2.0 + 16 * zmax / (2 * xmax)) * len(frames)), squeeze=False)
for ax, f in zip(axs[:, 0], frames):
    X = np.load(os.path.join(run, "x_%d.npy" % (f * 10))); C = X[:2 * N * nv].reshape(2, N, nv, 3); H = X[2 * N * nv:]
    yc = np.median(C[..., 1]); m = (np.abs(H[:, 1] - yc) < 0.04) & (H[:, 2] < zmax * 1e-3 * 1.2)
    ax.scatter(H[m, 0] * 1e3, H[m, 2] * 1e3, s=0.4, c="#999999")
    for k, col in enumerate(("tab:red", "tab:blue")):
        for i in range(N):
            c = C[k, i][idx]; ax.plot(c[:, 0] * 1e3, c[:, 2] * 1e3, color=col, lw=0.6, alpha=0.9)
            if i >= lab:
                j = np.argmax(c[:, 2]); ax.text(c[j, 0] * 1e3, c[j, 2] * 1e3, "%s%d" % ("LR"[k], i), fontsize=6, color=col)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-2, zmax); ax.set_aspect("equal"); ax.set_title("sim frame %d" % f, fontsize=9)
plt.tight_layout(); plt.savefig(out, dpi=70); print("wrote", out)
