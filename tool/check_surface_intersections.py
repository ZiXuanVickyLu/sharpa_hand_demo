#!/usr/bin/env python3
"""Independent geometric test of a save_npy run: do any surface edges pierce surface triangles?
   python3 tool/check_surface_intersections.py output/<run> <frame> [<frame> ...] [--min-dist]
Segment-triangle intersection (Moller-Trumbore) for every surface edge against every surface
triangle that shares no vertex with it, with an AABB cull. Also reports, with --min-dist, the
smallest vertex-triangle distance between different connected components (coarse: vertices only)."""
import sys, os, numpy as np
run = sys.argv[1]; frames = [int(a) for a in sys.argv[2:] if a.lstrip("-").isdigit()]
F = np.load(os.path.join(run, "surface_faces.npy")).astype(np.int64)
E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
# connected components of the surface (one per rod)
parent = np.arange(F.max() + 1)
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for a, b in E:
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
comp = np.array([find(i) for i in range(len(parent))]); _, comp = np.unique(comp, return_inverse=True)
print("%d surface triangles, %d edges, %d components" % (len(F), len(E), comp[np.unique(F)].max() + 1))
for f in frames:
    X = np.load(os.path.join(run, "x_%d.npy" % f)).astype(np.float64)
    T0, T1, T2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    tlo = np.minimum(np.minimum(T0, T1), T2); thi = np.maximum(np.maximum(T0, T1), T2)
    hits = []
    for s in range(0, len(E), 400):
        e = E[s:s + 400]; P, Q = X[e[:, 0]], X[e[:, 1]]
        elo, ehi = np.minimum(P, Q), np.maximum(P, Q)
        ov = np.all((elo[:, None, :] <= thi[None]) & (ehi[:, None, :] >= tlo[None]), axis=2)
        ei, ti = np.nonzero(ov)
        if not len(ei): continue
        share = (F[ti][:, :, None] == e[ei][:, None, :]).any(axis=(1, 2))
        ei, ti = ei[~share], ti[~share]
        if not len(ei): continue
        p, d = P[ei], Q[ei] - P[ei]
        a, e1, e2 = T0[ti], T1[ti] - T0[ti], T2[ti] - T0[ti]
        h = np.cross(d, e2); det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-30; inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = p - a; u = np.einsum("ij,ij->i", sv, h) * inv
        q = np.cross(sv, e1); v = np.einsum("ij,ij->i", d, q) * inv; t = np.einsum("ij,ij->i", e2, q) * inv
        m = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
        for k in np.nonzero(m)[0]: hits.append((int(e[ei[k], 0]), int(e[ei[k], 1]), int(ti[k])))
    same = sum(1 for a, b, t in hits if comp[a] == comp[F[t, 0]])
    msg = "frame %d: %d edge-triangle intersections (%d within one rod, %d between rods)" % (f, len(hits), same, len(hits) - same)
    if hits:
        a, b, t = hits[0]; c = 0.5 * (X[a] + X[b]); msg += "; first at (%.3f, %.3f, %.3f)" % tuple(c)
    if "--min-dist" in sys.argv:
        V = np.unique(F); best = 1e9
        for cc in range(comp[V].max() + 1):
            vs = V[comp[V] == cc]; ts = np.nonzero(comp[F[:, 0]] != cc)[0]
            cen = (T0[ts] + T1[ts] + T2[ts]) / 3.0
            for s in range(0, len(vs), 200):
                dd = np.linalg.norm(X[vs[s:s + 200]][:, None, :] - cen[None], axis=2); best = min(best, dd.min())
        msg += "; closest vertex to another rod's triangle centroid %.2f mm" % (best * 1e3)
    print(msg)
