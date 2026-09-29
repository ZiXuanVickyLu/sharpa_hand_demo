#!/usr/bin/env python3
"""Sharpa-hand riffle shuffle: the C-IPC card packets driven by a baked robot-hand motion
(tool/bake_hand_shuffle.py -> asset/<asset>/) instead of Dirichlet rows and plates.

  python3 tool/gen_hand_shuffle.py --asset hand_shuffle_t60 --name hand_shuffle_t60 [...]

Frame of reference: the rig's (Z up, table z = 0), metres. The cards are the C-IPC card mesh
scaled to the recorded card size, 27 per packet at the recorded pitch, card 0 (the lowest, the
first off the thumb) on the recorded edges T0/G0 and the others stacked along `stack_dir`. The
simulation starts at the recording's `hold` phase and runs to the end of `withdraw`.

What is physical and what is not:
  * finger pads, table/mat, rake, push, squeeze: frictional contact with the keyframed hand;
  * the thumb's grip is adhesion: each card's upper-inner vertex rows ride the recorded thumb-tip
    target (down with it in the bow, along the card only in the riffle) and are released at the
    first frame where the thumb collider's surface has actually left that card's upper edge,
    so a card is held exactly as long as the thumb is on it;
  * a short synthetic pre-roll slides the non-thumb links in horizontally from --preroll-dist
    away, so that the first state is penetration-free whatever the decimation did to the pads;
  * the packet is shifted along the card just enough for the thumb to clear the upper edges by
    --thumb-clear; the playing mat is put just under the lowest point the hand ever reaches
    in the rake, so the fingertips cannot ride over cards lying flat."""
import argparse, copy, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def minjerk(s):
    s = np.clip(s, 0.0, 1.0); return s * s * s * (10 - 15 * s + 6 * s * s)

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def load_obj(p):
    V, F = [], []
    for l in open(p):
        if l.startswith("v "): V.append([float(t) for t in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    return np.array(V), F

def pt_tri_dist(P, T, chunk=256):
    """distance from each point P [n,3] to the nearest of the triangles T [m,3,3] (Ericson 5.1.5)."""
    if len(T) == 0: return np.full(len(P), np.inf)
    out = np.empty(len(P))
    a, b, c = T[:, 0][None], T[:, 1][None], T[:, 2][None]
    ab, ac = b - a, c - a
    for i0 in range(0, len(P), chunk):
        p = P[i0:i0 + chunk, None]
        ap = p - a
        d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
        bp = p - b; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
        cp = p - c; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
        den = va + vb + vc; den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        q = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]
        def seg(o, d, t): return o + d * np.clip(t, 0, 1)[..., None]
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], seg(a, ab, d1 / np.where(d1 - d3 == 0, 1, d1 - d3)), q)
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], seg(a, ac, d2 / np.where(d2 - d6 == 0, 1, d2 - d6)), q)
        e = (d4 - d3) + (d5 - d6)
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], seg(b, c - b, (d4 - d3) / np.where(e == 0, 1, e)), q)
        q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
        q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
        out[i0:i0 + chunk] = np.linalg.norm(p - q, axis=-1).min(1)
    return out

def ray_tri(O, d, T):
    """Moller-Trumbore, rays O [m,3] + t d against triangles T [k,3,3]: t [m,k], inf where missed."""
    a = T[:, 0]; e1 = T[:, 1] - a; e2 = T[:, 2] - a
    pvec = np.cross(d[None], e2); det = (e1 * pvec).sum(-1)
    ok = np.abs(det) > 1e-16; inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = O[:, None, :] - a[None]
    uu = (tvec * pvec[None]).sum(-1) * inv[None]
    qvec = np.cross(tvec, e1[None])
    vv = (qvec * d[None, None]).sum(-1) * inv[None]
    tt = (qvec * e2[None]).sum(-1) * inv[None]
    hit = ok[None] & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 0)
    return np.where(hit, tt, np.inf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="hand_shuffle", help="directory under asset/ holding the bake (hand_*_kf.npy, hand_meta.json)")
    ap.add_argument("--name", default="hand_shuffle")
    ap.add_argument("--cards", type=int, default=27)
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--preroll", type=int, default=15, help="synthetic frames before the hold in which the non-thumb links slide in")
    ap.add_argument("--preroll-dist", type=float, default=4.0e-3, help="their horizontal displacement at the start of the pre-roll (m); grown until the first state is clear")
    ap.add_argument("--thumb-clear", type=float, default=0.3e-3, help="each held card's upper edge rides this far under the thumb tip's surface, along the card (m)")
    ap.add_argument("--thumb-perp", type=float, default=0.15e-3, help="and never closer than this to it in any direction (m)")
    ap.add_argument("--ride-stride", type=int, default=1, help="evaluate the thumb ride every this many frames and interpolate (slow riffles)")
    ap.add_argument("--thumb-reach", type=float, default=4.0e-3, help="a held card follows the thumb surface up to this far along the card; beyond it the thumb has left it (m)")
    ap.add_argument("--hold-rows", type=int, default=2, help="vertex rows at the upper-inner edge that ride the thumb (1 = a hinge)")
    ap.add_argument("--mat", default="auto", help="playing surface height (m), or 'auto': just under the lowest point of the hand in the rake")
    ap.add_argument("--exclude-links", default="thumb_", help="comma-separated substrings of hand links left out of the collider. The thumb is out by "
                    "default: its grip is modelled by the rows riding its surface, and a prescribed row against a keyframed collider "
                    "has no free DOF to resolve a contact (a run with the thumb in stalled at alpha = 0 for 500 s on one step)")
    ap.add_argument("--thumb-surface", choices=("tip", "collider"), default="tip",
                    help="surface the held rows ride: the rendered full-resolution thumb tip (thumb_tip_*.npz), or the thumb faces of the "
                         "decimated hand mesh itself. Use 'collider' with --exclude-links '' (thumb kept as a collider) so the prescribed "
                         "rows stay clear of the surface the solver sees")
    ap.add_argument("--release-all-at", default=None,
                    help="release every card's held rows at this recorded frame ('bow_end' = last frame of the bow) instead of one by one "
                         "as the thumb tip leaves them; with the thumb in the collider, the riffle is then physical")
    ap.add_argument("--thumb-collider", action="store_true",
                    help="the thumb links form their own keyframed collider thumb_<side> (the hand collider then never holds them), "
                         "with --thumb-thickness and --mu-thumb: a rigid stand-in for the soft pad. Combine with --thumb-surface collider, "
                         "--thumb-clear/--thumb-perp >= thickness + the clearances, and --release-all-at <hold start> for a physical grip")
    ap.add_argument("--thumb-thickness", type=float, default=0.0, help="contact offset of the thumb collider (m): the cards see the thumb's "
                    "surface pushed out by this much, a bigger and blunter tip whose facets meet the card edges at smaller angles")
    ap.add_argument("--mu-thumb", type=float, default=None, help="card-thumb friction (default --mu-hand)")
    ap.add_argument("--thumb-cap", type=float, default=0.0, help="radius (m) of the thumb tip's cap, made a collider of its own with "
                    "--mu-cap (0 = the whole thumb is one collider)")
    ap.add_argument("--mu-cap", type=float, default=0.5, help="card friction of the thumb cap collider (--thumb-cap)")
    ap.add_argument("--pinch", type=float, default=0.0, help="extra inward displacement of the non-thumb links (m) reached at the end of the "
                    "pre-roll and kept through the hold, fading to zero over the bow: presses the packet onto the thumb before the bow")
    ap.add_argument("--physical-from", type=int, default=None,
                    help="two-stage run for a physical grip (needs --thumb-collider). Writes <name>_setup.json: the held rows ride the thumb "
                         "(put them inside its contact offset with --thumb-clear/--thumb-perp below --thumb-thickness) with card-thumb contact "
                         "excluded, checkpoints every --setup-checkpoint-every steps, stops at this step; and <name>.json: no prescribed card "
                         "vertices at all, restarted from the setup's checkpoint at this step with the thumb contact on, so the packet is "
                         "pinched from its first step and everything after is contact")
    ap.add_argument("--setup-checkpoint-every", type=int, default=50, help="steps between the setup run's checkpoints")
    ap.add_argument("--hinge-frames", type=int, default=0,
                    help="staged hand-over: the second held row is released this many frames before the edge row, so each card relaxes "
                         "from clamped to pinned about its still-prescribed edge, and the edge is then freed without a lateral snap")
    ap.add_argument("--mu-hand", type=float, default=0.8); ap.add_argument("--mu-card", type=float, default=0.02)
    ap.add_argument("--mu-mat", type=float, default=0.5, help="friction of the playing mat. Contact-solver feedback: with the thumb as a "
                    "collider the inner cards' feet rest on the mat, and the thumb's push along a 45-degree card slides them inward "
                    "unless the mat's friction is at least 1")
    ap.add_argument("--xi", type=float, default=5e-5); ap.add_argument("--d-hat", type=float, default=2e-4)
    ap.add_argument("--eps-v", type=float, default=1e-4); ap.add_argument("--density", type=float, default=800.0)
    ap.add_argument("--mem-scale", type=float, default=1.0, help="scale of the membrane modulus only (bending stays physical). The loads on a card are "
                    "grams: at 1/30 the in-plane strain under the 0.4 N buckling load is 5e-4 (0.05 mm over the card), and the stiff-shell "
                    "systems that cost 400-1700 CG iterations per Newton step become much better conditioned")
    ap.add_argument("--bend-scale", type=float, default=1.0, help="scale of the bending stiffness only (1 = the physical E t^3 / 12 (1 - nu^2))")
    ap.add_argument("--no-rigid-coarse", action="store_true", help="plain block-Jacobi PCG (IS 6.6): 4x more CG iterations in the push, and the cards' sliding modes are resolved worse")
    ap.add_argument("--mu-max", type=float, default=0.03, help="upper bound of the AL penalty estimate (MS 11.1). The estimate (1.35 kg for these cards) leaves a "
                    "resting contact a gap window lambda/mu of 0.06 um per card weight, below the solver's accuracy: every contact flickers, the "
                    "cards jitter and bounce and the pushed deck arches. 0.03 (window 2.5 um) makes the deck static; below 0.01 the bottom of a "
                    "pile needs more violation than the CCD allows and the bouncing returns. 0 = no bound")
    ap.add_argument("--prebend", action="store_true", help="start each card bent (sine bow) between its stacked foot and its held edge at "
                    "the ride position, one mesh per card, instead of sliding the straight card down its axis (needed with a stepped pad)")
    ap.add_argument("--foot-bar", type=float, default=0.0, help="mm inside card 0's foot: a 3 mm plate on the mat that stops the inner feet "
                    "sliding toward the centre, present until the riffle is done (0 = none)")
    ap.add_argument("--guide-plate", type=float, default=0.0, help="mm outside the top card's outer face: a static plate (70 mm wide, keyframed "
                    "collider) the held packet leans on during the bow and the riffle, so a swept thumb pad slides under the card ends "
                    "instead of carrying them along (0 = none)")
    ap.add_argument("--guide-span", type=float, nargs=2, default=[3.0, 23.0], help="mm below the cards' thumb edge the guide plate spans")
    ap.add_argument("--guide-mu", type=float, default=0.3, help="friction of the guide plates")
    ap.add_argument("--guide-out", type=int, default=-1, help="recording frame at which the guide plates retract (default: the end of riffle_done)")
    ap.add_argument("--walls", action="store_true", help="two frictionless squaring walls (thin plates, 132 x 60 mm, keyframed colliders) parked at "
                    "--walls-park either side of the deck's depth centre from frame 0, sliding in to --walls-in between the sim frames --walls-close "
                    "(after every card is down, before the push) and staying there: the third pair of contacts a human supplies with the other fingers")
    ap.add_argument("--walls-park", type=float, default=0.150); ap.add_argument("--walls-in", type=float, default=0.038)
    ap.add_argument("--walls-centre", type=float, default=None, help="depth (y, m) the walls close around; default: the packets' depth centre from the motion")
    ap.add_argument("--walls-close", type=int, nargs=2, default=[440, 529], help="sim frames over which the walls close")
    ap.add_argument("--abort-capped", type=int, default=5, help="simulation.abort_after_capped_steps: stop after this many consecutive capped steps (0 = never)")
    ap.add_argument("--save-bgeo", action="store_true", help="also write the surfaces as Houdini .bgeo every save_every steps")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="float64 restart checkpoints every this many FRAMES (x substeps steps); 0 = none")
    ap.add_argument("--k-min", type=int, default=2, help="minimum outer iterations per step (the paper uses 2 without friction, 6 with)")
    ap.add_argument("--left-lag", type=int, default=0, help="extra frames the LEFT hand's releases wait (the right hand already leads by one frame in the recording)")
    ap.add_argument("--newton-rule", choices=("paper", "cipc"), default="paper")
    ap.add_argument("--inner-max", type=int, default=0); ap.add_argument("--newton-tol", type=float, default=5e-4)
    a = ap.parse_args()

    A = os.path.join(ROOT, "asset", a.asset)
    meta = json.load(open(os.path.join(A, "hand_meta.json"))); sc = meta["scene_cm"]; first = meta["first"]
    assert "thumb_contact" in sc, "this generator reads the motion format with card0 edges, stack_dir and world-frame targets"
    motion_path = meta["motion"] if os.path.isabs(meta["motion"]) else os.path.join(A, meta["motion"])   # relative to the asset directory
    zm = np.load(motion_path, allow_pickle=True)
    ph = {str(k): (int(s0), int(s1)) for k, s0, s1 in zip(zm["phase_names"], zm["phase_start"], zm["phase_end"])}
    start, end = ph["hold"][0], ph["withdraw"][1]
    bow0, bow_end, r0, rd1 = ph["bow"][0], ph["bow"][1], ph["riffle"][0], ph["riffle_done"][1]
    if a.mu_thumb is None: a.mu_thumb = a.mu_hand
    rake0, rake1 = ph.get("lower", ph.get("regrip", ph["push"]))[0], ph["back_off"][1]   # direct-push motions have no 'lower' 
    n = a.substeps; N = a.cards; P = a.preroll
    card_v, card_f = load_obj(os.path.join(ROOT, "asset", "card_shuffle", "card15x7.obj"))
    xs = np.unique(np.round(card_v[:, 0], 9))
    rows = [int(i) for i in np.where(card_v[:, 0] < xs[a.hold_rows - 1] + 1e-9)[0]]        # the upper-inner edge and the rows next to it
    top_row = [int(i) for i in np.where(np.abs(card_v[:, 0] - xs[0]) < 1e-9)[0]]
    Lref = card_v[:, 0].max() - card_v[:, 0].min()
    pitch = sc["packet_thickness_cm"] * 0.01 / sc.get("cards_per_packet", 27)

    bodies, colliders, table, report, sidecar = [], [], [], {}, {}
    mat_candidates = []
    sides = (("left", np.array([1.0, 1, 1]), "target_left"), ("right", np.array([-1.0, 1, 1]), "target_right"))
    hand = {}
    for side, M, tkey in sides:                                       # ---- hand collider first (the mat needs both)
        kf = np.load(os.path.join(A, "hand_%s_kf.npy" % side)).astype(np.float64)
        Vh, Fh = load_obj(os.path.join(A, "hand_%s.obj" % side))
        links = meta["links"][side]
        keep = np.ones(len(Vh), bool); thumb = np.zeros(len(Vh), bool); tip = np.zeros(len(Vh), bool)
        for k, (i0, i1) in links.items():
            if any(t and t in k for t in a.exclude_links.split(",")): keep[i0:i1] = False
            if "thumb_" in k: thumb[i0:i1] = True
            if "thumb_DP" in k or "thumb_elastomer" in k: tip[i0:i1] = True
        if a.thumb_collider: keep &= ~thumb
        rec = kf[start - first: end - first + 1]
        hand[side] = dict(rec=rec, keep=keep, thumb=thumb, tip=tip, F=np.array(Fh))
        mat_candidates.append(rec[rake0 - start: rake1 - start + 1][:, keep, 2].min())
    mat = max(0.0, float(min(mat_candidates)) - 1.0e-4) if a.mat == "auto" else float(a.mat)

    for side, M, tkey in sides:
        H = hand[side]; rec, keep, thumb, tipm, Fh = H["rec"], H["keep"], H["thumb"], H["tip"], H["F"]
        T0, G0 = np.array(sc["T0"]) * 0.01 * M, np.array(sc["G0"]) * 0.01 * M
        nrm = np.array(sc["stack_dir"]) * M; u = unit(G0 - T0); e = np.array([0.0, 1.0, 0.0])
        L = np.linalg.norm(G0 - T0); S = L / Lref; hl, hd = 0.5 * L, 0.5 * S * (card_v[:, 2].max() - card_v[:, 2].min())
        inward = np.array([1.0, 0, 0]) * M
        tgt = np.asarray(zm[tkey], float)[:, :3] * 0.01
        assert np.isfinite(tgt[start:rd1 + 1]).all(), "thumb target missing between hold and riffle_done"
        offs = [i * pitch for i in range(N)]
        if a.thumb_surface == "collider":                                    # the decimated thumb faces, as the solver sees them
            th_faces = Fh[thumb[Fh].all(1)]
            tip_tris = lambda f: rec[f - start][th_faces]
        else:
            tipz = np.load(os.path.join(A, "thumb_tip_%s.npz" % side))          # the rendered thumb tip, full resolution
            tipV, tipF, tip0 = tipz["V"].astype(np.float64), tipz["F"], int(tipz["first"])
            tip_tris = lambda f: tipV[f - tip0][tipF]
        # ---- thumb contact per card and frame: a ray from each upper edge along the card toward the thumb
        # finds the tip's surface; the held rows ride `thumb_clear` under it. D(f) is the thumb target's
        # displacement (all of it up to the end of the bow, then frozen: in the riffle the tip slides
        # across the edges), delta_i(f) the along-card correction that keeps edge i on the surface.
        base = 0.5 * (T0 + G0) + S * card_v[:, 0:1] * u + S * card_v[:, 2:3] * e
        y_s = np.arange(-14e-3, 14.01e-3, 1e-3)
        y_t = tgt[start][1] - T0[1]                                     # the thumb's depth relative to the card centre
        edge = np.stack([T0 + o_ * nrm + (y_t + y_s)[:, None] * e for o_ in offs])          # [N, ns, 3] upper-edge samples
        ckey = "%s_%d_%g_%g_%g_%d_%d" % (side, N, a.thumb_clear, a.thumb_perp, a.thumb_reach, a.left_lag, a.ride_stride) + \
               ("" if a.thumb_surface == "tip" else "_" + a.thumb_surface)
        cpath = os.path.join(A, "thumb_ride_%s.npz" % ckey)
        if os.path.exists(cpath):
            cz = np.load(cpath); delta, rel, gap, held = cz["delta"], [int(x) for x in cz["rel"]], cz["gap"], cz["held"]
            frames_f = list(range(start, rd1 + 1))
        else:
            reach = 8e-3
            frames_f = list(range(start, rd1 + 1))
            gap = np.full((len(frames_f), N), np.inf)
            stride = max(1, a.ride_stride)                              # a slow riffle has hundreds of frames: sample, then interpolate
            coarse = sorted(set(list(range(0, len(frames_f), stride)) + [len(frames_f) - 1, r0 - start, bow_end - start]))
            for k in coarse:
                f = frames_f[k]
                D = tgt[min(f, bow_end)] - tgt[start]
                O = (edge + D + reach * u).reshape(-1, 3)
                tris = tip_tris(f)
                lo_b, hi_b = O.min(0) - 2.5e-2, O.max(0) + 2.5e-2
                tris = tris[((tris.max(1) >= lo_b) & (tris.min(1) <= hi_b)).all(1)]
                t = np.concatenate([ray_tri(O[c0:c0 + 128], -u, tris).min(1) for c0 in range(0, len(O), 128)]).reshape(N, len(y_s)).min(1)
                gap[k] = t - reach                                          # > 0: the surface is that far above the edge
            for k0, k1 in zip(coarse[:-1], coarse[1:]):                 # frames between samples: linear in the gap, lost if either end is
                for k in range(k0 + 1, k1):
                    w = (k - k0) / float(k1 - k0)
                    both = np.isfinite(gap[k0]) & np.isfinite(gap[k1])
                    gap[k] = np.where(both, (1 - w) * np.where(both, gap[k0], 0.0) + w * np.where(both, gap[k1], 0.0), np.inf)
            held = np.isfinite(gap) & (gap <= a.thumb_reach)
            rel = []
            for i in range(N):
                k_r = r0 - start
                lost = np.flatnonzero(~held[k_r:, i])
                rel.append(r0 + (int(lost[0]) if len(lost) else rd1 - r0))
            if side == "left" and a.left_lag: rel = [r + a.left_lag for r in rel]
            order = np.argsort(rel, kind="stable")                          # cards the tip never covers in the riffle: one per frame
            for rank, i in enumerate(order):
                if rel[i] == r0: rel[i] = r0 + min(rank, 2 * i)
            delta = np.zeros((len(frames_f), N))
            for k in range(len(frames_f)):
                ok = held[k]
                if not ok.any(): delta[k] = delta[k - 1] if k else 0.0; continue
                d_ok = a.thumb_clear - gap[k]
                idx = np.arange(N)
                delta[k] = np.interp(idx, idx[ok], d_ok[ok])                # cards beside the tip's footprint follow their neighbours
            # the ray measures along the card; on the tip's flanks the true distance is smaller, and the solver
            # activates contact below d_hat + xi: back each edge off until it is a.thumb_perp from the surface
            for k in coarse:
                f = frames_f[k]
                D = tgt[min(f, bow_end)] - tgt[start]; tris = tip_tris(f)
                lo_b, hi_b = (edge + D).reshape(-1, 3).min(0) - 1.2e-2, (edge + D).reshape(-1, 3).max(0) + 1.2e-2
                tris = tris[((tris.max(1) >= lo_b) & (tris.min(1) <= hi_b)).all(1)]          # only the tip's faces over the packet's top
                todo = np.flatnonzero(held[k])
                for _ in range(12):
                    if not len(todo) or not len(tris): break
                    pts = (edge[todo] + D + delta[k][todo][:, None, None] * u).reshape(-1, 3)
                    dperp = pt_tri_dist(pts, tris, chunk=2048).reshape(len(todo), len(y_s)).min(1)
                    todo = todo[dperp < a.thumb_perp]
                    delta[k][todo] += 0.2e-3
            for k0, k1 in zip(coarse[:-1], coarse[1:]):                 # the backed-off delta, interpolated the same way
                for k in range(k0 + 1, k1):
                    w = (k - k0) / float(k1 - k0); delta[k] = (1 - w) * delta[k0] + w * delta[k1]
            np.savez_compressed(cpath, delta=delta, rel=np.array(rel), gap=gap, held=held)
        if a.release_all_at is not None:                                # one release frame for the packet: the thumb collider does the riffle
            rel = [bow_end if a.release_all_at == "bow_end" else int(a.release_all_at)] * N
        thumb_backoff = 0.0
        if a.thumb_collider and a.physical_from is None:               # the packet backs off along the card, as a whole, until every card
            Ft0 = Fh[thumb[Fh].all(1)]; T_th0 = rec[0][Ft0]; V_th0 = rec[0][thumb]   # vertex / thumb vertex clears the offset in the first state
            cf0 = np.array([[f[0], f[1], f[2]] for f in card_f] + [[f[0], f[2], f[3]] for f in card_f if len(f) == 4])
            need = a.thumb_thickness + a.xi + 0.05e-3                       # the pair's distance floor, plus a margin (contact may be active)
            while True:
                P_all = np.concatenate([base + o_ * nrm + (delta[0][i] + thumb_backoff) * u for i, o_ in enumerate(offs)])
                lo_, hi_ = P_all.min(0) - 6e-3, P_all.max(0) + 6e-3
                nt = ((T_th0.max(1) >= lo_) & (T_th0.min(1) <= hi_)).all(1)
                nv_ = ((V_th0 >= lo_) & (V_th0 <= hi_)).all(1)
                d = pt_tri_dist(P_all, T_th0[nt]).min() if nt.any() else np.inf
                if nv_.any():
                    tris_c = np.concatenate([(base + o_ * nrm + (delta[0][i] + thumb_backoff) * u)[cf0] for i, o_ in enumerate(offs)])
                    d = min(d, pt_tri_dist(V_th0[nv_], tris_c).min())
                if d >= need or thumb_backoff > 12e-3: break
                thumb_backoff += 0.2e-3
            delta = delta + thumb_backoff
        shift = delta[0]                                                # per-card initial shift along the card
        Vc = base
        with open(os.path.join(A, "%s_card_%s.obj" % (a.name, side)), "w") as o:
            for v in Vc: o.write("v %.9f %.9f %.9f\n" % tuple(v))
            for f in card_f: o.write("f " + " ".join(str(i + 1) for i in f) + "\n")
        bot_row = [int(i) for i in np.where(np.abs(card_v[:, 0] - xs[-1]) < 1e-9)[0]]
        Vb, card_files = [], []
        if a.prebend:
            # --prebend: each card starts bent (a sine bow) with its foot where the straight stack puts it and its held
            # edge where the ride wants it, instead of being slid down the card by that amount. With a stepped thumb pad
            # the outer treads sit millimetres deeper, and sliding the whole card that far puts its foot under the mat.
            Lc = float(np.linalg.norm(base[top_row].mean(0) - base[bot_row].mean(0)))
            cmp_ = []
            for i, o_ in enumerate(offs):
                Vs = base + o_ * nrm; foot = Vs[bot_row].mean(0); top = Vs[top_row].mean(0) + shift[i] * u
                us = Vs[top_row].mean(0) - foot; us /= np.linalg.norm(us)
                s_par = np.clip(((Vs - foot) @ us) / Lc, 0.0, 1.0); w = Vs - (foot + np.outer(s_par * Lc, us))
                dch = top - foot; d = float(np.linalg.norm(dch)); uh = dch / d
                c = max(0.0, Lc - d); amp = np.sqrt(4.0 * d * c / np.pi ** 2)
                nh = nrm - (nrm @ uh) * uh; nh /= np.linalg.norm(nh)
                V = foot + np.outer(s_par * d, uh) + np.outer(amp * np.sin(np.pi * s_par), nh) + w
                Vb.append(V); cmp_.append(c)
                fn = "%s_card_%s_%02d.obj" % (a.name, side, i)
                with open(os.path.join(A, fn), "w") as o:
                    for v in V: o.write("v %.9f %.9f %.9f\n" % tuple(v))
                    for f in card_f: o.write("f " + " ".join(str(k + 1) for k in f) + "\n")
                card_files.append("%s/%s" % (a.asset, fn))
            print("%s pre-bent packet: initial compression %.2f (card 0) .. %.2f mm (card %d), max %.2f" % (side, cmp_[0] * 1e3, cmp_[-1] * 1e3, N - 1, max(cmp_) * 1e3))
        def ride(f, i):
            f = min(max(f, start), rd1); k = f - start
            return tgt[min(f, bow_end)] - tgt[start] + delta[k, i] * u
        tops0 = np.concatenate([(base + o_ * nrm + shift[i] * u)[top_row] for i, o_ in enumerate(offs)])
        n_kf = P + (rd1 - start) + 2
        regions = [(rows, "", 0)] if a.hinge_frames <= 0 else [(top_row, "a", 0), ([r for r in rows if r not in top_row], "b", a.hinge_frames)]
        for i, o_ in enumerate(offs):
            dl = []
            for sel, tag, early in regions:
                if a.prebend:      # the bent rows already sit at the frame-0 ride position: add only the ride's changes
                    rest = Vb[i][sel]
                    kfr = np.stack([rest + ride(start + j - P, i) - shift[i] * u for j in range(n_kf)]).astype(np.float32)
                else:
                    rest = (Vc + o_ * nrm)[sel]
                    kfr = np.stack([rest + ride(start + j - P, i) for j in range(n_kf)]).astype(np.float32)
                kname = "%s/%s_rows_%s%02d%s.npy" % (a.asset, a.name, side[0].upper(), i, tag)
                np.save(os.path.join(ROOT, "asset", kname), kfr)
                dl.append({"select": sel, "motion": {"type": "keyframes", "file": kname, "substeps": n},
                           "release_frame": max(0, (P + rel[i] - start - early) * n)})
            bodies.append({"name": "card_%s%02d" % (side[0].upper(), i), "type": "cloth",
                           "file": card_files[i] if a.prebend else "%s/%s_card_%s.obj" % (a.asset, a.name, side),
                           "material": "card", "move_to_origin": False, "scale": [1, 1, 1], "rotation": [0, 0, 0],
                           "translation": [0.0, 0.0, 0.0] if a.prebend else list(o_ * nrm + shift[i] * u), "initial_velocity": [0, 0, 0],
                           "thickness": a.xi, "friction": a.mu_card, "self_collision": False, "dirichlet": dl})
        # ---- hand collider with the pre-roll: non-thumb links slide in horizontally, from far enough to be clear
        cards0 = np.concatenate(Vb) if a.prebend else np.concatenate([Vc + o_ * nrm + shift[i] * u for i, o_ in enumerate(offs)])
        remap = -np.ones(len(keep), int); remap[keep] = np.arange(keep.sum())
        Fk = remap[Fh[keep[Fh].all(1)]]
        cf = np.array([[f[0], f[1], f[2]] for f in card_f] + [[f[0], f[2], f[3]] for f in card_f if len(f) == 4])
        card_tris = np.concatenate([V[cf] for V in Vb]) if a.prebend else np.concatenate([(Vc + o_ * nrm + shift[i] * u)[cf] for i, o_ in enumerate(offs)])
        dist0 = a.preroll_dist
        while True:
            first = rec[0].copy(); first[~thumb] -= dist0 * inward
            K0 = first[keep]; T_all = K0[Fk]
            lo, hi = cards0.min(0) - 6e-3, cards0.max(0) + 6e-3
            th_k = thumb[keep]; th_t = th_k[Fk].all(1)
            near = ((T_all.max(1) >= lo) & (T_all.min(1) <= hi)).all(1)
            nv = ((K0 >= lo) & (K0 <= hi)).all(1)
            def clear(sel_t, sel_v):
                d1 = pt_tri_dist(cards0, T_all[near & sel_t]).min() if (near & sel_t).any() else np.inf
                d2 = pt_tri_dist(K0[nv & sel_v], card_tris).min() if (nv & sel_v).any() else np.inf
                return min(d1, d2)
            d_fing = clear(~th_t, ~th_k)
            if d_fing >= 0.4e-3 or dist0 > 20e-3: break
            dist0 += 1e-3
        Ft = Fh[thumb[Fh].all(1)]                                        # the thumb's own faces, full-mesh indexing
        T_th = first[Ft]; near_t = ((T_th.max(1) >= lo) & (T_th.min(1) <= hi)).all(1)
        V_th = first[thumb]; nv_t = ((V_th >= lo) & (V_th <= hi)).all(1)
        d_thumb = min(pt_tri_dist(cards0, T_th[near_t]).min() if near_t.any() else np.inf,
                      pt_tri_dist(V_th[nv_t], card_tris).min() if nv_t.any() else np.inf)
        if a.thumb_collider and a.physical_from is None:
            need = a.thumb_thickness + a.xi
            if d_thumb < need: print("WARNING %s: the cards start %.3f mm from the thumb, inside the pair's %.3f mm distance floor" % (side, d_thumb * 1e3, need * 1e3))
        pre = np.repeat(rec[:1], P, axis=0)
        for j in range(P):
            s_ = minjerk(j / float(P))
            pre[j, ~thumb] += (-dist0 * (1.0 - s_) + a.pinch * s_) * inward
        recp = rec.copy()
        if a.pinch:                                                       # the pinch: full through the hold, gone by the end of the bow
            for k in range(len(recp)):
                f = start + k
                w = 1.0 if f <= bow0 else (1.0 - minjerk((f - bow0) / float(bow_end - bow0)) if f <= bow_end else 0.0)
                recp[k, ~thumb] += a.pinch * w * inward
        K = np.concatenate([pre, recp], 0)[:, keep]
        np.save(os.path.join(A, "%s_%s_kf.npy" % (a.name, side)), K.astype(np.float32))
        with open(os.path.join(A, "%s_%s.obj" % (a.name, side)), "w") as o:
            for v in K[0]: o.write("v %.9f %.9f %.9f\n" % tuple(v))
            for f in Fk: o.write("f %d %d %d\n" % tuple(f + 1))
        colliders.append({"name": "hand_" + side, "type": "tri", "file": "%s/%s_%s.obj" % (a.asset, a.name, side), "thickness": 0.0,
                          "friction": 0.5, "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0, 0, 0],
                          "motion": {"type": "keyframes", "file": "%s/%s_%s_kf.npy" % (a.asset, a.name, side), "substeps": n}})
        table += [["card_%s%02d" % (side[0].upper(), i), "hand_" + s2, a.mu_hand] for i in range(N) for s2 in ("left", "right")]
        if a.thumb_collider:                                              # the thumb: its own collider, static through the pre-roll
            remap_t = -np.ones(len(thumb), int); remap_t[thumb] = np.arange(thumb.sum())
            Ft_l = remap_t[Ft]
            Kt = np.concatenate([np.repeat(rec[:1], P, axis=0), rec], 0)[:, thumb]
            parts = [("thumb", Ft_l, a.mu_thumb)]
            if a.thumb_cap > 0:
                # --thumb-cap R --mu-cap MU: the tip's cap (the thumb vertices within R of the point furthest down the card axis at
                # the riffle's first frame) is a collider of its own with friction MU; the rest of the thumb keeps --mu-thumb.
                # Contact-solver feedback: the ends under the cap are metered at its inner shoulder, where the surface is steeper
                # than the friction cone, and are carried along once they sit within it (about 5 mm of the bottom at 0.7); the
                # flank needs its friction to keep the outer cards through the bow, the cap needs less to let them go.
                ref = Kt[P + (r0 - start)]
                cap_v = np.linalg.norm(ref - ref[np.argmax(ref @ u)], axis=1) < a.thumb_cap
                in_cap = cap_v[Ft_l].all(1)
                parts = [("thumb", Ft_l[~in_cap], a.mu_thumb), ("thumbcap", Ft_l[in_cap], a.mu_cap)]
                report.setdefault("thumb_cap", {})[side] = dict(cap_faces=int(in_cap.sum()), flank_faces=int((~in_cap).sum()), cap_vertices=int(cap_v.sum()))
            for pname, Fp, mu in parts:
                used = np.zeros(len(Kt[0]), bool); used[np.unique(Fp)] = True
                remap_p = -np.ones(len(used), int); remap_p[used] = np.arange(used.sum())
                Kp = Kt[:, used]
                np.save(os.path.join(A, "%s_%s_%s_kf.npy" % (a.name, pname, side)), Kp.astype(np.float32))
                with open(os.path.join(A, "%s_%s_%s.obj" % (a.name, pname, side)), "w") as o:
                    for v in Kp[0]: o.write("v %.9f %.9f %.9f\n" % tuple(v))
                    for f in remap_p[Fp]: o.write("f %d %d %d\n" % tuple(f + 1))
                colliders.append({"name": pname + "_" + side, "type": "tri", "file": "%s/%s_%s_%s.obj" % (a.asset, a.name, pname, side),
                                  "thickness": a.thumb_thickness, "friction": mu, "scale": [1, 1, 1], "rotation": [0, 0, 0],
                                  "translation": [0, 0, 0],
                                  "motion": {"type": "keyframes", "file": "%s/%s_%s_%s_kf.npy" % (a.asset, a.name, pname, side), "substeps": n}})
                table += [["card_%s%02d" % (side[0].upper(), i), pname + "_" + s2, mu] for i in range(N) for s2 in ("left", "right")]
        if a.guide_plate > 0:
            # --guide-plate: a thin static plate along the packet's outer face (the top card's face), a little below the
            # thumb, from the first frame until the riffle is done. Contact-solver feedback: a flat pad swept across the
            # stack carries the held card ends along by friction (the cards only resist by their weight), so its inner
            # edge never passes a card; with the stack leaning on this plate the pad slides under the ends instead and
            # the edge frees one card per staircase step. Like the squaring walls, it is a fixture, not a card constraint.
            g0, g1 = [v * 1e-3 for v in a.guide_span]
            pb = T0 + (float(N - 1) * pitch + a.guide_plate * 1e-3) * nrm
            GV = np.array([pb + g0 * u - 0.035 * e, pb + g1 * u - 0.035 * e, pb + g1 * u + 0.035 * e, pb + g0 * u + 0.035 * e])
            nk = P + (end - start) + 1; k_out = P + ((rd1 if a.guide_out < 0 else a.guide_out) - start)
            Kg = np.zeros((nk, 4, 3), np.float32)
            for k in range(nk):
                tt = min(max((k - k_out) / 10.0, 0.0), 1.0)
                Kg[k] = GV + 0.06 * tt * nrm
            np.save(os.path.join(A, "%s_guide_%s_kf.npy" % (a.name, side)), Kg)
            with open(os.path.join(A, "%s_guide_%s.obj" % (a.name, side)), "w") as o:
                o.write("# guide plate along the packet's outer face (metres)\n")
                for v in GV: o.write("v %.7f %.7f %.7f\n" % tuple(v))
                o.write("f 1 2 3\nf 1 3 4\n")
            colliders.append({"name": "guide_" + side, "type": "tri", "file": "%s/%s_guide_%s.obj" % (a.asset, a.name, side), "thickness": 0.0,
                              "friction": a.guide_mu, "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0, 0, 0],
                              "motion": {"type": "keyframes", "file": "%s/%s_guide_%s_kf.npy" % (a.asset, a.name, side), "substeps": n}})
            table += [["card_%s%02d" % (side[0].upper(), i), "guide_" + s2, a.guide_mu] for i in range(N) for s2 in ("left", "right")]
            print("%s guide plate: %.1f mm outside the top card's face, %g..%g mm below the thumb edge, retracts at sim frame %d" % (side, a.guide_plate, g0 * 1e3, g1 * 1e3, k_out))
        if a.foot_bar > 0:
            # --foot-bar: a low plate on the mat just inside card 0's foot, from the first frame until the riffle is done.
            # Contact-solver feedback: under the bow the inner cards' feet slide toward the centre (the packet's inner face
            # is free), which bows the inner cards 10 mm and the middle ones 2 mm; with the feet held, the compression
            # follows the pad's treads and the cards let go in order.
            gb = a.foot_bar * 1e-3; foot0 = base[bot_row].mean(0); xb = foot0[0] + inward[0] * gb
            BV = np.array([[xb, foot0[1] - 0.035, mat + 1e-4], [xb, foot0[1] + 0.035, mat + 1e-4], [xb, foot0[1] + 0.035, mat + 3e-3], [xb, foot0[1] - 0.035, mat + 3e-3]])
            nk = P + (end - start) + 1; k_out = P + ((rd1 if a.guide_out < 0 else a.guide_out) - start)
            Kb = np.zeros((nk, 4, 3), np.float32)
            for k in range(nk):
                tt = min(max((k - k_out) / 10.0, 0.0), 1.0); Kb[k] = BV - np.array([0.0, 0.0, 0.012 * tt])
            np.save(os.path.join(A, "%s_bar_%s_kf.npy" % (a.name, side)), Kb)
            with open(os.path.join(A, "%s_bar_%s.obj" % (a.name, side)), "w") as o:
                o.write("# foot bar on the mat inside the packet's inner foot (metres)\n")
                for v in BV: o.write("v %.7f %.7f %.7f\n" % tuple(v))
                o.write("f 1 2 3\nf 1 3 4\n")
            colliders.append({"name": "bar_" + side, "type": "tri", "file": "%s/%s_bar_%s.obj" % (a.asset, a.name, side), "thickness": 0.0,
                              "friction": a.guide_mu, "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0, 0, 0],
                              "motion": {"type": "keyframes", "file": "%s/%s_bar_%s_kf.npy" % (a.asset, a.name, side), "substeps": n}})
            table += [["card_%s%02d" % (side[0].upper(), i), "bar_" + s2, a.guide_mu] for i in range(N) for s2 in ("left", "right")]
            print("%s foot bar: %.1f mm inside card 0's foot (x %.1f mm), 3 mm tall, sinks at sim frame %d" % (side, a.foot_bar, xb * 1e3, k_out))
        low_edge = cards0[:, 2].min()
        report[side] = dict(hand_vertices=int(keep.sum()), thumb_vertices=int(thumb.sum()) if a.thumb_collider else 0,
                            keyframes=list(K.shape), preroll_dist_mm=round(dist0 * 1e3, 1),
                            clearance_start_mm=dict(fingers=round(1e3 * d_fing, 3), thumb=round(1e3 * d_thumb, 3)),
                            thumb_backoff_mm=round(1e3 * thumb_backoff, 2),
                            card_shift_mm=[round(1e3 * shift.min(), 2), round(1e3 * shift.max(), 2)],
                            thumb_gap_at_hold_mm=[round(1e3 * np.nanmin(np.where(np.isfinite(gap[0]), gap[0], np.nan)), 2), int(held[0].sum())],
                            lowest_card_edge_mm=round(low_edge * 1e3, 2), release_frames=[rel[0], rel[N // 2], rel[-1]],
                            release_order_ok=bool(all(rel[i] <= rel[i + 1] for i in range(N - 1))))
        sidecar[side] = dict(release_frames=[int(x) for x in rel], shift=[float(x) for x in shift], preroll_dist=float(dist0))
        assert low_edge > mat + 2e-4, "the mat (%.2f mm) is above the packet's lowest edge (%.2f mm)" % (mat * 1e3, low_edge * 1e3)

    E, nu, t = 3e9, 0.3, 3e-4
    mu_mem = a.mem_scale * E * t / (2.0 * (1.0 + nu)); k_bend = a.bend_scale * E * t ** 3 / (24.0 * (1.0 - nu ** 2))
    frames = (P + (end - start)) * n
    if a.walls:
        yc = a.walls_centre if a.walls_centre is not None else float(T0[1])   # the packets' depth centre (run m6 used the measured 0.2977)
        WV = np.array([[-0.066, 0.0, 0.0], [0.066, 0.0, 0.0], [0.066, 0.0, 0.060], [-0.066, 0.0, 0.060]])
        with open(os.path.join(A, "%s_wall.obj" % a.name), "w") as o:
            o.write("# squaring wall, x-z plane at y = 0 (metres)\n")
            for v in WV: o.write("v %.4f %.4f %.4f\n" % tuple(v))
            o.write("f 1 2 3\nf 1 3 4\n")
        nk = (frames + n - 1) // n + 1; f0, f1 = a.walls_close
        for wname, sgn in (("wall_a", -1), ("wall_b", 1)):
            K = np.zeros((nk, 4, 3), np.float32)
            for k in range(nk):
                tt = min(max((k - f0) / float(f1 - f0), 0.0), 1.0)
                K[k] = WV + np.array([0.0, yc + sgn * (a.walls_park + (a.walls_in - a.walls_park) * tt), 0.0])
            np.save(os.path.join(A, "%s_%s_kf.npy" % (a.name, wname)), K)
            colliders.append({"name": wname, "type": "tri", "file": "%s/%s_wall.obj" % (a.asset, a.name), "thickness": 0.0, "friction": 0.0,
                              "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0.0, yc + sgn * a.walls_park, 0.0],
                              "motion": {"type": "keyframes", "file": "%s/%s_%s_kf.npy" % (a.asset, a.name, wname), "substeps": n}})
        report["walls"] = dict(depth_centre_mm=round(yc * 1e3, 1), park_mm=a.walls_park * 1e3, in_mm=a.walls_in * 1e3, close_frames=list(a.walls_close))
    cipc = a.newton_rule == "cipc"
    cfg = {
        "simulation": {"dt": 0.02 / n, "frames": frames, "gravity": [0, 0, -9.81], "output_dir": "output/" + a.name, "save_bgeo": bool(a.save_bgeo), "save_npy": True, "save_every": n,
                       "checkpoint_every": a.checkpoint_every * n, "abort_after_capped_steps": a.abort_capped},
        "materials": {"card": {"model": "cloth_stvk", "mu_mem": mu_mem, "nu": nu, "density": a.density, "thickness": t, "k_bend": k_bend}},
        "bodies": bodies, "colliders": colliders,
        "planes": [{"point": [0, 0, mat], "normal": [0, 0, 1], "friction": a.mu_mat}],
        "contact_table": {"exclude": [], "friction": table},
        "contact": {"enable": True, "d_hat": a.d_hat, "epsilon": 1e-3, "K_min": a.k_min, "max_outer_iters": 500,
                    "decay_factor": 0.9, "decay_remove_threshold": 0.01, "mu_mode": "diag_max", "mu_scale": 0.1, "alpha_lower_bound": 1e-6,
                    **({"mu_max": a.mu_max} if a.mu_max > 0 else {}),
                    "stall": {"iters": 50, "alpha": 1e-4, "mu_factor": 2.0, "d_hat_factor": 0.5, "max_adaptations": 0},
                    "ccd": {"s": 0.1, "max_iter": 100, "float_screen": False, "rebuild_quality_ratio": 1.5},
                    "inversion_free": "off", "self_collision_default": False,
                    "friction": {"enable": True, "eps_v": a.eps_v, "normal_force": "paper"}, "drop_parallel_edge_pairs": False},
        "newton": {"inner_max_iters": a.inner_max if a.inner_max > 0 else (50 if cipc else 8),
                   "line_search": {"max_halvings": 30, "energy_tolerance": 1e-12, "batched": True},
                   "early_accept_velocity_tol": 0.0, "increment_velocity_tol": a.newton_tol if cipc else 0.0},
        "linear_solver": {"type": "pcg", "preconditioner": "block_jacobi", "rel_tol": 1e-4, "max_iters": 10000, "check_interval": 8,
                          "graph": True, "fused": True, "fused_max_rows": 65536, "rigid_coarse": not a.no_rigid_coarse},
        "logging": {"level": "info", "stats_csv": "stats.csv", "timing": True, "debug_check_penetration": False},
    }
    if a.physical_from is not None:
        assert a.thumb_collider, "--physical-from needs --thumb-collider"
        S_ = a.physical_from
        setup = copy.deepcopy(cfg)
        for b in setup["bodies"]:
            for dd in b.get("dirichlet", []): dd["release_frame"] = -1                       # held throughout the setup
        setup["contact_table"]["exclude"] = [["card_%s%02d" % (sd, i), pn + "_" + s2] for sd in "LR" for i in range(N) for s2 in ("left", "right")
                                             for pn in (("thumb", "thumbcap") if a.thumb_cap > 0 else ("thumb",))]
        setup["simulation"].update({"frames": S_, "checkpoint_every": a.setup_checkpoint_every, "output_dir": "output/" + a.name + "_setup"})
        json.dump(setup, open(os.path.join(ROOT, "app", "config", a.name + "_setup.json"), "w"), indent=1)
        for b in cfg["bodies"]: b.pop("dirichlet", None)                                      # the physical run: cards are plain free bodies
        cfg["simulation"]["restart"] = {"step": S_, "positions": "output/%s_setup/ckpt_x_%d.npy" % (a.name, S_),
                                        "velocities": "output/%s_setup/ckpt_v_%d.npy" % (a.name, S_)}
        print("wrote app/config/%s_setup.json (%d steps, rows ride the thumb, card-thumb contact off)" % (a.name, S_))
    out = os.path.join(ROOT, "app", "config", a.name + ".json")
    json.dump(cfg, open(out, "w"), indent=1)
    sidecar.update(dict(motion=os.path.basename(meta["motion"]), thumb_surface=a.thumb_surface, release_all_at=a.release_all_at,
                        thumb_collider=a.thumb_collider, thumb_thickness=a.thumb_thickness, mu_thumb=a.mu_thumb, thumb_cap=a.thumb_cap, mu_cap=a.mu_cap, pinch=a.pinch,
                        physical_from=a.physical_from, hinge_frames=a.hinge_frames, start=start, end=end, preroll=P, substeps=n, mat=mat, cards=N,
                        phases={k: list(v) for k, v in ph.items()}))
    json.dump(sidecar, open(os.path.join(A, a.name + "_sim.json"), "w"), indent=1)
    print("wrote %s: %d cards, %d steps (%d pre-roll + %d recorded frames x %d substeps), mat %.2f mm, tilt %s deg" % (
        os.path.relpath(out, ROOT), 2 * N, frames, P, end - start, n, mat * 1e3, sc.get("packet_tilt_deg")))
    for s_, r in report.items(): print("  %s: %s" % (s_, r))

if __name__ == "__main__":
    main()
