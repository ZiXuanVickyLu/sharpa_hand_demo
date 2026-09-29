#!/usr/bin/env python
"""Bake a recorded hand-shuffle motion into moving triangle colliders (one vertex position per frame)
plus the per-link ranges and the scene record the generator reads. Uses the robot model of asset/robot/
through tool/render (warp/newton, packages of tool/render/requirements.txt):

  python3 tool/bake_hand_shuffle.py --motion <recording>/shuffle_motion.npz --out asset/<asset>
Every hand link that carries a mesh (palm, thumb, four fingers; not the arm) is decimated by
vertex clustering, merged into one mesh per hand, and transformed per frame by the recorded
world link transform: x_world = T_body(f) * T_shape * (scale * v). Units go from the rig's
centimetres to metres; the rig frame is kept (Z up, table z = 0). Output:

  hand_left.obj / hand_right.obj      mesh at the first baked frame
  hand_left_kf.npy / hand_right_kf.npy  float32 [frames, n, 3]
  hand_meta.json                      frame range, fps, phases, the generator's scene record in
                                      metres, per-link vertex ranges
"""
import argparse, json, os, sys
import numpy as np

def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def cluster(V, F, cell):
    key = np.floor(V / cell).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    W = np.zeros((len(cnt), 3)); np.add.at(W, inv, V); W /= cnt[:, None]
    G = inv[F]
    ok = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
    G = G[ok]
    _, first = np.unique(np.sort(G, axis=1), axis=0, return_index=True)
    G = G[np.sort(first)]
    used = np.unique(G); remap = -np.ones(len(W), np.int64); remap[used] = np.arange(len(used))
    return W[used], remap[G]

def cell_for(link):
    if "elastomer" in link or link.endswith("_DP"): return 0.22
    if "C_MC" in link: return 0.55
    return 0.36

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=-1, help="first baked frame (default: start of the approach phase)")
    ap.add_argument("--last", type=int, default=-1, help="last baked frame (default: end of the withdraw phase)")
    ap.add_argument("--thumb-pad", type=float, nargs=2, default=None, metavar=("ACROSS", "WIDTH"),
                    help="add a flat rectangular pad (cm across the card stack, cm along the cards' width) to each thumb's distal "
                         "link: perpendicular to the card axis at the end of the bow and --pad-offset beyond the fingertip's deepest "
                         "point. A flat face holds the bowed cards' ends without friction and, as it slides across the stack, its "
                         "inner edge frees each card while it is still fully bowed (the rounded fingertip lets a card slip off only "
                         "once it has nearly straightened, so it falls slowly and meets the other hand's card in the air)")
    ap.add_argument("--pad-inner", type=float, default=0.8, help="cm of the pad inward (toward the packet's inner face) of the fingertip's deepest point")
    ap.add_argument("--pad-offset", type=float, default=0.05, help="cm the pad sits beyond the fingertip's deepest point along the card axis")
    ap.add_argument("--pad-lip", type=float, default=0.0, help="cm of a lip along the pad's outer edge, standing off the pad toward "
                    "the cards: the held packet leans on it, so the pad can slide under the card ends instead of carrying "
                    "them along by friction (0 = no lip)")
    ap.add_argument("--pad-frame", type=int, default=-1, help="recording frame at which the pad is placed perpendicular to the card "
                    "axis (default: the end of the bow; the middle of the riffle halves the tilt the thumb's own turn gives it)")
    ap.add_argument("--pad-steps", type=int, default=0, help="make the pad a staircase across the stack: this many treads, each "
                    "perpendicular to the card axis and --pad-step-depth deeper than the one inside it, from --pad-steps-start "
                    "outward; the flat inner margin and the flat outer margin stay. Contact-solver feedback: with the feet on the "
                    "mat the inner cards are bowed most (10 mm against 2 mm), so any unloading release frees the middle cards first; "
                    "treads that descend outward give the inner card the least compression, and the risers hold every end in place, "
                    "so a plain retreat up the card axis frees the cards from the inside one at a time (0 = flat pad)")
    ap.add_argument("--pad-step-across", type=float, default=0.035, help="cm width of each tread across the stack")
    ap.add_argument("--pad-step-depth", type=float, default=0.035, help="cm each tread lies deeper (along the card axis) than the one inside it")
    ap.add_argument("--pad-steps-start", type=float, default=-0.5, help="cm from the fingertip's deepest point (across the stack, + = outward) "
                    "where the first tread begins")
    ap.add_argument("--pad-tilt", type=float, default=0.0, help="degrees the pad is turned about the cards' width axis so that its inner edge "
                    "sits deeper along the card axis (0 = perpendicular to the card axis; a few degrees stop the held "
                    "packet creeping toward the inner edge along the pad and make the freed card snap harder)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "render"))
    import warp as wp
    wp.config.quiet = True
    from robot_scene import YamSharpaBench
    z = np.load(a.motion, allow_pickle=True)
    body_q = z["body_q"].astype(np.float64); names = [str(n) for n in z["body_names"]]
    ph = {str(n): (int(s0), int(s1)) for n, s0, s1 in zip(z["phase_names"], z["phase_start"], z["phase_end"])}
    if a.first < 0: a.first = ph["approach"][0]
    if a.last < 0: a.last = ph["withdraw"][1]
    bench = YamSharpaBench(device="cpu", with_table=False)
    m = bench.model
    assert [str(n) for n in bench.body_labels] == names, "body order differs from the recording"
    sb = m.shape_body.numpy(); ss = m.shape_scale.numpy(); sx = m.shape_transform.numpy()
    frames = np.arange(a.first, a.last + 1)
    meta = dict(first=int(a.first), last=int(a.last), fps=float(z["fps"]),
                phases=[dict(name=str(n), start=int(s), end=int(e)) for n, s, e in zip(z["phase_names"], z["phase_start"], z["phase_end"])],
                links={})
    sc = json.loads(str(z["scene"]))
    # snapshot the recording next to the bake: the generator's output directory gets regenerated
    import shutil
    os.makedirs(a.out, exist_ok=True)
    snap = os.path.join(os.path.abspath(a.out), "motion.npz")
    if os.path.abspath(a.motion) != snap: shutil.copyfile(a.motion, snap)
    cams = os.path.join(os.path.dirname(os.path.abspath(a.motion)), "cams.json")
    if os.path.exists(cams): shutil.copyfile(cams, os.path.join(os.path.abspath(a.out), "cams.json"))
    gen = os.path.join(os.getcwd(), "gen_shuffle_motion.py")   # the IK generator as it was when this motion was baked
    if os.path.exists(gen): shutil.copyfile(gen, os.path.join(os.path.abspath(a.out), "gen_shuffle_motion.snapshot.py"))
    meta["motion"] = os.path.basename(snap)               # relative to the asset directory
    meta["scene_cm"] = sc                       # the generator's record, as written (cm, rig frame)
    meta["scene_m"] = {k: (np.array(v) * 0.01).tolist() if isinstance(v, (list, float)) and k not in ("scale", "packet_tilt_deg") else v for k, v in sc.items()}
    os.makedirs(a.out, exist_ok=True)
    for side in ("left", "right"):
        Vs, Fs, owner = [], [], []
        off = 0; ranges = {}
        for i in range(m.shape_count):
            link = names[sb[i]].split("/")[-1]
            if not link.startswith(side + "_"): continue
            if not any(t in link for t in ("hand_C_MC", "thumb_", "index_", "middle_", "ring_", "pinky_")): continue
            src = m.shape_source[i]
            V = np.asarray(src.vertices, dtype=np.float64) * ss[i]
            F = np.asarray(src.indices, dtype=np.int64).reshape(-1, 3)
            V = V @ quat_R(sx[i][3:7]).T + sx[i][:3]          # into the body frame
            Vd, Fd = cluster(V, F, cell_for(link))
            ranges[link] = [off, off + len(Vd)]
            Vs.append(Vd); Fs.append(Fd + off); owner += [int(sb[i])] * len(Vd); off += len(Vd)
        if a.thumb_pad is not None:
            # the pad, in the distal thumb link's frame: placed in world coordinates at the end of the bow, then carried back
            th = np.radians(float(sc["packet_tilt_deg"])); sgn = 1.0 if side == "left" else -1.0
            u = np.array([-np.cos(th) * sgn, 0.0, -np.sin(th)])          # down the cards of this side (from their held end to their feet)
            n_p = np.array([-np.sin(th) * sgn, 0.0, np.cos(th)])         # across the stack, from the inner face outward
            fb = ph["bow"][1] if a.pad_frame < 0 else a.pad_frame; ow = np.array(owner)
            dp = [b for b in np.unique(ow) if names[b].split("/")[-1] == side + "_thumb_DP"][0]
            tipmask = np.array([names[b].split("/")[-1] in (side + "_thumb_DP", side + "_thumb_elastomer") for b in ow])
            Vall = np.vstack(Vs); Xw = np.empty_like(Vall)
            for b in np.unique(ow):
                msk = ow == b
                Xw[msk] = Vall[msk] @ quat_R(body_q[fb, b, 3:7]).T + body_q[fb, b, :3]
            apex = Xw[tipmask][np.argmax(Xw[tipmask] @ u)]                # the fingertip's deepest point along the card axis
            across, width = a.thumb_pad
            tl = np.radians(a.pad_tilt); e1 = np.cos(tl) * n_p - np.sin(tl) * u     # across the pad; the inner (-e1) side deeper (+u) when tilted
            P0 = apex + a.pad_offset * u                                              # the pad's deepest-point pivot
            ey = np.array([0.0, 1.0, 0.0])
            if a.pad_steps > 0:
                # the profile across the stack, (across, depth) pairs: inner margin, treads and risers, outer margin
                A, Dp = a.pad_step_across, a.pad_step_depth
                prof = [(-a.pad_inner, 0.0), (a.pad_steps_start, 0.0)]
                for j in range(a.pad_steps):
                    prof += [(a.pad_steps_start + (j + 1) * A, j * Dp), (a.pad_steps_start + (j + 1) * A, (j + 1) * Dp)]
                prof.append((across - a.pad_inner, a.pad_steps * Dp))
                pts = [P0 + aa * e1 + dd * u for aa, dd in prof]
                corners = np.array([q + s2 * (width / 2) * ey for q in pts for s2 in (-1, 1)])
                faces = np.array([[2 * i, 2 * i + 2, 2 * i + 3] for i in range(len(pts) - 1)] + [[2 * i, 2 * i + 3, 2 * i + 1] for i in range(len(pts) - 1)])
                if a.pad_lip > 0:      # the lip stands on the outer edge (the last two vertices)
                    nm = len(corners); lip = np.array([corners[nm - 2] + a.pad_lip * u, corners[nm - 1] + a.pad_lip * u])
                    corners = np.vstack([corners, lip]); faces = np.vstack([faces, [[nm - 2, nm, nm + 1], [nm - 2, nm + 1, nm - 1]]])
            else:
                corners = np.array([P0 + a1 * e1 + s2 * (width / 2) * ey for a1, s2 in ((-a.pad_inner, -1), (across - a.pad_inner, -1), (across - a.pad_inner, 1), (-a.pad_inner, 1))])
                faces = np.array([[0, 1, 2], [0, 2, 3]])
                if a.pad_lip > 0:      # a quad standing on the outer edge (corners 1 and 2), off the pad along the card axis
                    lip = np.array([corners[1] + a.pad_lip * u, corners[2] + a.pad_lip * u])
                    corners = np.vstack([corners, lip]); faces = np.vstack([faces, [[1, 4, 5], [1, 5, 2]]])
            pad_local = (corners - body_q[fb, dp, :3]) @ quat_R(body_q[fb, dp, 3:7])      # R^T (x - p), row-wise
            ranges[side + "_thumb_DP_pad"] = [off, off + len(corners)]
            Vs.append(pad_local); Fs.append(faces + off); owner += [int(dp)] * len(corners); off += len(corners)
            meta.setdefault("thumb_pad", {})[side] = dict(across_cm=across, width_cm=width, inner_cm=a.pad_inner, offset_cm=a.pad_offset,
                                                          tilt_deg=a.pad_tilt, lip_cm=a.pad_lip, steps=a.pad_steps, step_across_cm=a.pad_step_across, step_depth_cm=a.pad_step_depth,
                                                          steps_start_cm=a.pad_steps_start, apex_cm=apex.tolist(), placed_at_frame=int(fb))
            print("%s thumb pad: %.1f x %.1f cm%s, apex at frame %d = (%.2f, %.2f, %.2f) cm" % (
                side, across, width, (", %d treads of %.3f x %.3f cm from %.2f cm" % (a.pad_steps, a.pad_step_across, a.pad_step_depth, a.pad_steps_start)) if a.pad_steps else "", fb, *apex))
        V = np.vstack(Vs); F = np.vstack(Fs); owner = np.array(owner)
        kf = np.empty((len(frames), len(V), 3), np.float32)
        for j, f in enumerate(frames):
            X = np.empty_like(V)
            for b in np.unique(owner):
                msk = owner == b
                X[msk] = V[msk] @ quat_R(body_q[f, b, 3:7]).T + body_q[f, b, :3]
            kf[j] = (X * 0.01).astype(np.float32)
        np.save(os.path.join(a.out, "hand_%s_kf.npy" % side), kf)
        with open(os.path.join(a.out, "hand_%s.obj" % side), "w") as o:
            for v in kf[0]: o.write("v %.7f %.7f %.7f\n" % tuple(v))
            for t in F: o.write("f %d %d %d\n" % tuple(t + 1))
        meta["links"][side] = ranges
        # the thumb tip at FULL resolution over hold..riffle_done: the scene generator lets the held card edges ride
        # this surface (the thumb's grip is adhesion, not contact), so it must be the surface the renderer shows
        f0, f1 = ph["hold"][0], ph["riffle_done"][1]
        tv, tf, tb, toff = [], [], [], 0
        for i in range(m.shape_count):
            link = names[sb[i]].split("/")[-1]
            if link not in (side + "_thumb_DP", side + "_thumb_elastomer"): continue
            src = m.shape_source[i]
            Vt = np.asarray(src.vertices, dtype=np.float64) * ss[i]
            Vt = Vt @ quat_R(sx[i][3:7]).T + sx[i][:3]
            tv.append(Vt); tf.append(np.asarray(src.indices, dtype=np.int64).reshape(-1, 3) + toff); tb += [int(sb[i])] * len(Vt); toff += len(Vt)
        tv = np.vstack(tv); tf = np.vstack(tf); tb = np.array(tb)
        tip = np.empty((f1 - f0 + 1, len(tv), 3), np.float32)
        for j, f in enumerate(range(f0, f1 + 1)):
            X = np.empty_like(tv)
            for b in np.unique(tb):
                msk = tb == b
                X[msk] = tv[msk] @ quat_R(body_q[f, b, 3:7]).T + body_q[f, b, :3]
            tip[j] = (X * 0.01).astype(np.float32)
        np.savez_compressed(os.path.join(a.out, "thumb_tip_%s.npz" % side), V=tip, F=tf.astype(np.int32), first=f0)
        print("%s hand: %d links, %d vertices, %d triangles, keyframes %s (%.0f MB)" % (side, len(ranges), len(V), len(F), kf.shape, kf.nbytes / 1e6))
    json.dump(meta, open(os.path.join(a.out, "hand_meta.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
