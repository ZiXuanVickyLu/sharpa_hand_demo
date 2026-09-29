#!/usr/bin/env python
"""Tri-view video of a hand-shuffle run: the robot arms and hands replaying the recorded joint
trajectory, with the SIMULATED cards of the run. Runs from anywhere with the packages of
tool/render/requirements.txt and a GPU with EGL:

  EGL_DEVICE=0 python3 tool/render_hand_shuffle.py \
      --run output/output/hand_shuffle_phys --config app/config/hand_shuffle_phys.json \
      --sim asset/hand_shuffle_phys/hand_shuffle_phys_sim.json --cams asset/hand_shuffle_phys/cams.json \
      --out output/demo/hand_shuffle_phys.mp4 --isolated        [--frames 0,60,120 --png-dir DIR]

writes the three-view composite and, with --isolated, one video per camera (<out stem>_CAM1/_CAM2/_HEAD.mp4);
--sim is the generator's sidecar, which carries the start frame, the pre-roll and the motion file (the hand IK
recording, asset/<asset>/motion.npz) that the robot replays. Same for hand_shuffle_mb with its own names.
The run must have been written with simulation.save_npy (x_<step>.npy, surface_faces.npy).
Sim frame j (= step j * substeps) shows recorded frame start + max(0, j - preroll)."""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "render"))
os.environ.setdefault("EGL_DEVICE", "0")
import render_triview as rt                      # noqa: E402  (headless EGL set up on import)
from render_triview import look_at, read_frame_gl, Layout, disable_distance_fog, draw_tag, font  # noqa: E402
import imageio.v2 as iio                         # noqa: E402
import warp as wp                                # noqa: E402
from newton.viewer import ViewerGL               # noqa: E402
from PIL import Image, ImageDraw                 # noqa: E402
from robot_scene import YamSharpaBench           # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--sim", default=None, help="the generator's sidecar (asset/<asset>/<name>_sim.json): start frame, pre-roll, motion file")
    ap.add_argument("--motion", default=None); ap.add_argument("--cams", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=145); ap.add_argument("--preroll", type=int, default=30)
    ap.add_argument("--show-preroll", action="store_true")
    ap.add_argument("--frames", default=None, help="comma list of sim frames: PNGs only"); ap.add_argument("--png-dir", default=None)
    ap.add_argument("--panel", type=int, nargs=2, default=[640, 340]); ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--isolated", action="store_true", help="also write one stand-alone video per camera, <out stem>_<CAM>.mp4, at --iso-size")
    ap.add_argument("--iso-size", type=int, nargs=2, default=[1280, 680], help="stand-alone video size (same aspect as --panel)")
    ap.add_argument("--no-labels", action="store_true", help="no camera-name tag on the stand-alone videos")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--near", type=float, default=5.0, help="camera near plane (cm). The viewer's default 0.01 cm leaves a depth resolution of ~0.9 mm at the "
                    "cards' distance, coarser than the 0.3 mm card pitch: stacked cards z-fight and their colours mix")
    ap.add_argument("--card-thickness", type=float, default=0.2e-3, help="draw each card as a slab of this thickness (m) around its mid-surface, with off-white "
                    "edges like paper, so the deck's sides do not average red and blue into purple; 0 = flat surfaces")
    ap.add_argument("--mat-gap", type=float, default=1e-3, help="the drawn mat's top sits this far (m) below the contact plane, so a card resting at the "
                    "contact offset (0.25 mm above the plane) does not z-fight with the mat's surface")
    ap.add_argument("--title", default="YAM + SHARPA  ·  riffle shuffle, simulated cards"); ap.add_argument("--case-id", default="06")
    a = ap.parse_args()
    if a.sim:
        sim = json.load(open(a.sim)); a.start, a.preroll = int(sim["start"]), int(sim["preroll"])
        m = sim["motion"]; a.motion = a.motion or (m if os.path.isabs(m) else os.path.join(os.path.dirname(os.path.abspath(a.sim)), m))
        cj = os.path.join(os.path.dirname(a.motion), "cams.json")
        if a.cams is None and os.path.exists(cj): a.cams = cj
    assert a.motion, "--motion or --sim is required"
    cfg = json.load(open(a.config)); n_sub = cfg["simulation"]["save_every"]
    n_card_v = sum(1 for b in cfg["bodies"]) * 105
    F = np.load(os.path.join(a.run, "surface_faces.npy")).astype(np.int32)
    Fc = F[(F < n_card_v).all(1)]
    half = n_card_v // 2
    FL, FR = Fc[(Fc < half).all(1)], Fc[(Fc >= half).all(1)] - half
    # colliders other than the two hands (e.g. squaring walls): drawn as a grey mesh from the run's own vertices
    # the hands are the colliders named hand_*; their vertex counts come from their OBJ files (paths as in the config,
    # relative to asset/), so the walls or any collider listed after them start at n_card_v + n_hand
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_hand = 0
    for col in cfg.get("colliders", []):
        if not col.get("name", "").startswith("hand"): continue
        fp = col["file"]
        for base in (os.path.dirname(os.path.abspath(a.config)), os.path.join(root, "asset"), root):
            if os.path.exists(os.path.join(base, fp)): fp = os.path.join(base, fp); break
        n_hand += sum(1 for l in open(fp) if l.startswith("v "))
    n_extra0 = n_card_v + n_hand
    FX = F[(F >= n_extra0).all(1)] if n_hand else np.zeros((0, 3), np.int32)
    if len(FX): print("[hand-shuffle] drawing %d extra collider triangles" % len(FX))
    steps = sorted(int(m.group(1)) for m in (re.match(r"x_(\d+)\.npy$", f) for f in os.listdir(a.run)) if m)
    sim_frames = [s // n_sub for s in steps if s % n_sub == 0]
    with np.load(a.motion, allow_pickle=True) as z:
        Q = np.asarray(z["joint_q"], dtype=np.float64); fps_rec = float(z["fps"])
        phases = [(str(n), int(s), int(e)) for n, s, e in zip(z["phase_names"], z["phase_start"], z["phase_end"])]
    cams = json.load(open(a.cams)) if a.cams else rt.DEFAULT_CAMS
    disable_distance_fog()
    pw, ph = a.panel
    # each view is rendered once per frame at the largest size needed; the composite's panels are downsampled from it
    vw_, vh_ = a.iso_size if a.isolated else (pw, ph)
    assert abs(vw_ / vh_ - pw / ph) < 0.01, "--iso-size must have the panel's aspect"
    bench = YamSharpaBench(with_table=True)
    viewer = ViewerGL(width=vw_ * a.ss, height=vh_ * a.ss, headless=True)
    viewer.set_model(bench.model)
    rnd = viewer.renderer
    rnd.draw_sky = False; rnd.sky_upper = rnd.sky_lower = rnd.background_color = (0.0, 0.0, 0.0)
    rnd.draw_shadows = True; rnd.spotlight_enabled = False
    try: rnd.shadow_extents = 220.0
    except AttributeError: rnd._shadow_extents = 220.0
    viewer.begin_frame(0.0); viewer.log_state(bench.state); viewer.end_frame()
    # slab topology from the card mesh: faces of one card (Fc0, 168 x 3), its boundary edges (40), replicated per card
    nvc = 105
    Fc0 = FL[(FL < nvc).all(1)]
    Ec = np.sort(np.concatenate([Fc0[:, [0, 1]], Fc0[:, [1, 2]], Fc0[:, [2, 0]]]), axis=1)
    ue, cnt = np.unique(Ec, axis=0, return_counts=True); Eb = ue[cnt == 1]
    n_side = half // nvc          # cards per side
    def slab_indices(n_cards):
        # vertex layout per side: [top of card 0..n-1 (n*nvc), bottom of card 0..n-1 (n*nvc)]
        off_t = np.arange(n_cards)[:, None, None] * nvc; off_b = n_cards * nvc + off_t
        top = (Fc0[None] + off_t).reshape(-1, 3); bot = (Fc0[None, :, ::-1] + off_b).reshape(-1, 3)
        a, b = Eb[:, 0], Eb[:, 1]
        q1 = np.stack([a[None] + off_t[:, 0], b[None] + off_t[:, 0], b[None] + off_b[:, 0]], -1).reshape(-1, 3)
        q2 = np.stack([a[None] + off_t[:, 0], b[None] + off_b[:, 0], a[None] + off_b[:, 0]], -1).reshape(-1, 3)
        return np.concatenate([top, bot]).astype(np.int32), np.concatenate([q1, q2]).astype(np.int32)
    slab = a.card_thickness > 0
    if slab:
        fF, fE = slab_indices(n_side)
        fFw, fEw = wp.array(fF.flatten(), dtype=wp.int32), wp.array(fE.flatten(), dtype=wp.int32)
        print("[hand-shuffle] cards drawn as %.2f mm slabs: %d faces + %d edge triangles per side" % (a.card_thickness * 1e3, len(fF), len(fE)))
    def slab_points(Xs):   # Xs: [n_cards * nvc, 3] in cm -> [2 * n_cards * nvc, 3]
        P = Xs.reshape(n_side, nvc, 3)
        v0, v1, v2 = P[:, Fc0[:, 0]], P[:, Fc0[:, 1]], P[:, Fc0[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)                                   # face normals (area weighted)
        N = np.zeros_like(P)
        for k in range(3): np.add.at(N, (slice(None), Fc0[:, k]), fn)
        N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-12)
        h = 0.5 * a.card_thickness * 100.0                                 # half thickness in cm
        return np.concatenate([(P + h * N).reshape(-1, 3), (P - h * N).reshape(-1, 3)]).astype(np.float32)
    fL, fR = wp.array(FL.flatten(), dtype=wp.int32), wp.array(FR.flatten(), dtype=wp.int32)
    fX = wp.array((FX - n_extra0).flatten(), dtype=wp.int32) if len(FX) else None
    mat_z = (float(cfg["planes"][0]["point"][2]) - a.mat_gap) * 100.0     # the playing mat (cm): a felt slab whose top is --mat-gap under the contact plane
    mx, my0, my1 = 34.0, 12.0, 48.0
    mat_v = wp.array(np.array([[sx * mx, y, z] for z in (0.02, mat_z) for y in (my0, my1) for sx in (-1, 1)], np.float32), dtype=wp.vec3)
    mat_f = wp.array(np.array([[4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int32).flatten(), dtype=wp.int32)
    lay = Layout(pw, ph); names = [c["name"] for c in cams]

    extra_first = None
    if fX is not None:
        for j0 in sim_frames:
            X0 = np.load(os.path.join(a.run, "x_%d.npy" % (j0 * n_sub))).astype(np.float32) * 100.0
            if len(X0) > n_extra0: extra_first = X0[n_extra0:]; print("[hand-shuffle] extra colliders first present at sim frame %d; earlier frames show them there" % j0); break
    def rec_frame(j): return a.start + max(0, j - a.preroll)
    def phase_at(f):
        for nme, s, e in phases:
            if s <= f <= e: return nme
        return ""
    def render(j):
        f = rec_frame(j)
        bench.joint_q[:] = Q[f]; bench._fk()
        Xall = np.load(os.path.join(a.run, "x_%d.npy" % (j * n_sub))).astype(np.float32) * 100.0   # m -> cm
        X = Xall[:n_card_v]
        if slab:
            vL, vR = wp.array(slab_points(X[:half]), dtype=wp.vec3), wp.array(slab_points(X[half:]), dtype=wp.vec3)
        else:
            vL, vR = wp.array(np.ascontiguousarray(X[:half]), dtype=wp.vec3), wp.array(np.ascontiguousarray(X[half:]), dtype=wp.vec3)
        # frames written before the extra colliders existed (a restart with added colliders): show them where the first
        # frame that has them puts them, which is their parked position if they are keyframed to move only later
        if fX is not None and len(Xall) <= n_extra0 and extra_first is not None: Xall = np.concatenate([Xall, extra_first])
        vX = wp.array(np.ascontiguousarray(Xall[n_extra0:]), dtype=wp.vec3) if (fX is not None and len(Xall) > n_extra0) else None
        views = []
        for cam in cams:
            viewer.set_camera(*look_at(cam["pos"], cam["target"])); viewer.camera.fov = float(cam["fov"]); viewer.camera.near = a.near
            viewer.begin_frame(j / fps_rec); viewer.log_state(bench.state)
            if mat_z > 0.05:
                viewer.log_mesh("/mat", mat_v, mat_f, color=(0.10, 0.36, 0.24), roughness=0.95, backface_culling=False)
            if slab:
                edge = (0.93, 0.91, 0.86)
                viewer.log_mesh("/cards_left", vL, fFw, color=(0.80, 0.14, 0.12), roughness=0.8, backface_culling=False)
                viewer.log_mesh("/cards_left_edges", vL, fEw, color=edge, roughness=0.9, backface_culling=False)
                viewer.log_mesh("/cards_right", vR, fFw, color=(0.13, 0.30, 0.82), roughness=0.8, backface_culling=False)
                viewer.log_mesh("/cards_right_edges", vR, fEw, color=edge, roughness=0.9, backface_culling=False)
            else:
                viewer.log_mesh("/cards_left", vL, fL, color=(0.80, 0.14, 0.12), roughness=0.8, backface_culling=False)
                viewer.log_mesh("/cards_right", vR, fR, color=(0.13, 0.30, 0.82), roughness=0.8, backface_culling=False)
            if vX is not None: viewer.log_mesh("/extra_colliders", vX, fX, color=(0.62, 0.62, 0.66), roughness=0.6, backface_culling=False)
            viewer.end_frame()
            im = Image.fromarray(read_frame_gl(viewer))
            views.append(im.resize((vw_, vh_), Image.LANCZOS) if a.ss > 1 else im)
        clock = "recorded t = %5.2f s   |   sim frame %d" % (f / fps_rec, j)
        cap = "[%s]   Simulated cards (barrier-free AL contact, 54 shells) driven by the IK-generated hand trajectory as a moving boundary." % phase_at(f)
        panels = [v if v.size == (pw, ph) else v.resize((pw, ph), Image.LANCZOS) for v in views]
        comp = lay.compose(panels, names, "%s   %s" % (a.case_id, a.title), clock, "cards ARE simulated  |  hands: recorded joint trajectory", cap)
        return comp, views

    tag_k = max(1, round(vw_ / pw)); f_tag = font("DejaVuSansMono-Bold.ttf", 12 * tag_k)
    def standalone(view, name):
        if a.no_labels: return view
        view = view.copy(); draw_tag(ImageDraw.Draw(view), 8 * tag_k, 8 * tag_k, name, f_tag, tag_k)
        return view

    todo = [j for j in sim_frames if a.show_preroll or j >= a.preroll]
    if a.frames:
        d = a.png_dir or os.path.join(a.run, "triview_png"); os.makedirs(d, exist_ok=True)
        for j in [int(s) for s in a.frames.split(",")]:
            comp, views = render(j)
            comp.save(os.path.join(d, "triview_%04d.png" % j)); print("frame", j)
            if a.isolated:
                for nme, v in zip(names, views): standalone(v, nme).save(os.path.join(d, "%s_%04d.png" % (nme, j)))
        viewer.close(); return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    def writer(path): return iio.get_writer(path, fps=a.fps, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    w = writer(a.out)
    stem = os.path.splitext(a.out)[0]
    iso = {nme: writer("%s_%s.mp4" % (stem, re.sub(r"[^A-Za-z0-9_-]", "_", nme))) for nme in names} if a.isolated else {}
    for k, j in enumerate(todo):
        comp, views = render(j)
        w.append_data(np.asarray(comp))
        for nme, v in zip(names, views):
            if nme in iso: iso[nme].append_data(np.asarray(standalone(v, nme)))
        if k % 25 == 0: print("[hand-shuffle] frame %d/%d" % (k, len(todo)))
    w.close()
    for x in iso.values(): x.close()
    viewer.close(); print("wrote", a.out)
    for nme in iso: print("wrote %s_%s.mp4 (%dx%d)" % (stem, nme, vw_, vh_))

if __name__ == "__main__":
    main()
