#!/usr/bin/env python
"""Three-view animation of a tear run in the simple GL style.

Black background, light table, the YAM arms + Sharpa hands in the viewer's
colorful per-link palette, the bread as a two-tone (crust / crumb) surface.
Three synchronized cameras side by side -- CAM2 (over the shoulder), HEAD
(front, high) and CAM1 (low, from the side) -- under a title bar with the
simulation clock, in the layout of the "YAM + SHARPA" case videos.

    EGL_DEVICE=0 uv run python render_triview.py outputs/tear_fiber_hi8x_v2
    # composite + one stand-alone video per camera (2x the panel resolution)
    EGL_DEVICE=0 uv run python render_triview.py <run> --isolated
    # camera tuning / spot checks: composite + per-panel PNGs, no video
    EGL_DEVICE=0 uv run python render_triview.py <run> --frames 10,190,222,334
    # robot-only review of a generated joint trajectory (no particles):
    # <out_dir> receives the videos, motion.npz holds joint_q [F, n_q] + fps
    EGL_DEVICE=0 uv run python render_triview.py <out_dir> --motion motion.npz

Bread surfaces are cached per frame under <run>/triview_cache/, so camera or
layout changes re-render in a fraction of the first pass.
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
# render_mesh defaults EGL_DEVICE to 1 (its author's 8-GPU box); on a
# single-GPU machine that is a black frame, so default to device 0 here
os.environ.setdefault("EGL_DEVICE", "0")
# render_mesh flips pyglet to headless EGL on import (what we want here)
from render_mesh import look_at, read_frame_gl, read_ply  # noqa: E402

import imageio.v2 as iio  # noqa: E402
import warp as wp  # noqa: E402
from newton.viewer import ViewerGL  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from robot_scene import YamSharpaBench  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402

# name, camera position, look-at target (rig cm; tabletop z=0, bread at
# (0, 30), robot bases at (-/+26.7, 0)), vertical fov in degrees
DEFAULT_CAMS = [
    # over the shoulder: above and just in front of the elbows, yawed left so
    # the forearms frame the hands and the void shows past the far table
    # corners; high enough (80 cm) to see the hands OVER the wrist barrels
    {"name": "CAM2", "pos": [6.0, 6.0, 80.0], "target": [-5.0, 35.0, 2.0], "fov": 42.0},
    # front, high, on the centerline, looking back at the robot
    {"name": "HEAD", "pos": [0.0, 95.0, 80.0], "target": [0.0, 24.0, 4.0], "fov": 39.0},
    # low front-right diagonal across the table
    {"name": "CAM1", "pos": [60.0, 94.0, 25.0], "target": [2.0, 29.0, 12.0], "fov": 29.5},
]
CRUMB_COL = (0.95, 0.86, 0.55)  # pale yellow interior
CRUST_COL = (0.85, 0.47, 0.13)  # orange-brown shell
BG = (3, 5, 8)
TEAL = (38, 150, 140)
FRAME_DT = 1.0 / 30.0  # run_tear saves one frame per 1/30 s of sim time
CACHE_VERSION = 1  # bump when surface() or the crust split changes


def disable_distance_fog():
    """The shape shader fogs fragments between 20 and 200 WORLD UNITS from the
    camera -- tuned for meter-scale scenes. This rig is in centimeters, so a
    camera 130 units away loses ~60% of every color to the fog. Push the fog
    range out of reach (must run before the viewer compiles its shaders)."""
    import newton._src.viewer.gl.shaders as sh
    src = sh.shape_fragment_shader
    new = src.replace("float fog_start = 20.0;", "float fog_start = 1.0e8;") \
             .replace("float fog_end   = 200.0;", "float fog_end   = 1.0e9;")
    if new == src:
        print("[triview] WARN: fog constants not found; colors will be fogged")
    sh.shape_fragment_shader = new


def capsule_crust_mask(p0_cm, shell_cm):
    """Particles starting within `shell_cm` of the loaf surface. The loaf is
    an x-aligned capsule squashed in z; recover it from the frame-0 bounding
    box and measure the distance analytically (a convex-hull plane test
    would need an N x n_facets matrix -- hopeless at 900k particles)."""
    lo, hi = p0_cm.min(0), p0_cm.max(0)
    c = 0.5 * (lo + hi)
    r = 0.5 * (hi[1] - lo[1])
    hl = 0.5 * (hi[0] - lo[0]) - r
    sq = (hi[2] - lo[2]) / (2.0 * r)
    q = p0_cm - c
    q[:, 2] /= sq
    q[:, 0] -= np.clip(q[:, 0], -hl, hl)
    return (r - np.linalg.norm(q, axis=1)) < shell_cm, float(
        np.pi * r * r * sq * (2.0 * hl + 4.0 / 3.0 * r))


def surface(p_cm, voxel, sigma, level):
    """Marching-cubes surface of a particle cloud: CIC splat (bincount --
    np.add.at is ~10x slower at 900k points) on a voxel grid anchored to
    absolute coordinates, Gaussian blur, fixed absolute iso level."""
    if len(p_cm) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    lo = np.floor((p_cm.min(0) - 4 * voxel) / voxel) * voxel
    dims = np.maximum(((p_cm.max(0) + 4 * voxel - lo) / voxel).astype(int) + 2, 8)
    f = (p_cm - lo) / voxel
    i0 = np.clip(np.floor(f).astype(np.int64), 0, dims - 2)
    fr = (f - i0).astype(np.float32)
    grid = np.zeros(int(np.prod(dims)), dtype=np.float64)
    for dx in (0, 1):
        wx = fr[:, 0] if dx else 1.0 - fr[:, 0]
        for dy in (0, 1):
            wy = fr[:, 1] if dy else 1.0 - fr[:, 1]
            for dz in (0, 1):
                wz = fr[:, 2] if dz else 1.0 - fr[:, 2]
                idx = ((i0[:, 0] + dx) * dims[1] + (i0[:, 1] + dy)) * dims[2] \
                    + (i0[:, 2] + dz)
                grid += np.bincount(idx, weights=wx * wy * wz,
                                    minlength=grid.size)
    grid = gaussian_filter(grid.reshape(dims).astype(np.float32), sigma)
    if grid.max() <= level:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v, fc, _, _ = marching_cubes(grid, level=level)
    return (v * voxel + lo).astype(np.float32), fc.astype(np.int32)


def draw_tag(d, x, y, name, fnt, k=1):
    """Camera-name tag (dark box, teal outline) with its corner at (x, y);
    `k` scales the box for the higher-resolution stand-alone videos."""
    tw = d.textlength(name, font=fnt)
    d.rectangle([x, y, x + 12 * k + tw, y + 19 * k], fill=(8, 14, 18),
                outline=TEAL, width=k)
    d.text((x + 6 * k, y + 2 * k), name, font=fnt, fill=(215, 228, 232))


def font(name, size):
    import matplotlib
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name)
    return ImageFont.truetype(path, size)


class Layout:
    """Title bar + three labelled panels + caption, on a near-black canvas."""

    def __init__(self, pw, ph, header=40, footer=34):
        self.pw, self.ph, self.header, self.footer = pw, ph, header, footer
        self.W, self.H = 3 * pw, header + ph + footer
        self.f_title = font("DejaVuSans-Bold.ttf", 18)
        self.f_clock = font("DejaVuSansMono.ttf", 15)
        self.f_small = font("DejaVuSans.ttf", 13)
        self.f_label = font("DejaVuSansMono-Bold.ttf", 12)

    def compose(self, panels, names, title, clock, right, caption):
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        for k, (p, n) in enumerate(zip(panels, names)):
            x0 = k * self.pw
            img.paste(p, (x0, self.header))
            d.rectangle([x0, self.header, x0 + self.pw - 1,
                         self.header + self.ph - 1], outline=TEAL, width=1)
            draw_tag(d, x0 + 8, self.header + 8, n, self.f_label)
        cy = self.header // 2
        d.rectangle([14, cy - 6, 26, cy + 6], fill=TEAL)
        d.text((38, cy), title, font=self.f_title, fill=(238, 242, 244),
               anchor="lm")
        d.text((self.W // 2, cy), clock, font=self.f_clock,
               fill=(205, 214, 218), anchor="mm")
        d.text((self.W - 16, cy), right, font=self.f_small,
               fill=(128, 142, 148), anchor="rm")
        d.text((16, self.header + self.ph + self.footer // 2), caption,
               font=self.f_small, fill=(128, 142, 148), anchor="lm")
        return img


def card_guides(z):
    """Cartoon card proxies for a shuffle motion file (gen_shuffle_motion.py):
    NOT a simulation, only where the generator assumed the cards to be, so
    the hands can be judged against them. Two tented packets (straight slabs
    on card 0's recorded edges, `packet_thickness_cm` along `stack_dir`) that
    thin out from the bottom during the riffle, the released cards as a flat
    pile per side on the table, and the pile's outer ends following the
    raking fingertips.
    Returns frame -> [(verts (8,3), faces (12,3), color)] for 4 boxes."""
    sc = json.loads(str(z["scene"]))
    if "stack_dir" not in sc:
        raise SystemExit("--guides: this motion file predates the scene keys "
                         "the guides read; regenerate it")
    tl = np.asarray(z["target_left"], dtype=float)
    ph = {str(n): (int(a), int(b)) for n, a, b in
          zip(z["phase_names"], z["phase_start"], z["phase_end"])}
    L, D = sc["card_cm"]
    thick = sc["packet_thickness_cm"]
    T0, G0 = np.array(sc["T0"]), np.array(sc["G0"])
    # the recorded targets are contact points somewhere on the end faces (the
    # finger one on the middle finger's line): card 0's edges move with them
    off_T = T0 - np.array(sc["thumb_contact"])
    off_G = (G0 - np.array(sc["finger_contact"])
             - np.array([0.0, sc["finger_dy"]["middle"], 0.0]))
    T1, G1 = np.array(sc["T_bow"]), np.array(sc["G_bow"])
    yc, floor = sc["yc"], sc["floor_z"]
    F = len(tl)
    box_f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5],
                      [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
                      [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    hidden = np.tile(np.array([[0.0, 0.0, -500.0]], np.float32), (8, 1))

    def box(c, ax, half):
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                            for k in (-1, 1)], dtype=float) * half
        return (c + corners @ np.array(ax)).astype(np.float32)

    # outer end of the left pile: the fingertips stop it during the riffle,
    # then the raking fingertips push it in
    x_out = np.full(F, G1[0])
    low = (tl[:, 6] == 1.0) & (tl[:, 5] < sc["rake_z"] + 0.6)
    for f in range(1, F):
        x_out[f] = max(x_out[f - 1], tl[f, 3]) if low[f] else x_out[f - 1]
    r0, r1 = ph["riffle"]

    def frame(i):
        r = float(np.clip((i - r0 + 1) / (r1 - r0 + 1), 0.0, 1.0))
        if i < ph["bow"][0]:
            T, G = T0, G0
        elif i <= ph["bow"][1]:
            T, G = tl[i, 0:3] + off_T, tl[i, 3:6] + off_G
        else:
            T, G = T1, G1
        out = []
        for sx, col in ((1.0, (0.78, 0.13, 0.11)), (-1.0, (0.12, 0.27, 0.80))):
            M = np.array([sx, 1.0, 1.0])
            if r < 1.0:   # the cards still held: the upper (1 - r) of the packet
                u = (G - T) / np.linalg.norm(G - T)
                n = np.array([u[2], 0.0, -u[0]])      # stacking direction
                h = 0.5 * thick * (1.0 - r)
                c = 0.5 * (T + G) + n * (thick - h)
                v = box(c, [u, [0, 1, 0], n], [0.5 * np.linalg.norm(G - T), 0.5 * D, h])
                out.append((v * M, box_f, col))
            else:
                out.append((hidden, box_f, col))
            if r > 0.0:   # the released cards, flat on the table
                h = 0.5 * thick * r
                z0 = floor + 0.05 + (0.0 if sx > 0 else 0.6 * thick * r)
                c = np.array([x_out[i] + 0.5 * L, yc, z0 + h])
                v = box(c, np.eye(3), [0.5 * L, 0.5 * D, h])
                out.append((v * M, box_f, col))
            else:
                out.append((hidden, box_f, col))
        return out

    return frame


def load_bread_run(run, args):
    """A tear run directory: particle plys + per-frame hand files. Returns
    (frame_ids, n_frames, joints_at, bread_meshes, right_text)."""
    def by_number(pattern, rx):
        out = {}
        for path in glob.glob(os.path.join(run, pattern)):
            m = re.search(rx, os.path.basename(path))
            if m:
                out[int(m.group(1))] = path
        return out

    # pair particle and hand files by the frame NUMBER in their names (a
    # list index would mis-pair runs saved with --save-every > 1)
    plys = by_number("sim_*.ply", r"sim_(\d+)\.ply$")
    hands = by_number("hand_*.npz", r"hand_(\d+)\.npz$")
    frame_ids = sorted(set(plys) & set(hands))
    n_frames = len(frame_ids)
    h0 = np.load(hands[frame_ids[0]])
    offset, scale = h0["map_offset"], float(h0["map_scale"])
    def to_cm(p_dom):
        return ((p_dom - offset) / scale) * 100.0

    p0 = to_cm(read_ply(plys[frame_ids[0]]))
    crust, loaf_vol = capsule_crust_mask(p0.copy(), args.crust_cm)
    spacing = (loaf_vol / len(p0)) ** (1.0 / 3.0)
    level = args.iso_frac * (args.voxel / spacing) ** 3
    print(f"[triview] {len(p0)} particles ({crust.sum()} crust), spacing "
          f"{spacing:.3f} cm, voxel {args.voxel} cm, iso level {level:.2f}")
    cache = os.path.join(run, "triview_cache",
                         f"v{args.voxel:.3f}_s{args.sigma:.2f}_i{args.iso_frac:.2f}"
                         f"_c{args.crust_cm:.2f}")
    os.makedirs(cache, exist_ok=True)

    def src_id(path):
        st = os.stat(path)
        return [os.path.basename(path), st.st_size, st.st_mtime_ns]

    def stamp(i):
        # everything a cached surface depends on: the source particles (and
        # frame 0, which defines the crust mask and the spacing), the
        # reconstruction parameters, and the code version. A re-simulated
        # run, changed parameters or an old cache layout all miss.
        return json.dumps({
            "version": CACHE_VERSION, "ply": src_id(plys[i]),
            "ply0": src_id(plys[frame_ids[0]]), "voxel": args.voxel,
            "sigma": args.sigma, "iso_frac": args.iso_frac,
            "crust_cm": args.crust_cm}, sort_keys=True)

    def bread_meshes(i):
        path = os.path.join(cache, f"f{i:04d}.npz")
        want = stamp(i)
        try:
            with np.load(path) as z:
                if str(z["stamp"]) == want:
                    return [(z["v0"], z["f0"], CRUMB_COL),
                            (z["v1"], z["f1"], CRUST_COL)]
        except (FileNotFoundError, KeyError, ValueError, OSError,
                zipfile.BadZipFile):
            pass  # missing, stale, or truncated by an interrupted run
        p = to_cm(read_ply(plys[i]))
        keep = p[:, 2] > -0.2  # cull sub-table stragglers
        v0, f0 = surface(p[keep & ~crust], args.voxel, args.sigma, level)
        # the crust is a thin shell, not a solid: its blurred density peaks
        # lower, so its surface sits at half the interior level
        v1, f1 = surface(p[keep & crust], args.voxel, args.sigma, 0.5 * level)
        tmp = f"{path}.{os.getpid()}.tmp.npz"  # atomic: never a half file
        np.savez(tmp, v0=v0, f0=f0, v1=v1, f1=f1, stamp=np.array(want))
        os.replace(tmp, path)
        return [(v0, f0, CRUMB_COL), (v1, f1, CRUST_COL)]

    def joints_at(i):
        return np.load(hands[i])["joint_q"]

    right = f"keyframe replay  |  CD-MPM bread, {len(p0) / 1000.0:.0f}k particles"
    return frame_ids, n_frames, joints_at, bread_meshes, right


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="mp4 (default <run>/triview.mp4)")
    ap.add_argument("--frames", default=None,
                    help="comma list: render only these frames as PNGs "
                         "(composite + one per panel) into --png-dir")
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--cams", default=None,
                    help="json list of {name,pos,target,fov} overriding the "
                         "three default cameras")
    ap.add_argument("--panel", type=int, nargs=2, default=[640, 340])
    ap.add_argument("--motion", default=None,
                    help="npz with joint_q [F, n_q] and fps: render the robot "
                         "alone replaying it (run_dir is then only the "
                         "output directory)")
    ap.add_argument("--guides", action="store_true",
                    help="with --motion from gen_shuffle_motion.py: draw "
                         "cartoon card proxies (packets, pile) at the "
                         "generator's contact targets -- not a simulation")
    ap.add_argument("--isolated", action="store_true",
                    help="also write one stand-alone video per camera, "
                         "<run>/triview_<NAME>.mp4, at --iso-size")
    ap.add_argument("--iso-size", type=int, nargs=2, default=[1280, 680],
                    help="stand-alone video size (same aspect as --panel)")
    ap.add_argument("--no-labels", action="store_true",
                    help="no camera-name tag on the stand-alone videos")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="30 = real time (one saved frame per 1/30 s)")
    ap.add_argument("--voxel", type=float, default=0.20, help="cm")
    ap.add_argument("--sigma", type=float, default=0.9, help="voxels")
    ap.add_argument("--iso-frac", type=float, default=0.5,
                    help="iso level as a fraction of the interior density")
    ap.add_argument("--crust-cm", type=float, default=0.7)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--shadow-extents", type=float, default=220.0,
                    help="half-size (cm) of the camera-centered shadow box")
    ap.add_argument("--case-id", default="05")
    ap.add_argument("--title", default="YAM + SHARPA")
    ap.add_argument("--right-text", default=None)
    ap.add_argument("--caption", default=None)
    args = ap.parse_args()

    run = args.run_dir
    cams = json.load(open(args.cams)) if args.cams else DEFAULT_CAMS
    assert len(cams) == 3, "need exactly three cameras"
    if args.motion:
        os.makedirs(run, exist_ok=True)
        with np.load(args.motion, allow_pickle=True) as z:
            motion_q = np.asarray(z["joint_q"], dtype=np.float64)
            frame_dt = 1.0 / float(z["fps"])
            phases = ([(str(n), int(a), int(b)) for n, a, b in
                       zip(z["phase_names"], z["phase_start"], z["phase_end"])]
                      if "phase_names" in z.files else [])
            guides_at = card_guides(z) if args.guides else None
        frame_ids = list(range(len(motion_q)))
        n_frames = len(frame_ids)
        right_default = "IK-generated joint trajectory  |  robot only"
        caption_default = ("Kinematic preview: YAM arms + Sharpa hands "
                           "replaying an IK-generated joint trajectory "
                           "(no objects simulated).")

        def joints_at(i):
            return motion_q[i]

        def bread_meshes(i):
            return []

        def phase_at(i):
            for n, a, b in phases:
                if a <= i <= b:
                    return n
            return ""
    else:
        frame_dt = FRAME_DT
        guides_at = None
        phase_at = lambda i: ""  # noqa: E731
        (frame_ids, n_frames, joints_at, bread_meshes,
         right_default) = load_bread_run(run, args)
        caption_default = (
            "Simulation: keyframe-authored grasp, lift and tear; bread is a "
            "CD-MPM phase-field fracture solid.   YAM arms: recorded "
            "joint-space trajectory replay.")

    disable_distance_fog()
    pw, ph = args.panel
    # each view is rendered ONCE per frame at the largest size needed; the
    # composite panels are downsampled from the stand-alone frames, so the
    # four videos are pixel-consistent
    vw_, vh_ = args.iso_size if args.isolated else (pw, ph)
    if abs(vw_ / vh_ - pw / ph) > 0.01:
        raise SystemExit("--iso-size must have the same aspect as --panel "
                         "(the cameras are framed for it)")
    bench = YamSharpaBench(with_table=True)
    viewer = ViewerGL(width=vw_ * args.ss, height=vh_ * args.ss, headless=True)
    viewer.set_model(bench.model)
    rnd = viewer.renderer
    rnd.draw_sky = False
    rnd.sky_upper = rnd.sky_lower = rnd.background_color = (0.0, 0.0, 0.0)
    rnd.draw_shadows = not args.no_shadows
    # Two more meter-scale defaults that break in this centimeter rig: the
    # camera-relative spotlight cone leaves the far cameras without direct
    # light (grey table), and the shadow map only spans +-10 units around
    # the camera, so no shadow ever reaches the scene.
    rnd.spotlight_enabled = False
    try:
        rnd.shadow_extents = args.shadow_extents
    except AttributeError:
        rnd._shadow_extents = args.shadow_extents
    viewer.begin_frame(0.0)  # pipeline warm-up: the first end_frame is blank
    viewer.log_state(bench.state)
    viewer.end_frame()

    MAX_V, MAX_F = 600000, 1200000

    def render_views(i):
        bench.joint_q[:] = joints_at(i)
        bench._fk()
        meshes = bread_meshes(i)
        padded = []
        for v, f, col in meshes:
            if len(v) > MAX_V or len(f) > MAX_F:
                raise RuntimeError(f"surface too dense: {len(v)} v {len(f)} f")
            if len(v) == 0:
                v = np.zeros((1, 3), np.float32)
            vp = np.repeat(v[:1], MAX_V, axis=0); vp[: len(v)] = v
            fp = np.zeros((MAX_F, 3), np.int32); fp[: len(f)] = f
            padded.append((wp.array(vp, dtype=wp.vec3),
                           wp.array(fp.flatten(), dtype=wp.int32), col))
        boxes = [(wp.array(np.ascontiguousarray(v, dtype=np.float32), dtype=wp.vec3),
                  wp.array(f.flatten(), dtype=wp.int32), col)
                 for v, f, col in (guides_at(i) if guides_at else [])]
        out = []
        for cam in cams:
            viewer.set_camera(*look_at(cam["pos"], cam["target"]))
            viewer.camera.fov = float(cam["fov"])
            viewer.begin_frame(i * frame_dt)
            viewer.log_state(bench.state)
            for k, (vw, fw, col) in enumerate(padded):
                name = f"/bread{k}"
                q = viewer._qualify(name) if hasattr(viewer, "_qualify") else name
                if q in getattr(viewer, "objects", {}):
                    # MeshGL uploads indices only once; marching cubes changes
                    # topology every frame, so force the re-upload
                    viewer.objects[q].indices = None
                viewer.log_mesh(name, vw, fw, color=col, roughness=0.85,
                                backface_culling=False)
            for k, (vw, fw, col) in enumerate(boxes):
                viewer.log_mesh(f"/guide{k}", vw, fw, color=col, roughness=0.9,
                                backface_culling=False)
            viewer.end_frame()
            im = Image.fromarray(read_frame_gl(viewer))
            if args.ss > 1:
                im = im.resize((vw_, vh_), Image.LANCZOS)
            out.append(im)
        return out

    def to_panels(views):
        return [v if v.size == (pw, ph) else v.resize((pw, ph), Image.LANCZOS)
                for v in views]

    lay = Layout(pw, ph)
    names = [c["name"] for c in cams]
    title = f"{args.case_id}   {args.title}".strip()
    right = args.right_text or right_default
    caption = args.caption or caption_default

    def composite(i, views):
        clock = (f"simulation t = {i * frame_dt:6.2f} s   |   "
                 f"{args.fps * frame_dt:.2f}x playback")
        ph_name = phase_at(i)
        cap = f"[{ph_name}]   {caption}" if ph_name else caption
        return lay.compose(to_panels(views), names, title, clock, right, cap)

    tag_k = max(1, round(vw_ / pw))
    f_tag = font("DejaVuSansMono-Bold.ttf", 12 * tag_k)

    def standalone(view, name):
        if args.no_labels:
            return view
        view = view.copy()
        draw_tag(ImageDraw.Draw(view), 8 * tag_k, 8 * tag_k, name, f_tag, tag_k)
        return view

    if args.frames:
        png_dir = args.png_dir or os.path.join(run, "triview_png")
        os.makedirs(png_dir, exist_ok=True)
        for i in [int(s) for s in args.frames.split(",")]:
            views = render_views(i)
            for n, v in zip(names, views):
                (standalone(v, n) if args.isolated else v).save(
                    os.path.join(png_dir, f"{n}_{i:04d}.png"))
            composite(i, views).save(
                os.path.join(png_dir, f"triview_{i:04d}.png"))
            print(f"[triview] frame {i} -> {png_dir}")
        viewer.close()
        return

    def open_writer(path):
        return iio.get_writer(path, fps=args.fps, codec="libx264", quality=9,
                              macro_block_size=1, pixelformat="yuv420p")

    out = args.out or os.path.join(run, "triview.mp4")
    writer = open_writer(out)
    iso = {}
    if args.isolated:
        stem = os.path.splitext(out)[0]
        iso = {n: open_writer(f"{stem}_{re.sub(r'[^A-Za-z0-9_-]', '_', n)}.mp4")
               for n in names}
    for k, i in enumerate(frame_ids):
        views = render_views(i)
        writer.append_data(np.asarray(composite(i, views)))
        for n, v in zip(names, views):
            if n in iso:
                iso[n].append_data(np.asarray(standalone(v, n)))
        if k % 25 == 0:
            print(f"[triview] frame {k}/{n_frames}")
    writer.close()
    for w in iso.values():
        w.close()
    viewer.close()
    print(f"[triview] wrote {out} ({n_frames} frames @ {args.fps} fps, "
          f"{lay.W}x{lay.H})")
    for n in iso:
        print(f"[triview] wrote {os.path.splitext(out)[0]}_{n}.mp4 "
              f"({vw_}x{vh_})")


if __name__ == "__main__":
    main()
