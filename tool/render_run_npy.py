#!/usr/bin/env python
"""Single-view video of any run written with simulation.save_npy (x_<step>.npy + surface_faces.npy),
rendered head-less with the Newton GL viewer (no Houdini needed). Packages: tool/render/requirements.txt;
a GPU with EGL.

  EGL_DEVICE=0 python3 tool/render_run_npy.py --run output/output/<run> --config app/config/<config>.json \
      --out output/demo/<clip>.mp4 --up z --cam-pos 0.3 0.6 0.4 --cam-target 0 0.3 0

Bodies are coloured per body, colliders light grey. --cut "x>0.0" (axis, < or >, value; scene
coordinates) removes the COLLIDER triangles whose centroid satisfies it (a look inside a container);
--every N renders every N-th saved frame."""
import argparse, json, os, re, sys
import numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), d) for d in ("render", "robot")]
os.environ.setdefault("EGL_DEVICE", "0")
import render_triview as rt                      # noqa: E402  (headless EGL set up on import)
from render_triview import look_at, read_frame_gl, disable_distance_fog   # noqa: E402
import imageio.v2 as iio                         # noqa: E402
import warp as wp                                # noqa: E402
import newton                                    # noqa: E402
from newton.viewer import ViewerGL               # noqa: E402
from PIL import Image, ImageDraw                 # noqa: E402

PALETTE = [(0.85, 0.47, 0.16), (0.20, 0.45, 0.80), (0.30, 0.65, 0.35), (0.75, 0.25, 0.30), (0.55, 0.40, 0.75), (0.85, 0.75, 0.25)]

def count_obj_vertices(path):
    return sum(1 for l in open(path) if l.startswith("v "))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--up", default="z", choices=["y", "z"])
    ap.add_argument("--cam-pos", type=float, nargs=3, required=True); ap.add_argument("--cam-target", type=float, nargs=3, required=True)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720]); ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--fps", type=float, default=50.0); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--cut", default=None); ap.add_argument("--frames", default=None)
    ap.add_argument("--ground", type=float, default=None, help="draw a ground slab with its top at this height (scene units, along --up)")
    ap.add_argument("--ground-size", type=float, default=4.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--n-collider-vertices", type=int, default=-1, help="override (default: counted from the collider obj files)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(a.config))
    F = np.load(os.path.join(a.run, "surface_faces.npy")).astype(np.int32)
    steps = sorted(int(m.group(1)) for m in (re.match(r"x_(\d+)\.npy$", f) for f in os.listdir(a.run)) if m)
    X0 = np.load(os.path.join(a.run, "x_%d.npy" % steps[0]))
    n = len(X0)
    n_coll = a.n_collider_vertices
    if n_coll < 0:
        n_coll = 0
        for c in cfg.get("colliders", []):
            if c.get("type", "tri") != "tri": continue
            p = c["file"]
            for base in (os.path.dirname(a.config), os.path.join(root, "asset"), root):
                if os.path.exists(os.path.join(base, p)): p = os.path.join(base, p); break
            n_coll += count_obj_vertices(p)
    is_coll = (F >= n - n_coll).all(1)
    Fb, Fc = F[~is_coll], F[is_coll]
    def to_view(X):                                   # scene -> viewer (Z up)
        return X if a.up == "z" else np.stack([X[:, 0], -X[:, 2], X[:, 1]], axis=1)
    if a.cut and len(Fc):
        m = re.match(r"\s*([xyz])\s*([<>])\s*(-?[0-9.eE+-]+)", a.cut); ax = "xyz".index(m.group(1)); v = float(m.group(3))
        cen = X0[Fc].mean(1)[:, ax]
        Fc = Fc[~((cen > v) if m.group(2) == ">" else (cen < v))]
    # split the body surface by connected body: bodies are contiguous vertex blocks only in the free range, so
    # colour by connected component instead
    parent = np.arange(n)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for t in Fb:
        r0 = find(t[0])
        for k in (1, 2):
            r1 = find(t[k])
            if r1 != r0: parent[r1] = r0
    comp = np.array([find(t[0]) for t in Fb]) if len(Fb) else np.zeros(0, int)
    groups = [Fb[comp == c] for c in sorted(set(comp.tolist()))]
    disable_distance_fog()
    W, H = a.size
    viewer = ViewerGL(width=W * a.ss, height=H * a.ss, headless=True)
    viewer.set_model(newton.ModelBuilder().finalize())
    rnd = viewer.renderer
    rnd.draw_sky = False; rnd.sky_upper = rnd.sky_lower = rnd.background_color = (0.04, 0.045, 0.055)
    rnd.draw_shadows = True; rnd.spotlight_enabled = False
    ext = float(np.abs(to_view(X0)).max()) * 2.5
    try: rnd.shadow_extents = ext
    except AttributeError: rnd._shadow_extents = ext
    g_arrays = [wp.array(g.flatten(), dtype=wp.int32) for g in groups]
    c_array = wp.array(Fc.flatten(), dtype=wp.int32) if len(Fc) else None
    if a.ground is not None:
        s, z1 = a.ground_size, a.ground; z0 = z1 - 0.02 * s
        gv = wp.array(np.array([[sx * s, sy * s, z] for z in (z0, z1) for sy in (-1, 1) for sx in (-1, 1)], np.float32), dtype=wp.vec3)
        gf = wp.array(np.array([[4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int32).flatten(), dtype=wp.int32)
    cam_pos, cam_tgt = to_view(np.array([a.cam_pos]))[0], to_view(np.array([a.cam_target]))[0]
    def render(step):
        X = to_view(np.load(os.path.join(a.run, "x_%d.npy" % step)).astype(np.float32))
        V = wp.array(np.ascontiguousarray(X), dtype=wp.vec3)
        viewer.set_camera(*look_at(cam_pos.tolist(), cam_tgt.tolist())); viewer.camera.fov = a.fov
        viewer.begin_frame(step * float(cfg["simulation"]["dt"]))
        if a.ground is not None:
            viewer.log_mesh("/ground", gv, gf, color=(0.22, 0.24, 0.27), roughness=0.95, backface_culling=False)
        for i, g in enumerate(g_arrays):
            viewer.log_mesh("/body_%d" % i, V, g, color=PALETTE[i % len(PALETTE)], roughness=0.7, backface_culling=False)
        if c_array is not None:
            viewer.log_mesh("/colliders", V, c_array, color=(0.72, 0.74, 0.78), roughness=0.5, backface_culling=False)
        viewer.end_frame()
        im = Image.fromarray(read_frame_gl(viewer))
        if a.ss > 1: im = im.resize((W, H), Image.LANCZOS)
        if a.label:
            d = ImageDraw.Draw(im); d.text((16, H - 34), "%s   frame %d   t = %.2f s" % (a.label, step, step * float(cfg["simulation"]["dt"])), fill=(235, 235, 235))
        return im
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.frames:
        stem = os.path.splitext(a.out)[0]
        for s in [int(x) for x in a.frames.split(",")]:
            render(s).save("%s_f%04d.png" % (stem, s)); print("wrote %s_f%04d.png" % (stem, s))
        viewer.close(); return
    w = iio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    todo = steps[::a.every]
    for k, s in enumerate(todo):
        w.append_data(np.asarray(render(s)))
        if k % 50 == 0: print("[render] %d/%d" % (k, len(todo)))
    w.close(); viewer.close(); print("wrote", a.out)

if __name__ == "__main__":
    main()
