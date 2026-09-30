#!/usr/bin/env python
"""Mesh-render a coupling run: robot URDF meshes (Newton ViewerGL, headless EGL)
plus the MPM particles as a point cloud, composited per saved frame."""

import argparse
import glob
import math
import os
import re
import sys

import os as _os
import pyglet
pyglet.options["headless"] = True  # must precede newton.viewer import
# EGL device 0 on this machine renders black (wedged); 1-7 work
pyglet.options["headless_device"] = int(_os.environ.get("EGL_DEVICE", "1"))

import imageio.v2 as iio
import numpy as np
import warp as wp

sys.path[:0] = [os.path.dirname(os.path.realpath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "robot")]
from newton.viewer import ViewerGL  # noqa: E402
from robot_scene import YamSharpaBench  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage.measure import marching_cubes  # noqa: E402


def crust_mask(p0_cm, shell_cm=0.7):
    """True for particles that start within `shell_cm` of the loaf surface.

    The initial loaf is convex (a capsule), so distance to the convex hull's
    planes is the distance to the surface. Particle order is stable across
    saved frames, so a frame-0 mask stays valid for the whole run."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(p0_cm)
    # hull.equations: [A | b] with A x + b <= 0 inside; max over planes is
    # the (negative) distance to the surface for interior points
    d = (p0_cm @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
    return d > -shell_cm


def particles_to_surface(p_cm, voxel=0.7, sigma=1.3, level=None):
    """Marching-cubes surface (cm) from a particle cloud (cm).

    The voxel grid is anchored to absolute voxel-grid coordinates (not the
    frame's AABB) and the iso level is a fixed absolute density, so the
    reconstructed surface does not flicker frame to frame."""
    lo = np.floor((p_cm.min(0) - 4 * voxel) / voxel) * voxel
    hi = p_cm.max(0) + 4 * voxel
    dims = np.maximum(((hi - lo) / voxel).astype(int) + 2, 8)
    grid = np.zeros(dims, dtype=np.float32)
    # trilinear (CIC) splatting: nearest-voxel binning makes the density field
    # jump whenever a particle crosses a voxel boundary, so deforming regions
    # pop frame to frame; spreading each particle over its 8 neighbours makes
    # sub-voxel motion change the field (and the extracted surface) smoothly
    f = (p_cm - lo) / voxel
    i0 = np.clip(np.floor(f).astype(int), 0, dims - 2)
    fr = f - i0
    for dx in (0, 1):
        wx = fr[:, 0] if dx else 1.0 - fr[:, 0]
        for dy in (0, 1):
            wy = fr[:, 1] if dy else 1.0 - fr[:, 1]
            for dz in (0, 1):
                wz = fr[:, 2] if dz else 1.0 - fr[:, 2]
                np.add.at(grid, (i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz),
                          (wx * wy * wz).astype(np.float32))
    grid = gaussian_filter(grid, sigma)
    if level is None:
        # expected interior density: particle spacing 0.25 cm -> ~22 particles
        # per 0.7 cm voxel, blurred; a fixed fraction of that is stable
        level = 1.3
    verts, faces, normals, _ = marching_cubes(grid, level=level)
    return (verts * voxel + lo).astype(np.float32), faces.astype(np.int32), \
        (-normals).astype(np.float32)


def read_frame_gl(viewer):
    """Direct glReadPixels readback (newton 1.5 get_frame hangs on CUDA-GL
    interop when the EGL context and warp device sit on different GPUs)."""
    import ctypes
    from newton._src.viewer.viewer_gl import RendererGL
    gl = RendererGL.gl
    w, h = viewer.renderer._screen_width, viewer.renderer._screen_height
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, viewer.renderer._frame_fbo)
    gl.glFinish()
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    buf = (ctypes.c_ubyte * (w * h * 3))()
    gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, buf)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)[::-1].copy()


def look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return wp.vec3(*pos), pitch, yaw


def read_ply(path):
    with open(path, "rb") as f:
        h = b""
        while not h.endswith(b"end_header\n"):
            h += f.readline()
        n = int(re.search(rb"element vertex (\d+)", h).group(1))
        return np.fromfile(f, dtype=np.float32, count=n * 3).reshape(n, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20, help="stride 1 @ 20 fps = real time")
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[-58.0, -18.0, 42.0])
    ap.add_argument("--cam-target", type=float, nargs=3, default=[-9.0, 30.0, 12.0])
    ap.add_argument("--point-radius", type=float, default=0.55, help="cm")
    ap.add_argument("--voxel", type=float, default=0.5, help="cm")
    ap.add_argument("--iso-frac", type=float, default=0.75,
                    help="iso level as fraction of interior density; higher cuts "
                         "through the low-density fuzz that BC shear leaves on "
                         "deformed surfaces")
    ap.add_argument("--sigma", type=float, default=2.0, help="voxels")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--points", action="store_true",
                    help="render particles as points instead of a reconstructed surface")
    ap.add_argument("--bread", action="store_true",
                    help="two-tone bread look: golden crust shell (particles "
                         "starting near the loaf surface) over pale rough "
                         "crumb, exposed at torn faces")
    ap.add_argument("--crust-cm", type=float, default=0.7,
                    help="initial shell thickness classified as crust")
    args = ap.parse_args()

    plys = sorted(glob.glob(os.path.join(args.run_dir, "sim_*.ply")))
    h0 = np.load(os.path.join(args.run_dir, "hand_000000.npz"))
    offset, scale = h0["map_offset"], float(h0["map_scale"])

    # multi-object recordings carry per-object particle slices + colors
    meta_path = os.path.join(args.run_dir, "meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        slices = [tuple(s) for s in meta["slices"]]
        obj_colors = [tuple(c) for c in meta["colors"]]
        spacing = float(meta.get("spacing_cm", 0.5))
    else:
        slices, obj_colors, spacing = None, [(0.93, 0.55, 0.18)], 0.5

    crust = None
    if args.bread:
        p0 = read_ply(plys[0])
        p0_cm = ((p0 - offset) / scale) * 100.0
        crust = crust_mask(p0_cm, shell_cm=args.crust_cm)
        print(f"[bread] {crust.sum()} crust / {(~crust).sum()} crumb particles")
    CRUMB_COL = (0.93, 0.88, 0.73)  # pale porous interior
    CRUST_COL = (0.79, 0.52, 0.22)  # baked golden-brown shell
    # iso level = half the expected interior particle count per voxel; this is
    # what the old hard-coded 1.3 was for 0.5 cm spacing (0.5*(0.7/0.5)^3=1.37)
    VOXEL = args.voxel
    iso_level = args.iso_frac * (VOXEL / spacing) ** 3

    bench = YamSharpaBench(with_table=True)
    viewer = ViewerGL(width=1280, height=960, headless=True)
    viewer.set_model(bench.model) if hasattr(viewer, "set_model") else None
    viewer.set_camera(*look_at(args.cam_pos, args.cam_target))
    if hasattr(viewer, "renderer") and hasattr(viewer.renderer, "draw_shadows"):
        viewer.renderer.draw_shadows = not args.no_shadows
    # pipeline warm-up: the first end_frame renders nothing
    viewer.begin_frame(0.0)
    viewer.log_state(bench.state)
    viewer.end_frame()

    n_max = len(read_ply(plys[0]))
    pts_wp = wp.zeros(n_max, dtype=wp.vec3)
    colors = wp.array(np.tile(np.array([[0.95, 0.60, 0.15]], dtype=np.float32),
                              (n_max, 1)), dtype=wp.vec3)

    out = args.out or os.path.join(args.run_dir, "mesh_render.mp4")
    writer = iio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1)
    frames = 0
    for k, ply in enumerate(plys[::args.stride]):
        i = k * args.stride
        hand = os.path.join(args.run_dir, f"hand_{i:06d}.npz")
        if not os.path.exists(hand):
            continue
        h = np.load(hand)
        bench.joint_q[:] = h["joint_q"]
        bench._fk()

        p_dom = read_ply(ply)
        p_all = ((p_dom - offset) / scale) * 100.0  # domain -> rig m -> cm
        # (points, color, iso_scale): the crust is a thin shell, not a solid
        # interior, so its reconstruction level must sit lower
        if crust is not None:
            groups = [(p_all[~crust], CRUMB_COL, 1.0),
                      (p_all[crust], CRUST_COL, 0.5)]
        elif slices is None:
            groups = [(p_all, obj_colors[0], 1.0)]
        else:
            groups = [(p_all[s:e], obj_colors[k], 1.0)
                      for k, (s, e) in enumerate(slices)]

        viewer.begin_frame(i / 30.0)
        viewer.log_state(bench.state)
        if args.points:
            p_cm = p_all[p_all[:, 2] > -0.2]
            pts_wp.assign(np.resize(p_cm.astype(np.float32), (n_max, 3)))
            viewer.log_points("/mpm", pts_wp, radii=args.point_radius, colors=colors)
        else:
            # log_mesh reuses GPU buffers by name and rejects size changes;
            # pad to fixed-size buffers with degenerate triangles
            MAX_V, MAX_F = 400000, 800000
            for k, (p_cm, col, iso_scale) in enumerate(groups):
                p_cm = p_cm[p_cm[:, 2] > -0.2]  # cull sub-table stragglers
                v, f, n = particles_to_surface(p_cm, voxel=VOXEL,
                                               sigma=args.sigma,
                                               level=iso_level * iso_scale)
                if len(v) > MAX_V or len(f) > MAX_F:
                    raise RuntimeError(f"surface too dense: {len(v)} verts {len(f)} tris")
                vp = np.repeat(v[:1], MAX_V, axis=0); vp[: len(v)] = v
                fp = np.zeros((MAX_F, 3), dtype=np.int32); fp[: len(f)] = f
                # newton's MeshGL uploads the index buffer only on the FIRST
                # log_mesh ("no topology changes") — but marching cubes changes
                # topology every frame, and stale frame-0 indices over fresh
                # vertices render as shredded scanline stripes. Force the EBO
                # to re-upload, and let the viewer recompute normals to match.
                name = f"/soft{k}"
                qname = viewer._qualify(name) if hasattr(viewer, "_qualify") else name
                if qname in getattr(viewer, "objects", {}):
                    viewer.objects[qname].indices = None
                viewer.log_mesh(name, wp.array(vp, dtype=wp.vec3),
                                wp.array(fp.flatten(), dtype=wp.int32),
                                color=col, roughness=1.0 if args.bread else 0.7,
                                backface_culling=False)
        viewer.end_frame()
        writer.append_data(read_frame_gl(viewer))
        frames += 1
    writer.close()
    viewer.close()
    print("wrote", out, f"({frames} frames)")


if __name__ == "__main__":
    main()
