#!/usr/bin/env python3
"""IPC's "Rods twist (100 s)" (Li et al. 2020, Fig. 4 and Table 1) on IPC's low-resolution rod.

   python3 tool/gen_rod_twist.py [--name rod_twist] [--seconds 100] [--newton-rule ipc|paper]

The set-up is IPC's own script (IPC/input/otherExamples/typical/rodsTwist.txt, the low-resolution
twin of paperExamples/4_rodsTwist.txt), read together with its source:

  * four rods `asset/rod/rod.msh` (= IPC's input/tetMeshes/rod.msh: 1 m long, radius 2 cm, 630
    vertices, 1722 tets) along x at (y, z) = (+-0.1, +-0.1); `size 1` rescales the scene so that
    its largest extent is 1 m, which it already is;
  * neo-Hookean, E = 1e4 Pa, nu = 0.4, density 1000 kg/m^3, no gravity, h = 0.025 s, 100 s;
  * script `twist` (AnimScripter.cpp, AST_TWIST): the vertices within handleRatio = 1 % of the
    scene's x-range from either end (the two end caps of every rod) are Dirichlet and rotate about
    the x axis through the scene's bounding-box centre at -0.4 pi rad/s (x min) and +0.4 pi rad/s
    (x max): 72 deg/s each way, 40 relative turns in 100 s;
  * self-collision on, no friction; d_hat = 1e-3 l and Newton tolerance 1e-2 l (m/s) with l the
    scene's bounding-box diagonal (1.056 m).

Ours: the same numbers. d_hat is our contact offset (the rods are kept 1.06 mm apart where IPC's
barrier starts acting at that distance), the Newton tolerance maps to newton.increment_velocity_tol
(an RMS over the free vertices where IPC takes the maximum), and the outer loop is [Z25]'s.
`--newton-rule paper` uses [Z25]'s inner rule instead (one accepted Newton step per outer iteration)."""
import argparse, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_msh_bbox(path):
    L = open(path).read().split("\n"); i = L.index("$Nodes"); hdr = [int(t) for t in L[i + 1].split()]
    p = i + 2; V = []
    for _ in range(hdr[0]):
        nn = int(L[p].split()[3]); p += 1 + nn
        V += [[float(t) for t in L[p + k].split()] for k in range(nn)]; p += nn
    V = np.array(V); return V.min(0), V.max(0)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", default="rod_twist")
    ap.add_argument("--mesh", default="rod/rod.msh",
                    help="rod mesh under asset/: rod/rod.msh (IPC's low-resolution rod, 630 vertices) or rod/rod300x33.msh\n"
                         "(the paper's: 13 333 vertices and 50 511 tets per rod, 53 k / 202 k for the four, IPC Table 1)")
    ap.add_argument("--seconds", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--offset", type=float, default=0.1, help="|y| = |z| of the rod axes (IPC: 0.1)")
    ap.add_argument("--rate", type=float, default=0.4, help="end-cap angular speed in units of pi rad/s (IPC: 0.4)")
    ap.add_argument("--model", default="NH", choices=["NH", "SNH"],
                    help="NH (default) is IPC's model. It needs the end-state inversion guard of MS 10.4 (2026-09-21): with the\n"
                         "earlier first-root rule the run died at the first buckling snap (frame 1624). SNH (stable neo-Hookean,\n"
                         "what [Z25] and libuipc use) needs no guard but is much softer in compression here: 29 k active pairs\n"
                         "and 1.7 s per step at frame 1300 against 2 k and 0.1 s with NH.")
    ap.add_argument("--mu-fixed", type=float, default=None,
                    help="fix the AL penalty stiffness (kg) instead of [Z25]'s estimate, a tenth of the largest diagonal entry of\n"
                         "M + h^2 K. With NH one strongly compressed element sets that maximum (0.077 at rest, 7.7 at J = 0.17,\n"
                         "4e5 and more once an element is nearly flat), and the huge penalty then crushes further elements.\n"
                         "0.08 is the rest-state value of this scene.")
    ap.add_argument("--mu-max", type=float, default=8.0,
                    help="upper bound of [Z25]'s penalty estimate (MS 11.1); 8 = 100 x the rest value of this scene. 0 = none.\n"
                         "Without it one nearly flat NH element after a buckling snap drives mu to 4e5 and beyond.")
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="seconds over which the end caps' rate rises from 0 (motion.ramp_time). Needed on the paper's mesh, where\n"
                         "the full first increment (4.4 mm) exceeds the 3.3 mm elements next to the caps; IPC starts at full rate.")
    ap.add_argument("--newton-rule", default="ipc", choices=["ipc", "paper"])
    ap.add_argument("--inner-max", type=int, default=8)
    ap.add_argument("--k-min", type=int, default=2, help="[Z25]: 2 without friction")
    ap.add_argument("--epsilon", type=float, default=1e-3,
                    help="outer termination: stop when beta = prod(1 - alpha_k) <= epsilon, the share of the step the anchor has\n"
                         "not taken ([Z25]: 1e-3, kept). The outer count is ln(epsilon) / ln(1 - alpha): 95 at alpha = 0.07 for 1e-3,\n"
                         "73 for 5e-3.")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--no-bgeo", action="store_true")
    a = ap.parse_args()
    lo, hi = read_msh_bbox(os.path.join(ROOT, "asset", a.mesh))
    ext = hi - lo
    scene_ext = np.array([ext[0], 2 * a.offset + ext[1], 2 * a.offset + ext[2]])
    l = float(np.linalg.norm(scene_ext))
    assert abs(scene_ext.max() - 1.0) < 1e-3, "IPC's `size 1` would rescale this scene"
    frames = int(round(a.seconds / a.dt)); deg = a.rate * 180.0
    bodies = []
    for k, (y, z) in enumerate([(-a.offset, -a.offset), (-a.offset, a.offset), (a.offset, -a.offset), (a.offset, a.offset)]):
        bodies.append({
            "name": "rod_%d" % k, "type": "tet", "file": a.mesh, "material": "rod", "move_to_origin": False,
            "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0, y, z], "initial_velocity": [0, 0, 0],
            "thickness": 0.0, "friction": 0.0, "self_collision": True,
            "dirichlet": [
                {"select": {"bbox_min": [0.0, 0.0, 0.0], "bbox_max": [0.01, 1.0, 1.0]},
                 "motion": {"type": "rotation", "axis": [1, 0, 0], "center": [0, 0, 0], "deg_per_second": -deg, "start_frame": 0, "end_frame": frames, "ramp_time": a.ramp}},
                {"select": {"bbox_min": [0.99, 0.0, 0.0], "bbox_max": [1.0, 1.0, 1.0]},
                 "motion": {"type": "rotation", "axis": [1, 0, 0], "center": [0, 0, 0], "deg_per_second": deg, "start_frame": 0, "end_frame": frames, "ramp_time": a.ramp}}]})
    cfg = {
        "simulation": {"dt": a.dt, "frames": frames, "gravity": [0, 0, 0], "output_dir": a.name,
                       "save_bgeo": not a.no_bgeo, "save_npy": True, "save_every": a.save_every,
                       "checkpoint_every": 400, "abort_after_capped_steps": 5},
        "materials": {"rod": ({"model": "SNH", "E": 1e4, "nu": 0.4, "density": 1000.0, "snh_lambda_reparam": True} if a.model == "SNH"
                              else {"model": "NH", "E": 1e4, "nu": 0.4, "density": 1000.0})},
        "bodies": bodies, "colliders": [], "planes": [],
        "contact": {"enable": True, "d_hat": 1e-3 * l, "epsilon": a.epsilon, "K_min": a.k_min, "max_outer_iters": 500,
                    "decay_factor": 0.9, "decay_remove_threshold": 0.01, "mu_mode": "diag_max", "mu_scale": 0.1,
                    "alpha_lower_bound": 1e-6,
                    "stall": {"iters": 50, "alpha": 1e-4, "mu_factor": 2.0, "d_hat_factor": 0.5, "max_adaptations": 0},
                    "ccd": {"s": 0.1, "max_iter": 100, "float_screen": False, "rebuild_quality_ratio": 1.5},
                    "inversion_free": "auto", "self_collision_default": True,
                    "friction": {"enable": False, "eps_v": 1e-3, "normal_force": "paper"},
                    "drop_parallel_edge_pairs": False},
        "newton": {"inner_max_iters": a.inner_max, "line_search": {"max_halvings": 30, "energy_tolerance": 1e-12, "batched": True},
                   "early_accept_velocity_tol": 0.0,
                   "increment_velocity_tol": (1e-2 * l if a.newton_rule == "ipc" else 0.0)},
        "linear_solver": {"type": "pcg", "preconditioner": "block_jacobi", "rel_tol": 1e-4, "max_iters": 10000, "check_interval": 8,
                          "graph": True, "fused": True, "fused_max_rows": 65536},
        "contact_table": {"exclude": [], "friction": []},
        "logging": {"level": "info", "stats_csv": "stats.csv", "timing": True, "debug_check_penetration": False}}
    if a.mu_max > 0: cfg["contact"]["mu_max"] = a.mu_max
    if a.mu_fixed is not None:
        cfg["contact"]["mu_mode"] = "fixed"; cfg["contact"]["mu_fixed"] = a.mu_fixed
    out = os.path.join(ROOT, "app", "config", a.name + ".json")
    json.dump(cfg, open(out, "w"), indent=1)
    print("wrote %s: 4 rods, %d frames at h = %g s, l = %.4f m, d_hat = %.3e m, Newton tol %s, end caps +-%.0f deg/s" % (
        out, frames, a.dt, l, 1e-3 * l, ("%.3e m/s" % (1e-2 * l)) if a.newton_rule == "ipc" else "paper rule", deg))

if __name__ == "__main__":
    main()
