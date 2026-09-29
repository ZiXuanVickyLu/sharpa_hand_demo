#!/usr/bin/env python3
"""Derive a restart variant of a hand-shuffle config (IS 5.x restart):
   python3 tool/make_restart_variant.py app/config/hand_shuffle_mb.json hand_shuffle_mb_b1 \
       --step 4290 --from hand_shuffle_mb/x_4290.npy [--mu-card 0.1] [--mu-hand 0.8] [--mu-plane 0.5] [--frames N]
The variant keeps the scene (bodies, colliders, keyframes) and changes friction only; it writes
app/config/<name>.json with simulation.restart set and output_dir <name>. Frames of the base run
before the restart step are linked into the new output directory so the renderer sees one sequence."""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("name")
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--from", dest="src", required=True, help="positions npy, relative to output/")
ap.add_argument("--velocities", default=None)
ap.add_argument("--mu-card", type=float, default=None)
ap.add_argument("--mu-hand", type=float, default=None)
ap.add_argument("--mu-plane", type=float, default=None)
ap.add_argument("--frames", type=int, default=None)
ap.add_argument("--abort-capped", type=int, default=5)
ap.add_argument("--inner-max", type=int, default=None)
ap.add_argument("--link-from", default=None, help="base run directory (under output/) whose earlier frames are linked")
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = json.load(open(a.base))
sim = c["simulation"]
sim["output_dir"] = a.name
sim["restart"] = {"step": a.step, "positions": a.src, "velocities": a.velocities}
sim["abort_after_capped_steps"] = a.abort_capped
if a.frames is not None: sim["frames"] = a.frames
if a.inner_max is not None: c["newton"]["inner_max_iters"] = a.inner_max
if a.mu_card is not None:
    for b in c["bodies"]: b["friction"] = a.mu_card
if a.mu_plane is not None:
    for p in c["planes"]: p["friction"] = a.mu_plane
if a.mu_hand is not None:
    for f in c["contact_table"]["friction"]: f[2] = a.mu_hand
out = os.path.join(root, "app", "config", a.name + ".json")
json.dump(c, open(out, "w"), indent=1)
print("wrote", out)
if a.link_from:
    src = os.path.join(root, "output", a.link_from); dst = os.path.join(root, "output", a.name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in os.listdir(src):
        if f.startswith("x_") and f.endswith(".npy") and int(f[2:-4]) < a.step or f == "surface_faces.npy":
            t = os.path.join(dst, f)
            if not os.path.lexists(t): os.symlink(os.path.join(src, f), t); n += 1
    print("linked", n, "earlier frames from", src)
