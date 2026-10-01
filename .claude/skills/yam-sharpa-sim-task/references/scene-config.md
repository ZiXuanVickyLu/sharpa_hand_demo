# The scene config and the solver settings that worked

The full schema is `doc/al-ipc-implementation-spec.md` section 5.3; unknown keys are errors, keys starting
with `_` are comments. Start every new scene from the solver blocks of `app/config/hand_shuffle_phys.json`
and change only what the task needs. Write a generator script for the scene (as `tool/gen_hand_shuffle.py`)
so every parameter has one place and the config regenerates byte for byte.

## Blocks and the values used by the demos

```jsonc
"simulation": { "dt": 0.002, "frames": N, "gravity": [0, 0, -9.81], "output_dir": "output/<name>",
                "save_bgeo": true, "save_npy": true, "save_every": 10, "checkpoint_every": 500,
                "abort_after_capped_steps": 5, "restart": null }
"materials":  { "card": { "model": "cloth_stvk", "mu_mem": 3.46e5, "nu": 0.3, "density": 800,
                          "thickness": 0.0003, "k_bend": 1.85e-3 } }        // k_bend = half the paper card
"bodies":     [ { "name": "card_L00", "type": "cloth", "file": "<asset>/<name>_card_left.obj",
                  "material": "card", "thickness": 0.0003, "friction": 0.02, "self_collision": false,
                  "move_to_origin": false, "scale": [1,1,1], "rotation": [...], "translation": [...],
                  "dirichlet": [ { "select": [vertex ids], "motion": { "type": "keyframes",
                                   "file": "<asset>/<name>_rows_L00a.npy", "substeps": 10 },
                                   "release_frame": 550 } ] } ]
"colliders":  [ { "name": "hand_left", "type": "tri", "file": "<asset>/<name>_left.obj", "thickness": 0.0,
                  "friction": 0.5, "motion": { "type": "keyframes", "file": "<asset>/<name>_left_kf.npy",
                  "substeps": 10 } }, { "name": "thumb_left", ... "friction": 0.7 } ]
"planes":     [ { "point": [0, 0, 0.005], "normal": [0, 0, 1], "friction": 0.5 } ]   // the 5 mm mat
"contact_table": { "exclude": [], "friction": [["card_L00", "thumb_left", 0.7], ...] }  // per pair
"contact":    { "d_hat": 2e-4, "epsilon": 1e-3, "K_min": 2, "max_outer_iters": 500, "mu_mode": "diag_max",
                "mu_scale": 0.1, "mu_max": 0.03, "stall": { "max_adaptations": 0, ... },
                "friction": { "enable": true, "eps_v": 1e-4, "normal_force": "paper" },
                "self_collision_default": false, "inversion_free": "off" }
"newton":     { "inner_max_iters": 4, "increment_velocity_tol": 5e-4, "line_search": { "batched": true } }
"linear_solver": { "type": "pcg", "preconditioner": "block_jacobi", "rigid_coarse": true, "graph": true,
                   "fused": true, "rel_tol": 1e-4, "max_iters": 10000 }
```

Why these: `dt` 2 ms with 10 substeps per 20 ms recording frame keeps the keyframed colliders from
jumping more than a card thickness per step; `mu_max` 0.03 kg bounds the AL penalty so resting cards do not
jitter (the paper's estimate leaves a multiplier window below the Newton accuracy); the "cipc" rule
(`inner_max_iters` 4 with the increment tolerance) is the accepted-step rule that made the wound rods and the
resting cards work; `abort_after_capped_steps` 5 turns an infeasible configuration into a stopped run
instead of hours at the cap; the mat (`planes`, 5 mm) exists because with a thinner one the feet of
released cards wedge under the rounded fingertips.

## Materials for other objects

- Thin flexible things (cards, lace strips, cloth): `cloth_stvk`; `mu_mem` sets in-plane stiffness,
  `k_bend` the bending. Cards: 0.3 mm thick, `k_bend` 1.85e-3 (the user wanted them softer than paper's
  3.7e-3). A lace is a strip 5-8 mm wide, 1-2 mm thick, `k_bend` higher than a card's; test it resting and
  hanging over an edge before any hand touches it. `self_collision: true` where the object folds on itself.
- Soft volumes: `tet` bodies with `SNH` (E 1e5-1e6, nu 0.3, density 1000, `snh_lambda_reparam`) as in the
  trapped-balls demo; `inversion_free` "auto" turns on for NH bodies.
- Rigid fixtures (a shoe body, a board, a container): static `tri` colliders (`motion: null`), with their
  own friction and a `contact_table` row per object pair. There are no rigid dynamic bodies; a "rigid"
  object that must move is either a stiff tet body or a keyframed collider.

## The grasp-formation trick (prescribe to form, then release)

Pulling a thin object into a hand by contact alone is fragile. The demos prescribe the object's first
contact rows for the grasp's formation, then free them:

- The scene generator selects the rows (vertex ids) that touch the hand (for cards: the two upper-inner
  rows), computes their trajectory by ray-casting the rows onto the moving collider surface for each
  frame (`thumb_ride_*` cache in the asset dir), and writes a keyframe file per row group:
  float [frames, n_selected, 3] in metres, one row per recording frame including the synthetic pre-roll.
- Each group gets a `release_frame` (in simulation steps). Release in two stages: the inner row first
  (`--hinge-frames` before), the edge row last (`--release-all-at F`: for the shuffle, frame 185 = step 550,
  20 frames into the bow). Freed with the object fully loaded, a row leaves the hand with a snap; freed
  one row at a time it hinges first and the second stage is gentle.
- The friction model's normal force lags by one step (spec section 13), so the first free step of a row has
  no friction: a hand-over between a prescribed row and a contact-held row must be staged (the second row
  freed 10 frames before the edge), or the object slips at the switch.
- The synthetic pre-roll: the non-thumb links slide in horizontally from 4-5 mm out over the first 15
  frames so the first simulated state has no interpenetration between the hand and the prescribed rows
  (`--preroll`, `--preroll-dist`). Keep it for any grasp that starts closed on the object.

The honest demo prescribes nothing after the grasp forms. The scripted alternative (every row prescribed
until a scheduled release, the thumb not even a collider) is the moving-boundary demo; say which one a
new scene is in its README row.

## Colliders from the bake

`--thumb-collider` makes the thumb links a collider of their own with `--mu-thumb` (0.7 in the demo) so its
friction differs from the pads' (0.5); the generator can also split a region of a collider into its own
collider (`--thumb-cap R --mu-cap MU`, by distance from a reference point at a reference frame) when one
part needs another friction. Collider `thickness` 0 and `--thumb-clear 0.2e-3 --thumb-perp 0.15e-3` were the
clearances the prescribed rows kept from the thumb surface; an inflated collider (3 mm offset) intrudes into
the packet's faces before reaching the edges, so keep offsets at zero and handle clearance in the rows.

## Restarting instead of resimulating

`checkpoint_every` 500 writes `ckpt_x/v_<step>.npy`. For a change that only affects the sequence after step
S (a different push, a different friction from S on): `tool/make_restart_variant.py <base config> <name>
--step S --from output/<run>/ckpt_x_S.npy --velocities output/<run>/ckpt_v_S.npy` writes a config with
`simulation.restart` (paths relative to `output/`) and output dir `<name>`; link the base run's earlier
frames into the new output directory so the sequence reads as one. The restart requires the same vertex
layout: a new bake of the same rig and the same object meshes qualifies; a changed mesh does not.
