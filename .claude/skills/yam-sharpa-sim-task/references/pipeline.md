# The pipeline: commands, timings, file formats

All commands run from the repository root. The Python tools use the environment of `pyproject.toml`
(`uv sync` once); the solver binaries come from `cmake --preset release` (+ `release-gui` for the viewer).

## Stages at a glance

| stage | command | time | output |
| --- | --- | --- | --- |
| IK | `uv run python tool/ik/<generator>.py [options] --out output/ik/<name>` | 2-3 min, CPU | `shuffle_motion.npz`, `report.json` |
| bake | `uv run python tool/bake_hand_shuffle.py --motion output/ik/<name>/shuffle_motion.npz --out asset/<asset>` | 1 min | asset directory (below) |
| scene | `python3 tool/<generator>.py --asset <asset> --name <name> [options]` | 1-10 min | `app/config/<name>.json` + per-run files in the asset dir |
| run | `build/release/cs_run app/config/<name>.json [--frames N]` | ~0.2 s/step alone | `output/output/<name>/` |
| viewer | `DISPLAY=:1 build/release-gui/cs_view app/config/<name>.json --play` | same pace | same output |
| restart | `python3 tool/make_restart_variant.py <base.json> <newname> --step S --from output/<run>/ckpt_x_S.npy --velocities output/<run>/ckpt_v_S.npy` then run `app/config/<newname>.json` | from S on | `output/output/<newname>/` |
| render | `EGL_DEVICE=0 uv run python tool/render_hand_shuffle.py --run output/output/<name> --config app/config/<name>.json --sim asset/<asset>/<name>_sim.json --cams asset/<asset>/cams.json --out output/demo/<name>.mp4 --isolated` | minutes | composite + 3 views |

The per-step cost grows with the number of contact pairs (`|C|` in the log line) and collapses when
the solver struggles: a healthy step is 2-5 outer iterations; 500 outer iterations with `alpha_mean
0.000` means the line search fails, and `abort_after_capped_steps` (5) ends the run with exit code 4.

## The recording (`shuffle_motion.npz`)

Lengths in cm, rig frame (z up, table z = 0, the hands at y around 30), 50 frames per second.

| key | shape | meaning |
| --- | --- | --- |
| `joint_q` | [F, 56] | left arm 6 + hand 22, then right arm 6 + hand 22 (the bench's joint order) |
| `fps` | scalar | 50 |
| `phase_names`, `phase_start`, `phase_end` | [P] | inclusive frame ranges of the segments |
| `body_q` | [F, 86, 7] | world link transforms, xyz (cm) + quaternion xyzw; `body_names` [86] |
| `joint_names` | [56] | |
| `target_left`, `target_right` | [F, 7] | what the hand was steered to (NaN where a phase does not steer that contact) |
| `table_exact_cm`, `arm_gap_cm`, `arm_gap_exact_cm`, `thumb_finger_gap_cm` | [F] | clearances by mesh vertices / covering spheres (NaN on frames not checked) |
| `scene` | json string | the generator's record: object pose, contact points, every option (the bake copies it into `hand_meta.json` as `scene_cm` / `scene_m`) |

`report.json` is a per-phase table: worst IK error (cm), table clearance, left-right gap, thumb-to-finger
gap, smallest joint-limit margin, bound events, peak joint rates. Read it before baking: an IK error
above a millimetre or a bound event inside a tracked phase is a kink the cards will feel.

Frame to step: the simulation starts at recording frame `start - preroll` (the shipped demos: hold frame
145 minus a 15-frame synthetic pre-roll, so recording frame f is simulation step 10 (f - 130)); the
sidecar `<name>_sim.json` written by the scene generator carries `start`, `preroll`, `substeps`.

## The asset directory (bake output + generator output)

From the bake: `hand_left.obj`, `hand_right.obj` (full-resolution hand meshes at the first baked frame),
`hand_left_kf.npy`, `hand_right_kf.npy` (float32 [frames, n_vertices, 3], metres), `hand_meta.json`
(`first`, `last`, `fps`, `phases`, `links` = per-link vertex ranges per side, `motion` = the recording's
file name relative to the directory, `scene_cm`, `scene_m`), `motion.npz` (a copy of the recording),
`cams.json` (the renderer's cameras, copied if present next to the recording), `thumb_tip_*.npz` (the
rendered thumb-tip surface per frame, used by the scripted-grip mode only).

From the scene generator (`<name>_*`): the decimated hand colliders and their keyframes
(`<name>_left.obj`, `<name>_left_kf.npy` ...), separate thumb colliders when `--thumb-collider`, the object
meshes, one keyframe file per prescribed vertex group (`<name>_rows_L00a.npy`: [frames, n_selected, 3]),
and the sidecar `<name>_sim.json` (start, preroll, substeps, release frames per object, the options).

## The run directory (`output/output/<name>/`)

`x_<step>.npy` every `save_every` steps: float32 [n_vertices, 3] in metres, global numbering = the bodies in
config order, then the colliders in config order (count each file's vertices to split it; `scripts/
body_state.py` does). `surface_faces.npy` once (int32 [m, 3]). `surface_<step>.bgeo` when `save_bgeo`.
`ckpt_x_<step>.npy` / `ckpt_v_<step>.npy` every `checkpoint_every` steps (float64). `stats.csv` one row per
step: outer/inner iterations, PCG iterations, pair counts, CCD hits, `alpha_mean`, `mu`, `max_penetration`,
per-phase timings. `resolved_config.json`: the config with every default filled in.

The viewer (`cs_view`) writes the same output as `cs_run` and runs at the same pace; use it for the user's
final look, `cs_run` for everything else.

## Reading a run quickly

- Progress: the last `frame N  outer k  newton m  pcg/avg  |C| pairs  alpha_mean` line of the log.
- Health: `outer` 2-6 and `alpha_mean` near 1 are normal; `outer 500` or `alpha_mean 0.00` for consecutive
  steps means an infeasible configuration (something prescribed pushes into something that cannot move),
  not a slow solve. The abort rule will stop the run; find the pinch instead of raising the cap.
- Objects: `scripts/body_state.py <run> <config> <steps...>` for per-body heights, extents and motion;
  `scripts/section_view.py` for an x-z (or y-z) section of bodies and colliders around a plane.
- Two recordings or two runs that should agree: `scripts/motion_diff.py a.npz b.npz`.

## Packaging

Ship `app/config/<name>.json`, `asset/<asset>/` (bake inputs + per-run files, recording named relatively),
the clips in `output/demo/`, and the README row with steps, time, and the prescribed/physical statement.
Regenerate the config once more from the shipped asset and diff it against the shipped one (it must be
identical up to the name); render once from the repository root with the documented command. Do not ship
run outputs or caches (`.gitignore` covers `output/*` except `output/demo/`).
