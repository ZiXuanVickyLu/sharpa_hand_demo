# contact_solver demos

A CUDA implementation of the barrier-free, augmented-Lagrangian contact solver of
*Robust and Efficient Penetration-Free Elastodynamics without Barriers* (Zheng, Luo, Li, 2025,
arXiv 2512.12151), packaged with four demo scenes and the assets they need:

| demo | config | what it is | steps | time on an RTX 4090 / 5090 |
| --- | --- | --- | --- | --- |
| rods | `app/config/rod_twist_paper.json` | IPC's "rods twist" (Li et al. 2020, Table 1) on IPC's 53 k-vertex `rod300x33.msh`, 100 s | 4000 | 0.1-0.5 s per step (30 s of the 100 s verified) |
| animal well | `app/config/animal_well.json` | [Z25] Fig. 12: 71 MB tet mesh of animals dropped into a pool | 300 | about 13 min |
| trapped balls | `app/config/trapped_balls.json` | [Z25] Fig. 1 / 21: squishy balls pressed in a cylinder | 300 | about 4 min |
| hand shuffle, moving boundary | `app/config/hand_shuffle_mb.json` | two Sharpa hands riffle-shuffle 54 playing cards; the hands are keyframed colliders from a recorded motion, but the thumb is not one: each card's top rows are a Dirichlet boundary condition that rides the thumb until that card's scheduled release, so the grip and the release timing are prescribed and only the fall, landing and push are contact | 6660 | about 32 min |
| hand shuffle, physical grip | `app/config/hand_shuffle_phys.json` | the same shuffle by real physical interaction: the thumb is a collider and the cards are held and released by its contact and friction, not by a scripted animation | 3630 | about 13 min |


## Requirements

* Linux, an NVIDIA GPU with at least 4 GB free (the paper-mesh rods use 3 GB), a driver for the toolkit below.
* CUDA toolkit 12.8 or newer (developed on 13.0), CMake 3.24 or newer, Ninja, gcc 12 or newer (C++20).
* Network access during the first configure: Eigen, nlohmann/json, spdlog, cuda-api-wrappers and
  (for the viewer) polyscope are fetched by CMake's FetchContent into `build/*/_deps`.
* For the viewer (`cs_view`): OpenGL and X11 development packages
  (`libgl1-mesa-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`).
* Python 3 with numpy for the scene generators (`tool/gen_*.py`). The motion generator, the bake and the clip renderer
  under `tool/` need more (warp, newton, scipy, imageio, ...), declared in `pyproject.toml` and installed with
  [uv](https://docs.astral.sh/uv/): `uv sync` once, then `uv run python tool/...`.

The CUDA architecture is auto-detected from the installed GPU at configure time. If you move the
folder to a machine with a different GPU, delete `build/` and configure again.

## Build

```bash
cmake --preset release                       # headless runner, build/release
cmake --build --preset release -j

cmake --preset release-gui                   # the polyscope viewer, build/release-gui
cmake --build --preset release-gui --target cs_view -j
```

Asset, config and output paths are compiled into the binaries relative to this folder
(`asset/`, `app/config/`, `output/`); the config argument is a path relative to the current directory, so run the
commands below from the repository root.

## Run

Headless:

```bash
build/release/cs_run app/config/animal_well.json
build/release/cs_run app/config/trapped_balls.json
build/release/cs_run app/config/hand_shuffle_mb.json
build/release/cs_run app/config/hand_shuffle_phys.json          # also writes .bgeo surfaces
build/release/cs_run <config> --frames 50      # stop early
```

Each run writes to `output/<output_dir of the config>/`: `stats.csv` with one row per step
(outer and Newton iterations, PCG iterations, contact pairs, CCD hits, the accepted fraction
alpha, beta, the penalty mu, energy, maximum penetration, per-phase timings, guard counters),
Houdini `.bgeo` surfaces when `save_bgeo` is on, `x_<step>.npy` positions when `save_npy` is on,
and `ckpt_x/v_<step>.npy` checkpoints every `checkpoint_every` steps. `tool/summarize_stats.py`
prints a per-run summary of `stats.csv`.

Live, in the viewer:

```bash
build/release-gui/cs_view app/config/rod_twist_paper.json --play
build/release-gui/cs_view app/config/hand_shuffle_mb.json --play --record output/frames
```

`--play` starts stepping at once; the panel has pause, single step, steps per drawn frame and a
"pause at frame" field. `--record DIR` saves one PNG per drawn frame (make a video with ffmpeg).

Restart from a checkpoint: set `simulation.restart` in the config (`tool/make_restart_variant.py`
derives such a config from a run and a step) and run as usual.

## Regenerating the scene configs

The configs are generated files; the generators are the reference for every parameter.

```bash
python3 tool/gen_rod_twist.py --name rod_twist_paper --mesh rod/rod300x33.msh --mu-max 6.67 --ramp 0.5 --save-every 4 --no-bgeo
python3 tool/gen_hand_shuffle.py --asset hand_shuffle_mb --name hand_shuffle_mb \
        --ride-stride 3 --newton-rule cipc --inner-max 4 --mat 0.005 --mu-card 0.05      # about 10 min (numpy ray casting)
python3 tool/gen_hand_shuffle.py --asset hand_shuffle_phys --name hand_shuffle_phys \
        --ride-stride 3 --newton-rule cipc --inner-max 4 --mat 0.005 --mu-card 0.02 --mu-mat 0.5 --bend-scale 0.5 \
        --thumb-collider --mu-thumb 0.7 --thumb-surface collider --thumb-clear 0.2e-3 --thumb-perp 0.15e-3 \
        --release-all-at 185 --hinge-frames 10 --save-bgeo
```

## The hand shuffle, in more detail

**What is prescribed in the moving-boundary demo.** The two hands are keyframed triangle colliders,
but the thumb is left out of them. Each card instead carries one Dirichlet set of 14 vertices, its two
upper-inner rows, whose keyframes make them ride the rendered thumb tip; those vertices stay prescribed
until the card's own `release_frame`, and the release frames are scheduled per card, from step 860 to
4190, one card at a time alternating hands. So in this demo the grip and the timing of every release are
boundary conditions; what the contact solver decides is everything after a release: the fall, the
landing on the other hand's cards, and the push. The physical-grip demo below removes that: it prescribes
the same rows only to form the grasp (released at steps 450 and 550, 20 frames into the bow) and makes
the thumb a collider, so from then on no card has a prescribed vertex.

`asset/hand_shuffle_mb/` holds everything the scene needs:

* `hand_left.obj`, `hand_right.obj`, `hand_left_kf.npy`, `hand_right_kf.npy`: the two Sharpa hands as
  triangle colliders with one vertex position per recorded frame, baked from a recorded
  inverse-kinematics motion (`motion.npz`, `hand_meta.json`, `cams.json`). The motion itself was
  produced by a separate robot-simulation pipeline that is not part of this folder; the baked
  keyframes are all the solver needs.
* `thumb_tip_*.npz`, `thumb_ride_*.npz`: the thumb-tip surface per frame and a cache used by the
  generator to let the top card rows ride the thumb until released.
* `hand_shuffle_mb_*`: the per-run files the shipped config points at (decimated hand
  colliders and keyframes, card meshes, per-card release rows). `gen_hand_shuffle.py` rewrites
  them together with the config.

The scene runs at 2 ms substeps with a small card-to-card friction, a bounded contact penalty
(`contact.mu_max`) and a thin mat; each was chosen against a measured failure (cards catching tip to tip
in the push, resting cards jittering, card feet wedging under the fingertips). Its clips are `output/demo/hand_shuffle_mb.mp4` (three views side by side)
and `hand_shuffle_mb_CAM1/_CAM2/_HEAD.mp4`. To look at a run here, use `cs_view` or load the `.bgeo` files in Houdini.

**Rendering the clips.** `tool/render_hand_shuffle.py` made every clip in `output/demo/`: it replays the
hand IK recording (`asset/<asset>/motion.npz`) on the robot model (`asset/robot/`, URDF and meshes) and draws
the simulated cards from the run's `x_<step>.npy`, into the three-view composite and, with `--isolated`, one
video per camera. It needs the Python environment of `pyproject.toml` (`uv sync`) and a GPU with EGL:

```bash
EGL_DEVICE=0 uv run python tool/render_hand_shuffle.py \
    --run output/output/hand_shuffle_phys --config app/config/hand_shuffle_phys.json \
    --sim asset/hand_shuffle_phys/hand_shuffle_phys_sim.json --cams asset/hand_shuffle_phys/cams.json \
    --out output/demo/hand_shuffle_phys.mp4 --isolated
```

and the same with `hand_shuffle_mb` in the four names for the moving-boundary demo (`--frames 0,60,120
--png-dir DIR` renders single frames instead). `tool/bake_hand_shuffle.py`, which bakes a recording into the
hand colliders and keyframes of an asset directory, uses the same environment (`uv run python tool/bake_hand_shuffle.py ...`).

`asset/hand_shuffle_phys/` is a second bake of the same rig for the physical-grip variant, from a motion
generated with extra options of that pipeline's IK: the bow is done by the thumb alone, the riffle ends
with the thumb's last knuckle opening, and the hands push the piles together without lifting off. The
bare thumb is a collider of its own; the cards' top rows ride it only to form the grasp and are freed
early in the bow, and from then on holding, releasing, falling, landing and the push are contact and
friction. The directory holds the files the run reads (`hand_shuffle_phys_*`) and the inputs
`gen_hand_shuffle.py` needs to regenerate them (`motion.npz`, `hand_meta.json`, the full-resolution bake
`hand_left/right.obj` and `_kf.npy`). The run writes `surface_<step>.bgeo` every 10 steps; its four
clips are `output/demo/hand_shuffle_phys.mp4` (three views side by side) and `_CAM1`, `_CAM2`, `_HEAD`.

## Regenerating the hand motions (IK)

`tool/ik/gen_shuffle_motion.py` is the inverse-kinematics generator that produced both recordings: it plans the two
arms and hands (the robot model of `asset/robot/`) around the card packets, frame by frame at 50 Hz, and writes
`shuffle_motion.npz` (joint trajectory, link transforms, targets, clearances) plus a report. The bake then turns a
recording into an asset directory, and `tool/gen_hand_shuffle.py` builds the config from that. Both shipped
recordings reproduce from this folder alone (the physical-grip one bit for bit, the moving-boundary one to
below 1e-6 rad):

```bash
uv sync                                                    # once: the Python environment of pyproject.toml
# the moving-boundary demo's recording (0.24 s per card, the left hand 0.12 s behind)
uv run python tool/ik/gen_shuffle_motion.py --riffle-step 0.24 --release-lag 0.12 --out output/ik/mb
# the physical-grip demo's recording
uv run python tool/ik/gen_shuffle_motion.py \
    --riffle-step 0.32 --release-lag 0.4 --packet-thick 0.65 --thumb-inset 0.45 --bow-dir card \
    --bow-fingers 0 --bow-thumb 0.7 --riffle-lift 0.3 --direct-push --push-pause 0.5 --square-slack 1.6 \
    --riffle-knuckle 30 --riffle-knuckle-from 25 --riffle-knuckle-frames 200 --riffle-knuckle-mode freeze \
    --riffle-knuckle-press 1.0 400 --riffle-knuckle-press-from 45 --riffle-cut 85 --out output/ik/phys
uv run python tool/bake_hand_shuffle.py --motion output/ik/phys/shuffle_motion.npz --out asset/hand_shuffle_phys
```

Each run takes about 2.5 minutes on the CPU. The physical-grip options, in order: the riffle's pace and the lag
between the hands, the packet stacked at the cards' own thickness, the thumb target 4.5 mm over the packet's end
face, the bow done by the thumb alone along the card axis, the thumb's 3 mm withdrawal during the slide, the push
with no lift-off (a 0.5 s pause, stopping 16 mm short of the card's half length), the last knuckle opening 30
degrees over 200 frames from frame 25 of the riffle with a 1 cm descent from frame 45, and the riffle cut after 85
frames. `gen_shuffle_motion.py --help` documents every option; each was added for a measured effect on the cards.

## Documents

* `doc/al-ipc-math-spec.md`: the algorithm as implemented, with every deviation from the paper dated and measured.
* `doc/al-ipc-implementation-spec.md`: data model, kernels, the JSON configuration schema (section 5.3).
* `doc/acceleration-inner-rule-and-mu-bound.md`: the two changes that made the wound rods and the resting cards work (inner Newton rule, penalty bound), written as a reproduction guide.
* `doc/references.md`: the papers and repositories referenced.
* The paper: Zheng, Luo, Li 2025, https://arxiv.org/abs/2512.12151 (not included; see `doc/references.md`).

## Layout

```
CMakeLists.txt CMakePresets.json cmake/   build system (presets: release, release-gui, debug, ...)
ext/        vendored HouGeoIO (bgeo I/O), cnpy, warp_svd; FetchContent for the rest
src/        the solver: core, ccd, geometry, scene, gpu, fem, contact, linsys, solver, io
app/        cs_run (headless), cs_view (viewer), config/ (the demo scenes)
tool/       scene generators and analysis scripts; ik/ (the motion generator), robot/ (the robot model loader),
            render/ (the clip renderer's modules); pyproject.toml at the root declares their Python environment
asset/      meshes and keyframes of the demos
doc/        the math and implementation specifications, the acceleration guide, references
```

`asset/animal_well/fetch.sh` and `asset/trapped_balls/fetch.sh` record where the two large
meshes came from (libuipc's assets and the paper's supplementary repository); the files are
already here.

## License

This project's code, configurations and documents are released under the MIT License (`LICENSE`).
Third-party components keep their own licenses, found in their directories: `ext/cnpy` (MIT),
`ext/warp_svd` (NVIDIA Warp's SVD, Apache-2.0), `src/libculbvh` and `src/ccd/libcuGTE` (see their
`LICENSE` files), `ext/HouGeoIO`; the demo meshes came from the sources recorded in `asset/*/fetch.sh`
and `doc/references.md`, and the robot description under `asset/robot/` belongs to its manufacturers. `THIRD_PARTY_NOTICES.md` lists every
third-party component with its origin and license.
