---
name: yam-sharpa-sim-task
description: End-to-end recipe for making the two YAM arms with Sharpa hands perform a manipulation task in simulation with this repository's tools: the IK motion generator (tool/ik), the collider bake, the barrier-free AL contact solver (cs_run / cs_view) and the clip renderer. Use it whenever the user asks for a new hand-manipulation demo or task on this rig (shuffling, lacing a shoe, folding, stacking, pushing, pinching, tying, pouring cards or cloth...), asks to reproduce, modify or extend the card-shuffle demos, asks to "generate a motion" or "a recording" for the hands, or wants deformable objects (cards, cloth, laces, soft bodies) simulated against hand motion. Also use it when the user describes a target behaviour in words and expects the agent to get there autonomously through IK and simulation, even if they never say "IK", "contact solver" or "skill".
---

# Making the YAM + Sharpa rig do a task in simulation

This skill is the distilled method of the card-shuffle work in this repository: a recorded hand motion
drives the contact solver as moving colliders, deformable objects respond through contact and friction,
and the loop "plan the motion, simulate, measure, adjust" is run until the task is done. It is written
for a new task given as a prompt (for example "use the YAM + Sharpa to lace up a shoe"), so it says what
to decide, what to build, how to check each stage cheaply, and which failures are already known.

Read the references as you reach each stage; they hold the details this file only names:

- `references/pipeline.md`: the commands, timings, file formats and outputs of every stage.
- `references/ik-framework.md`: how to write a task-specific IK generator on `tool/ik/yam_kin.py`.
- `references/scene-config.md`: the scene JSON, the grasp-formation trick, the solver settings that worked.
- `references/lessons.md`: the measured failure modes of this rig and solver, with numbers.
- `references/task-mapping.md`: how to map a new task onto the recipe, worked for lacing and folding.
- `scripts/`: `motion_diff.py`, `body_state.py`, `section_view.py`, `watch_run.sh` (analysis and run helpers).

## 1. The ground truth you work in

- Two 6-DoF YAM arms with 22-joint Sharpa hands, mirrored about x = 0, bases 0.5334 m apart, a table at
  z = 0. The IK works in centimetres in the rig frame (z up); the solver works in metres in the same
  frame, the task objects usually sitting on a thin mat plane just above z = 0.
- The hands reach the solver only as keyframed triangle colliders (one vertex position per 20 ms frame,
  interpolated over 10 substeps of 2 ms). They have infinite force: whatever they are driven into, the
  objects must get out of the way or the solver fails. Objects only ever feel the hands through contact
  and friction.
- Stage costs on one RTX-class GPU: IK 2 to 3 minutes (CPU), bake 1 minute, scene generation 1 to 10
  minutes, simulation about 0.2 s per step when it runs alone (a 3600-step sequence in 13 minutes), the
  clip render a few minutes. Two simulations on one GPU each run at half speed, so run one at a time.
- Everything is deterministic: the same config reproduces the same run, and the IK reproduces a recording
  bit for bit, which is what makes "keep the verified frames, change what comes after" a workable method.

## 2. Decide what is physical before building anything

Write down, for the task, which interactions must be the result of contact and which may be scripted.
The honest version of a task is one where the behaviour that defines it is physical: in the shuffle, the
cards are held and released by the thumb's friction, not by a release schedule. Scripting is still used in
one place, and should be: forming the grasp. Pulling a thin object into a hand by contact alone is
fragile and slow to tune, so the first contact rows of each object are prescribed to ride the hand for the
first second and then freed in two stages (the second row first, the edge last), after which nothing on the
object's side is prescribed. State this split in the deliverable's README; the user will ask.

Then split the motion into phases with a verb each (reach, approach, grasp, bow, slide, open, withdraw,
push, home...). Every phase is a segment of the IK schedule and a frame range in the simulation, and you
will measure each one separately.

## 3. Build the scene side first, cheaply

Before any hand moves, make the objects exist and rest: a config with the objects on the mat and the hands
parked, 100 to 300 steps. This validates the meshes, the units, the material (`cloth_stvk` for cards, laces
and cloth; `SNH`/`NH` tets for soft bodies), the mat height, self-collision and the contact settings, for a
minute of GPU time. Reuse the solver block of `app/config/hand_shuffle_phys.json` as the starting point; its
values were chosen against failures (see `references/scene-config.md`). Write a generator script for the
scene (as `tool/gen_hand_shuffle.py` does) rather than editing JSON by hand: you will regenerate it dozens of
times with one parameter changed, and the generator is the reproducible record of every parameter.

## 4. Write the IK generator for the task

Copy the skeleton of `tool/ik/gen_shuffle_motion.py` and keep its structure: a scene record (object pose,
contact points, in cm), key poses solved with multi-start and settled, a schedule of segments
`(name, seconds, kind, payload)`, a per-frame tracking loop that solves each frame from the previous one
with a continuity term, hand-overs that carry a residual and fade it over 15 frames, and the diagnostics
(table clearance, arm-to-arm clearance, joint-limit margins, bound events). The framework's terms are few
(`Point`, `Axis`, `Above`, `Posture`) and enough: a fingertip pad at a target point with its pad normal
facing the object, the palm kept above the table, a posture preference to keep joints off their limits.

Make every new behaviour an additive option with a default that reproduces the previous motion. This is not
pedantry: once the user has looked at a run and liked its first N frames, every later variant must keep
those frames identical, and `scripts/motion_diff.py` must say "first differing frame = where you intended".
A change that alters earlier frames by accident costs a run and the user's trust.

Keep the generated motion plausible as a hand motion: the diagnostics must show no table penetration, the
two hands apart, and no joint pinned at a bound while it tracks (a joint arriving at a bound kinks every
other joint). A grasp that only works with the fingertips 1 cm into the table is a cheat the user will
reject, as the final push of the shuffle showed.

## 5. Bake, generate, run short, measure

`tool/bake_hand_shuffle.py` turns a recording into the asset directory (colliders, keyframes, metadata);
your scene generator turns the asset into the config; `cs_run` runs it. Do not run the whole sequence the
first time: run to the end of the first phase you doubt (`--frames N`), then measure. Measurement means
numbers from `x_<step>.npy`, not impressions from a video: per-object heights, extents, which objects moved
when, section views of the contact geometry (`scripts/body_state.py`, `scripts/section_view.py`). The
questions that decided the shuffle were all geometric ("the card ends sit 8 to 14 mm above the tip on the
flank", "the fingertip is 16 mm across and the face it pushes is 15 mm tall"), and they came from these
views, never from the clips.

Iterate one variable at a time, and stop a run the moment it has shown its failure; a run that has failed
has nothing more to teach, and the GPU is the bottleneck. Keep every run's output until the user has looked
at it (they inspect the `.bgeo` sequences themselves); delete only what they say to delete. When a change
only affects the sequence after step S, restart from the checkpoint at S (`tool/make_restart_variant.py`)
instead of resimulating from zero: a 10-minute test instead of 40.

Expect the hard part to be tuning, not building. Friction is the most sensitive quantity in the system
(0.02 against 0.05 between cards changed the bow itself), and most "it should trivially work" ideas fail
on a geometric detail. Budget for it, say so to the user, and report what was measured when something does
not work, in numbers, so that the next decision is theirs with the facts in hand.

## 6. Finish and package

The deliverable is a config plus an asset directory plus the recording, with the clips, and a README row
that says in one line what is prescribed and what is physical. Before shipping: no absolute paths in any
metadata (the recording is named relative to its asset directory), the generator reproduces the config
byte for byte, the render command works from the repository root, and every number quoted in the README
was measured on the shipped run alone on the GPU. Keep failed variants out of the package but keep their
measurements in the notes: the user values knowing why something does not work almost as much as the
thing that does.

## 7. Working with the user

They steer. Report each run's result as soon as it is known, in the units they think in (steps, mm,
cards), and offer the next step rather than asking permission for routine ones; ask only when two readings
of the task would lead to materially different work, when an action is destructive, or when a change would
touch frames they have verified. When they say stop, stop at once and keep the files.
