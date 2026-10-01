# Writing a task-specific IK generator on `tool/ik/yam_kin.py`

`tool/ik/gen_shuffle_motion.py` is the worked example (the card shuffle); `tool/ik/yam_kin.py` is the
framework it is built on. A new task gets its own generator script next to it, built the same way.

## Frames, units, mirroring

- Rig frame: z up, table top at z = 0, the robot bases on the x axis at x = ±26.67 cm, yawed 90°, the
  workspace in front of them around y = 30 cm. The IK works in centimetres; the bake converts to metres.
- Every target is written for the LEFT hand in "task coordinates" and mirrored for the right hand by
  `M = (-1, 1, 1)`: `target_world = target_task * M`. Keep this convention; the bake and the renderer
  rely on the two hands being mirror images of one schedule.
- FK results (`k.fk(q).point(...)`) are in the solver frame of that hand; for the right hand multiply by `M`
  to compare with a task-frame target, and never store a mirrored point as a "world" record (see the
  `Ts.append` lines of the example for which frame goes where).

## The robot in `ArmKin`

`ArmKin(bench, side)` with `bench = YamSharpaBench(device="cpu", with_table=False)` from `tool/robot`.
28 joints per side (`k.n_q`), in this order: `arm1..arm6`, then the thumb `thumb_CMC_FE, thumb_CMC_AA,
thumb_MCP_FE, thumb_MCP_AA, thumb_IP`, then for index, middle, ring: `MCP_FE, MCP_AA, PIP, DIP`, and for the
pinky `CMC, MCP_FE, MCP_AA, PIP, DIP`. `k.ARM` = [0..5], `k.THUMB` = [6..10], `k.jidx[name]` = index,
`k.lo`, `k.hi` = limits (rad). Bodies by name through `k.b("thumb_DP")`, `k.b("index_DP")`,
`k.b("hand_wrist")`, etc.; fingertip markers used by the example: `FINGER_TIP = (2.3, 0.5, 0)` on `<finger>_DP`,
`THUMB_TIP = (3.1, -0.2, 0)` on `thumb_DP`, pad normal `PAD_N`, tip direction `TIP_DIR = (1, 0, 0)` (all cm
in the link frame).

Limits that shape what the hand can do (degrees): arm4 ±90/−97, arm5 ±90, arm6 ±120; thumb CMC flexion
−10..110, abduction ±20, MCP −30..80, IP 0..100; finger MCP −10..90. The DIP joints are tied to the PIPs
(`DIP_TIES`: DIP = 0.6 PIP) so a finger curls naturally with one variable.

## Terms (the whole vocabulary)

- `Point(body, local_cm, target_world_cm, w=1.0, mask=(1,1,1))`: put a link point at a target; the mask
  frees axes (the example pins the index/ring/pinky tips in x and z only and lets y follow the hand).
- `Axis(body, local_dir, target_dir, w)`: align a link direction (a fingertip pointing down, a pad normal
  facing the object). Weight 0.2 to 0.5 against Point weight 1: direction terms cost tip position fast.
- `Above(body, local_cm, normal, min_value, w=5.0)`: a one-sided clearance (the fingertip caps above the
  table gap, the thumb outside a keep-out line). It only acts when violated.
- `Posture(q_ref, w)`: pull toward a reference pose; weight 0.02 for naturalness, 0.1 as a continuity anchor
  during tracking (every frame solved from the previous frame's solution), 2 to 5 to freeze joints.

`k.solve(q_init, vars, terms, ties=DIP_TIES, margin=0.02)` solves only the joints in `vars` (bounded least
squares, the others stay at `q_init`); `k.solve_multi(q_init, vars, terms, n_starts=24, seed=0, accept=...)`
restarts from random poses for a key pose whose branch matters (the grasp). `Solution` reports `pos_err`
(worst hard Point error, cm), `axis_err` (deg), `hinge` (worst Above violation), `limit_margin` (rad).

Clearance by meshes, not by terms: `Proxies(bench, k)` fits covering spheres to the link meshes;
`px.exact_table(fk)` is the lowest mesh vertex above the table, `exact_gap(px_l, fk_l, px_r, fk_r)` the
left-right mesh gap. Report both per phase; they are the plausibility check of the motion.

## The generator's architecture (keep it)

1. **Scene record**: the object's geometry in task coordinates (for cards: the packet's end faces, the
   contact points `T0`/`G0` and their bowed versions) from a few options (`--tilt`, `--zlift`, `--top-x`...).
   Everything derived from it goes into the `scene` json so the bake and the scene generator read the same
   numbers.
2. **Key poses**: `h.preshape(curl)` as a start, `k.solve_multi` for the grasp pose (filter candidates by
   `pos_err < 0.4`, `hinge < 0.1` and mesh table clearance), then `h.settle(q, vars, terms_fn)`: the pose
   that is its own optimum under the continuity anchor, so tracking starts without a hop.
3. **Schedule**: a list of `(name, seconds, kind, payload)`. Kinds in the example: `joint` (interpolate
   joints to a key pose with min-jerk), `hold` (stay), `task` (track targets per frame: thumb tip T and
   fingertips G, or the rake points P), `thumb` (thumb joints only, the rest frozen), `regrip`, and the
   `lag=True` payload that delays one hand relative to the other. Add kinds for a new task rather than
   bending these.
4. **Tracking loop**: for each frame, build the terms for the current target, solve from the previous
   frame's `q` with `Posture(q, 0.1)`, store `q` and the target record. Targets move with `cruise()` (constant
   speed, short smooth ends; for anything that contacts an object, since min-jerk hits at twice the mean
   speed) or `minjerk()` (free motions).
5. **Hand-overs**: when one problem hands to another (grasp to thumb-only), measure the residual
   `e0 = tip_now - target(0)` and fade it over `HANDOVER = 15` frames; otherwise the first frame of the
   new problem closes the residual in one step and the colliders jump.
6. **Diagnostics** per frame: IK error (by FK, not the solver's word), table clearance, arm gap, thumb-finger
   gap, joint-limit margins and bound events; the report prints them per phase. A joint reaching or leaving
   a bound while tracked is a kink; pin abduction joints near mid-range with a stiff Posture rather than
   letting the solver use them as cheap extra axes.
7. **Output**: `np.savez_compressed(out/shuffle_motion.npz, joint_q, fps, phase_*, body_q, body_names,
   joint_names, target_left, target_right, table_exact_cm, arm_gap_cm, arm_gap_exact_cm, thumb_finger_gap_cm,
   scene)` plus `report.json`. Keep these keys: the bake and the renderer read them.

## Options are additive, by design

Every new behaviour is an option whose default reproduces the previous motion. Then a variant that starts at
riffle frame F is bit-identical before F, and `scripts/motion_diff.py` proves it ("first differing frame").
Count frames in the riffle's own frame numbers (frame 25 of the riffle = step 1100 in the shipped demo)
and say so in the help text; the user thinks in simulation steps, and the mapping is what they check.

Options that exist already and generalise: `--release-lag` (one hand behind the other), `--riffle-lift`
(move along the object's axis during a slide), `--riffle-knuckle ... --riffle-knuckle-mode freeze|slide`
(prescribe one joint from a frame on, the rest frozen or still tracking), `--riffle-knuckle-press`
(a descent while a joint opens), `--riffle-cut` (end a phase early, schedule unchanged), `--direct-push
--push-pause` (skip a lift-off), `--square-slack` (how far a push stops short), `--push-z` (rise or dip of
the fingertips during a push; negative also lowers the table clearance of that phase, which is how the
"finger straight segment pushes" variant was made).

## Checks before baking

- `report.json`: IK error per phase below 0.15 cm in the tracked phases (the example: 0.112 to 0.133),
  `table` never below the gap (0.2 cm), `gap=` (exact mesh gap between the hands) positive everywhere,
  no bound events inside phases that touch the objects.
- `scripts/motion_diff.py new.npz previous.npz`: the first differing frame is where the new option starts.
- The tip trajectory makes sense in numbers (print the thumb tip and a fingertip per phase; compare with
  the object's geometry from the scene record): the user's questions are always about millimetres.
