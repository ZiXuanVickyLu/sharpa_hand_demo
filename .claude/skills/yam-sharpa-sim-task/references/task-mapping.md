# Mapping a new task onto the recipe

Work through these eight questions before writing code; their answers are the design, and most failures
of the shuffle were answers that had been assumed instead of decided.

## 1. The objects and what they are made of

Which things deform, which are fixed, which must move rigidly? Deformable things are `cloth` shells
(cards, strips, laces, sheets) or `tet` bodies (soft balls, pads); fixed things are static `tri` colliders
(a shoe, a board, a tray); anything rigid that must move is a keyframed collider driven by the IK (the
hands) or by a scripted keyframe file. There is no rigid-body dynamics and no object pushes the hands
back. Choose materials from `references/scene-config.md` and test each object alone first (resting,
hanging, folded) for a minute of GPU time.

## 2. The fixtures and the frame

Place the task in the rig frame: table at z = 0, the workspace around y = 30 cm, the two hands mirrored
about x = 0. Fixtures that both hands work on sit on the centreline; things one hand holds and the other
works on need the arms' reach checked early with a throwaway key-pose solve (`solve_multi` from the
preshape, filter by table clearance). The arms' comfortable reach at table height is a band around the
bases; a palm facing the table edge is not reachable there (lessons item 3).

## 3. The contacts

For each phase, which hand part touches which object surface, and with what face? Fingertip pads
(`<finger>_DP`, pad normal), the thumb pad or tip, the backs of the phalanges, the palm. Write each contact
as a Point target on the link plus an Axis for the pad normal, and remember the contact surfaces are round:
a fingertip is a 16 mm bulb; its vertical band is a few millimetres high. A pinch (thumb pad against a
finger pad) holds a strip by friction when the gap between the two prescribed pads equals the strip's
thickness plus the contact offset (`d_hat`); set that gap in the IK from the object's thickness, and give
the pinch a friction of 0.7-1.0 on both colliders.

## 4. The grasp formation

Decide which object vertices are prescribed while the grasp forms and when they are released (two stages,
the inner ones first). For a lace end pinched between two pads: the rows under the pads ride the thumb
pad for the first second; for a card packet: the two upper rows ride the thumb. The release step is the
first step the task is physical, and it goes into the README's one-line statement.

## 5. The phases

List them with a verb, a duration and a hand (reach 2 s, approach 1 s, grasp 0.4 s, lift, thread, pull,
release, withdraw, home). Each becomes a segment of the schedule and a frame range to measure. Phases that
touch objects move at constant speed (`cruise`); free motions with min-jerk. Nothing faster than about
5 cm/s while in contact: the shipped push runs at 5 cm/s, the riffle slide at 0.8 mm/s.

## 6. What to measure per phase

Before the first run, write the number that says the phase worked: the object's vertices that moved, a
height, a gap, a count (cards released, eyelets threaded), an alignment (edges within x mm). Put it in a
small script run on `x_<step>.npy` (start from `scripts/body_state.py`), so every run is judged the same
way and the user reads a table, not a feeling.

## 7. What can go wrong (and is known)

Read `references/lessons.md`. The recurring ones: a prescribed hand pinches something (abort), a round
surface sorts objects by height instead of pushing them, friction glues a released object to a held one,
a straight object leans and topples instead of leaving, the hand dips into the table to reach a face.

## 8. The honest statement

One sentence for the README: what is prescribed (the grasp's formation, for how many frames), what is
physical (everything after), and what the result is in numbers. If part of the task could only be done
by scripting (a knot, a thread through a hole), say so rather than hiding it in a keyframe file.

---

## Worked mapping: "lace up a shoe with the YAM + Sharpa"

- **Objects**: the shoe is a static `tri` collider (its upper with the eyelet holes, friction 0.5); the lace is
  a `cloth` strip (6 mm wide, 1.5 mm thick, `self_collision: true`, friction 0.4 against the shoe, 0.8-1.0
  against the pads), long enough to thread. Test the lace alone: it must hang through an eyelet and slide
  when pulled by a prescribed vertex before any hand exists.
- **Contacts**: each lace end is pinched between the thumb pad and the index pad (gap = 1.5 mm + `d_hat`),
  the pad normals facing each other. Threading uses the lace's own stiffness: the pinched end is pushed
  through the eyelet along the hole's axis, released, and re-pinched on the other side by the other hand.
- **Grasp formation**: the lace vertices under the pads ride the thumb pad for the first second after the
  pads close, then are freed in two stages; from then on the pinch holds by friction alone.
- **Phases per eyelet pair**: approach above the eyelet, pinch the end, insert (constant speed along the
  hole axis, 2-3 cm), release, regrasp from below with the other hand, pull through (constant speed, the
  length that must pass), tension (a short pull along the shoe's axis), release; then the next pair.
- **Measurements**: lace vertices on each side of each eyelet's plane (threaded or not), the lace's tension
  as the distance between its anchored end and the free end, slip of the pinched vertices relative to the
  pads during a pull (more than 2 mm per pull means the pinch is too weak: more friction or a tighter gap).
- **Likely trouble**: the pinch slipping under a long pull (friction, gap), the lace end buckling at the hole
  instead of entering (insert slower, from closer, with a stiffer lace or a shorter free length past the
  pads), the fingers of the two hands colliding at the regrasp (check `arm_gap_exact_cm` per phase), the
  pads pinching the lace against the shoe's upper (an infeasible configuration: approach from above the
  hole, not along the shoe).
- **Honest statement**: "the lace is prescribed only while each pinch forms (N frames per grasp); every
  insertion, pull and release is contact and friction; k of the 2m eyelets were threaded, the lace slipped
  x mm per pull".

## Worked mapping: "fold a cloth sheet in half"

- A `cloth` sheet (30 x 30 cm, 0.5 mm, `self_collision: true`) on the mat; two pinches at two adjacent
  corners (one per hand), lift to 8 cm, carry over the far edge in an arc (constant speed), lower, release,
  press flat with the backs of the fingers (flat surfaces, lessons item 2), withdraw.
- Measure: the corners' height, the fold line's position against the sheet's midline, the sheet's maximum
  height after the press, the overhang of the top layer past the bottom one.

## Worked mapping: "stack loose cards into one squared deck"

- The shuffle's last phase alone: cards flat on the mat, the hands push from both sides with a tall flat
  surface. Measure the deck's length and width and the outer-edge spread per card. The round fingertips
  stop at a deck length of about 150 mm (lessons items 2 and 17); a squared deck needs the flat backs of
  the fingers or a flat tool.
