# Measured lessons from the card-shuffle work

Each item cost runs to learn. Numbers are from the shipped rig (cards 116 x 90 mm at scale 1.25, 0.3 mm
thick, 27 per packet) but the mechanisms are general. When a new task shows the same symptom, start here.

## About prescribed colliders

1. **A prescribed collider has infinite force.** Anything caught between two parts of a hand, between a hand
   and the table, or between two hands cannot escape; the solver's line search fails on every iteration
   (`outer 500`, `alpha_mean 0.000`) and the abort rule stops the run. Raising the cap never helps: find the
   pinch (section views at the last good step) and change the motion.
2. **Rounded fingertips cannot push a face taller than their vertical band.** The fingertip bulb is about
   16 mm across; pushing a 15 mm-tall stack of cards, the cards at the bulb's upper slope rode up and were
   pinched under the next finger segment, locking the solver at 141 mm of deck length; raising the
   fingertips 4 mm made the lower slope lift the cards instead. Only a flat, tall surface (the finger's
   straight segment, the back of the fingers) squares the deck, and the straight segment is reachable only
   with the bulb below the table, which is physically wrong (the user rejected it). Plan pushes with the
   contact surface's height in mind from the start.
3. **The hand model's workspace limits:** at table height the arm cannot present the palm vertically to the
   table edge (arm4 saturates, the fingers droop 13° or more); flat vertical surfaces that ARE reachable
   there are the backs of the distal phalanges with the hand turned over (a lift, flip, lower sequence of
   about 3.5 s keeping 12 cm above the objects). The thumb's opening swings its tip about 0.5 mm per
   degree toward the centre and rises slowly.
4. **The IK keeps the hand above the table (0.2 cm gap on the fingertip caps); the solver does not.** A
   motion that dips into the table runs fine and looks wrong. Watch `table_exact_cm` in the report.

## About the grasp and the release

5. **Compression is what makes a release a snap.** A card released fully bowed drops 20-30 steps to the mat;
   released as it straightens over a rounded tip it falls 66 to 50 mm in 70 steps and then hangs on its
   held neighbour. A straight object (zero compression) leaning on a support does not leave on its own:
   remove the support and the whole leaning stack topples together (30 cards in 20 steps).
6. **Too much bow is as bad as none.** 4-5 mm of compression along a 116 mm card holds and snaps well;
   8-9 mm bends each card to 90° and a freed card stands up instead of falling inward.
7. **A rigid flank entering a stack sorts the objects by geometry, not by intent.** With the thumb's flank
   at 45° to the packet's end face, the inner cards were compressed 10 mm, the middle 2-4, the outer 0,
   and the outer five were shoved 7-16 mm out of the cap's reach during the bow; nothing after the bow
   could meter them. Where the contact surface meets the stack decides the whole release; measure the
   per-object offsets right after the grasp forms.
8. **Friction transport:** a surface moving along a held edge carries the edge with it (measured in isolation:
   a bowed card's loaded end on a static perpendicular plate does not creep; on a moving plate it follows
   ~80 %). A release by sliding only works where the object has its own spring to leave with; under a cap
   that keeps deepening, the ends migrate into the friction cone (within ~5 mm of the bottom at friction 0.7)
   and are carried for the rest of the motion.
9. **Opening a joint to release works only if the objects are loaded.** Freezing the thumb and opening the
   last knuckle dropped every remaining straight card at about 5° of opening; opening while still sliding
   rotated the pad into the ends and pushed the outer cards up the flank instead.

## About friction and materials

10. **Friction is the most sensitive quantity.** Thumb friction: 2 releases cards in bunches, 1.0 and 0.7 one
    at a time but slow falls, 0.5 a real snap but one card per hand slips off during the bow. Card-to-card
    friction: 0.02 lets released cards land in 40-60 steps; 0.05 and 0.1 glue a released card to its held
    neighbour through the lagged friction force (it hangs and creeps) and even change the bow itself (two
    cards moved 10 mm at step 500, 12 instead of 20 released by step 1160). Lowering the thumb's cap
    friction to 0.5 or 0.3 to improve the release wrecked the grasp (7 cards per hand slipped off before the
    riffle): the same friction holds and lets go. Mat friction 0.5 keeps landed objects in place; 1.5 pins
    the feet of cards that tent against others so they only creep.
11. **The lagged normal force (friction model "paper")** means every switch from prescribed to free, and
    every new contact, starts with one friction-less step: stage hand-overs and expect a small slip.
12. **Softer bending (the user wanted cards at half the paper's k_bend)** makes release gentler and drag
    easier; it does not change where an object leaves a rigid surface, which is geometry and the friction
    cone.

## About the solver's limits

13. **Many parallel, coincident edges** (a squaring deck, a stack of thin shells edge to edge) raise the
    edge-edge pair count several-fold and slow steps to seconds; `drop_parallel_edge_pairs` exists for
    cloth-like configurations. Thin shells meeting exactly edge to edge at the same height cannot pass
    each other; interleave or offset them.
14. **`mu_max` 0.03** keeps resting thin objects still; the paper's estimate (1.35 kg) lets them jitter.
15. **A run's cost tracks the pair count:** 0.2 s per step at a few hundred pairs, seconds at thousands;
    the log's `|C|` is the early warning that a configuration is becoming expensive.

## About the landing and the end state

16. **A block drop lands fanned.** Fourteen cards falling together landed rotated 15-40° in the plane, and a
    push that squares only along one axis then squeezed the rotated ones out sideways (over 55 cm of
    scatter in the worst run); five cards dropped together landed straight and squared cleanly. The
    quality of the end depends on the size of the last block.
17. **Pushes stop short on purpose.** The shipped push stops 16 mm short of the card's half length per
    side; closer, the fingertips lock (item 2). A push that must square fully needs a different contact
    surface, not more travel.

## About the process

18. **Change one thing, keep the verified frames.** Every variant was checked with the motion diff before
    simulating; the one time a base motion was swapped by mistake cost a run and the user's confidence.
19. **Stop at the first sign.** The user's rule: a run that has shown its failure is stopped at once; nothing
    is learned by finishing it, and the GPU is the bottleneck. One simulation at a time.
20. **Keep outputs until inspected, delete only on instruction.** The user reads the `.bgeo` sequences
    themselves; a deleted run output before they looked was a real complaint.
21. **Measure before explaining.** Every mechanism above was found by printing offsets along the card axis
    and across the stack, contact heights and normals, and section views; none from watching clips.
22. **Report numbers in the user's units** (simulation steps, millimetres, cards, hand changes) and say what
    was tried and what it showed, including the failures: the decisions that mattered (ship, change the
    bow, accept a dip) were theirs.
