# Two solver accelerations (week of 2026-09-21): the inner Newton rule and the bounded penalty

Written for an agent that has to reproduce both changes in this solver from a clean checkout of
an earlier commit, or port them to another AL-IPC implementation. Everything below is stated
so that it can be done without reading the conversation that produced it. Spec sections are
MS = `doc/al-ipc-math-spec.md`, IS = `doc/al-ipc-implementation-spec.md`.

| change | what it does | where | commit | measured effect |
| --- | --- | --- | --- | --- |
| 1. Inner Newton rule | The subproblem loop ends when the *energy* line search accepted its first trial, instead of when the step factor is literally 1 | `Stepper::Impl::solve_subproblem`, `src/solver/stepper.cu`; MS §7 | fe6020c | rod twist, wound: 470 to 277 Newton steps per time step, 1.23 to 0.82 s per step |
| 2. Bounded penalty | `contact.mu_max` clamps [Z25]'s per-step penalty estimate from above | `Stepper::Impl::estimate_mu`, scene schema; MS §11.1 | b03d6ab, 87b03a0 | rod twist passes the buckling snap it used to crash on (4-8 outer iterations instead of 75-100); card shuffle deck comes to rest (motion per frame 0.43 mm to 0.07 mm) |

Both are contained in the stepper and the scene schema. Neither touches the kernels, the
linear solver or the contact code. The full test suite (`ctest` in `build/release`, 34 tests
then, 36 now) passed unchanged after each.

---

## 0. Where the changes sit in the algorithm

One time step of the AL-IPC solver (MS §6) is

```text
mu <- estimate_mu()                                  # change 2 clamps this
repeat (outer iteration k):
    x_hat <- SolveSubproblem(x_hat, x_anchor, C, mu)  # change 1 ends this loop earlier
    alpha <- CCD from x_anchor toward x_hat (and the NH end-state test, MS §10.4)
    x_anchor <- x_anchor + alpha (x_hat - x_anchor)
    lambda_i <- max(0, lambda_i - mu c_i(x_hat));  decay / remove inactive pairs
    beta <- beta (1 - alpha)
until beta <= epsilon and k >= K_min
```

`SolveSubproblem` is a projected Newton loop: gradient and Hessian of the augmented
Lagrangian at `x_hat`, PCG for the direction `p`, a backtracking line search on the energy,
`x_hat <- x_hat + r p`. The Newton loop has a cap (`newton.inner_max_iters`, 8 for the rod
twist, 4 for the card shuffle).

The two quantities that the changes act on:

* `r`, the accepted line-search factor of one Newton step. Its first trial is 1, except that
  for a non-invertible material (neo-Hookean, NH) the first trial is `min(1, 0.9 t_inv)` where
  `t_inv` is the smallest step along `p` at which some tetrahedron's `det D_s` reaches zero
  (MS §10.4, `fem_inversion_toi`). Halving only happens afterwards, when the energy does not
  decrease.
* `mu`, the AL penalty stiffness, one number per time step, from [Z25] Eq. 20:
  `mu = 0.1 max over free vertices and axes of diag(M + h^2 K(x^t))`, the largest diagonal entry
  of the elasto-inertial Hessian at the start of the step (`fem_max_diagonal`).

---

## 1. Inner Newton rule: an inversion-limited first trial counts as accepted

### 1.1 The rule before and after

[Z25]'s stopping rule for the subproblem, and ours until 2026-09-21, is "repeat Newton until
a full step is accepted". The implementation tested `r == 1`. With the NH inversion bound in
the first trial that test has a defect: whenever the full Newton direction would invert an
element, `r` is `0.9 t_inv < 1` before the energy is even looked at, and the loop repeats
although the line search did not backtrack at all.

New rule (MS §7, revised 2026-09-21): the loop repeats only when the **energy** line search had
to halve its first trial. An inversion-limited first trial is accepted, with one safeguard:
a first trial below 0.1 is not accepted (the Newton direction inverts an element almost at
once; the loop repeats so that the subproblem does not end with next to no motion).

```text
before:   accepted  :=  (r == 1)
after:    accepted  :=  (halvings == 0)  and  (r >= 0.1)
```

where `halvings` is the number of energy halvings the line search performed in this Newton
step (0 when its first trial passed the energy test, whatever that first trial was).

With the optional stricter rule `newton.increment_velocity_tol > 0` (C-IPC's PNTol, used by
the card shuffle: an accepted step ends the loop only when the RMS increment velocity is below
the tolerance), the increment actually applied is `r p`, not `p`; the RMS is now computed from
`r p`. Before the change `r` was 1 whenever this test ran, so this is only a correction for
the new case `r < 1`.

### 1.2 Why it was worth doing (the evidence)

Rod twist (IPC's scene, `app/config/rod_twist_paper.json`; the low-resolution rod config the measurements below were made on is no longer shipped but `tool/gen_rod_twist.py` regenerates it; NH, `d_hat = 1 mm`, no friction),
tightly wound at frame 800-1000 with 17-22 k active contact pairs: a step took about 100 outer
iterations and 740-790 Newton steps. Reading the debug log per Newton step (`line-search
factor`, `RMS increment velocity`, added in the same commit) showed:

* the inversion bound was 0.35-0.88 in almost every Newton step, and there was **no energy
  backtracking at all**;
* so every outer iteration ran the capped 8 Newton steps (7.7 on average), and the increment
  did not shrink within them (RMS 7.5e-3 to 1.0e-2 m/s from the first to the last of the 8).

The repeats bought nothing: the subproblem was as converged after the first inversion-limited
step as after the eighth. Frames 750-958, NH, `mu` fixed at 0.08, per time step:

| setting | outer | Newton | Newton per outer | s per step | solver time to frame 958 |
| --- | --- | --- | --- | --- | --- |
| epsilon 1e-3, old rule | 77 | 593 | 7.7 | 1.53 | 416 s |
| epsilon 5e-3, old rule | 61 | 470 | 7.7 | 1.23 | 334 s |
| epsilon 5e-3, new rule | 72 | 277 | 3.9 | 0.82 | 230 s |

(The remaining 3.9 per outer iteration come from the scene's own Newton tolerance,
`increment_velocity_tol = 1e-2 l` as IPC uses, which still asks for repeats when the increment
sits just above it. `tool/gen_rod_twist.py --newton-rule paper` drops that and leaves [Z25]'s
rule alone.)

The outer count is unchanged by design: the outer loop is governed by `beta = prod(1 - alpha)`,
and `alpha` comes from the CCD, not from the Newton loop. The change only removes Newton steps
that did not move the iterate.

### 1.3 The code

`src/solver/stepper.cu`, `Stepper::Impl::solve_subproblem` (the loop as it is now; the
diff of fe6020c changed the two lines marked):

```cpp
void solve_subproblem() {
    const double vtol = scene.desc.newton.increment_velocity_tol;
    const bool single = single_newton_next;          // MS §10.4 pull-back: exactly one step
    for (int it = 0; it < scene.desc.newton.inner_max_iters; ++it) {
        int pcg_iters = 0, halvings = 0;
        const real r = newton_step(pcg_iters, halvings);   // r = accepted factor, halvings = energy halvings
        stats.inner_newton += 1;
        stats.pcg_iters_total += pcg_iters;
        stats.line_search_halvings += halvings;
        if (log().should_log(spdlog::level::debug) && sb.n_free > 0) {
            const double v = std::sqrt(reduce_sum_sq(p.data(), sb.n_free, scratch_d) / double(sb.n_free)) / double(h);
            log().debug("    newton {}: line-search factor {:.3g}, RMS increment velocity {:.4e} m/s (tol {:.3e}), CG {}",
                        it, double(r), double(r) * v, vtol, pcg_iters);
        }
        if (single) break;
        // CHANGED: was `if (r != real(1)) continue;`
        const bool accepted = (halvings == 0) && double(r) >= 0.1;
        if (!accepted) continue;
        if (vtol <= 0.0 || sb.n_free == 0) break;
        // CHANGED: the accepted increment is r p (was p, correct only while r was 1 here)
        const double rms_v =
            double(r) * std::sqrt(reduce_sum_sq(p.data(), sb.n_free, scratch_d) / double(sb.n_free)) / double(h);
        if (rms_v < vtol) break;
    }
}
```

What `newton_step` must provide for this to be correct:

* `halvings` counts **only energy halvings**. In our line search the first trial is
  `r = min(1, t_inv)` (`t_inv` already carries the 0.9 factor); `halvings` is incremented once
  per `r *= 0.5` that follows a failed energy test. The batched path (three trials `r, r/2,
  r/4` evaluated in one go on small systems) adds the index of the accepted slot, `halvings +=
  k`, which is the same count. Nothing is added for the inversion bound itself.
* The energy test is `E(x + r p) <= E(x) + tol max(1, |E(x)|)` with
  `newton.line_search.energy_tolerance` (1e-12).
* `r` returned is the factor actually applied to `x_hat`, so that `r p` is the increment.

Nothing else changed. In particular the pull-back path (`single_newton_next`, MS §10.4) still
breaks after exactly one step, and the outer loop's own inversion test on the anchor path is
untouched.

### 1.4 How to reproduce and verify

1. Apply the two changed lines above (and the debug log if you want the per-step numbers).
2. Build (`cmake --build build/release`), run `ctest --test-dir build/release`; the suite is
   unaffected (nothing in the tests depends on `r == 1` as a stopping rule).
3. Generate the wound rod twist and run 200 steps from a checkpoint or from frame 0 with the
   ramp (`tool/gen_rod_twist.py --mesh rod/rod.msh --model NH --epsilon 1e-3` (mesh path relative to `asset/`); the
   run's `stats.csv` has `outer_iters`, `inner_newton`, `ls_halvings`,
   `inversion_limited` per step). Expected: `inner_newton / outer_iters` drops from about 7.7
   to about 4 with the scene's Newton tolerance, to 1-2 with `--newton-rule paper`;
   `outer_iters` unchanged within noise; the trajectory the same to visual accuracy (the
   iterate that ends the subproblem is the same point either way when the repeats were idle).
4. With `--log-level debug` (or the equivalent) confirm on a wound frame that the line-search
   factor is below 1 with `halvings = 0` in most Newton steps: that is the case the rule
   addresses. If instead you see halvings, the rule does nothing and the cost is elsewhere.

### 1.5 Pitfalls when porting

* If the target solver applies its inversion bound as a *separate* shortening after the energy
  line search, the fix is the same: the loop's repeat condition must look at the energy
  search's outcome, not at the final factor.
* Do not drop the 0.1 floor. A direction that inverts an element within a tenth of its length
  is a bad direction, and accepting it as "converged" ends the subproblem with the iterate
  practically where it started; the outer loop would then blend toward a stale iterate.
* If the solver uses a residual-based Newton tolerance instead of [Z25]'s rule the change is
  moot; the change matters exactly because the rule is "full step accepted".

---

## 2. Bounded penalty: `contact.mu_max`

### 2.1 The estimate and its two failure modes

[Z25]'s per-step penalty (MS §11.1, `mu_mode = "diag_max"`, `mu_scale = 0.1`):

```text
mu = 0.1 * max_{v free, c in xyz} [ M + h^2 K(x^t) ]_{(v,c),(v,c)}
```

computed by `fem_max_diagonal` from the per-vertex 3x3 diagonal blocks of the projected
element Hessians at the start-of-step positions plus the lumped mass. It is a **maximum over
all vertices**, so one element sets the penalty for the whole scene.

**Failure mode A, NH under strong compression (the rod twist).** The NH Hessian grows like
`1/J^2` as an element flattens. On IPC's rod twist the estimate is 0.08 kg at rest, 0.27 at
25 s, 2.8 at 40 s (tightest element `J = 0.17`); that is fine, the accepted fraction stays
near 0.6 with 5-8 outer iterations. But when a buckling snap leaves one element nearly flat
(`J = 5e-4`) the estimate jumps to 4e5 and later 1e18. A penalty that stiff crushes further
elements in the next step (the contact term dominates the elasticity by orders of magnitude,
the Newton system loses conditioning), a feedback that ends with CG failing.

Fixing `mu` at its rest value (`mu_mode = "fixed"`, `mu_fixed = 0.08`) is **not** a remedy: the
penalty is then too soft once the elements have stiffened, the iterate overshoots the
constraints, and the accepted fraction falls from 0.57 to 0.05-0.07 (75-100 outer iterations
and four times as many active pairs at 25 s). The estimate tracks the stiffness correctly in
the normal regime; only its tail is wrong.

**Failure mode B, light contacts under a stiff estimate (the card shuffle).** The multiplier
update `lambda <- max(0, lambda - mu c(x_hat))` makes a resting pair inactive as soon as its
gap exceeds `lambda / mu`. A resting contact must therefore keep its gap inside that window
from step to step, and the window must be wider than the solver's positional accuracy. For the
3 GPa playing cards (54 shells, 1.9 g each) the estimate is `mu = 1.35 kg` while a card's
weight is `lambda = h^2 F = 7.5e-8 kg m`: a window of 0.06 um against a Newton tolerance of
1 um per substep. Every resting contact flickered between `lambda = 0` and twice the weight,
105 of 1560 pairs were dropped and re-found every step, 400 CCD hits per step limited the
anchor, and the cards jittered by 0.4-0.5 mm per frame (measured on the hand-shuffle cards demo).

### 2.2 The rule

Keep the estimate, clamp it from above:

```text
mu = min( 0.1 * max diag(M + h^2 K),  mu_max )
```

`mu_max` is a scene parameter in kg (the unit of `mu`), optional, off by default (`null`),
so that every existing scene is unchanged. It only touches the per-step estimate; the stall
adaptation (MS §11.2, `mu <- 2 mu` after 50 stalled outer iterations) is applied after it and
is **not** clamped (it is off, `max_adaptations = 0`, in both scenes that use the bound; if you
turn both on, decide whether the doubling should respect the bound).

### 2.3 How to choose `mu_max`

The bound has to be set per scene from two sides.

* **Upper side, stiffness.** `mu` should stay at the level of the *legitimate* elasto-inertial
  diagonal, so that the contact Hessian `mu grad d grad d^T` (spectral contribution at most
  `2 mu`) does not dominate the conditioning. For the rod twist that is 100 times the rest
  value: `mu_max = 8 kg` (rest 0.08, 2.8 at the tightest legitimate compression; the snap's
  4e5 is cut off). `tool/gen_rod_twist.py --mu-max 8` (default).
* **Lower side, resolution of the multipliers.** For resting contacts carrying a force `F`
  over a substep `h`, the per-weight window `lambda / mu = h^2 F / mu` must be well above the
  solver's positional accuracy `eps_Newton`, and the violation at the bottom of a pile of `n`
  such contacts, `n lambda / mu`, must stay below the room the CCD leaves, `d_hat - xi`
  (contact distance minus thickness), or the multipliers never converge and the pile bounces:

  ```text
  eps_Newton  <<  h^2 F / mu        and        n h^2 F / mu  <  d_hat - xi
  ```

  For the card shuffle (`h = 2 ms` substep, ten per 20 ms frame, `F = 19 mN`, `n = 27`, CCD room 0.2 mm,
  `eps_Newton ~ 1 um`) that leaves `mu` in 0.01-0.03 kg. Measured on the resting piles
  (40 frames, motion of a card per 20 ms frame):

  | mu (kg) | window per weight | motion per frame | CCD hits per step |
  | --- | --- | --- | --- |
  | 1.35 (estimate) | 0.06 um | 0.43 mm | 1110 |
  | 0.03 (`mu_max`) | 2.5 um | 0.07 mm | 4 |
  | 0.01 | 7.5 um | 0.16 mm | 13 |
  | 0.003 | 25 um (0.7 mm at the pile's bottom) | 0.54 mm | 129 |
  | 0.001 | 75 um (2 mm at the bottom) | 0.82 mm | 296 |

  `mu_max = 0.03` is the scene's setting (`tool/gen_hand_shuffle.py --mu-max`, default 0.03).
  The alternative, converging Newton tightly enough for the estimate's window (cap 32,
  tolerance 1e-4), gives the same result at five times the cost.

A per-pair `mu_i` chosen from the pair's own `lambda_i` would remove the guess. It is not
built; `mu_max` is the one-number version.

### 2.4 The code

Four places, all in b03d6ab (the card-shuffle value and the analysis are 87b03a0, no code).

`src/scene/scene_desc.h`, in `ContactDesc` next to `mu_mode`, `mu_scale`, `mu_fixed`:

```cpp
std::optional<double> mu_max;       // kg, upper bound of the estimate (MS §11.1); none = unbounded
```

`src/scene/scene_desc.cpp`, the contact block's parser, validator and serializer:

```cpp
// parse (next to mu_fixed)
if (const json* v = r.get("mu_max")) d.mu_max = as_number(*v, r.key_path("mu_max"));
// validate
if (d.mu_max) check(*d.mu_max > 0, r.key_path("mu_max"), "must be > 0");
// serialize (round trip of the resolved config)
s["mu_max"] = c.mu_max ? json(*c.mu_max) : json(nullptr);
```

`src/solver/stepper.cu`, `Stepper::Impl::estimate_mu`:

```cpp
real estimate_mu() {
    const auto& cd = scene.desc.contact;
    if (cd.mu_mode == "fixed" && cd.mu_fixed) return real(*cd.mu_fixed);   // the bound does not apply
    fem_element_cache(sb, sb.x_prev.data(), cache, /*need_hessian=*/true);
    const real dmax = fem_max_diagonal(sb, cache, h2, scratch_r);
    real m = real(cd.mu_scale) * dmax;
    // MS §11.1 upper bound: one nearly flat NH element must not set the whole scene's penalty
    if (cd.mu_max && double(m) > *cd.mu_max) m = real(*cd.mu_max);   // ADDED
    return m;
}
```

`test/test_scene_config.cpp`, the defaults block: `CHECK(!d.contact.mu_max.has_value());`.

Config (`"contact"` block of the scene JSON, IS §5.3 schema listing):

```json
"mu_mode": "diag_max",
"mu_scale": 0.1,
"mu_max": 8.0
```

`stats.csv` already has a `mu` column per step (`stats.mu = double(mu)` right after the
estimate), which is how the clamp is observed.

### 2.5 How to reproduce and verify

1. Add the field, parser, validator, serializer, clamp and the test line; build; run the
   suite (`test_scene_config` covers the default and the round trip; nothing else changes).
2. Rod twist: `tool/gen_rod_twist.py --mesh rod/rod.msh --model NH --mu-max 8`
   (`--mu-max 0` for the unbounded estimate, `--mu-fixed 0.08` for the fixed value). Run to
   frame 1900 (after `tool/gen_rod_twist.py` has written `app/config/rod_twist.json`: `build/release-gui/cs_view app/config/rod_twist.json --play` or `build/release/cs_run app/config/rod_twist.json`). Expected in
   `stats.csv`: `mu` at 8.0 through the snap frames (1624 and 1712 were the failures), `alpha`
   0.5-0.6, 4-8 outer iterations, 0.02-0.1 s per step, no capped step to frame 1874. Unbounded:
   `mu` reaches 1e5 and more at the snap, then CG fails within a few steps. Fixed 0.08: 75-100
   outer iterations at 25 s.
3. Card shuffle: `tool/gen_hand_shuffle.py` (default `--mu-max 0.03`, `--mu-max 0` for the
   estimate) and `tool/analyze_hand_shuffle_pile.py` on the resting frames. Expected: median
   card motion per 20 ms frame 0.07 mm bounded against 0.43 mm unbounded, CCD hits per step
   single digits against about 1000.

### 2.6 Pitfalls when porting

* The bound must be applied to the **estimate**, not replace it. A fixed `mu` fails in the
  other direction (§2.1, failure mode A, second paragraph).
* `mu` has the unit of mass (kg) in this formulation (the constraint is a distance, the
  multiplier `lambda` has units kg m, the penalty energy is `mu c^2 / 2` in kg m^2, an
  `h^2`-scaled energy). A port whose penalty is in N/m or scaled differently needs the bound
  in its own unit; the ratios above (100 x rest for the rod, and the window formula for the
  cards) transfer.
* The two failure modes need bounds of opposite character: a bound near the legitimate
  stiffness for A, a bound near the contact force scale for B. There is no single default,
  which is why the parameter is off unless set.
* If the solver estimates `mu` per body or per pair, apply the same clamp per entry; the
  window argument of §2.3 is then per pair.

---

## 3. Order of application and how to tell each apart

The two changes are independent. Change 1 is a pure cost reduction (same iterate, fewer
Newton steps); apply it first, it has no parameter. Change 2 changes the trajectory (a
different penalty gives a different subproblem), needs a value per scene, and is what
turns a crash (rod twist snap) or a wrong end state (jittering cards) into a correct run.

To attribute an observed speed-up to the right change, look at `stats.csv`:

* change 1 shows as `inner_newton / outer_iters` dropping with `outer_iters` unchanged;
* change 2 shows as `mu` sitting at the bound, `alpha` rising and `outer_iters` dropping
  (rod twist), or as `hits` (CCD hits) and `kept`/`removed` pairs per step collapsing (cards).

References: MS §7 (rule), §10.4 (inversion bound `t_inv`), §11.1 (estimate, bound, window
table), §11.2 (stall); IS §6.7 (line search and inner loop), §5.3 (schema);
the card measurements above come from the hand-shuffle cards demo (the rod-twist reproduction note was removed with the low-resolution rods demo).
