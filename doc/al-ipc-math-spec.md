# Barrier-Free Augmented-Lagrangian Contact — Mathematical Specification

**Status:** v0.1 — for review, pre-implementation. Nothing in this document is built yet.
**Basis:** Zheng, Luo, Li, *Robust and Efficient Penetration-Free Elastodynamics without
Barriers*, arXiv:2512.12151 (Dec 2025), referred to below as **[Z25]**; the official
integration into libuipc contributed by Genesis AI (branch `AL-release` of
`wiso-enoji/libuipc`, and `spiriMirror/libuipc` main), referred to as **[UIPC-AL]**.
Local checkouts: `../libuipc` (both branches), `../coupled_solver` (infrastructure reference).
**Companion document:** `al-ipc-implementation-spec.md` (GPU layout, kernels, project structure).

This document restates the method in the exact form we will implement. Every equation here
is either derived in place or attributed; where [Z25] and [UIPC-AL] differ, §16 records the
decision. Section numbers are referenced from the implementation spec.

---

## 0. Scope and reading guide

**v1 scope.** Linear tetrahedral bodies (stable Neo-Hookean, Neo-Hookean, fixed corotated),
kinematic triangle-mesh colliders and half-space planes, prescribed (moving) Dirichlet
vertices, point–triangle and edge–edge contact with offset, semi-implicit Coulomb friction,
implicit Euler, double precision.
**Deferred (specified only where cheap):** codimensional shells/rods (thickness offsets are
carried in the formulas), affine bodies, adaptive time stepping.

Reading order for implementers: §1–§2 (state and objective), §5–§9 (the solver), §10
(CCD), then §3–§4 (energies and geometry), then §11–§14.

---

## 1. Notation and state

| Symbol | Meaning |
| --- | --- |
| $N$ | number of vertices in the scene (all bodies, colliders included) |
| $\mathcal F \subset \{0..N-1\}$ | free vertices (degrees of freedom); $n_f = \lvert\mathcal F\rvert$ |
| $\mathcal P = \{0..N-1\}\setminus\mathcal F$ | prescribed vertices: Dirichlet-driven mesh vertices and collider vertices |
| $x \in \mathbb R^{3N}$ | stacked positions; $x_{\mathcal F}$ free part, $x_{\mathcal P}$ prescribed part |
| $x^t, v^t$ | state at the beginning of the step; $h$ time step |
| $M$ | lumped mass matrix (per vertex $m_v = \sum_{e\ni v} \rho_e V_e/4$), diagonal |
| $\tilde x$ | inertia target, §2 |
| $\hat x$ | current (possibly penetrating) solver iterate |
| $x$ (inside a step) | current intersection-free anchor, always a valid state |
| $\mathcal C$ | active constraint set; element $i$ carries $(\text{pair}_i, \lambda_i, n_i)$ |
| $\lambda_i \ge 0$ | multiplier estimate of pair $i$ |
| $n_i \in \mathbb N$ | consecutive-inactive counter; decay weight $\gamma_i = \Gamma^{\,n_i}$ |
| $\mu$ | augmented-Lagrangian penalty stiffness (units of mass, §11) |
| $\delta_i$ | separation offset of pair $i$, §4.4 |
| $\xi_i$ | geometric thickness of pair $i$ (CCD minimum separation), §4.4 |
| $d_i(x)$ | unsigned distance of pair $i$ at $x$; $\nabla d_i(x)\in\mathbb R^{12}$ (or $\mathbb R^{3}$ for planes) |
| $c_i(\hat x)$ | linearized signed constraint, §5.1 |
| $\alpha$ | CCD step fraction of one outer iteration |
| $\beta$ | termination weight, §15 |
| $K_{\min},\ \varepsilon,\ \Gamma,\ \gamma_{\min},\ C_\mu$ | algorithm parameters, §15.3 |

All vectors indexed by a pair are ordered `[p, t0, t1, t2]` (point–triangle), `[a0, a1, b0, b1]`
(edge–edge), `[p]` (point–plane); a per-vertex 3-block of a 12-vector is written
$(\nabla d_i)_v$.

---

## 2. Time integration and the incremental potential

Implicit Euler on the free DOFs:

$$
x^{t+1}_{\mathcal F} = x^t_{\mathcal F} + h\,v^{t+1}_{\mathcal F},\qquad
M\,(v^{t+1}-v^{t}) = h\,\big(f_{\text{int}}(x^{t+1}) + f_{\text{ext}}\big).
$$

With the inertia target $\tilde x = x^t + h v^t + h^2 M^{-1} f_{\text{ext}}$ (gravity and any
constant external force), the step is the minimizer of the incremental potential

$$
E(\hat x) \;=\; \tfrac12\,(\hat x-\tilde x)^\top M\,(\hat x-\tilde x) \;+\; h^2\,U(\hat x),
\qquad U = U_{\text{el}} + U_{\text{f}},
$$

restricted to $\hat x_{\mathcal F}$ (prescribed vertices are constants inside the step, §12).
$U_{\text{el}}$ is the elastic energy (§3), $U_{\text{f}}$ the lagged friction potential
(§13). No other damping is added; implicit Euler's numerical damping is the only dissipation
besides friction. Units: $[E] = \text{kg·m}^2$; the contact terms of §5 carry the same units.

The contact constraint set is **not** part of $E$; contact enters through the augmented
Lagrangian of §5 and the intersection-free path construction of §6.

---

## 3. Elastic energies and analytic eigen-systems

### 3.1 Element kinematics

Tetrahedron $e$ with vertices $x_0..x_3$ and rest edge matrix $D_m = [X_1-X_0, X_2-X_0, X_3-X_0]$,
$V_e = \lvert\det D_m\rvert/6$, deformation gradient

$$
F_e = D_s D_m^{-1},\quad D_s = [x_1-x_0,\; x_2-x_0,\; x_3-x_0].
$$

Writing $g_m^\top$ for row $m$ of $D_m^{-1}$ ($m=1,2,3$), $F_e = \sum_{a=0}^{3} x_a\, a_a^\top$
with $a_m = g_m$ and $a_0 = -(g_1+g_2+g_3)$. The constant vectors $a_a\in\mathbb R^3$ are the
"shape gradients"; $\partial(V_e\psi)/\partial x_a = V_e\,P(F_e)\,a_a$ with $P$ the first
Piola–Kirchhoff stress. These $a_a$ are exactly the row vectors [Z25] calls $A_{i'}$ in their
Eq. 19.

### 3.2 Constitutive models (isotropic, in singular values)

$F = U\,\mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)\,V^\top$ (signed SVD, $\det U=\det V = 1$,
at most one negative $\sigma$), $I_C = \sum\sigma_i^2$, $J = \prod\sigma_i$.
Lamé parameters from $(E,\nu)$: $\mu_L = E/(2(1+\nu))$, $\lambda_L = E\nu/((1+\nu)(1-2\nu))$.

| Model | $\Psi(\sigma)$ | Notes |
| --- | --- | --- |
| **SNH** (stable Neo-Hookean, log-free form) | $\tfrac{\mu_L}{2}(I_C-3) + \tfrac{\hat\lambda}{2}(J-\hat\alpha)^2$, $\hat\alpha = 1+\mu_L/\hat\lambda$ | $\hat\lambda=\lambda_L+\mu_L$ (re-parameterized so the small-strain moduli match linear elasticity; config flag can select $\hat\lambda=\lambda_L$). Rest state is stress-free for any $\hat\lambda$. Invertible: no inversion guard. |
| **NH** (compressible Neo-Hookean) | $\tfrac{\mu_L}{2}(I_C-3) - \mu_L\ln J + \tfrac{\lambda_L}{2}\ln^2 J$ | Requires $J>0$ on every iterate and on the path: inversion guard §10.4. |
| **COR** (fixed corotated) | $\mu_L\sum_i(\sigma_i-1)^2 + \tfrac{\lambda_L}{2}(J-1)^2$ | The "corotated linear" model of [Z25] Table 2. |

Energy per element $V_e\Psi$; $U_{\text{el}} = \sum_e V_e \Psi(F_e)$.

### 3.3 Analytic eigen-system of $\partial^2\Psi/\partial F^2$

For any isotropic $\Psi(\sigma)$ the $9\times9$ Hessian has nine analytic eigenpairs
(Smith–De Goes–Kim 2018/2019; the same construction as `libcwheels/energy/stable_neohookean_tet.h`,
which serves as the CPU oracle). Let $\Psi_i = \partial\Psi/\partial\sigma_i$, $\Psi_{ij}$ the
second derivatives, $e_i$ the canonical basis of $\mathbb R^3$.

1. **Scaling modes** ($s = 1,2,3$): the symmetric $3\times3$ matrix $A_{ij} = \Psi_{ij}$ has
   eigenpairs $(a_s, q_s)$; eigenvector $Q_s = U\,\mathrm{diag}(q_s)\,V^\top$, eigenvalue $a_s$.
2. **Twist modes**, one per index pair $(i,j)$, $k$ the third index:
   $Q^{\mathrm T}_{ij} = \tfrac{1}{\sqrt2}\,U\,(e_i e_j^\top - e_j e_i^\top)\,V^\top$, eigenvalue
   $\tau_{ij} = \dfrac{\Psi_i + \Psi_j}{\sigma_i + \sigma_j}$.
3. **Flip modes**, one per pair: $Q^{\mathrm L}_{ij} = \tfrac{1}{\sqrt2}\,U\,(e_i e_j^\top + e_j e_i^\top)\,V^\top$,
   eigenvalue $\phi_{ij} = \dfrac{\Psi_i - \Psi_j}{\sigma_i - \sigma_j}$, with the limit
   $\Psi_{ii}-\Psi_{ij}$ when $\sigma_i\to\sigma_j$.

Sanity checks used as unit-test oracles: for $\Psi=\tfrac{\mu}{2}\|F\|_F^2$ all nine eigenvalues
equal $\mu$; at the rest state of any rotation-invariant model the twist eigenvalues vanish.

Specializations (write $J_\alpha := \hat\lambda(J-\hat\alpha)$ for SNH):

| Model | $\Psi_i$ | $A_{ii}$ | $A_{ij}\ (i\ne j)$ | $\tau_{ij}$ | $\phi_{ij}$ |
| --- | --- | --- | --- | --- | --- |
| SNH | $\mu_L\sigma_i + J_\alpha\,\sigma_j\sigma_k$ | $\mu_L + \hat\lambda\sigma_j^2\sigma_k^2$ | $\sigma_k\,\hat\lambda\,(2J-\hat\alpha)$ | $\mu_L + J_\alpha\sigma_k$ | $\mu_L - J_\alpha\sigma_k$ |
| NH | $\mu_L\sigma_i + (\lambda_L\ln J-\mu_L)/\sigma_i$ | $\mu_L + (\mu_L - \lambda_L\ln J + \lambda_L)/\sigma_i^2$ | $\lambda_L/(\sigma_i\sigma_j)$ | $\mu_L + (\lambda_L\ln J-\mu_L)/(\sigma_i\sigma_j)$ | $\mu_L - (\lambda_L\ln J-\mu_L)/(\sigma_i\sigma_j)$ |
| COR | $2\mu_L(\sigma_i-1) + \lambda_L(J-1)\sigma_j\sigma_k$ | $2\mu_L + \lambda_L\sigma_j^2\sigma_k^2$ | $\sigma_k\lambda_L(2J-1)$ | $2\mu_L - \dfrac{4\mu_L}{\sigma_i+\sigma_j} + \lambda_L(J-1)\sigma_k$ | $2\mu_L - \lambda_L(J-1)\sigma_k$ |

**PSD projection.** Replace every eigenvalue by $(\cdot)^+ = \max(\cdot, 0)$. The projected
element Hessian is $H_e^+ = V_e \sum_{k=0}^{8} \lambda_k^+\, g_k g_k^\top$ with
$g_k \in \mathbb R^{12}$, $(g_k)_a = Q_k\, a_a$ (chain rule through $F=\sum_a x_a a_a^\top$).
The per-vertex-block closed form used on the GPU is derived in Appendix B.

### 3.4 Gradient

$\nabla_{x_a}(V_e\Psi) = V_e\,P\,a_a$ with $P = U\,\mathrm{diag}(\Psi_1,\Psi_2,\Psi_3)\,V^\top$;
for SNH explicitly $P = \mu_L F + J_\alpha\,\partial J/\partial F$ with
$\partial J/\partial F = [f_1\times f_2,\ f_2\times f_0,\ f_0\times f_1]$ (columns $f_c$ of $F$).

### 3.5 Codimensional shells (cloth): kinematics and membrane energy

Added for the cloth scenes of [Z25] (Figs. 22 and 23); everything above is unchanged for
tetrahedra. A cloth body is a triangle mesh with thickness $t$ and volumetric density $\rho$;
its lumped mass is $\rho\,t\,A_e/3$ per vertex of every triangle $e$ of area $A_e$.

**Kinematics.** Triangle $e$ with rest vertices $X_0,X_1,X_2$ and current $x_0,x_1,x_2$. Fix a
rest orthonormal frame in the triangle's plane, $\bar e_1 = (X_1-X_0)/|X_1-X_0|$,
$\bar e_2 = \bar n\times\bar e_1$, and write the rest edges in it,
$D_m = \begin{bmatrix}\bar e_1^\top(X_1-X_0) & \bar e_1^\top(X_2-X_0)\\ 0 & \bar e_2^\top(X_2-X_0)\end{bmatrix}\in\mathbb R^{2\times2}$,
$A_e = |\det D_m|/2$. The deformation gradient is $F_e = D_s D_m^{-1}\in\mathbb R^{3\times2}$
with $D_s = [x_1-x_0,\;x_2-x_0]$. With $g_m^\top$ the rows of $D_m^{-1}$ ($m=1,2$),
$F_e = \sum_{a=0}^2 x_a a_a^\top$ with $a_m = g_m\in\mathbb R^2$ and $a_0 = -(g_1+g_2)$:
the shape gradients of §3.1 with two components instead of three. The element energy is
$t\,A_e\,\Psi(F_e)$ and $\partial(tA_e\Psi)/\partial x_a = tA_e\,P(F_e)\,a_a$ with
$P\in\mathbb R^{3\times2}$.

**Membrane models.** Thin SVD $F = U\,\mathrm{diag}(\sigma_1,\sigma_2)\,V^\top$ with
$U\in\mathbb R^{3\times2}$ orthonormal columns, $V\in SO(2)$, $\sigma_1\ge\sigma_2\ge0$
(no signed convention is needed: a triangle cannot invert, $\det(F^\top F)\ge0$ by Cauchy–Schwarz).
$I_C = \sigma_1^2+\sigma_2^2$, $J = \sigma_1\sigma_2$ (the area ratio). The models are the
two-dimensional restrictions of §3.2, with the membrane shear modulus $\mu_m$ and a second
Lamé-like constant $\lambda_m$:

| Model | $\Psi(\sigma)$ | $\Psi_i$ |
| --- | --- | --- |
| **cloth-COR** | $\mu_m\sum_i(\sigma_i-1)^2 + \tfrac{\lambda_m}{2}(J-1)^2$ | $2\mu_m(\sigma_i-1) + \lambda_m(J-1)\sigma_j$ |
| **cloth-SNH** | $\tfrac{\mu_m}{2}(I_C-2) + \tfrac{\hat\lambda_m}{2}(J-\hat\alpha)^2$, $\hat\alpha = 1+\mu_m/\hat\lambda_m$ | $\mu_m\sigma_i + \hat\lambda_m(J-\hat\alpha)\sigma_j$ |
| **cloth-StVK** | $\mu_m\|E\|_F^2 + \tfrac{\lambda_m}{2}\operatorname{tr}^2E$, $E = \tfrac12(F^\top F - I)$ | $\mu_m\sigma_i(\sigma_i^2-1) + \tfrac{\lambda_m}{2}\sigma_i(I_C-2)$ |

[Z25] state their cloth by $(\mu_{\text{mem}}, k_{\text{bend}})$ only (Table 2) and take the
scenes from OGC's examples, whose `tri_ke` is an **areal** stiffness in N/m: $\mu_{\text{mem}}
= \mu_m t$. The element energy is $tA_e\Psi(F)$ with $\Psi$ in $\mu_m$, so $\mu_m =
\mu_{\text{mem}}/t$; $\lambda_m = 2\mu_m\nu/(1-\nu)$ (plane stress). Reading $\mu_{\text{mem}}$
as a Pa modulus makes the sheet $10^3\times$ too soft at $t = 1$ mm, and the penalty stiffness
$\mu = C\max\operatorname{diag}$ (§11) then cannot separate a folded sheet -- that is how the
twisting cloth stalled before this was settled. Which of the three models OGC's energy
corresponds to is still not stated; cloth-StVK is the default and the model is a config key.

### 3.6 Eigen-system of the $3\times2$ membrane Hessian

For an isotropic $\Psi(\sigma_1,\sigma_2)$ the $6\times6$ Hessian $\partial^2\Psi/\partial F^2$
has six analytic eigenpairs (Kim–Eberle 2020, the shell case of §3.3). Let $\hat n = u_1\times u_2$
complete $U$ to a rotation $\bar U = [u_1,u_2,\hat n]\in SO(3)$, and $v_i$ the columns of $V$.

1. **Scaling modes** ($s = 1,2$): eigenpairs $(a_s,q_s)$ of the $2\times2$ matrix
   $A_{ij} = \Psi_{ij}$; $Q_s = U\,\mathrm{diag}(q_s)\,V^\top$.
2. **Twist mode**: $Q^{\mathrm T} = \tfrac1{\sqrt2}U(e_1e_2^\top - e_2e_1^\top)V^\top$, eigenvalue
   $\tau = (\Psi_1+\Psi_2)/(\sigma_1+\sigma_2)$.
3. **Flip mode**: $Q^{\mathrm L} = \tfrac1{\sqrt2}U(e_1e_2^\top + e_2e_1^\top)V^\top$, eigenvalue
   $\phi = (\Psi_1-\Psi_2)/(\sigma_1-\sigma_2)$, limit $\Psi_{11}-\Psi_{12}$ at $\sigma_1\to\sigma_2$.
4. **Out-of-plane modes** ($i = 1,2$): $Q^{\mathrm N}_i = \hat n\,v_i^\top$, eigenvalue
   $\nu_i = \Psi_i/\sigma_i$. These are the modes a tetrahedron does not have: bending the
   membrane out of its plane at no in-plane strain.

PSD projection as in §3.3. **Reuse of the tet kernel.** Pad the shell to the tet form:
$\bar F = [F\ \ 0]$ with singular values $(\sigma_1,\sigma_2,0)$, $\bar U$ as above,
$\bar V = \mathrm{diag}(V,1)$, and shape gradients $\bar a_a = (a_a, 0)\in\mathbb R^3$. Because
$\bar V^\top\bar a_a$ has a zero third component, every mode of §3.3 whose $V$-side support is
$e_3$ alone drops out of the per-vertex blocks, and the tet twist and flip modes for the pairs
$(1,3)$ and $(2,3)$ both reduce to $\mp\hat n\,v_i^\top/\sqrt2$; assigning both of them the
eigenvalue $\nu_i$ therefore reproduces the out-of-plane pair exactly, and the $(1,2)$ twist and
flip, and the $2\times2$ scaling block padded with a zero, reproduce the rest. Appendix B and
the per-vertex-block kernel apply verbatim with $V_e := t A_e$; only the cache (a $3\times2$
SVD, two $\Psi_i$, the $3\times2$ PK1) is new.

### 3.7 Bending

Discrete-shell bending (Grinspun et al. 2003; the form used by Codim-IPC and by libuipc's
`discrete_shell_bending`) over every interior edge $\bar e = (x_1,x_2)$ with opposite vertices
$x_0,x_3$ and dihedral angle $\theta$:

$$
U_{\text{bend}} = \sum_{\text{hinges}} k_b\,\frac{|\bar e|}{\bar h}\,(\theta-\bar\theta)^2,\qquad
\bar h = \frac{A_1+A_2}{3|\bar e|},
$$

with rest length $|\bar e|$, rest areas of the two triangles, rest angle $\bar\theta$ (0 for a
flat sheet), and $k_b = k_{\text{bend}}$ of Table 2. $\nabla\theta\in\mathbb R^{12}$ is the standard
dihedral gradient. Two Hessians are specified and switchable:

* **Gauss–Newton** (default): $2k_b\frac{|\bar e|}{\bar h}\,\nabla\theta\nabla\theta^\top$, which
  drops $\partial^2\theta$. It is PSD and rank one, so it enters the linear system through the
  same matrix-free rank-one path as a contact pair (implementation spec §6.5) and touches no
  BSR pattern. The Newton rate it costs is invisible when the outer loop is bounded by $K_{\min}$
  and $\epsilon$, which is the case in every cloth scene of [Z25] (Fig. 22 runs exactly two
  Newton steps per time step).
* **Full**: $2k_b\frac{|\bar e|}{\bar h}\,[\nabla\theta\nabla\theta^\top + (\theta-\bar\theta)\nabla^2\theta]$
  projected to PSD by a $12\times12$ eigen-decomposition, assembled into the BSR through hinge
  adjacency (a hinge couples $x_0$ and $x_3$, a vertex pair no triangle produces, so the pattern
  is the union of triangle and hinge adjacency).

### 3.8 What a shell does not need

* **No inversion guard** (§10.4): $\det(F^\top F)\ge0$ always, so $J$ can reach zero but never
  change sign. A line-search rejection of $A(\hat x)/A_e < 10^{-3}$ per triangle stands in for it.
* **No signed SVD.** The thin SVD is computed from the $2\times2$ eigen-decomposition of
  $F^\top F$, which is cheaper and better conditioned than the $3\times3$ Jacobi of §3.3.

---

## 4. Contact geometry

### 4.1 Primitives and candidate pairs

Each tetrahedral body exposes its boundary surface (outward-oriented triangles, their edges
and vertices); each cloth body (§3.5) exposes all of its triangles, edges and vertices, with
the primitive thickness $\xi = t/2$ on each; each collider is a triangle mesh; each plane is
$(o, n)$. Distances are unsigned and the point-triangle gradient is two-sided (§4.2), so a shell
needs no inside. Candidate pair types:

* **PT**: surface vertex $p$ vs surface triangle $(t_0,t_1,t_2)$, $p\notin\{t_m\}$;
* **EE**: surface edge $(a_0,a_1)$ vs surface edge $(b_0,b_1)$, no shared vertex;
* **PH**: surface vertex vs plane.
* **PS**: surface vertex vs analytic sphere $(c, r, \text{inverted})$, added for the trapped
  squishy balls of [Z25] Fig. 21, whose container is an inverted sphere of prescribed radius.
  With $u = p - c$: $d = |u| - r$ and $\nabla d = u/|u|$ for a solid sphere; $d = r - |u|$ and
  $\nabla d = -u/|u|$ for an inverted one (the body lives inside). $\nabla d$ is a unit vector,
  so every PH statement below holds for PS with $n$ replaced by $\nabla d$.

Pairs within one body are candidates only if that body has self-collision enabled; pairs
whose two bodies are excluded by the contact table are never candidates. Pairs sharing a
vertex are never candidates (distance identically zero). Pairs whose vertices are all
prescribed are not candidates either: no free DOF can resolve them, so two colliders (or a
collider and a Dirichlet region) intersecting is a scene error. Because such pairs are not
candidates, a prescribed region driven into a collider passes through it silently and the
free vertices attached to it are then dragged into contacts the outer loop cannot advance
(the CCD fraction falls to zero). The implementation therefore still runs the CCD on these
pairs, counts the hits without admitting them, and warns per step (IS §5.x, 2026-09-17). Collider vertices against body triangles
are candidates like any other PT pair.

### 4.2 Unsigned distance and its gradient

Let $q(x)$ and $r(x)$ be the closest points of the two primitives (unique whenever $d>0$ and
the primitives are non-parallel). With barycentric weights of the closest points frozen:

* PT: $q = p$, $r = \sum_m w_m t_m$ ($w\ge0$, $\sum w_m = 1$),
  $d = \|p - r\|$, $\hat n = (p-r)/d$,
  $\nabla_p d = \hat n$, $\nabla_{t_m} d = -w_m\hat n$.
* EE: $q = (1-s)a_0 + s a_1$, $r = (1-u)b_0 + u b_1$, $\hat n = (q-r)/d$,
  $\nabla_{a_0} d = (1-s)\hat n$, $\nabla_{a_1} d = s\hat n$,
  $\nabla_{b_0} d = -(1-u)\hat n$, $\nabla_{b_1} d = -u\hat n$.
* PH: $d = n^\top(p-o)$ (signed, $n$ unit), $\nabla_p d = n$.

**Justification.** $d(x) = \min_{(\theta)} \|q(x,\theta) - r(x,\theta)\|$ over the compact
parameter domain; by Danskin's theorem the derivative w.r.t. $x$ equals the partial derivative
at the minimizing parameters, which is the closed-point formula above. This coincides with the
sub-case (PP/PE/PT, and the nine EE sub-cases) gradients of the IPC toolkit where the minimizer
is unique, and provides a valid sub-gradient at sub-case boundaries and for parallel edges.
The device implementation uses the GTE closest-point queries already vendored in
`coupled_solver/src/collision/ccd/distance.cu` (they return $w$ / $(s,u)$ and $\hat n$);
implementation-side unit tests compare $\nabla d$ against central finite differences.

**Degenerate distance.** $d \le d_{\text{tiny}}$ (default $10^{-12}$ m) can only occur if the
anchor $x$ is not strictly separated; then $\hat n$ falls back to the triangle normal (PT) or
$\pm\,\widehat{(a_1-a_0)\times(b_1-b_0)}$ oriented by the previous linearization (EE), and the
event is logged. The CCD of §10 keeps $d_i(x) \ge \xi_i + s\,(d_i^{\text{prev}}-\xi_i) > \xi_i$
on every anchor, so this is a safety net, not a code path in normal operation.

### 4.3 Linearization point

All distances and gradients used by the constraints are evaluated at the **anchor $x$**
(intersection-free), never at $\hat x$. This is what makes the linearized constraint a
signed quantity with a consistent orientation even when $\hat x$ penetrates.

### 4.4 Offsets and thicknesses

Each primitive carries a thickness $\xi$ (0 for tetrahedral surfaces, half-thickness for
shells/rods, user value for colliders). For a pair, $\xi_i = \xi_A + \xi_B$ and the constraint
offset is

$$
\delta_i = \hat d + \xi_i,
$$

where $\hat d$ is the scene-level contact offset ([Z25] "$\delta$", Table 2: $10^{-3}$ m
typical). CCD (§10) enforces $d \ge \xi_i$ along the path; the constraints ask for $d \ge \delta_i$
at the subproblem solution. (This split is the one used by [UIPC-AL]: CCD with `thickness`,
constraint offset `thickness + d_hat`.)

---

## 5. Linearized constraints and the augmented Lagrangian

### 5.1 Constraint

For $i\in\mathcal C$ with linearization at the anchor $x$:

$$
c_i(\hat x) \;=\; g_i + \nabla d_i(x)^\top(\hat x - x),
\qquad g_i := d_i(x) - \delta_i .
$$

Feasibility of the subproblem is $c_i(\hat x)\ge0$ for all $i\in\mathcal C$. Storing
$(g_i, \nabla d_i)$ per pair makes $c_i$ a 12-term dot product on the displacement
$\hat x - x$. The algebraically equivalent absolute form $d^0_i + \nabla d_i^\top\hat x$ with
$d^0_i = d_i(x) - \nabla d_i^\top x - \delta_i$ (the form [UIPC-AL] stores) subtracts
scene-scale coordinates to recover a millimetre gap; it must not be used: in single
precision it destroys the gap entirely, and in double it discards six to seven digits for
nothing.

### 5.2 Augmented Lagrangian with slacks

Introduce slacks $s_i \ge 0$ so that $c_i - s_i = 0$, and

$$
\mathcal L(\hat x, s;\lambda,\gamma) \;=\; E(\hat x) + \sum_{i\in\mathcal C}\gamma_i
\Big[\tfrac{\mu}{2}\,(c_i(\hat x)-s_i)^2 - \lambda_i\,(c_i(\hat x)-s_i)\Big].
$$

($\gamma_i$ is the decay weight of §8; $\mu$ is scene-wide in v1, per-vertex-minimum is a
straightforward generalization used by [UIPC-AL].)

### 5.3 Slack elimination (closed form)

Minimizing over $s_i\ge0$ at fixed $\hat x$ gives (Appendix A)

$$
s_i^\star = \max\!\big(0,\; c_i(\hat x) - \lambda_i/\mu\big),
\qquad
\mathcal L(\hat x, s^\star) = E(\hat x) + \sum_i \gamma_i\,\tfrac{\mu}{2}\Big(\max(0,\,\lambda_i/\mu - c_i(\hat x))\Big)^2 + \text{const}.
$$

This is the Powell–Hestenes–Rockafellar functional for inequality constraints: a one-sided
quadratic on the *shifted* gap $c_i - \lambda_i/\mu$. A pair is **active** at $\hat x$ iff
$c_i(\hat x) < \lambda_i/\mu$ (equivalently $s_i^\star = 0$).

### 5.4 Gradient and the Newton matrix

$$
G(\hat x) = \nabla E(\hat x) + \sum_i \gamma_i\,\mu\,\min\!\big(0,\; c_i(\hat x)-\lambda_i/\mu\big)\,\nabla d_i(x),
$$

$$
H(\hat x) = \big[\nabla^2 E(\hat x)\big]^{+} + \sum_{i\in\mathcal C} \gamma_i\,\mu\,\nabla d_i(x)\nabla d_i(x)^\top .
$$

$[\nabla^2E]^+ = M + h^2\big(\sum_e H_e^+ + H_{\text{f}}^+\big)$ with the projected element
Hessians of §3.3 and the projected friction Hessian of §13. The contact term is kept for
**every** pair in $\mathcal C$, active or not ([Z25] §4.2): for inactive pairs it is not the
true Hessian (which is zero) but a PSD regularizer whose weight $\gamma_i$ decays with
inactivity (§8). $H$ is SPD on the free DOFs whenever $M\succ0$.

Each pair contributes a rank-one $12\times12$ block $\gamma_i\mu\,\nabla d_i\nabla d_i^\top$:
four diagonal and six upper off-diagonal $3\times3$ blocks.

### 5.5 Line-search model

Inside one Newton step the slacks are frozen at their value $s_i$ computed at the step's
start. The energy compared by the line search is

$$
\mathcal L_s(\hat x) = E(\hat x) + \sum_i \gamma_i\,\tfrac{\mu}{2}\,\big(c_i(\hat x) - s_i - \lambda_i/\mu\big)^2 ,
$$

which equals $\mathcal L(\hat x, s)$ of §5.2 up to a constant. For active pairs
$\mathcal L_s$ agrees with the eliminated functional along any direction that keeps them
active; for inactive pairs it is a two-sided well centered at the current gap, consistent with
the Hessian choice of §5.4. ([UIPC-AL] implements exactly this: `update_slack` folds
$s_i + \lambda_i/\mu$ into $d^0_i$ before the energy/gradient/Hessian kernels.)

---

## 6. The time step (outer loop)

This is [Z25] Algorithms 1–3 written in the operational order that shares the CCD query
between the step bound and the active-set expansion (the order [UIPC-AL] uses). One outer
iteration = one call of the subproblem solver (§7) + one CCD (§10) + one active-set update
(§9). $k$ counts completed outer iterations.

```text
STEP(x^t, v^t, C^t):
  1  x̃ ← x^t + h v^t + h² M⁻¹ f_ext  (free);  targets x_P^* ← scripts/keyframes at t+h   (§12)
  2  x ← x^t;  x̂ ← x^t;  x̂_P ← x_P^*                                                   (§12)
  3  μ ← C_μ · max_{v∈F} max_{c} [ M + h² ∇²U_el(x^t) ]_{(v,c),(v,c)}   unless fixed   (§11)
  4  friction snapshot F from C^t at x^t                                                 (§13)
  5  C ← C^t;  β ← 1;  k ← 0;  stall ← 0
  6  loop
  6.1   for i ∈ C: (d_i, ∇d_i) at x;  g_i ← d_i − δ_i                                     (§5.1)
  6.2   x̂ ← SolveSubproblem(x̂, x, C, μ)                                                  (§7)
  6.3   for i ∈ C: active_i ← (λ_i − μ c_i(x̂) > 0);  λ_i ← max(0, λ_i − μ c_i(x̂));
                    n_i ← 0 if active_i else n_i + 1                                          (§8)
  6.4   (α, {T_j}_{j∈I'}) ← CCD(x → x̂)   [I' = hit candidates ∉ C; α ∈ [0,1]]           (§10)
  6.5   C ← C ∪ Filter(I', {T_j}) with λ=0, n=0;   C ← C \ { i : Γ^{n_i} < γ_min }      (§9, §8)
  6.6   if α > α_lb:  x ← x + α (x̂ − x);  stall ← 0   else  stall ← stall + 1
  6.7   k ← k + 1;   if k ≥ K_min:  β ← (1 − α) β
  6.8   if stall ≥ N_stall: μ ← 2μ; d̂ ← d̂/2; stall ← 0                                   (§11.2)
  6.9   if β ≤ ε or k ≥ k_max: break
  7  x^{t+1} ← x;  v^{t+1} ← (x − x^t)/h;  C^{t+1} ← C;  x̂ discarded
```

Remarks.

* The anchor $x$ moves only along segments that CCD certified; $x^{t+1} = x$ is therefore
  intersection-free (and inversion-free for NH). $\hat x$ is never exported.
* Line 6.1 re-linearizes **every** pair each outer iteration (the anchor moved). Inside §7 the
  linearization is fixed.
* Line 6.3 uses the final $\hat x$ of the subproblem; the updated $\lambda$ is what the
  **next** subproblem sees, and what the friction snapshot of the next step will read.
* Line 6.4/6.5: candidates discovered by the CCD from $x$ to $\hat x$ enter $\mathcal C$
  immediately and act in the next subproblem. [Z25] Alg. 1 calls `UpdateActiveSet` with
  $(x^{[k]}, \hat x^{[k]})$, the tail of the previous CCD segment; the pairs found there are
  exactly the pairs that blocked that CCD, but a literal reading inserts them one subproblem
  later than we do (they would first act in the solve producing $\hat x^{[k+2]}$). Sharing the
  query removes that lag; [UIPC-AL] does the same. Moving-boundary pairs ([Z25] Alg. 1
  line 3 + line 8 at $k=0$) are discovered by the first CCD here because $\hat x_{\mathcal P}$
  already sits at the target.
* The initial guess of the first subproblem is $x^t$ (not $\tilde x$), as in [Z25] Alg. 1.
* **Guarded Dirichlet advance (NH, 2026-09-21).** Line 2 puts the prescribed vertices of the
  iterate at their targets at once. An element that joins prescribed and free vertices is then
  deformed by the whole increment before the first Newton step, and when the increment exceeds
  the element (IPC's rod twist on the paper's mesh: the end caps move 4.4 mm per step, the
  543 tetrahedra next to them are 0.8-2.7 mm high) it is inverted in the iterate, where the NH
  energy is undefined; the anchor then cannot follow (§10.4) and the zone next to the driven
  region is destroyed within 20 steps (J from 0.001 to 240 at the caps, the rest of the rod
  untouched). IPC moves its boundary nodes inside the line search. Here, when the inversion
  guard is on: let $d_P = x_P^* - \hat x_P$ (zero on free vertices) and $\sigma \le 1$ the largest
  fraction for which no guarded element falls below half of the volume it had at the start of
  the step, $f_e(\theta) \ge \tfrac12 f_e(x^t)$ for $\theta\le\sigma$ (a first root per element, like
  §10.4; relative to the start of the step so that repeated advances cannot compound, and an
  element already below the bound does not constrain; going 0.9 of the way to inversion
  instead outran the relaxation of the free vertices and collapsed the elements geometrically
  in `test_guarded_dirichlet`). The step first tries a *joint start*:
  $\hat x \leftarrow x^t + \sigma_J\,w\,(\tilde x - x^t)$: the driven vertices on their targets
  ($w = 1$) and the free vertices near them at their inertial prediction, $w$ being a smoothstep
  of the rest distance to the nearest prescribed vertex of the body that reaches 0 at 5 % of
  the body's bounding-box diagonal (further away the velocities can be noisy from contact; those
  vertices start at $x^t$ as always), with $\sigma_J$ from the same rule against $f_e(0)$. In a
  steady scripted motion the material next to the driven region already moves with it, the joint
  displacement is nearly rigid there, $\sigma_J = 1$, and the iterate satisfies the boundary
  condition from the start (moving the driven vertices alone against fixed neighbours needed
  hundreds of advances on the paper's rod mesh, because an inexact CG solve relaxes only the
  nearest layers). A start from rest needs `motion.ramp_time`. If $\sigma_J < 1$, then before
  every subproblem solve while $\hat x_P \ne x_P^*$: $\hat x_P \leftarrow \hat x_P + \sigma d_P$.
  The solve relaxes the free vertices around the new boundary positions, so the next advance
  reaches further. An outer iteration whose solve ran with $\hat x_P \ne x_P^*$ resets
  $\beta \leftarrow 1$: the termination weight then counts only iterates that satisfied the
  boundary condition, and the step cannot end before the targets are reached. When
  $\sigma = 1$ at the start of the step, which is every scene so far except the one above, the
  algorithm is unchanged. A pull-back (§10.4) resets the iterate's prescribed vertices to the
  anchor as well, so it re-arms the advance (before this rule the rest of that step's
  prescribed increment was silently deferred to the next step). Statistics: `bc_partial_iters`.
* $k_{\max}$ is a safety cap (default 500); hitting it is logged as a failure of the step,
  the step still returns the valid $x$.

---

## 7. Subproblem solver (projected Newton with line search)

```text
SolveSubproblem(x̂, x, C, μ):
  repeat
     for i ∈ C:  s_i ← max(0, c_i(x̂) − λ_i/μ);   fold: g̃_i ← g_i − s_i − λ_i/μ
     G ← ∇E(x̂) + Σ_i γ_i μ (g̃_i + ∇d_iᵀ (x̂ − x)) ∇d_i     (only free DOFs)
     H ← [∇²E(x̂)]⁺ + Σ_i γ_i μ ∇d_i ∇d_iᵀ                  (free DOF rows/cols)
     p ← −H⁻¹ G   via PCG, block-Jacobi, ‖r‖ ≤ tol_rel ‖G‖
     r ← 1;  [NH only: r ← min(1, 0.9·t_inv(x̂, p))]
     while L_s(x̂ + r p) > L_s(x̂) + tol_E and r > r_min:  r ← r/2
     x̂ ← x̂ + r p
  until the first trial r_0 was accepted (no energy backtracking) or inner cap reached
  return x̂
```

Here $r_0$ is the first trial of the line search: 1, or for NH the inversion bound
$\min(1, 0.9\,t_{\text{inv}})$ (revised 2026-09-21, below).

* $\mathcal L_s$ is §5.5; $\nabla E$, $\nabla^2E$ are §2/§3/§13. Prescribed DOFs are not
  unknowns: their rows/columns are dropped (diagonal identity, zero right-hand side), their
  values inside $c_i$ and inside element energies are the targets.
* Stopping on "full step accepted" is [Z25]'s rule: for quadratic $E$ (linear elasticity) the
  loop runs once; for nonlinear elasticity it takes extra steps only when the line search had
  to backtrack. [UIPC-AL] implements the same rule (`flag_clamped` → repeat Newton without
  CCD). The inner cap (default 8) prevents pathological loops; hitting it proceeds to CCD.
* **An inversion-limited step counts as accepted (2026-09-21).** Until then the loop asked for
  $r = 1$ literally, and the NH bound $r \le 0.9\,t_{\text{inv}}$, which [Z25] does not have,
  made that impossible whenever the full Newton step would invert an element: on IPC's rod
  twist (tightly wound, frame 800) the bound was 0.35-0.88 in almost every Newton step with no
  energy backtracking at all, so every outer iteration ran the 8 capped Newton steps (7.7 on
  average, 780 Newton steps per time step for 100 outer iterations) while the increment did
  not shrink within them (7.5e-3 to 1.0e-2 m/s). The rule is about the *energy* line search:
  the loop repeats only when that search had to halve its first trial. Safeguard: a bound
  below 0.1 is not accepted as a step (the loop repeats), so that a Newton direction that
  inverts an element almost at once cannot end the subproblem with next to no motion.
  With the stricter rule below, the tolerance is tested on the accepted increment $r\,p$.
* Newton residual is **not** a stopping criterion; accuracy is controlled by $K_{\min}$ and
  $\varepsilon$ (§15). An optional displacement-based early accept
  ($\max_v\|p_v\|/h < v_{\text{tol}}$, as in [UIPC-AL]) is exposed but off by default.
* PCG tolerance: relative residual $10^{-4}$ ([Z25] §6), maximum iterations 2000; failure to
  converge is logged, the last iterate is used.
* Optional stricter rule (2026-09-07, `newton.increment_velocity_tol` $v_{\text{tol}} > 0$): a
  full step is still required, but the loop continues while
  $\sqrt{\sum_{v\,\text{free}} |p_v|^2 / n_{\text{free}}}\,/h \ge v_{\text{tol}}$, i.e. while the
  accepted increment's RMS velocity is above the tolerance (C-IPC's PNTol, $5\times10^{-4}$ m/s
  for the card shuffle), up to the inner cap. Needed when a stiff shell's elastic snap must
  complete within a step: under the [Z25] rule a released bowed card (E = 3 GPa) took two
  Newton steps per frame and straightened by creep over ~50 frames instead of one.

---

## 8. Multiplier, decay and removal

After the subproblem, at its final $\hat x$:

$$
\lambda_i \leftarrow \max\!\big(0,\ \lambda_i - \mu\, c_i(\hat x)\big),
\qquad
n_i \leftarrow \begin{cases} 0 & \text{if } \lambda_i - \mu\,c_i(\hat x) > 0 \ (\text{active}),\\ n_i + 1 & \text{otherwise (inactive)},\end{cases}
\qquad \gamma_i = \Gamma^{\,n_i}.
$$

The first line is $\lambda_i \leftarrow \lambda_i - \mu(c_i - s_i^\star)$ evaluated with
$s_i^\star = \max(0, c_i-\lambda_i/\mu)$: it equals $\lambda_i - \mu c_i$ when active and $0$
when inactive ([Z25] Eq. 13 in both branches). The activity test uses the pre-update
$\lambda_i$; equivalently, a pair is active iff its updated multiplier is positive.

**Decay semantics.** $\gamma_i$ is multiplied by $\Gamma$ for each consecutive outer iteration
in which the pair is inactive and reset to 1 as soon as it is active again; a pair is removed
from $\mathcal C$ when $\gamma_i < \gamma_{\min}$ ($\gamma_{\min}=0.01$, i.e. after
$\lceil \ln\gamma_{\min}/\ln\Gamma\rceil = 44$ consecutive inactive iterations at $\Gamma=0.9$).
This is the prose of [Z25] §4.2–4.3 and their Appendix A ("any constraint with $\gamma_i<1$
necessarily yields $s_i>0$"); the printed listing of their Algorithm 2 (lines 20–26) shows the
two $\gamma$ assignments swapped and is treated as a typesetting error. [UIPC-AL] implements
the prose semantics with an integer counter (`cnt`) and weight `decay^cnt`, removing at
`|cnt| > 25`; we keep the paper's threshold and expose both knobs.

$\mathcal C$ persists across time steps with its $\lambda_i$ and $n_i$ (warm start of the
multipliers is what gives the finite-step termination its bite).

---

## 9. Active-set expansion with earliest-impact filtering

Input: candidates $I'$ = pairs reported as **hits** by the CCD of the current outer iteration
(§10) that are not already in $\mathcal C$, each with its time of impact $T_j\in[0,1)$.

1. For every vertex $v$: $T_v = \min\{T_j : j\in I',\ v\in j\}$ ($T_v = +\infty$ if none).
2. Keep $j\in I'$ iff $\exists\, v\in j:\ T_j \le T_v + \tau_T$ (i.e. $j$ is the earliest impact
   of at least one of its vertices; $\tau_T = 10^{-6}$ absorbs floating-point ties).
3. Deduplicate against $\mathcal C$ by pair identity (type + canonical primitive ids) and
   insert with $\lambda_j = 0$, $n_j = 0$.

$T_v$ ranges over the **new** hits $I'$ only, as in [Z25] Alg. 3 (which removes members of
$\mathcal C$ before taking the per-vertex minimum). [UIPC-AL] takes the minimum over all
candidates including members of $\mathcal C$, a stricter keep rule that admits fewer pairs per
iteration; the two are not equivalent and the choice is exposed as
`contact.toi_filter_domain` (default: paper).

Property (from [Z25] §4.3): a pair that keeps blocking the CCD will eventually be the earliest
impact of one of its vertices (once the earlier pairs of those vertices are in $\mathcal C$ and
resolved), so filtering never causes stagnation; it prevents a fast primitive that sweeps
through several layers from inserting all layers at once. [UIPC-AL] (`filter_new_candidates`)
implements steps 1–2 with an atomic-min per vertex; [UIPC-AL] main branch skips the filter.

---

## 10. Continuous collision detection and the step bound

### 10.1 Query

Given the anchor $x$ (intersection-free) and the iterate $\hat x$, the linear path
$x(\theta) = x + \theta(\hat x - x)$, $\theta\in[0,1]$, is tested for all candidate pairs of
§4.1.

* Broad phase: swept AABBs of every surface vertex, edge and triangle over $[x, \hat x]$,
  padded by the primitive thickness plus a margin; vertex-vs-triangle tree query and
  edge-vs-edge tree query (LBVH, §impl). Pairs sharing a vertex and pairs excluded by the
  contact table are discarded.
* Narrow phase: additive CCD (ACCD, Li et al. 2021) with minimum separation $\xi_i$ and
  early-stop ratio $s$ (default 0.1): conservative advancement with a bound on the relative
  displacement returns $t_i\le1$, a lower bound on the first $\theta$ at which the gap
  $d_i(x(\theta)) - \xi_i$ would shrink to $s$ times its initial value $d_i(x)-\xi_i$
  (so $d_i \ge \xi_i + s\,(d_i(x)-\xi_i)$ holds on $[0,t_i]$). A pair is a **hit** iff $t_i < 1$.
* Planes: closed form. With gap $g = n^\top(p-o) - \xi$ and $\Delta = \hat x_p - x_p$:
  no hit if $n^\top\Delta \ge 0$, else $t = \min(1, (1-s)\,g / (-n^\top\Delta))$.
* Spheres: closed form, with the radius allowed to move linearly across the step, $r(\theta) =
  r_0 + \theta(r_1 - r_0)$ (§12.3). Let $u = p - c$, $g_0 = d(0) - \xi$ and, for the inverted
  case, $q(\theta) = r(\theta) - s\,g_0 - \xi$ (for a solid sphere $q(\theta) = r(\theta) + s\,g_0
  + \xi$). The gap reaches $s\,g_0$ exactly when $|u + \theta\Delta| = q(\theta)$, i.e. at a root
  of $(r'^2 - |\Delta|^2)\theta^2 + 2(r' q_0 - u^\top\Delta)\theta + (q_0^2 - |u|^2) = 0$ with
  $r' = r_1 - r_0$, $q_0 = q(0)$. $t$ is the smallest root in $(0, 1]$ with $q(t) \ge 0$, or 1 if
  none; $g_0 \le 0$ with an approaching vertex gives $t = 0$ as for planes. This is exact, not
  a conservative-advancement bound, because $|u + \theta\Delta|$ is known in closed form.

### 10.2 Step bound

$$
\alpha_{\text{ccd}} = \min\Big(1,\ \min_{i\ \text{hit}} t_i\Big),\qquad
\alpha = \text{the largest fraction} \le \alpha_{\text{ccd}} \text{ whose end state passes §10.4 (NH only, else } \alpha_{\text{ccd}}).
$$

The anchor is advanced by $\alpha$ (§6 line
6.6). Because ACCD stops strictly before the minimum separation, $d_i(x) > \xi_i$ holds for
every candidate pair (every pair with at least one free vertex, §4.1) on every anchor, which
keeps the linearization of §4.2 well defined.

### 10.3 What is reported

For every hit pair: identity and $t_i$ (feeds §9). $\alpha$ is the global minimum. Both come
from the same query; no second CCD is run.

### 10.4 Inversion guard (NH only)

Two different things need $J > 0$, and they get different rules (revised 2026-09-21).

**The Newton line search (§7)** moves the iterate along a search direction, which is only
locally meaningful, so IPC's rule is used as it stands: for each tetrahedron the smallest
$\theta\in(0,1]$ with $\det D_s(x(\theta)) = 0$ is the first root of a cubic, and
$t_{\text{inv}}(\hat x, p) = 0.9\,\min_e \theta_e$ bounds the step; the energy line search and the
next Newton direction then steer away from $J = 0$.

**The anchor blend (§6 line 6.4-6.6)** is different. Only the blended *state* must be free of
inverted elements (it becomes $x^{t+1}$, where the NH energy must be defined); its straight
path to the iterate needs no such property, because nothing is evaluated along it except the
CCD, which certifies the surface and knows nothing about interior elements. And the far end of
the path, the iterate, is valid by the rule above. So with $\alpha_{\text{ccd}}$ from the CCD
(run without an inversion cap) and $f_e(\theta) = \det D_s^e(x_a + \theta(\hat x - x_a))$ oriented so
that $f_e(0) > 0$:

$$
\alpha = \max\Big\{\theta\in\Theta:\ f_e(\theta)\ \ge\ \eta_J\,\min\big(f_e(0),\,f_e(1)\big)\ \ \forall e\Big\},\qquad
\Theta = \big\{\alpha_{\text{ccd}}\,(1 - j/8)\big\}_{j=0}^{7},\quad \eta_J = 0.1,
$$

(an element with $f_e(1)\le 0$ is compared with $f_e(0)$ alone). Almost always the first
candidate passes and the cost is the old guard's (one kernel, one reduction).

**Pull-back.** No candidate passes when the CCD keeps $\alpha_{\text{ccd}}$ below the far end of
some element's inverted interval: the valid states beyond it cannot be reached on this chord,
and creeping toward its near end (the first-root rule) only crushes the element. Then the
anchor does not move in this outer iteration ($\alpha = 0$), the iterate is reset to it,
$\hat x \leftarrow x_a$, and the next subproblem solve is a *single* guarded Newton step from
there. Its chord $x_a \to \hat x$ is that Newton step, inversion free by the line-search rule
above, so the next blend is limited by the CCD alone, and the anchor follows the Newton path
segment by segment, which is what IPC's own iterate does. The iterate's earlier progress is
discarded; the multipliers are kept. (The first-root rule is kept only as the guard of last
resort for that one chord, should it fail the test numerically.)

*Why the first-root rule fails on the anchor path.* It was the rule until 2026-09-21. When an
element has to deform and turn within one step so far that the straight path from the anchor to
the iterate crosses $J = 0$ (a buckling snap), the iterate stays on the far side while every
outer iteration moves the anchor 0.9 of the remaining way to the singular configuration:
$\alpha = \alpha_{\text{inv}}$ shrinks geometrically. Measured on IPC's rod twist with NH
(`app/config/rod_twist_nh.json`, frame 1624, 16 relative turns): 490 of 500 outer iterations
limited by the guard, $\alpha$ from 0.28 to $1.4\cdot10^{-7}$, 46 tetrahedra left at $J = 0$,
after which CG cannot converge. With the end-state test but the first-root rule still as its
fallback the same happened 88 frames later (frame 1712: 499 of 500 outer iterations in the
fallback), which is why the fallback is the pull-back. [UIPC-AL] has no inversion check (its FEM default is the stable
neo-Hookean model, which needs none). Tests: `test_block_assembly`, end-state block (a tetrahedron
on the straight path to diag(1, -s, -1) of itself, $f = (1-(1+s)\theta)(1-2\theta)$: the margin
is 1 at $\theta = 1$ where the first-root rule stops at 0.45, negative inside the inverted
interval, 0.125 at the candidate $\theta = 0.75$; an inverted end state is refused); scene
`app/config/rod_twist_nh.json`.

---

## 11. Penalty stiffness $\mu$

### 11.1 Per-step estimate

$$
\mu = C_\mu\ \max_{v\in\mathcal F}\ \max_{c\in\{x,y,z\}}\ \big[\,M + h^2\,\nabla^2U_{\text{el}}(x^t)\,\big]_{(v,c),(v,c)},
\qquad C_\mu = 0.1,
$$

i.e. a tenth of the largest diagonal entry of the (projected) elasto-inertial Hessian at the
start of the step ([Z25] Eq. 20; [UIPC-AL] `mu_scale_mode = "diag_norm"` computes the same
maximum over the $3\times3$ diagonal blocks of the assembled system with contact disabled).

**Regime of validity** (measured 2026-09-03). The estimate is a stiff penalty only when
$h^2 K \gg m$, which holds for every material in [Z25]. In a mass-dominated scene it degenerates:
with no elasticity at all, $\mu = 0.1\,m$ and the multiplier relaxes as
$\lambda \leftarrow (m\lambda + \mu\lambda^\star)/(m+\mu)$, i.e. at rate $m/(m+\mu) = 0.91$ per
outer iteration, so the step's $\beta$ criterion ends it long before $\lambda$ converges and the
residual multiplier error shows up as a slow drift. The trajectory stays penetration-free
throughout — only the accuracy degrades. The stall rule of §11.2 is the safeguard; a fixed
$\mu$ is the alternative for such scenes.
Rationale: the contact Hessian of a pair is $\mu\,\nabla d\nabla d^\top$ with $\|\nabla d\|\le\sqrt2$
(PT/EE) so its spectral contribution is at most $2\mu$, keeping $\kappa(H)$ at the level of
$\kappa(\nabla^2E)$. Units: kg. A fixed user value is accepted for experiments.

**Upper bound (2026-09-21, `contact.mu_max`, off by default).** The estimate is a maximum over
all vertices, so one element sets it for the whole scene, and with NH that element's Hessian
grows like $1/J^2$ under compression. On IPC's rod twist $\mu$ is 0.08 at rest, 0.27 at 25 s,
2.8 at 40 s (tightest element $J = 0.17$), which is fine: the accepted fraction stays near
0.6 with 5-8 outer iterations. But when a buckling snap leaves one element nearly flat
($J = 5\cdot10^{-4}$) the estimate jumps to $4\cdot10^{5}$ and later $10^{18}$, and a penalty
that stiff crushes further elements: a feedback that ends with CG failing. Fixing $\mu$ at its
rest value is no remedy: the penalty is then too soft once the elements have stiffened, the
iterate overshoots the constraints, and the accepted fraction falls from 0.57 to 0.05-0.07
(75-100 outer iterations and four times as many active pairs at 25 s). `mu_max` keeps the
estimate and clamps it, $\mu \leftarrow \min(\mu, \mu_{\max})$; the scene uses 100 times the rest
value, which legitimate compression stays under.

**Light contacts under a stiff estimate (2026-09-22, the card shuffle).** The estimate has a
second failure mode, opposite to the stall of §11.1's regime note: the multiplier update
$\lambda \leftarrow \max(0, \lambda - \mu\,c(\hat x))$ makes a pair inactive as soon as its gap exceeds
$\lambda/\mu$, so a resting contact must hold its gap inside that window from step to step.
For the 3 GPa playing cards (54 shells, 1.9 g each) the estimate is $\mu = 1.35$ kg while a
card's weight is $\lambda = h^2 F = 7.5\cdot10^{-8}$ kg m: a window of 0.06 um, against a
Newton tolerance of 1 um per substep. Every resting contact therefore flickered between
$\lambda = 0$ and twice the weight, 105 of 1560 pairs were dropped and re-found in every step,
400 CCD hits per step limited the anchor, and the cards jittered by 0.4-0.5 mm per frame with
their heights wandering over 2 mm, measured on the moving-boundary hand shuffle. The other bound is set by
the pile: the bottom contact of an $n$-card pile carries $n$ weights and its violation
$n\lambda/\mu$ must stay below $\delta - \xi$, the room the CCD leaves, or the multipliers
never converge and the pile bounces. Measured on the resting piles, 40 frames each (motion of a
card per 20 ms frame; the CCD room is 0.2 mm):

| $\mu$ | window per weight | motion per frame | CCD hits per step |
| --- | --- | --- | --- |
| 1.35 (estimate) | 0.06 um | 0.43 mm | 1110 |
| 0.03 (`mu_max`) | 2.5 um | 0.07 mm | 4 |
| 0.01 | 7.5 um | 0.16 mm (0.075 with Newton tolerance 1e-4) | 13 (1) |
| 0.003 | 25 um (0.7 mm at the pile's bottom) | 0.54 mm | 129 |
| 0.001 | 75 um (2 mm at the bottom) | 0.82 mm | 296 |

So $\mu$ has to sit between "window above the solver's accuracy" and "pile violation below
the CCD room": $\lambda/\mu \gg \varepsilon_{\text{Newton}} h$ and $n\lambda/\mu < \delta - \xi$, which
for this scene leaves 0.01-0.03. Converging Newton tightly (cap 32, tolerance 1e-4) achieves
the same with the estimate, at five times the cost. `mu_max` = 0.03 is the scene's setting. A
per-pair $\mu_i$ chosen from the pair's own $\lambda_i$ would remove the guess; not built.

### 11.2 Stall adaptation

If $\alpha < \alpha_{\text{stall}}$ ($10^{-4}$) for $N_{\text{stall}}$ (50) consecutive outer
iterations: $\mu \leftarrow 2\mu$, $\hat d \leftarrow \hat d/2$ (thicknesses unchanged), counter
reset; at most $N_{\text{adapt}}$ (10) adaptations per step. [Z25] report the rule never fired in
their experiments; it is kept as the theoretical safeguard.

---

## 12. Prescribed and moving boundaries

Prescribed vertices $\mathcal P$: Dirichlet-selected mesh vertices with scripted or keyframed
targets, and all collider vertices (static colliders have constant targets).

* At step start, $\hat x_{\mathcal P} = x^{\star}_{\mathcal P}(t+h)$ (the target), while the
  anchor keeps $x_{\mathcal P} = x^t_{\mathcal P}$. Every iterate $\hat x$ satisfies the boundary
  condition exactly; the anchor's boundary vertices move only through the blends of §6 line 6.6.
* Prescribed DOFs are removed from the linear system (§7). Their displacement
  $\hat x_{\mathcal P} - x_{\mathcal P}$ still enters $c_i(\hat x)$ for pairs touching them and
  enters the CCD of §10, so free vertices are pushed out of the way of an oncoming boundary
  within the same step ([Z25] §5.3: no auxiliary spring, no extra iterations).
* Position error at the end of the step: with $\beta_0$ the weight of $x^t$ in $x^{t+1}$
  (Appendix C), $x^{t+1}_{\mathcal P} = \beta_0 x^t_{\mathcal P} + (1-\beta_0)x^\star_{\mathcal P}$,
  hence $\|x^{t+1}_{\mathcal P} - x^\star_{\mathcal P}\| = \beta_0\,h\,\|v_{\mathcal P}\| \le \varepsilon\,h\,\|v_{\mathcal P}\|$
  (termination guarantees $\beta_0 \le \beta \le \varepsilon$). Targets of the next step are
  computed from the script, not from the lagging position, so the lag does not accumulate.
* Velocity of prescribed vertices: $(x^{t+1}_{\mathcal P} - x^t_{\mathcal P})/h$ (used by friction
  through the relative displacement, §13).

The set $\mathcal P$ is fixed for the whole simulation in v1 (release/attach events would
change the DOF layout; deferred).

---

### 12.3 Analytic colliders with a prescribed radius

A sphere collider carries a radius schedule $r(t)$, piecewise linear in the keyframes given
and held after the last one. Over the step $t \to t+h$ it is treated exactly as a prescribed
vertex is (§12.1): the constraint of a PS pair is linearized against the end-of-step sphere
$r_1 = r(t+h)$, so every iterate sees the boundary where it will be, and the CCD of §10.1
interpolates $r(\theta)$ from the wall's *anchor* radius $r_a$ to $r_1$ along the same $\theta$
as the vertices. $r_a$ starts the step at $r(t)$ and is blended with every accepted step exactly
as the vertex anchor is (§6 line 6.6): $r_a \leftarrow r_a + \alpha\,(r_1 - r_a)$. Without this
the second and later CCDs of a step would measure against a wall that has not moved while the
vertices have, and certify paths that cross the real one; with it the certified path is
intersection-free against the moving wall, and at termination $r_a$ lags $r_1$ by at most
$\epsilon\,|r_1 - r_0|$, the analogue of the prescribed-vertex bound of §12.1. The radius change is purely
normal to the wall, so the friction snapshot of §13 needs no collider-velocity term: the
tangential relative motion is the vertex's own. A sphere whose radius shrinks by more than a
vertex's distance to it in one step is a scene error of the same kind as a collider driven
through a body (§12.2), and is reported by the same debug sweep.

## 13. Friction (semi-implicit, lagged)

Same model as IPC (Li et al. 2020), with the normal force taken from the contact solution of
the previous step ([Z25] Appendix A).

**Snapshot at step start (§6 line 4).** Re-linearize every $i\in\mathcal C^t$ at $x^t$
(so $c_i(x^t) = d_i(x^t) - \delta_i$) and set

$$
\hat F_i = \max\!\big(0,\ \lambda_i - \mu\,(d_i(x^t)-\delta_i)\big)
\quad(\text{units of } E\text{-gradient, i.e. } h^2\times\text{force}),\qquad
F_i = \hat F_i/h^2 \ \ (\text{N}).
$$

Pairs with $\hat F_i > 0$ form the friction set $\mathcal F_{\text{fr}}$ for the step, each
with frozen closest-point weights ($w$ for PT, $(s,u)$ for EE) and a frozen orthonormal tangent
basis $T_i\in\mathbb R^{3\times2}$ of the contact plane at $x^t$ (PT: $e_1 \parallel t_1-t_0$,
$e_2 = e_1\times\hat n_\triangle$-orthogonalized; EE: $e_1\parallel a_1-a_0$,
$e_2 \perp e_1$ in the plane spanned by the two edges; PH: any basis of the plane).
$\mu$ here is the value the previous step ended with. ([UIPC-AL] uses $\hat F_i = \lambda_i$,
the converged limit of the same expression; a config switch selects it.)

**Tangential relative displacement over the step**, for PT:

$$
u_i(\hat x) = T_i^\top\Big[(\hat x_p - x^t_p) - \sum_m w_m (\hat x_{t_m} - x^t_{t_m})\Big]\in\mathbb R^2,
$$

for EE with $(1-s, s, -(1-u), -u)$ weights, for PH $u = T^\top(\hat x_p - x^t_p)$. $u_i = J_i(\hat x - x^t)$
with constant $J_i\in\mathbb R^{2\times12}$.

**Potential** (already scaled to $E$ units):

$$
h^2 U_{\text{f}}(\hat x) = \sum_{i\in\mathcal F_{\text{fr}}} \mu_{f,i}\,\hat F_i\, f_0\!\big(\|u_i\|;\ \epsilon_v h\big),
\qquad
f_0(y;\epsilon) = \begin{cases} y, & y\ge\epsilon,\\[2pt] -\dfrac{y^3}{3\epsilon^2} + \dfrac{y^2}{\epsilon} + \dfrac{\epsilon}{3}, & 0\le y<\epsilon,\end{cases}
$$

$C^1$ with $f_0'(y)/y = 1/y$ for $y\ge\epsilon$ and $(2\epsilon - y)/\epsilon^2$ below. Gradient
$J_i^\top\,\mu_f\hat F_i\,(f_0'(\|u\|)/\|u\|)\,u$; Hessian $J_i^\top H_2 J_i$ with the IPC $2\times2$
formula (exact tangential part for $\|u\|\ge\epsilon$, PSD-projected otherwise). $\mu_{f,i}$ is
the pair's Coulomb coefficient from the contact table (default: geometric mean of the two
bodies' coefficients).

Friction pairs are constants for the whole step; they are not part of $\mathcal C$'s update
and do not need CCD. Momentum: $\nabla d_i$ and $J_i$ both sum to zero over the pair's
vertices, so contact and friction forces are internal (test oracle §17.2).

---

## 14. End of step

$x^{t+1} = x$ (anchor), $v^{t+1} = (x^{t+1}-x^t)/h$ for all vertices, $\mathcal C^{t+1} = \mathcal C$
with $\lambda_i, n_i$; the friction snapshot of the next step is built from it (§13). Export
uses $x^{t+1}$ only.

---

## 15. Termination

### 15.1 Weights

With $\alpha^{[j]}$ the step fractions, the anchor after $k$ iterations is the convex
combination $x = \beta_0 x^t + \sum_{j=1}^{k}\beta_j \hat x^{[j]}$ with
$\beta_j = \alpha^{[j]}\prod_{l=j+1}^{k}(1-\alpha^{[l]})$, $\alpha^{[0]}:=1$ (Appendix C).
The stopping quantity is $\beta = \sum_{j<K_{\min}}\beta_j$: the total weight of $x^t$ and of
the first $K_{\min}-1$ iterates. The recursion in §6 line 6.7 (freeze at 1 until $k\ge K_{\min}$,
then multiply by $1-\alpha$) reproduces it exactly.

### 15.2 Guarantees (informal, from [Z25] §4.4)

* Every accepted anchor is intersection-free by construction (§10).
* Non-contacting regions keep at least $(1-\varepsilon)$ of their intended displacement, so
  TOI clamping in a contact-rich region cannot freeze the rest of the scene.
* With $\alpha$ bounded away from zero the weights decay geometrically and the loop terminates;
  $\alpha>0$ is produced because pairs that block the CCD enter $\mathcal C$ and the
  multiplier iteration drives their linearized gaps to $\delta_i>0$.
* First-order accuracy in $h$ (analysed for linear elasticity, verified numerically in §17.1).

### 15.4 Optional displacement continuation (2026-09-07, `newton.increment_velocity_tol`)

With $v_{\text{tol}} > 0$ (the same key as the inner rule of §7) the outer loop does not stop
at $\beta \le \varepsilon$ while the last accepted subproblem solution still moved the free
vertices at an RMS velocity $\sqrt{\sum_{v\,\text{free}} |\hat x_v - x_{a,v}|^2 / n_{\text{free}}}\,/h
\ge v_{\text{tol}}$; it re-linearizes $\mathcal C$ at the new anchor and solves again, at most
$K_{\min}$ extra iterations per step (a creeping or oscillating scene would otherwise run to
$k_{\max}$: the 54-card packets did). Rationale: the $\beta$ rule treats a full, unobstructed step as
converged, but the step solved a *linearized* contact problem; if that solution moved the
system, the linearization at the new anchor can differ. Measured on the card shuffle it does
not resolve the release jam (§15.5): the linearized problem's minimum there is stationary, so
no amount of re-linearization at the same anchor changes it. Off by default; the paper's
counts are unaffected.

### 15.5 A limitation of the linearized contact model (card shuffle, 2026-09-07)

A released, bowed card whose neighbour 0.31 mm away is still held cannot straighten: its top
row would have to slide past the neighbour's held top row (0.2 mm apart horizontally, 5 mm of
travel), and the edge-edge pair's gradient at the anchor, $\nabla d \propto (1,1)/\sqrt 2$, makes
the linearized gap decrease along that path, so the subproblem's constrained minimum is the
bowed card pressed under the row: a false equilibrium that persists over frames (the card
straightens by creep, ~0.3 mm per frame, until the neighbour is released). An isolated card
snaps flat in one frame. Five substeps, three times the spacing, $K_{\min} = 6$, PCG
tolerance $10^{-8}$, the §7 tolerance and §15.4 all leave it.
The barrier of C-IPC is evaluated at every Newton iterate, so its normals follow the
motion and the corner is passed within the step. The scenes of [Z25] slide along smooth or
flat surfaces, where the anchor gradient stays valid over the step.

### 15.3 Parameters and defaults

| Symbol | Config key | Default | Source |
| --- | --- | --- | --- |
| $\varepsilon$ | `contact.epsilon` | $10^{-3}$ | [Z25] §6 |
| $K_{\min}$ | `contact.K_min` | 2; 6 with friction or shells | [Z25] §6 |
| $\Gamma$ | `contact.decay_factor` | 0.9 | [Z25] §4.2 |
| $\gamma_{\min}$ | `contact.decay_remove_threshold` | 0.01 | [Z25] §4.3 |
| $C_\mu$ | `contact.mu_scale` | 0.1 | [Z25] §5.2 |
| $\hat d$ | `contact.d_hat` | scene ($10^{-3}$ m typical) | [Z25] Table 2 |
| $s$ (ACCD) | `contact.ccd.s` | 0.1 | ACCD |
| $\alpha_{\text{lb}}$ | `contact.alpha_lower_bound` | $10^{-6}$ | [UIPC-AL] |
| $N_{\text{stall}},\alpha_{\text{stall}}$ | `contact.stall.*` | 50, $10^{-4}$ | [Z25] §5.2 |
| $k_{\max}$ | `contact.max_outer_iters` | 500 | ours |
| inner cap | `newton.inner_max_iters` | 8 | ours |
| increment velocity tolerance | `newton.increment_velocity_tol` | 0 (off); 5e-4 m/s for the card shuffle | C-IPC PNTol |
| PCG tol | `linear_solver.rel_tol` | $10^{-4}$ | [Z25] §6 |
| $\epsilon_v$ | `contact.friction.eps_v` | $10^{-3}$ m/s | [Z25] Table 2 |
| $\tau_T$ | `contact.toi_tie_tolerance` | $10^{-6}$ | [UIPC-AL] |

---

## 16. Decisions and deviations

| Topic | [Z25] | [UIPC-AL] | This spec |
| --- | --- | --- | --- |
| Inner Newton loop | repeat until a full step is accepted | same (`flag_clamped`), plus `min_iter−1` pure Newton steps before the first CCD | paper rule with a cap; no pre-CCD Newton steps; an NH inversion-limited step (bound >= 0.1) counts as accepted (§7) |
| $\gamma$ update branches | prose: decay when inactive; printed Alg. 2 swapped | decay when inactive (counter) | decay when inactive (§8) |
| Removal | $\gamma<0.01$ | counter $>25$ | $\gamma<\gamma_{\min}$, default 0.01 (configurable) |
| Earliest-impact filter | yes | AL-release: yes; main: no | yes |
| $\beta$ accounting | $\sum_{j<K_{\min}}\beta_j$ | tracks $\beta_0$ only, `min_iter` floor | paper |
| $\mu$ estimate | $C\max\text{diag}$ | `diag_norm` mode = same; legacy per-vertex mass·scale·$h^2$ | paper; per-vertex reserved |
| Friction normal force | $h^{-2}\mu(c-s-\lambda/\mu)$ at $x^t$ | $\lambda_i$ | paper (switch to $\lambda_i$) |
| Moving boundaries | penalty-free elimination (§5.3) | soft constraints (WIP) | paper |
| Elastic Hessian assembly | per-vertex-block analytic projection, warp reduction | generic triplets (WIP) | paper's block formula, gather-by-block (impl spec) |
| Parallel edge–edge pairs | not discussed | dropped (mollifier test) | kept with sub-gradient; option to drop |
| One-ring surface pairs | not discussed | kept | kept (option to skip for tet surfaces) |
| Stall adaptation | specified, never triggered | absent | specified |
| CCD-then-update ordering | Alg. 1 indexing (see §6 remarks) | shared query | shared query |
| Stored constraint constant | not specified | absolute form $d^0_i + \nabla d_i^\top\hat x$ | displacement form $g_i + \nabla d_i^\top(\hat x - x)$ (§5.1) |
| $T_v$ domain in the earliest-impact filter | new pairs only (Alg. 3) | all candidates incl. $\mathcal C$ | paper (switchable, §9) |
| Pairs with all vertices prescribed | not discussed | kept as constraints | excluded from candidates (§4.1) |
| Broad-phase padding | not discussed | $\hat d$ + thickness on both sides | thickness only (hits, not proximity, §10.1) |
| Analytic sphere colliders | Fig. 21 uses one (inverted, shrinking) via Cubic Barrier's SDF | absent | PS pairs with closed-form moving-radius CCD (§4.1, §10.1, §12.3) |
| Cloth membrane model | $(\mu_{\text{mem}}, k_{\text{bend}})$ only; scenes from OGC | NeoHookeanShell | cloth-StVK by default with $\mu_m = \mu_{\text{mem}}$; COR and SNH shells selectable (§3.5); the mapping is an open decision |
| Cloth bending Hessian | not discussed | full $12\times12$ with PSD projection | Gauss–Newton rank-one by default, full behind a flag (§3.7) |
| Cloth $K_{\min}$ | 6 for cloth and friction scenes, 2 otherwise | `min_iter` | scene-level `K_min`; a per-body value is not needed while scenes are pure cloth or pure solid |

---

## 17. Validation oracles

1. **First-order convergence, 1D bar.** Linearly elastic bar (or a rod of tets) against a
   fixed plane: (a) initial velocity, no gravity; (b) gravity, no initial velocity. Reference:
   the same spatial discretization stepped with $h_{\text{ref}}$ four decades smaller.
   Accumulated position and velocity errors vs $h$ must show slope 1 on a log–log plot
   ([Z25] Fig. 7).
2. **Momentum.** Two bodies, one moving, frictional impact, no gravity, no prescribed
   vertices: total linear momentum constant to solver tolerance every step (§13).
3. **Penetration-free invariant.** After every step, a discrete distance sweep over all
   surface pairs reports $d \ge \xi$ (strictly) — checked in debug builds.
4. **Coulomb threshold.** Cube on a slope of angle $\theta$; sliding iff $\mu_f < \tan\theta$;
   acceleration $g(\sin\theta - \mu_f\cos\theta)$ while sliding ([Z25] Fig. 9).
5. **Static friction.** Masonry arch stands at $\mu_f = 0.5$ and collapses at $\mu_f = 0$
   ([Z25] Fig. 10).
6. **Termination bookkeeping.** $\beta$ recomputed from the logged $\alpha^{[j]}$ equals the
   running value; prescribed-vertex lag $\le \varepsilon h\|v_{\mathcal P}\|$.
7. **Unit-level oracles:** distance gradients vs finite differences; eigen-system vs Eigen's
   dense eigendecomposition of the $9\times9$ Hessian; per-block assembly vs the dense
   $12\times12$ projected Hessian; ACCD vs a brute-force sampled distance minimum.

---

## Appendix A — Slack elimination

Per pair, with $w := c_i - \lambda_i/\mu$ and $\phi(s) = \tfrac{\mu}{2}(c_i - s)^2 - \lambda_i(c_i - s)$
for $s\ge0$: $\phi'(s) = -\mu(c_i - s) + \lambda_i = -\mu(w - s)$, so the unconstrained minimizer
is $s = w$; with $s\ge0$, $s^\star = \max(0,w)$.
If $w\le0$: $\phi(0) = \tfrac\mu2 c_i^2 - \lambda_i c_i = \tfrac\mu2 w^2 - \lambda_i^2/(2\mu)$.
If $w>0$: $c_i - s^\star = \lambda_i/\mu$ and $\phi = \tfrac{\lambda_i^2}{2\mu} - \tfrac{\lambda_i^2}{\mu} = -\lambda_i^2/(2\mu)$.
Hence $\phi(s^\star) = \tfrac\mu2\,\min(0,w)^2 - \lambda_i^2/(2\mu)$, which is §5.3; its gradient
in $\hat x$ is $\mu\min(0,w)\nabla d_i$ (§5.4), and the multiplier update
$\lambda_i \leftarrow \lambda_i - \mu(c_i - s^\star) = \max(0,\lambda_i - \mu c_i)$ (§8).

## Appendix B — Per-vertex-block projected element Hessian

From §3.3, $H_e^+ = V_e\sum_k \lambda_k^+ g_k g_k^\top$ with $(g_k)_a = Q_k a_a$. For the
$3\times3$ block of local vertices $(a,b)$:

$$
(H_e^+)_{ab} = V_e\sum_k \lambda_k^+\,(Q_k a_a)(Q_k a_b)^\top
= V_e\,U\Big[\sum_k \lambda_k^+\,D_k\,B_{ab}\,D_k^\top\Big]U^\top,
\qquad B_{ab} := b_a b_b^\top,\ \ b_a := V^\top a_a,
$$

using $Q_k = U D_k V^\top$. This is [Z25] Eq. 19. Exploiting the structure of $D_k$:

* scaling mode $s$ ($D = \mathrm{diag}(q_s)$): $D B D^\top = (q_s q_s^\top)\circ B$ (Hadamard);
* twist $(i,j)$ ($D = \tfrac1{\sqrt2}(e_ie_j^\top - e_je_i^\top)$):
  $D B D^\top = \tfrac12\big[B_{jj}e_ie_i^\top + B_{ii}e_je_j^\top - B_{ji}e_ie_j^\top - B_{ij}e_je_i^\top\big]$;
* flip $(i,j)$ ($D = \tfrac1{\sqrt2}(e_ie_j^\top + e_je_i^\top)$): same with $+$ on the last two terms.

Summing twist and flip of the pair with projected eigenvalues $\tau^+_{ij},\phi^+_{ij}$:

$$
S_{ab} = \sum_{s} a_s^+\,(q_sq_s^\top)\circ B_{ab}
+ \sum_{(i,j)}\Big[\tfrac{\tau^+_{ij}+\phi^+_{ij}}{2}\big(B_{jj}e_ie_i^\top + B_{ii}e_je_j^\top\big)
+ \tfrac{\phi^+_{ij}-\tau^+_{ij}}{2}\big(B_{ji}e_ie_j^\top + B_{ij}e_je_i^\top\big)\Big],
\qquad (H_e^+)_{ab} = V_e\,U S_{ab} U^\top .
$$

Per block this costs one $3\times3$ outer product, nine scaled entries, and two $3\times3$
products with $U$ — the reduction [Z25] quantify as 2.34× fewer multiplications than forming
the $9\times9$ projection and transforming back. The per-element cache needed by a
block-parallel assembly is $(U, V, \{a_s^+, q_s\}, \{\tau^+_{ij},\phi^+_{ij}\}, V_e)$: 37 scalars
(9 + 9 + 3 + 9 + 3 + 3 + 1).

## Appendix C — Anchor weights

Let $x^{[0]} = x^t$ and $x^{[j]} = (1-\alpha^{[j]})x^{[j-1]} + \alpha^{[j]}\hat x^{[j]}$. By
induction $x^{[k]} = \beta_0^{[k]}x^t + \sum_{j=1}^k \beta_j^{[k]}\hat x^{[j]}$ with
$\beta_j^{[k]} = \alpha^{[j]}\prod_{l=j+1}^{k}(1-\alpha^{[l]})$ and $\alpha^{[0]}=1$; the
weights sum to one. Going from $k$ to $k+1$ multiplies every existing weight by $(1-\alpha^{[k+1]})$
and appends $\beta^{[k+1]}_{k+1} = \alpha^{[k+1]}$. Therefore $\beta = \sum_{j<K_{\min}}\beta_j$
stays equal to 1 while $k < K_{\min}$ (the appended weight is inside the sum and the sum is
convex) and afterwards obeys $\beta \leftarrow (1-\alpha)\beta$ — the update of §6 line 6.7.
With $K_{\min}=1$ and no contact ($\alpha=1$ always) the loop stops after one iteration and
the step is one Newton step from $x^t$: the semi-implicit Euler limit [Z25] discuss.
