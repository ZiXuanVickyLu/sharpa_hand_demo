# Barrier-Free AL Contact Solver — Implementation Specification

**Status:** v0.1 — for review, pre-implementation.
**Basis:** `al-ipc-math-spec.md` (referenced as **MS §n**), the [Z25] paper, the [UIPC-AL]
integration (`../libuipc`, branch `al/AL-release`), and the infrastructure of
`../coupled_solver` (CMake layout, presets, CUDA wheels, LBVH, ACCD, buffer pools).
**Target machine:** Linux, CUDA 13.0, RTX 4090 (sm_89) — the same GPU as [Z25]'s benchmarks,
so Table 1/2 timings are directly comparable.

This document fixes the project layout, the reuse inventory, the device data model, the
kernel-level design of every pipeline stage, the JSON configuration schema, the test plan and
the milestone order. It does not contain code.

---

## 1. Goals and non-goals

* Reproduce [Z25] for tetrahedral elastodynamics with contact, friction and moving boundaries,
  matching their iteration counts (Newton ≈ 6–30 per step, PCG ≈ 30–70 per solve, TOI ≈ 0.2–0.5
  average) before chasing their wall-clock numbers.
* Reuse the GPU wheels that already exist in the sibling repositories instead of rewriting them
  (§3); do **not** inherit `libcwheels`' CPU solver stack and its MKL/Embree/SuiteSparse
  dependency chain.
* Configurable through JSON (nlohmann::json, same convention as `coupled_solver`); every
  algorithmic knob of MS §15.3 is a key.
* Deterministic by default: no floating-point atomics in the assembly or the linear solve
  (only integer atomic-min in the CCD filter, which is order-independent). Optional
  atomic fast paths are opt-in.
* Non-goals for v1: cloth/rods, affine bodies, adaptive $h$, Python bindings, GUI beyond an
  optional polyscope viewer.

---

## 2. Repository layout

Mirrors `coupled_solver` (CMake presets, `cmake/` helpers, `ext/` FetchContent deps,
`src/` libraries, `test/` via `add_cu_test`, `app/` executables + configs).

```text
contact_solver_demos/
  CMakeLists.txt              project(contact_solver LANGUAGES CXX CUDA); options CS_USE_DOUBLE (ON),
                              CS_ENABLE_GUI (OFF), CS_BUILD_TESTS (ON), CS_BUILD_APPS (ON);
                              CUDA-13 CCCL include handling and arch autodetect copied from coupled_solver
  CMakePresets.json           linux {release, relwithdebinfo, debug} x {double, float} (Ninja + cmake/toolchain.cmake),
                              windows VS presets kept for parity
  cmake/                      auto_detect_cuda_arch.cmake, add_cuda_test.cmake, toolchain.cmake, sublist.cmake (verbatim copies)
  ext/                        CMakeLists.txt: FetchContent for Eigen 3.4 (upstream, no fork needed), nlohmann_json 3.11,
                              spdlog 1.14, fmt 10, cuda-api-wrappers v0.8.2-b1, optional polyscope;
                              vendored: HouGeoIO/ (bgeo I/O, copied from coupled_solver/ext), cnpy/ (+ system zlib) for .npy keyframes
  src/
    core/                     typedef.h/.cuh (real, real3, atomics — copied), vector_type_t.h, float3x3.h, double3x3.h,
                              svd3.cuh (Warp's templated McAdams SVD, vendored, + real3x3 adapter), device_buffer.cuh (DBuffer/HBuffer + loose_resize), buffer2d.cuh,
                              cuda_check.h, timer.h, log.h
    libculbvh/                stacklessbvh_lite.cu/.cuh, bound.h (copied from coupled_solver/src/libculbvh)
    ccd/                      accd.cu/.cuh (copied + fixes, §3.2), distance.cu/.cuh (copied), libcuGTE/ (copied headers),
                              plane_ccd.cuh, inversion_ccd.cuh (new)
    geometry/                 tet_mesh.h/.cpp, tri_mesh.h/.cpp (host, Eigen), loaders (bgeo via HouGeoIO, obj, tetgen .node/.ele),
                              placement.h (scale/rotate/translate), surface.h (boundary faces/edges/vertices, orientation),
                              selection.h (index list / all / bbox), motion.h (scripted rotation/translation/scale, keyframe playback)
    scene/                    config.h/.cpp (JSON schema → SceneDesc, validation, defaults), scene.h/.cpp (global vertex/element
                              numbering, free/prescribed partition, material table, contact table)
    gpu/                      scene_buffers.cuh/.cu (device SoA of §4), upload/download, prescribed-target upload per step
    fem/                      elastic_models.cuh (Ψ, Ψ_i, A, τ, φ per model), element_cache.cu (F, SVD, eigen data), gradient.cu,
                              hessian_blocks.cu (Appendix-B assembly), energy.cu
    contact/                  broadphase.cu (AABBs, LBVH wrappers, pair filtering), narrowphase.cu (ACCD → TOIs),
                              active_set.cu (merge, earliest-impact filter, decay/removal), linearize.cu (d⁰, ∇d),
                              al_terms.cu (slack, energy, gradient, Hessian/SpMV, λ update), friction.cu (snapshot, energy, grad, Hess),
                              planes.cu
    linsys/                   bsr.cuh/.cu (pattern, gather map), assemble.cu, spmv.cu, block_jacobi.cu, pcg.cu
    solver/                   al_stepper.h/.cu (MS §6 loop, §7 subproblem, §15 termination), stats.h
    io/                       bgeo_writer.cpp (surface + tets + attributes), csv_stats.cpp
    contact_solver.h          public facade: Simulator(config) → init(), step(), sync_to_host(), export(frame)
  test/                       CMakeLists.txt with add_cu_test targets (§8)
  app/                        cs_run (headless: config → frames), cs_view (polyscope, optional); app/config/*.json
  asset/                      small meshes committed (cube_tet, sphere_tet, grid); large scenes gitignored
  doc/                        this spec, the math spec, references.md, later: implementation reports
  knowledge/                  reference PDFs (gitignored, titles in doc/references.md)
```

Library targets: `cs_core`, `cs_culbvh`, `cs_ccd`, `cs_geometry`, `cs_scene`, `cs_gpu`, `cs_fem`,
`cs_contact`, `cs_linsys`, `cs_solver`, `cs_io`, aggregated into `libcontact_solver` (STATIC,
`CUDA_SEPARABLE_COMPILATION ON`, `CUDA_RESOLVE_DEVICE_SYMBOLS ON`, C++20/CUDA20) following the
per-directory pattern of `coupled_solver/src/bufferpool/CMakeLists.txt`.

---

## 3. Reuse inventory

| Component | Source | Reuse mode | Adaptation |
| --- | --- | --- | --- |
| Build skeleton, presets, `cmake/*.cmake` | `coupled_solver/` | copy | rename `PDSPAI_*` → `CS_*`; Linux presets first; drop MKL/TBB/embree/CHOLMOD |
| `ext/CMakeLists.txt` dependency block | `coupled_solver/ext` | copy, trim | keep Eigen (upstream tag), json, spdlog, fmt, cuda-api-wrappers, HouGeoIO, cnpy/zlib, polyscope(opt); drop libigl, spectra, fast_matrix_market, cereal, tinyxml2 |
| `typedef.h/.cuh`, `vector_type_t.h`, `float3x3.h`, `double3x3.h` | `coupled_solver/src` | copy | namespace `cs`; keep `atomicMin_real_nonneg` (used by the CCD filter) |
| 3×3 SVD | NVIDIA Warp `warp/native/svd.h` (Apache-2.0 with Eric Jang's MIT notice; templated McAdams/Sifakis port) | vendor | `_svd_config<double>`: 8 Jacobi sweeps, $10^{-12}$ epsilons, `::rsqrt` on device; adapter to `real3x3`; signed-SVD test (§8.1). coupled_solver's `svd3_cuda.h` is float-only (and its double build downcasts) — not used. libuipc's implicit-QR `qr_svd.hpp` is a test oracle only. Decision 2026-09-02. |
| `DBuffer`/`HBuffer`, `Buffer2D` | `coupled_solver/src/typedef.cuh`, `buffer2d.cuh` | copy | add `loose_resize` growth (×1.5) as in [UIPC-AL] |
| Aggregated multi-mesh pool pattern (free DOFs first, global topology offsets, Dirichlet index lists) | `coupled_solver/src/bufferpool/pd_aggregated_bufferpool.*` | pattern | rewrite lean (`SceneBuffers`), without PD-specific channels |
| `LBVHStacklessLite` (build/refit/self-query/cross-query, float AABBs) | `coupled_solver/src/libculbvh` = `culbvh` repo | copy | none; pad AABBs by a float-rounding margin because `real` is double |
| Swept-AABB kernels, broadphase pair buffers, adjacency filtering, capacity policy | `coupled_solver/src/collision/self_collision_handler.*` | pattern | keep `filter_adjacent_*` (shared-vertex removal); one-ring skip behind a flag |
| ACCD (`vertex_triangle_ccd`, `edge_edge_ccd`, smem variants) | `coupled_solver/src/collision/ccd/accd.*` | copy + fix | see §3.2 |
| Closest-point distance queries with barycentrics/normal | `coupled_solver/src/collision/ccd/distance.*`, `libcuGTE/` | copy | add the ∇d assembly on top (MS §4.2) |
| PCG structure (device dot/axpy, block results, per-component reductions) | `coupled_solver/src/solver/pcg_solver.*` | pattern | our matrix is BSR + matrix-free pair term (§6.6), coupled PCG on 3n scalars (not per-component) |
| Block-Jacobi preconditioner (3×3 inverse blocks, `update_with_collision`) | `coupled_solver/src/solver/preconditioner.*` | pattern | blocks include pair diagonal contributions |
| SNH eigen-system, `buildTwistAndFlipEigenvectors`, `buildScalingEigenvectors` | `coupled_solver/src/libcwheels/energy/stable_neohookean_tet.h`, `mat_utils.h` | CPU oracle | test-only dependency (Eigen, header-only parts) |
| bgeo loaders/writers (`BgeoTetLoader`, `BgeoTriLoader`, writers) | `coupled_solver/src/libcwheels/geometry/load_bgeo.h`, `write_bgeo.h` | copy | header-only over HouGeoIO; strip cwheels type aliases |
| Placement keys (`move_to_origin`, `scale`, `rotation`, `translation`), selection (`indices`/`all`/`bbox`), scripted motion (rotation/translation/scale, start/end frame), keyframe `.npy` sequences | `libcwheels/geometry/mesh.cpp`, `energy/dirichlet_boundary_condition.h`, `geometry/collider.h` | semantics | re-implemented in `geometry/` with the same JSON keys so scenes port easily |
| Test harness (`add_cu_test`, assert-based executables, `APP_WORKING_DIR`) | `coupled_solver/test` | copy | — |
| Active-set kernels (hash-sort-unique merge, per-vertex TOI atomic-min, slack/λ kernels) | `libuipc` (`global_active_set_manager.cu`, AL-release) | design reference | re-implemented on our buffers (libuipc uses muda/Eigen device types) |
| AL energy/gradient/Hessian per pair, friction helpers (bases, closest points, Jacobians, f0/f1) | `libuipc` (`al_contact_function.h`, `friction_utils.h`, `codim_ipc_contact_function.h`) | design reference | same formulas, `real3`-based |

### 3.1 What is deliberately not reused

`libcwheels` solvers, `SimMesh`/`Solver` class hierarchy, Embree broadphase, MKL/CHOLMOD linear
algebra, the PD/ADMM solver implementations, the MPM coupling stack, URDF robots. They pull
in heavy host dependencies and their abstractions (PD local/global steps) do not fit a Newton
contact solver.

### 3.2 Known defects to fix while vendoring

* `accd.cu::vertex_triangle_ccd`: the relative-motion bound uses `displacement[3]` instead of
  the mean-removed `dis_eval[3]`; must be `dis_eval[3]` (ACCD's $l_p = \|\bar p_0\| + \max_m\|\bar p_m\|$).
  Also the in-loop step uses a hard-coded `0.9` where `(1 - s)` is meant, and the "already
  inside minimum separation" fallback that `edge_edge_ccd` has is missing. Unit test §8.1
  covers all three.
* `accd.cu::triangle_triangle_ccd` passes its arguments in the wrong order to the
  point/edge routines (the minimum separation lands in `max_iter`, giving zero iterations); the
  smem variant has the right order. We do not use triangle–triangle CCD, but the function is
  removed rather than vendored broken.
* `distance.cu` returns a zero normal when $d^2 < 10^{-10}$; the linearization kernel must use
  its own fallback (MS §4.2) rather than propagate a zero gradient.
* `culbvh` AABBs are `float`. Swept boxes are computed from `double` positions and converted
  with **directed rounding** (`__double2float_rd` on minima, `__double2float_ru` on maxima,
  after adding $\xi$ in double). The tree's internal boxes are exact `fmin/fmax` unions of leaf
  floats, so every double point of a swept primitive lies inside its float box: broad-phase
  completeness is a certificate, not a heuristic. No relative pad and no config key.
* `stacklessbvh_lite.cu::mergeUpKernel` stores a child bound and then `atomicOr`s the flag
  without a release fence; add `__threadfence()` before the `atomicOr`. Pre-existing on the
  rebuild path too, but `refit()` (which we call every outer iteration, §6.10) runs it far more
  often. `refit()` has no caller and no test in `coupled_solver`: add a ground-truth query
  comparison after a refit to `test_accd`'s sibling `test_lbvh`.
* The self-query emits every unordered pair twice (only `node.data == objIdx` is excluded;
  the culbvh CSVs show exactly 2× counts). Reject `node.data >= objIdx` so each edge pair is
  emitted once as `(x < y)`, matching the canonical `pair_key`.
* The query kernel's result counter saturates at INT_MAX/2 (2026-09-07). A range of the
  compressor's release phase (frames 201+, the lid rising 0.3 m per frame over a pancake of
  seven million edges) produces billions of candidates in one launch; the plain `atomicAdd`
  wrapped negative, the room check `maxRes - globalIdx` then passed, and blocks wrote before
  the start of the result buffer, corrupting neighbouring device memory and surfacing as an
  illegal address in the narrow phase up to half an hour later. The saturated count is still
  >= the buffer size, so `chunked_pass` (§6.10, item 15) sees the overflow and narrows the
  chunk as before; it also treats a negative count as an overflow, defensively.

---

## 4. Device data model

All arrays are SoA device buffers (`DBuffer<T>` = `cuda::unique_span<T>`), sized once at init
except the pair/candidate arrays, which use `loose_resize` (capacity ×1.5 on growth, never
shrink). `real = double` by default (`CS_USE_DOUBLE`); `float` builds are for experiments only.

### 4.1 Vertices (size N, free vertices first: indices `[0, n_free)` are DOFs)

| Buffer | Type | Meaning |
| --- | --- | --- |
| `x_prev` | real3 | $x^t$ |
| `x_anchor` | real3 | $x$ (intersection-free anchor) |
| `x_hat` | real3 | $\hat x$ (iterate); prescribed part holds targets |
| `x_tilde` | real3 | $\tilde x$ (free part) |
| `x_ls0` | real3 | line-search start point |
| `v` | real3 | velocity |
| `mass` | real | lumped mass (0 for prescribed) |
| `thickness` | real | primitive thickness $\xi$ |
| `body_id`, `contact_id` | int | owning body; contact-table row |
| `is_prescribed` | uint8 | 1 for $\mathcal P$ (redundant with index range, kept for kernels that see global ids) |
| `x_target` | real3 | per-step targets for $\mathcal P$ (uploaded each step) |

### 4.2 Elements and surface

* Tets: `tet` int4 (global vertex ids), `DmInv` (9 real), `volume`, `material_id`; per-element
  cache `elem_U` (9), `elem_V` (9), `elem_eig` (3 scaling + 3 twist + 3 flip, projected),
  `elem_q` (9: the three scaling eigenvectors), `elem_P` (9: PK1, for the gradient) — 45 reals
  (the volume is a separate array).
* Shells (MS §3.5–3.7): `shell_tri` int3 (global vertex ids), `shell_DmInv` (4 real, the
  $2\times2$ inverse), `shell_area`, `shell_thickness`, `shell_material_id`; per-element cache
  in the SAME layout as the tet cache (`elem_U` holds $\bar U$, `elem_eig` the padded nine
  eigenvalues with the $(1,3)$ and $(2,3)$ twist and flip slots both set to $\nu_i$, `elem_P`
  the $3\times2$ PK1 padded with a zero column) so `k_bsr_assemble` and `k_elem_grad` consume
  it through an element-kind tag on the gather map; hinges `hinge_idx` int4 $(x_0,x_1,x_2,x_3)$,
  `hinge_rest` (rest length, $\bar h$, $\bar\theta$), and, for the Gauss–Newton form, the
  per-hinge scalar and 12-vector $\nabla\theta$ consumed by the rank-one SpMV path of §6.5.
* Surface: `surf_vert` (int), `surf_edge` (int2), `surf_tri` (int3, outward), each with
  `body_id`; per-body `self_collision` flag; vertex→incident-surface-primitive CSR for
  adjacency filtering.
* Planes: `plane_o`, `plane_n`, `plane_contact_id`.
* Spheres (MS §4.1 PS, §12.3): `sphere_c`, `sphere_inverted`, `sphere_contact_id`, and the
  per-step radii `sphere_r0` (at the anchor) and `sphere_r1` (at the target), re-uploaded by the
  stepper at the start of every step from the host-side keyframe schedule.

### 4.3 Linear system

* Static BSR over free vertices from tet adjacency (vertex–vertex pairs, both triangles stored,
  diagonal included): `row_ptr` (n_free+1), `col_idx`, `blocks` (real 3×3, row-major), plus a
  **gather map**: for every BSR slot, the list of (tet, local a, local b) contributions
  (`slot_ptr`, `slot_tet`, `slot_ab` packed) built once on the host from the element list.
* Diagonal mirror `diag_blocks` (n_free × 3×3) for the preconditioner (sum of static diagonal
  slot + pair contributions).
* PCG workspace: `r, z, p, Ap` (3·n_free reals), scalar reductions on device (`rz, pAp, rr, bb`),
  block partial sums.

### 4.4 Active set (SoA, capacity-managed; one struct per pair kind or a tagged union)

| Buffer | Type | Meaning |
| --- | --- | --- |
| `pair_key` | int64 | canonical identity: `(type<<60) \| (primA<<30) \| primB` (surface-primitive ids) |
| `pair_idx` | int4 | global vertex ids in MS §1 order (PH: `.x` vertex, `.y` plane id) |
| `pair_lambda` | real | $\lambda_i$ |
| `pair_ninact` | int | $n_i$ |
| `pair_gap` | real | $g_i = d_i(x) - \delta_i$, the gap at the anchor (MS §5.1; the constraint is evaluated on the displacement $\hat x - x$, never on absolute positions — §11) |
| `pair_gap_folded` | real | $\tilde g_i = g_i - s_i - \lambda_i/\mu$, rewritten at every Newton step |
| `pair_grad` | real (12) | $\nabla d_i(x)$ |
| `pair_slack` | real | $s_i$ of the current Newton step |
| `pair_hit` | uint8 | scratch: filter flag |

Candidate buffers from the CCD: `cand_key`, `cand_idx`, `cand_toi`; per-vertex `T_v` (real,
initialized to 2). Friction snapshot: `fr_idx` (int4), `fr_F` (real $\hat F_i$), `fr_w` (real2:
$w_1,w_2$ or $s,u$), `fr_basis` (real 6), `fr_mu` (real).

### 4.5 Per-vertex pair incidence (rebuilt whenever $\mathcal C$ changes)

CSR `vp_ptr` (n_free+1), `vp_pair`, `vp_slot` (which of the four slots the vertex occupies)
built by counting-sort on vertex id. Used by the matrix-free contact SpMV, the block-Jacobi
diagonal update and the gradient gather. Same structure for friction pairs (`vf_*`).

Memory estimate for the largest [Z25] scene (0.87 M vertices, 2.25 M tets, 1.45 M pairs):
vertices ≈ 0.25 GB, element cache ≈ 0.7 GB, static BSR (≈15 blocks/vertex, both triangles)
≈ 0.95 GB, pairs ≈ 0.25 GB, CCD scratch ≈ 0.5 GB — well inside 24 GB.

---

## 5. Host side: scene, configuration, I/O

### 5.1 Pipeline

`config.json` → `SceneDesc` (validated, defaults filled) → host meshes loaded and placed →
surfaces extracted (boundary faces of tets with outward orientation, unique edges, vertices)
→ global numbering (free vertices of all bodies first, then prescribed vertices of bodies, then
collider vertices) → materials (Lamé from $E,\nu$; density → lumped mass) → contact table
(pair enable + friction coefficient per body pair) → `SceneBuffers::upload`.

Per step the host evaluates the scripted/keyframed targets of every prescribed vertex (MS
§12) and uploads `x_target`; everything else stays on the device. Export downloads `x_anchor`
(and optionally $\lambda$ statistics) at the configured frame stride.

### 5.2 Mesh formats

bgeo tets/tris (HouGeoIO, the `coupled_solver` asset convention), Wavefront obj (triangles),
TetGen `.node/.ele`. gmsh `.msh` v2 is added when the libuipc `animal_well.msh` asset is used.

* MEDIT `.mesh` (2026-09-15): the `MeshVersionFormatted` / `Dimension` header and the per-record
  reference numbers are optional; a record's reference is read only when another token follows
  on its line. The authors' `Arm13K.mesh` (Figs. 14 and 15) has neither header nor references.
* gmsh `.msh` 4.1 ASCII (2026-09-21, IPC's `rod.msh`): `$Nodes` and `$Elements` come in entity
  blocks (`numEntityBlocks numNodes minTag maxTag`; per block `entityDim entityTag parametric
  numNodesInBlock`, the block's node tags, then its coordinates, with `entityDim` parametric
  coordinates per node when `parametric` is set; elements: `entityDim entityTag elementType
  numElementsInBlock`, then `tag node...`). Tetrahedra (types 4 and 11, first four nodes) are
  kept, every other element type is skipped. 4.0 (a different node layout) and binary files
  are rejected by version and file type. Test: `test_mesh_load` case 1c (a two-block file with a
  parametric surface block and triangles to skip).

### 5.3 Configuration schema (JSON)

Keys marked (†) map directly onto MS §15.3. Unknown keys are errors, except that a key whose
name starts with an underscore is an ignored comment at any level. Every default is recorded by
the loader into the run's `resolved_config.json`.

```jsonc
{
  "simulation": {
    "dt": 0.01, "frames": 300, "gravity": [0, -9.81, 0],
    "output_dir": "output/scene_name", "save_bgeo": true, "save_every": 1,
    "save_npy": false,            // 2026-09-19: also write x_<step>.npy (float32 [n_vertices, 3], global numbering) and once surface_faces.npy (int32 [m, 3]); for renderers without a bgeo reader
    "checkpoint_every": 0,        // 2026-09-19 (§5.x restart): every N steps write ckpt_x_<step>.npy and ckpt_v_<step>.npy (float64); 0 = off
    "restart": null,              // { "step": k, "positions": "<npy>", "velocities": "<npy>" | null }, §5.x restart
    "abort_after_capped_steps": 0, // stop the run (exit code 4) after this many consecutive steps at the outer iteration cap; 0 = off
    "seed_velocity": null                      // optional per-body initial velocities live in bodies[]
  },
  "contact": {
    "enable": true,
    "d_hat": 1e-3,                             // † contact offset (m), added to thicknesses
    "epsilon": 1e-3, "K_min": 2,               // † termination
    "max_outer_iters": 500,                    // † safety cap
    "decay_factor": 0.9, "decay_remove_threshold": 0.01,   // † Γ, γ_min
    "mu_mode": "diag_max", "mu_scale": 0.1, "mu_fixed": null,   // † μ estimate (MS §11.1) or fixed value (kg)
    "mu_max": null,                            // upper bound of the estimate (kg), MS §11.1; null = none
    "alpha_lower_bound": 1e-6, "toi_tie_tolerance": 1e-6,
    "stall": { "iters": 50, "alpha": 1e-4, "mu_factor": 2.0, "d_hat_factor": 0.5, "max_adaptations": 10 },
    "ccd": { "s": 0.1, "max_iter": 100, "float_screen": false, "rebuild_quality_ratio": 1.5 },
    "inversion_free": "auto",                  // "auto": on for NH bodies only; true/false forces
    "self_collision_default": true,
    "exclude_one_ring_pairs": false, "drop_parallel_edge_pairs": false,
    "toi_filter_domain": "paper",              // † "paper" (new hits only, MS §9) | "all" (fork rule)
    "friction": { "enable": true, "eps_v": 1e-3, "normal_force": "paper" }   // "paper" (MS §13) | "lambda"
  },
  "newton": {
    "inner_max_iters": 8,
    "line_search": { "max_halvings": 30, "energy_tolerance": 1e-12 },
    "early_accept_velocity_tol": 0,            // 0 = off (MS §7)
    "increment_velocity_tol": 0                // 0 = off; C-IPC PNTol-style RMS increment velocity, m/s (MS §7)
  },
  "linear_solver": { "type": "pcg", "preconditioner": "block_jacobi", "rel_tol": 1e-4, "max_iters": 2000,
                     "check_interval": 8,     // PCG iterations per captured graph replay (§6.6)
                     "graph": true,           // false = plain launch loop with host reductions
                     "rigid_coarse": false, "rigid_coarse_max_rows": 4096 },   // §6.6 two-level preconditioner (per-body rigid modes)
  "materials": {
    "rubber": { "model": "SNH", "E": 1e6, "nu": 0.3, "density": 1000, "snh_lambda_reparam": true },
    "stiff":  { "model": "NH",  "E": 1e7, "nu": 0.3, "density": 1000 }
  },
  "bodies": [
    {
      "name": "ball", "type": "tet" | "cloth", "file": "asset/basic/sphere_tet.bgeo", "material": "rubber",
      "move_to_origin": true, "scale": [1, 1, 1], "rotation": [0, 0, 0], "translation": [0, 1, 0],
      "initial_velocity": [0, 0, 0], "thickness": 0.0, "friction": 0.1, "self_collision": true,
      "dirichlet": [ { "select": "all" | [ids] | { "bbox_min": [..], "bbox_max": [..] },
                       "motion": { "type": "rotation", "axis": [0,1,0], "center": null, "deg_per_second": 180,
                                   "velocity": [0,0,0], "rate": 0, "start_frame": 0, "end_frame": -1,
                                   "ramp_time": 0 }   // s; 2026-09-21: the scripted rate rises linearly from 0 over this time
                       | { "type": "keyframes", "file": "seq.npy", "substeps": 1 } } ]
    }
  ],
  "colliders": [
    { "name": "container", "type": "tri", "file": "asset/funnel/container.bgeo", "thickness": 0.0, "friction": 0.5,
      "scale": [1,1,1], "rotation": [0,0,0], "translation": [0,0,0],
      "motion": null | { "type": "keyframes", "file": "container_up.npy", "substeps": 1 } | { "type": "translation", ... } }
  ],
  "materials": { "...": "tet materials as above; a cloth material is",
    "sheet": { "model": "cloth_stvk" | "cloth_cor" | "cloth_snh", "mu_mem": 1e3, "nu": 0.3, "k_bend": 1e-3,
               "thickness": 1e-3, "density": 1000, "bending_hessian": "gauss_newton" | "full" } },
  "planes": [ { "point": [0, 0, 0], "normal": [0, 1, 0], "friction": 0.5 } ],
  "spheres": [ { "center": [0, 0, 0], "radius": 1.0, "inverted": true, "friction": 0.0,
                 "radius_keyframes": [[0.0, 1.0], [2.0, 0.4], [3.0, 5.0]] } ],
  "contact_table": { "exclude": [["ball", "container"]], "friction": [["ball", "container", 0.3]] },
  "logging": { "level": "info", "stats_csv": "stats.csv", "timing": true, "debug_check_penetration": false }
}
```

Selection and motion keys are the ones `coupled_solver` scenes already use (`dirichlet_boundary_condition.h`
script block, `mesh.cpp` placement keys), so its assets and scene fragments port with minimal edits.

### 5.4 Statistics

Per step to CSV: outer iterations, inner Newton steps, PCG iterations (sum/avg), #pairs (PT/EE/PH),
#candidates, #hits, mean and min $\alpha$, final $\beta$, $\mu$, stall adaptations, timings per
stage (linearize, assemble, PCG, line search, CCD broad/narrow, active-set), max penetration
check (debug). These are the columns needed to reproduce [Z25] Tables 1–2.
Added 2026-09-17: `prescribed_hits`, the CCD hits between prescribed-only primitives summed
over the step (the diagnostic of §5.x below; 0 in a well-posed scene).

---

### 5.x Prescribed motion driven into a boundary (diagnostic, 2026-09-17)

MS §4.1 excludes pairs whose primitives are all prescribed from the candidate set (no free
DOF can resolve them). The CCD kernels used to skip them before the narrow phase, so a
Dirichlet region driven into a collider, or a prescribed vertex driven through a plane or
sphere, went through without any trace; the only symptom was the outer loop stalling at
$\alpha = 0$ on the free vertices attached to the region (the funnel scene of
the paper-scene reproductions did this for six set-ups before the cause was found).
Now the narrow phase evaluates the ACCD for prescribed-only pairs that pass the body and
contact-table rules, and a hit ($t < 1$) increments a device counter instead of being
reported; the plane and sphere kernels do the same for prescribed vertices. The counter is
packed into the status word next to $\alpha$ and the hit count (one copy), returned as
`CcdResult::prescribed_hits`, summed into `StepStats::prescribed_hits` (CSV column) and,
when non-zero, logged once per step at warn level. Nothing else changes: the pairs remain
non-candidates and $\alpha$ is not affected. Held free regions (`release_frame`) are not
covered while held (their vertices are in the free range and carry no prescribed flag).
Cost: an ACCD evaluation per prescribed-only candidate pair per CCD call; such pairs are the
overlapping boxes of distinct colliders or Dirichlet regions, a few hundred at most in the
scenes run so far. Test: `test_prescribed_drive` (a fully prescribed cube driven through the
ground plane and through a triangle collider counts hits on every crossing step and a free
cube counts none).

### 5.x Ramp-up of scripted motions (`motion.ramp_time`, 2026-09-21)

A scripted rotation, translation or scale starts at full rate in its first step. When that
first increment is larger than an element next to the driven region (IPC's rod twist on the
paper's mesh: the end caps move 4.4 mm per step, the elements are 3.3 mm), the prescribed
vertices, which sit at their targets from the start of the step (MS §6), shear their free
neighbours past the mesh spacing in one step: the CCD then reports hits between surface
primitives two rings apart, those enter the active set as self-contact at d_hat, and the cap
zone crumples (NH: J = 0 at the caps after 20 steps, with the rods still 16 cm apart). IPC
moves its boundary nodes inside the Newton line search and has no such start-up problem.
With `ramp_time` = T > 0 the motion's elapsed time t is replaced by
$\tau(t) = t^2/(2T)$ for $t < T$ and $t - T/2$ afterwards (constant acceleration to the
scripted rate, then the scripted rate; the motion lags the unramped one by T/2). Default 0:
unchanged. Validation: `ramp_time` >= 0, scripted types only. Test: `test_scene_config`
(round trip, rejected with keyframes) and `test_prescribed_ramp` (targets of a rotating region
against the closed form, with and without the ramp).

### 5.x Interactive viewer `cs_view` (2026-09-21, `CS_ENABLE_GUI`, preset `release-gui`)

`cs_view <config.json> [--play] [--frames N] [--output DIR] [--no-export]` runs the same
pipeline as `cs_run` (load, `build_scene`, one `Stepper`) inside a polyscope window, one solver
step per UI callback while playing, so what is on screen is the solver's state and nothing is
read back from files. The scene surface (`scene.surf_tris`, global numbering) is split by
`body_id` of the triangle's first vertex into one polyscope surface mesh per body and one per
collider; positions come from `Stepper::positions` after every step (one device-to-host copy of
all vertices; no CUDA/GL interop, the scenes this is meant for have under a million vertices).
Prescribed vertices of a body are shown as a point cloud (`dirichlet`, off by default) so
driven regions can be checked. Planes are drawn as polyscope ground/slabs only when axis
aligned (`planes[0]` sets the ground plane height and up direction), spheres are not drawn.

Panel: play / pause, single step, steps per UI frame (1, for slow scenes; more for fast ones),
stop at frame N, and the last step's statistics (`StepStats`: outer and Newton iterations, CG
per solve, pairs by kind, mean accepted alpha, time per step, capped flag) with running totals;
a capped step pauses the run and says so. Export follows the config (`save_bgeo`, `save_npy`,
`save_every`, the stats CSV) into `<output_dir>` exactly as `cs_run` does, unless
`--no-export`. `simulation.restart` and `abort_after_capped_steps` are honoured. The UI
blocks for the duration of a step; that is accepted (a step is the unit of work and cannot be
interrupted consistently), the window title shows the frame being computed.
No solver code depends on the viewer; the target is only built with `CS_ENABLE_GUI=ON`
(polyscope v2.3.0 fetched by `ext/CMakeLists.txt`). Not covered by CTest (needs a display);
checked by running a scene (`app/config/rod_twist_paper.json`).

### 5.x Restart from a saved state, checkpoints, capped-step guard (2026-09-19)

Long scripted scenes (the scripted hand shuffle: 6600 steps, two to three
hours) need to be continued from a step with a changed later part (another hand motion after
the riffle), and must not burn a night on a dead step (one run sat 12 h at $\alpha = 0$,
500 outer iterations and 150 s per step, before it was noticed).

* **State of a step boundary.** Between steps the stepper carries $x^t$ (`x_prev`), $v^t$ and
  the step index, plus the active set $\mathcal C^t$ with its multipliers (which decay from
  step to step, MS §8) and the friction snapshot (MS §13); $\mu$, trees and candidates are
  rebuilt per step. A checkpoint holds $x$, $v$ and the step index only: the first step after
  a restart starts from an empty set, exactly like step 0, finds its pairs through its own
  CCD, builds their multipliers from zero and carries no lagged friction. A restarted run is
  therefore bitwise equal to the uninterrupted one on a scene without contact and close to it
  otherwise (a soft cube settling on the plane: 4e-5 m after ten more steps; a body sliding
  with friction loses one step of friction impulse, $\mu_f g h$ of velocity).
* **`simulation.restart`** `{ "step": k, "positions": file, "velocities": file | null }`,
  k >= 1. `positions` is an npy of shape [n_vertices, 3], float32 or float64, in the global
  numbering (`x_<k>.npy` of `save_npy`, or `ckpt_x_<k>.npy`); `velocities` likewise, or
  absent/null for a start from rest (meant for states that are at rest anyway; `x_<k>.npy`
  files carry no velocity). Paths are absolute or relative to the output root. The scene is
  built as usual, rest shapes, masses, bending rest angles and the BSR pattern come from the
  mesh files, never from the restart file. Then `Stepper::set_state(X, V, k)` overwrites
  `x_prev`, the anchor and the iterate with X and `v` with V for the free range, and sets the
  step index to k, so step k is the next one taken, motions, keyframes (`motion.substeps`
  included) and `release_frame` are evaluated at their absolute step and the run ends at
  `simulation.frames` as before. Prescribed vertices take the scene's own target of step
  k - 1 (exact keyframes) instead of the file's value; if the two differ by more than 1e-6 m
  anywhere the loader warns with the largest difference (a motion that changed *before* the
  restart step; with a float32 file the difference is rounding, 1e-8 m). Output numbering
  continues (`x_<k+1>.npy`, ...; frame k itself is exported again as the restart state), the
  stats CSV holds the steps from k on.
* **`simulation.checkpoint_every`** N > 0: after every N-th step write `ckpt_x_<step>.npy` and
  `ckpt_v_<step>.npy`, float64 [n_vertices, 3], the exact pair for a later restart.
* **`simulation.abort_after_capped_steps`** N > 0: after N consecutive steps that ended at
  `contact.max_outer_iters` the run logs an error and returns exit code 4. A capped step is
  tolerated by the solver (it keeps the last feasible anchor), but a sequence of them means a
  prescribed motion is squeezing something that cannot give way (a card pinched between a
  keyframed fingertip and the ground plane: both sides prescribed, no feasible step), and
  every further step costs the full cap.

Tests (`test_restart`): (1) a free cube drifting and falling without contact, 20 steps,
against 10 steps + `set_state` on a fresh stepper + 10 steps: positions bitwise equal; (2) the
cube settling on the ground plane with friction: restarted and uninterrupted runs agree to
2e-4 m after 10 more steps; (3) step 0 and a wrong vertex count are rejected, by the loader
and by `set_state`; (4) `run_simulation`: checkpoint files appear, a run restarted from them
(float64, with velocities) and one restarted from `x_10.npy` (float32, at rest) both continue
the numbering at step 10 and end within 2e-4 m of the reference, a file of the wrong shape is
rejected; (5) the capped guard: with `max_outer_iters` 1 every step is capped and the run
returns code 4 at the third one.

### 5.x Releasing a held region at a frame (`bodies[].dirichlet[].release_frame`, 2026-09-07)

The C-IPC card shuffle scene holds two rows of every card,
moves them for 50 frames, then frees one card per frame. A region with `release_frame` >= 0
is a *held free region*: its vertices stay in the free range [0, n_free) with their real
masses, so the body's element and contact machinery is unchanged, and while the region is
held its vertices are projected: `k_predict` writes the region's target into x_hat and
x_tilde, the gradient rows, the SpMV rows and the preconditioner blocks of held vertices are
masked to the identity (the Newton increment there is zero), and the CCD treats them like any
other vertex (they move from anchor to target). From `release_frame` on the mask is cleared
and the vertices are ordinary free vertices carrying the velocity of their last prescribed
step. A region without `release_frame` (default -1) is placed in the prescribed range as
before. Validation: `release_frame` >= 0 requires a `motion`; the resolved config carries it.

Linear solver contract: `PcgOptions::held` is honoured by all three PCG paths (plain loop,
captured graph, fused kernel): held rows have their residual and preconditioned residual
zeroed, so the solve is of the free-free block with the held displacements pinned at zero.
Bug fixed 2026-09-07: the graph path captured a null mask (`Pcg::held_` was never assigned
from the options), so held rows were solved as free. Small scenes never showed it because the
fused kernel, which takes the mask as a kernel argument, serves every system whose grid is
co-resident; the 54-card shuffle (17k rows) fell back to the graph and its grip rows drifted
0.75 mm in the first frame. Regression tests: `test_bsr_pcg` "held rows" (all three paths
against a dense solve of the reduced system, held rows bitwise zero) and `test_release_region`
(the held sheet on the graph path). Debug switch: `CS_PCG_VERIFY=1` runs the plain loop next
to the selected path on every solve and logs both iteration counts and the relative
difference of the solutions.
`CS_SYNC_KERNELS=1` synchronises the device at every kernel check in a release build (an
asynchronous fault is then reported at its launch); `CS_TRACE_ALLOC=1` logs device
reallocations above 64 MB.
Test: a held row released at frame k must have the same position at k-1 as its target and a
non-zero elastic response at k.

## 6. Pipeline stages → kernels

Every stage lists: purpose, kernel granularity, inputs/outputs, and atomic/determinism notes.
Stages map one-to-one onto the lines of MS §6 and §7.

### 6.1 Step setup (MS §6 lines 1–5)

* `k_predict`: per free vertex, $\tilde x = x^t + hv + h^2 g$; copies `x_anchor = x_hat = x_prev`;
  prescribed part of `x_hat` ← `x_target`.
* `k_elem_cache(x_prev)` (§6.4) then `k_diag_max`: per free vertex, the diagonal of
  $M + h^2\sum_e (H_e^+)_{aa}$ from the gather map; device max-reduce → $\mu$ (MS §11.1).
  Skipped when `mu_fixed` is set.
* Friction snapshot (§6.9).
* $\beta=1$, $k=0$, stall $=0$ on the host.

### 6.2 Linearization (MS §5.1, line 6.1) — `k_linearize`

One thread per pair. Reads `x_anchor` of the 4 vertices, computes the closest points with the
GTE queries (PT: `vertex_triangle_distance_square` → $w$, $\hat n$; EE: `edge_edge_distance_square`
→ $(s,u)$, $\hat n$; PH: plane formula), $d = \sqrt{d^2}$, $\nabla d$ per MS §4.2 (degenerate
fallback), $g_i = d - \delta_i$ with $\delta_i = \hat d + \xi_A + \xi_B$. Writes `pair_gap`,
`pair_grad`. The absolute-form constant $d - \nabla d^\top x - \delta$ is never formed (§11).
No atomics.

### 6.3 Slack fold and contact gradient (MS §7 first lines) — `k_slack`, `k_contact_grad`

* `k_slack`: per pair, $c = g_i + \nabla d_i^\top(\hat x - x)$ (12-term dot product on the
  displacement, gathered from `x_hat` and `x_anchor`), $s = \max(0, c - \lambda/\mu)$,
  $\tilde g_i = g_i - s - \lambda/\mu$ (stored separately so `k_linearize` output survives the
  inner loop); writes `pair_slack`, `pair_gap_folded`.
* `k_contact_grad`: per free vertex via `vp_*` CSR: $G_v \mathrel{+}= \sum_{i\ni v}\gamma_i\mu\,(\tilde g_i + \nabla d_i^\top(\hat x - x))\,(\nabla d_i)_v$.
  Gather, deterministic. ($\gamma_i = \Gamma^{n_i}$ computed on the fly with `pow` or a small LUT.)

### 6.4 Elasticity (MS §3) — `k_elem_cache`, `k_elem_grad`, `k_bsr_assemble`

* `k_elem_cache`: per tet: $D_s$, $F$, 3×3 SVD in the build's `real` via the vendored Warp
  McAdams port (`svd3.cuh`; $\det U = \det V = 1$ with the sign carried by $\sigma_3$, verified by
  `test_svd3_double`; must handle $\sigma\to0$ and inverted elements),
  $\Psi_i$, $A$ (3×3 symmetric eigen-solve, analytic or Jacobi), $\tau_{ij},\phi_{ij}$ (with the
  $\sigma_i\approx\sigma_j$ limit), projection to $(\cdot)^+$, $P$. Writes the 39-real cache.
  Energy per tet written to `elem_energy` for the line search (`k_elem_energy` is a lighter
  kernel that recomputes only $\Psi$ from $F$ without the SVD for SNH/NH — SNH/NH need $J$ and
  $I_C$ only; COR needs the polar rotation, i.e. the SVD).
* `k_elem_grad`: per free vertex through the vertex→tet CSR: $G_v \mathrel{+}= h^2 V_e P_e a_a$.
  Gather, deterministic.
* `k_bsr_assemble`: **one thread per upper BSR slot** (`col >= row`, static pattern): loops
  over the slot's (tet, a, b) contributions, forms $b_a = V^\top a_a$, $b_b$, $B_{ab}$, the
  closed-form $S_{ab}$ of MS Appendix B, then $V_e h^2\,U S_{ab} U^\top$; adds the mass on
  diagonal slots; writes the block and its transpose into the mirror slot (`slot_mirror`,
  built once with the pattern). The matrix is therefore bitwise symmetric in every precision,
  only 10 of the 16 vertex-pair blocks per tet are formed, and the gather map is ~40% smaller.
  No atomics; this is the gather counterpart of [Z25]'s warp-reduced scatter and gives bitwise
  reproducibility. Cost per contribution ≈ 120 flops. Free vertices and tets are renumbered
  spatially at init (Morton order of rest positions, RCM as the alternative) so the per-slot
  gathers of a tet's cache hit L2; if measurement shows the gather still DRAM-bound, the
  fallback is a two-phase variant (per-tet block precompute, then slot gather) or [Z25]'s
  sorted-contribution warp-reduced scatter — this A/B is an M6 item (§12.3).
* Prescribed DOFs: never appear in the free BSR (they are outside `[0, n_free)`); element
  gradient/Hessian contributions to them are dropped by the gather kernels.

### 6.4b Shells (MS §3.5–3.8) — `k_shell_cache`, `k_hinge_cache`

* `k_shell_cache`: per triangle: $D_s$, $F = D_sD_m^{-1}\in\mathbb R^{3\times2}$, thin SVD from the
  $2\times2$ eigen-decomposition of $F^\top F$ ($\hat n = u_1\times u_2$), $\Psi_i$, the
  $2\times2$ $A$, $\tau,\phi$, $\nu_i = \Psi_i/\sigma_i$, projection, PK1; writes the padded
  tet-layout cache so the elasticity kernels of §6.4 run unchanged over a gather map tagged by
  element kind (tet or shell) with $V_e := tA_e$.
* `k_hinge_cache`: per interior edge: $\theta$, $\nabla\theta$, the Gauss–Newton scalar
  $2k_b|\bar e|/\bar h$; the gradient gathers per vertex through a vertex→hinge CSR; the
  rank-one Hessian goes through the two-pass path of §6.5 exactly as a contact pair does. With
  `bending_hessian = "full"`, a `k_hinge_hessian` forms and projects the $12\times12$ and the
  BSR pattern is built over triangle ∪ hinge adjacency.
* Surface for contact: a cloth body's own triangles, edges and vertices, thickness $t/2$; no
  extraction. `lumped_masses` adds $\rho tA_e/3$ per vertex; `fem_max_diagonal` and the
  gradient sum three element families (tet, shell, hinge).
* Line search: reject a trial with $A(\hat x)/A_e < 10^{-3}$ on any triangle (MS §3.8).

### 6.5 Contact Hessian (MS §5.4) — matrix-free

No pair blocks are inserted into the BSR. The pair term acts through a **two-pass rank-one
product** (exact algebraic rewrite of $\sum_i\gamma_i\mu\,\nabla d_i\nabla d_i^\top x$):

* Once per outer iteration (right after the active-set merge and `k_linearize`):
  `k_pair_coeff` writes $c_i = \gamma_i\mu$ (LUT of $\Gamma^n$, $n\le 44$); `k_vp_grad` copies the
  3-block $(\nabla d_i)_v$ of every incidence into `vp_grad` in CSR order (24 B, contiguous per
  vertex); `k_contact_diag` accumulates $\sum_{i\ni v}c_i(\nabla d_i)_v(\nabla d_i)_v^\top$ into
  `diag_contact` (symmetrized: six upper entries mirrored). None of these change inside the
  inner Newton loop ($\nabla d$ is frozen at the anchor, $\gamma$ changes only at MS §6 line 6.3),
  so the spec's earlier per-Newton-step `k_contact_diag` is retired.
* Per SpMV: `k_pair_sigma` (one thread per pair) $\sigma_i = c_i\,\nabla d_i^\top x_{in}$ (four
  24-B gathers of $x_{in}$, prescribed slots contribute 0); then the BSR row kernel's tail (or
  a separate `k_vertex_gather`) does $y_v \mathrel{+}= \sum_{i\ni v}\sigma_i\,\texttt{vp\_grad}[k]$: a
  coalesced 24-B stream plus one 8-B read of $\sigma$ per incidence. Deterministic, no atomics,
  and ~2.5× less traffic than the earlier single-pass gather that re-read all four vertices per
  incidence (≈0.85 → ≈0.35 ms per SpMV at 1.45 M pairs; ≈0.2–0.7 s per step at [Z25] Fig 1
  scale). Optional layout: store $\nabla d_i$ as unit normal + four slot weights (40 B instead
  of 96 B) when the degenerate-normal fallback is written consistently.
* Friction pairs: same two passes with $J_i^\top H_{2,i} J_i$ (rank ≤ 2; $H_{2,i}$ cached per
  Newton step by `k_friction_hess`, $J_i$ constant for the step).

Rejected for now (design review 2026-09-02): merging pair blocks into the BSR each outer
iteration ([Z25]'s single-matrix design). Its own cost model loses 6–10 ms per outer iteration
to the two-pass path at 1.45 M pairs and ~30 CG iterations and only wins at ≥100–250 CG
iterations per outer iteration (stiff-cloth regime) or with a MAS-type preconditioner; it
stays documented as the M6 alternative for that regime.

### 6.6 Linear solve (MS §7) — `pcg`

Preconditioned CG on the coupled 3·n_free vector:
$A = A_{\text{static}} + A_{\text{contact}} + A_{\text{friction}}$, applied as BSR gather SpMV
(one warp per block-row, lanes over the row's blocks, shuffle-reduce; both triangles stored so
no transpose scatter) plus the two-pass terms of §6.5. Preconditioner: inverse 3×3 of
`diag_blocks` (`k_block_inverse`, once per Newton step, symmetrized). Initial guess 0. All
reductions are two-level block reductions with fixed order (deterministic).

**Iteration structure.** Exactly three kernels per iteration: K1 = SpMV (BSR + pair tail)
with fused block-partial $p\cdot Ap$; K2 = finish $p\cdot Ap$, per row $\alpha$, $x \mathrel{+}= \alpha p$,
$r \mathrel{-}= \alpha Ap$, $z = D^{-1}r$, block partials of $r\cdot z$ and $r\cdot r$; K3 = finish, $\beta$,
$p = z + \beta p$, convergence flag ($\|r\|^2 \le \text{rel\_tol}^2\|b\|^2$). The three kernels are
captured once per Newton step as the body of a CUDA conditional while-node (CUDA ≥ 12.4;
we build on 13.0) with an iteration cap, so the loop runs without host launches or syncs;
fallback: block replay of `check_interval` (8) iterations with one host read between blocks.
Estimated −13% per iteration at 0.87 M vertices (vector traffic) and −35% at 50 k vertices
(launch gaps), consistent with libuipc's own measurements of its plain loop. The graph is
re-instantiated when pair capacity or the BSR capacity changes.

The symmetric-upper BSR storage with warp-reduced transpose contributions ([Z25] §5.1) halves
matrix memory at the price of atomics; the float block mirror of §12.4 halves traffic without
atomics and is preferred. Both are M6 options.

**As built (2026-09-06, `linsys/pcg.cu`, `linear_solver.graph`, default on).** The whole CG
recurrence lives on the device: a `PcgDeviceState` struct holds $r^\top z$, $p^\top Ap$,
$\alpha$, $\beta$, $\|r\|^2$, the iteration count and the done / converged / breakdown /
stagnated flags, and one iteration is the operator apply (BSR SpMV, contact two-pass, bending),
`k_partials` (block partials of $p\cdot Ap$ in the fixed reduce layout), `k_finish_pAp` (one
block folds the partials, forms $\alpha$, flags a non-SPD breakdown), `k_update_xrz` (the
$x, r, z$ update fused with the block partials of $r\cdot z$ and $r\cdot r$), `k_finish_beta`
(folds them, tests convergence in either criterion, forms $\beta$, counts the iteration, runs
the stagnation window) and `k_update_p`. Every kernel, the operator's included, takes a
pointer to the `done` flag and returns at once when it is set, so a replayed block costs only
launch time past convergence. `check_interval` (8) iterations are captured by stream capture
into a CUDA graph once per solve, the executable graph is refreshed with `cudaGraphExecUpdate`
(instantiated only when the topology changed, e.g. contact appearing) and replayed until the
flag is set, with one pinned-memory readback of the state per replay. The conditional
while-node is not used: with the early-exit guards the replay costs the same and the code
stays on the portable API. Consequences: the absolute criterion is now tested every
iteration (it was every `check_interval`), the CG coefficients for the Lanczos estimate are
written to device histories and read once at the end, and the reductions are the same
fixed-order two-level sums as before, so the path is bitwise reproducible run to run (its
last bits differ from the plain loop's because the partials fold in a different order; the
plain loop is kept as `graph: false`). The operator's kernels take the stream they launch on;
the solver's own stream is a blocking stream, so the legacy default stream orders itself
around the solve without events.

**Fused solve for small systems (2026-09-07, `linsys/pcg_fused.cu`, `linear_solver.fused`,
`fused_max_rows` 65 536).** Below that size one cooperative kernel runs the whole loop:
phase 0 (x, r, z, p and the partials of $r\cdot z$, $b\cdot b$), then per iteration phase 1
(BSR rows in 16-lane groups; the pair-side passes of contact, friction and bending through
device functions shared expression for expression with the kernel path,
`contact/contact_spmv_view.cuh`, `fem/bending_spmv_view.cuh`), phase 2 (the vertex-side
gathers into $Ap$ in the kernel path's order, partials of $p\cdot Ap$), phase 3 ($\alpha$
from partials every block folds identically; the $x, r, z$ update; partials of $r\cdot z$,
$r\cdot r$) and phase 4 ($\beta$, convergence, the $p$ update), with a grid barrier between
phases. One launch and one readback per solve; the graph path when the operator has no
views or the grid cannot be co-resident. Two rules the barriers impose, both found the hard
way: a 16-lane shuffle must carry its own half-warp mask (a full-warp mask deadlocks when the
warp's other group is elsewhere), and a partial array that every block reads after a barrier
must not be rewritten until two barriers later (three partial arrays, one per reduction
phase; with two, fast blocks overwrote what slow blocks were still folding and the solve was
not reproducible). Measured on the twisting cloth: 16.5 → 11.3 ms per step, the CG iteration
0.053 → 0.032 ms (optimization pass of 2026-09-07).

**Rigid-mode coarse correction (2026-09-19, `linear_solver.rigid_coarse`, default off).**
Scenes of many small stiff bodies that slide on each other (the 54 cards of the hand shuffle:
membrane stiffness 3.5e5 N/m against an inertia of $m/h^2$ = 4.5 N/m per vertex) are badly
conditioned under block-Jacobi: a body's rigid motion costs inertia only, while every vertex
block is scaled by the membrane stiffness, so $M^{-1}A$ has eigenvalues down to $3\cdot10^{-6}$
(measured by the Lanczos estimate: $\lambda_{\min}$ 1e-5 to 2e-6, $\lambda_{\max}$ 3.7,
$\kappa$ up to 1.7e6; 500 to 2000 iterations per solve in the push, 80 % of the run time).
The remedy is the standard two-level additive preconditioner with one coarse space per body,

$$M^{-1} = D^{-1} + \sum_b P_b\,G_b^{+}\,P_b^\top,\qquad
P_b\big|_i = [\,I_3\;\; -[\rho_i]_\times\,],\quad \rho_i = \hat x_i - c_b,$$

the six rigid modes of body $b$ about the centroid $c_b$ of its free, non-held rows at the
current iterate. The coarse matrix is the body-diagonal block of the operator,

$$G_b = P_b^\top (A_{\text{static}} + h^2 K_{\text{bend}})\,P_b \;+\; \sum_{i\in b} P_b|_i^\top\, C_i\, P_b|_i ,$$

where the first term is exact (elements and hinges never span bodies, so six applications of
the operator *without* its contact part to the six mode vectors of all bodies at once give
every body's block; a rigid motion therefore sees the inertia, the anchoring by prescribed
neighbours and the geometric stiffness, but no elastic or bending stiffness) and $C_i$ is the
per-vertex diagonal block of the matrix-free contact and friction terms that block-Jacobi
already uses (`ContactSystem::diagonal_blocks`). Dropping the cross terms between two vertices
of the same pair on the same body underestimates the pair's rigid stiffness by at most the
factor 3 of a triangle's barycentric weights, which bounds the largest eigenvalue; the contact
coupling *between* bodies is left to CG, as in block-Jacobi by body. $G_b$ is symmetrized and
pseudo-inverted on the host (6 x 6 eigen-decomposition, eigenvalues below 1e-12 of the largest
dropped: a body whose free rows are collinear or a single vertex), once per Newton step.
Held rows have $P|_i = 0$. Bodies with more than `rigid_coarse_max_rows` (4096) free rows get
no correction: their per-body loops are serial in one device thread (deterministic order), and
a large body has few rigid-mode problems relative to its size.

Per iteration: `k_coarse_restrict` (one thread per body: $y_b = G_b^{+}\sum_i P|_i^\top r_i$)
and `k_coarse_prolong` (one thread per row: $z_i \mathrel{+}= y_b^{t} + y_b^{r}\times\rho_i$), then
the block partials of $r\cdot z$ are formed again. Both carry the `done` guard and run inside
the captured block of the graph path; the plain path calls the same kernels; the fused
single-kernel path is bypassed while the correction is on. $M^{-1}$ stays symmetric positive
definite, so CG is unchanged; the relative criterion is measured in the new $M^{-1}$ norm.
Setup cost per Newton step: six SpMV and bending applications, twelve small kernels, one
download and upload of 36 doubles per body.
Tests (`test_rigid_coarse`): (1) bodies of stiff springs with small masses coupled by soft
springs: the solution equals the block-Jacobi one to 1e-8 relative and the iteration count
drops by more than 5x; (2) a held row stays bitwise zero and is left out of the modes; (3) a
single-vertex body and a collinear body do not break the solve (pseudo-inverse); (4) the
graph and plain paths agree to the usual last bits; (5) the correction off reproduces the old
iterates bitwise.

### 6.7 Line search and inner loop (MS §5.5, §7)

* `k_energy_*`: inertia (free), elastic (per tet, `elem_energy`), contact
  $\tfrac12\gamma_i\mu(\tilde d^0_i + \nabla d_i^\top\hat x)^2$ (per pair), friction (per friction
  pair) → one device reduction (fixed order) → $\mathcal L_s$.
* Host loop: $r=1$ (NH: $r \leftarrow \min(1, 0.9\,t_{\text{inv}})$ from `k_inversion_toi`),
  `k_step(x_hat = x_ls0 + r p)`, energy, halve until decrease (tolerance `energy_tolerance`),
  at most `max_halvings`; on failure keep $r$ = last tried and log.
* Inner loop repeats §6.3–§6.7 until $r=1$ or `inner_max_iters`. With
  `newton.increment_velocity_tol` > 0 an accepted full step ends the loop only when the RMS
  increment velocity $\sqrt{\sum_{\text{free}} |p_v|^2 / n_{\text{free}}}\,/h$ is below the
  tolerance (C-IPC's PNTol; one `reduce_sum_sq` per accepted step). The card shuffle needs it:
  under the [Z25] rule a released stiff card straightened by creep over ~50 frames
  (observed on the C-IPC card-shuffle scene).
* Outer loop (§6.9): with the same tolerance the loop continues past β ≤ ε while the accepted
  subproblem step's RMS velocity of the free vertices (from `k_displacement` before the blend)
  is at or above it, at most `K_min` extra iterations per step (MS §15.4); `stats.csv` column
  `outer_continued` counts those iterations.

### 6.8 Multiplier / decay update (MS §8) — `k_update_lambda`

Per pair at the subproblem's $\hat x$: $c = g_i + \nabla d_i^\top(\hat x - x)$ (unfolded gap),
$\lambda' = \max(0,\lambda-\mu c)$, `ninact` = 0 if $\lambda'>0$ else +1. Removal flag when
$\Gamma^{n} < \gamma_{\min}$ (compaction happens in the active-set merge, §6.10).

### 6.9 Friction snapshot (MS §13) — `k_friction_snapshot`

At step start, after `k_linearize` at $x^t$: per pair in $\mathcal C^t$: $\hat F = \max(0,\lambda - \mu(d - \delta))$
(or $\lambda$), keep if $>0$; closest-point weights and tangent basis at $x^t$; compaction by
CUB `DeviceSelect`. Per-step kernels `k_friction_energy/grad/hess` use $u = J(\hat x - x^t)$ and
the $f_0$ family of MS §13.

### 6.10 CCD, step bound and active-set expansion (MS §9–§10) — the contact stage

1. `k_swept_aabb_{vert,edge,tri}`: boxes over `[x_anchor, x_hat]` plus $\xi$, converted with
   directed rounding (§3.2), written **in place** into the buffer the trees were built on.
   Only the hit test matters to the AL method, so there is no $\hat d$ padding (libuipc pads
   every box by $\hat d$ + thickness on both sides, which turns the candidate set into a
   proximity set).
2. LBVH (`LBVHStacklessLite`): `compute()` at init; `refit_structure()` (Morton re-sort +
   rebuild) **once at the start of each step** for the body triangle and edge trees, when the
   sweeps are largest; **`refit()`** (bounds-only bottom-up merge, three kernels, no sort, no
   host sync) at every later outer iteration. The anchor always lies on the previous sweep
   (MS §6 line 6.6), so later sweeps are nested in the first one and the tree stays fitted;
   a normalized SAH ratio (Σ internal-node area / Σ leaf area vs. its value after the last
   rebuild, evaluated on device, acted on at the next iteration) triggers an extra rebuild
   above `rebuild_quality_ratio`. Kinematic collider trees: `refit_structure()` once per step;
   static collider trees built once. Queries per outer iteration: body-vertex boxes vs body
   triangle tree, edge-tree self-query (emitting each pair once), body vertices/edges vs each
   collider's trees, **and collider vertices vs the body triangle tree** (a sharp collider
   vertex against a body face would otherwise be missed). Query order reuses the Morton
   permutation computed at the step's rebuild (no per-query sort).
3. Admissibility inside the traversal's leaf test (predicates are state-independent, so the
   set is exactly the spec's): shared vertex, contact-table exclusion, same body with
   self-collision off, all-four-prescribed, and (optionally) tet-surface one-ring; no
   `remove_if` pass. Per-leaf primitive loads happen inside a latency-bound loop, so if
   measurement shows they slow traversal, the fallback keeps only the ordering fix and a
   per-primitive mask at the leaf and moves the shared-vertex test to the narrow phase.
   Pair capacity: count-then-allocate on the first CCD of a step and a high-water mark
   afterwards (overflow detected on device, re-run at 2×) instead of the 128/vertex and
   256/edge static multipliers, which would need ~7 GB at 1 M vertices.
4. `k_accd_pt`, `k_accd_ee`, `k_plane_toi`: ACCD with $\xi_i$ and $s$, in double, in the
   pair-local frame (§11.2), `max_iter` 100; the ACCD miss loop stops at $t = 1$ (libuipc runs
   to 1.1 with a 1000-iteration cap). **Hit-only output**: a miss writes nothing; a hit does an
   integer atomic-min of its TOI into the device $\alpha$ and into $T_v$ of its (up to four)
   vertices, tests membership in $\mathcal C$ by binary search on the sorted `pair_key`, and
   appends `(key, idx, t_i)` through a warp-aggregated integer atomic when not a member. Plane
   pairs likewise emit hits only (libuipc enumerates every vertex × plane pair per iteration),
   and so do sphere pairs, through `k_sphere_ccd` with the closed-form moving-radius root of
   MS §10.1 (pair type `kPS`, `pair_idx.x` vertex, `.y` sphere id).
   The optional float screen of §12.4 runs before the double ACCD when enabled. NH:
   `k_inversion_toi` over tets whose $\det D_s(\hat x)$ is below a threshold, folded into the
   same $\alpha$ atomic.
5. $\alpha$ is read once from the device (one host sync per outer iteration; the merge below
   keeps counts on device).
6. Earliest-impact filter: `k_keep_flag` over the compacted hits: `t_i <= max_{v in i} T_v + tol`,
   with $T_v$ taken over new hits only (MS §9; the libuipc fork minimizes over all candidates
   including members of $\mathcal C$, a stricter rule exposed as `contact.toi_filter_domain`).
7. Merge: $\mathcal C$ is kept permanently sorted by `pair_key`. Removed pairs are compacted
   out with a stable select; the kept hits are radix-sorted alone (≤ 10⁵ keys), deduplicated,
   and merged into $\mathcal C$ by a merge-path set union (new entries get
   $\lambda=0, n=0$; existing entries keep $\lambda, n$; a hit whose key is already in
   $\mathcal C$ takes $\mathcal C$'s record). Set-identical to [UIPC-AL]'s
   sort-unique-with-priority `update_active_set`, without re-sorting all of $\mathcal C$ each
   iteration. Then rebuild `vp_*` CSR and run the per-outer-iteration precompute of §6.5. When
   no pair was added or removed and the anchor did not move ($\alpha=0$), the CSR rebuild,
   `k_linearize` and the precompute are skipped (they must still run after a stall adaptation,
   MS §11.2, because $\delta_i$ changed).

   **As built (2026-09-06, `contact/active_set.cu`).** The union runs on the device as four
   thrust passes and one gather kernel: `copy_if` of the surviving indices of $\mathcal C$
   (stable, so the survivors stay sorted), `copy_if` + `sort_by_key` (radix, stable) +
   `unique_by_key` of the kept hits, `set_union_by_key` over the two sorted key arrays with
   tagged source indices as values (index into $\mathcal C$, or $|\mathcal C|$ + index into
   the hit arrays), and `k_gather_union`, which writes the six pair arrays of the union into a
   second set of arrays that is then swapped with the live one (so the steady state allocates
   nothing) and counts the types with block-partial integer atomics. Determinism: every pass is
   a deterministic primitive on unique keys, and the gather's only atomics are integer counts;
   `test_active_set_union` checks the result bitwise against a host implementation of the
   merge above, twice, on random sets with removals, duplicate hits and hits already in
   $\mathcal C$. Thrust's temporaries come from a bump pool (`DevicePool`) reset once per union,
   so no `cudaMalloc` happens inside the loop after the first outer iteration. *Deviation from
   the design line "sizes stay on device":* the survivor, hit and union sizes are read back by
   the thrust primitives (three short syncs) and the four type counts by one `cudaMemcpy`,
   because the kernels that follow are launched with those sizes; a fully device-driven
   variant (CUB with device-resident counts) would save about 50 µs per outer iteration and
   is not worth the code. The friction snapshot's compaction (§6.9, once per step) uses the same
   `copy_if` + gather (`compact_friction`). Until this date the merge was the M1 host stopgap
   (download all of $\mathcal C$ and the hits, merge on the CPU, upload): 6% of the animal
   well's step and 29% of the compressor's at 1.7 M pairs (measured on the squishy-press
   scenes), i.e. 77 M pair records through the CPU per step.
8. `k_advance_anchor`: if $\alpha > \alpha_{lb}$: `x_anchor += α (x_hat - x_anchor)` (all vertices,
   prescribed included).

Reported per outer iteration: $\alpha$, #candidates, #hits, #kept, #removed, |$\mathcal C$|.

### 6.11 Termination and end of step (MS §14–§15)

Host: $k$++, $\beta$ update when $k\ge K_{\min}$, stall logic (MS §11.2: rescale $\mu$, $\hat d$,
re-run `k_linearize` since $\delta_i$ changed), break on $\beta\le\varepsilon$ or $k\ge k_{\max}$.
Then `k_finish`: `v = (x_anchor - x_prev)/h`, `x_prev = x_anchor`. Export from `x_anchor`.

### 6.12 Debug checks (off by default)

`k_dcd_min_distance` over all candidate pairs with at least one free vertex after each step
(asserts $d\ge\xi$), a separate sweep over prescribed–prescribed pairs that reports collider
interpenetration as a scene error, NaN scans on `x_hat`/`G`, PCG residual history dump,
per-step `resolved_config.json`.

---

## 7. Host/device control flow of one step (summary)

```text
upload x_target
k_predict                         k_elem_cache(x_prev)  k_diag_max → μ
k_linearize(x_anchor)             k_friction_snapshot
loop outer (host):
  k_linearize(x_anchor)                                          // pairs changed or anchor moved
  loop inner (host):
    k_slack                       k_elem_cache(x_hat)
    G  ← k_elem_grad + k_contact_grad + k_friction_grad + inertia
    H  ← k_bsr_assemble ; diag ← static diag + k_contact_diag + k_friction_diag ; k_block_inverse
    pcg (spmv = bsr + k_contact_spmv + k_friction_spmv)  → p
    line search (energies) → r ; x_hat ← x_ls0 + r p
  until r == 1 or cap
  k_update_lambda
  CCD stage (6.10) → α, C' ; rebuild vp CSR
  k_advance_anchor(α) ; β, stall, termination (host)
k_finish ; stats ; export
```

Streams: a single stream in v1 (the pipeline is sequential); CUDA graphs are a later
optimization once shapes are stable inside the inner loop (PCG iterations are graph-friendly).

---

## 8. Test plan

Unit tests are `add_cu_test` executables under `test/`, assert-based, each with a host oracle.

### 8.1 Kernels and primitives

| Test | Oracle |
| --- | --- |
| `test_distance_gradient` | $\nabla d$ (PT/EE/PH, all sub-cases incl. parallel edges) vs central differences, $10^{-6}$ rel |
| `test_accd` | vendored ACCD (after §3.2 fixes) vs brute-force sampled TOI on random and adversarial cases; smem variant identical |
| `test_svd3_double` | $U\Sigma V^\top$ reconstruction $10^{-13}$, $\det U=\det V=1$, inverted and degenerate inputs |
| `test_elastic_eigensystem` | analytic $(\lambda_k, Q_k)$ vs Eigen's `SelfAdjointEigenSolver` of the finite-difference $9\times9$ Hessian; SNH vs `libcwheels` `buildEigensystem` |
| `test_block_assembly` | `k_bsr_assemble` blocks vs dense $12\times12$ projected Hessian from the eigen-system |
| `test_bsr_pcg` | random SPD BSR + rank-one pair terms vs Eigen dense solve; residual tolerance honoured |
| `test_al_single_constraint` | one free vertex vs a plane: iterate MS §6–§8 and compare $\lambda$, $x$ against the exact KKT solution; finite-step termination |
| `test_active_set_filter` | synthetic candidate sets: earliest-impact rule and dedup/merge preserve $\lambda,n$ |
| `test_friction_terms` | gradient/Hessian vs finite differences of $f_0$ potential; momentum sum zero |
| `test_moving_boundary` | prescribed cube pushing a free cube through the AL loop: no penetration, lag bound MS §12 |

### 8.2 Integration (scenes under `app/config/`, run by `cs_run`, checked by `test/check_*` tools)

| Scene | Check |
| --- | --- |
| `bar_drop_analytic` (MS §17.1) | slope-1 convergence of accumulated errors over $h\in\{10^{-2},...,10^{-5}\}$ |
| `two_cubes_momentum` (MS §17.2) | momentum drift $<10^{-8}$ relative |
| `cube_on_slope` (MS §17.4) | sliding threshold at $\tan\theta$, acceleration formula |
| `masonry_arch` (MS §17.5) | stands at 0.5, collapses at 0 |
| `sphere_drop_plane` | penetration-free invariant, rest height $= \delta$ above plane, bounce decay |
| `armadillo_container` (assets from `coupled_solver/asset/funnel`) | runs 300 frames penetration-free; iteration/PCG statistics logged |

### 8.3 Regression gates

Same numeric bgeo comparison approach as `coupled_solver/test/bgeo_diff.cpp` (tolerance-based
because CCD ties and the atomic-min are not bit-stable across driver versions), on the small
scenes, run by CTest.

---

## 9. Milestones and acceptance

| M | Deliverable | Accept when |
| --- | --- | --- |
| M0 | Repo skeleton, presets, `ext/`, vendored wheels compile on CUDA 13 / clang; `test_accd`, `test_distance_gradient`, `test_svd3_double` pass | CTest green |
| M1 | Elastic-only implicit Euler: scene load, `SceneBuffers`, element cache, BSR gather assembly, PCG, line search, bgeo export | `test_elastic_eigensystem`, `test_block_assembly`, `test_bsr_pcg`; a hanging bar matches a CPU Newton reference within 1e-6 |
| M2 | Planes: linearization, AL terms, λ/decay, plane TOI, anchor/β loop — the full MS §6 on PH pairs | `test_al_single_constraint`, `sphere_drop_plane`, `bar_drop_analytic` slope 1 |
| M3 | Self/inter-body contact: swept AABB + LBVH + ACCD, earliest-impact filter, merge, matrix-free contact SpMV | `two_cubes_momentum`, `armadillo_container` penetration-free |
| M4 | Friction snapshot and terms | `cube_on_slope`, `masonry_arch` |
| M5 | Moving boundaries (scripts, keyframes, kinematic colliders), NH inversion guard, stall adaptation | `test_moving_boundary`; twisting-rods-style scene stable |
| M9 | Shell membrane (MS §3.5, §3.6, §3.8): `"cloth"` body type, cloth materials, `k_shell_cache`, padded-cache reuse of the tet assembly, area-lumped mass, own-surface contact primitives with $\xi = t/2$ | `test_shell_eigensystem` against a finite-difference $6\times6$ at two stiffness scales and near $\sigma_1=\sigma_2$; `test_shell_block_assembly` against the dense projected Hessian; a hanging sheet matches a CPU Newton reference to 1e-6; `k_bend = 0` cloth drops onto a plane penetration-free |
| M10 | Mixed element families in the linear system: element-kind gather map, three-family gradient/diagonal/energy, `estimate_mu` on a pure-cloth scene | `estimate_mu` on cloth matches a host computation to 1e-10; a scene with one tet body and one cloth body assembles an SpMV that matches a dense reference |
| M11 | Bending (MS §3.7): hinge extraction, `k_hinge_cache`, Gauss–Newton rank-one through §6.5, full $12\times12$ behind `bending_hessian` | `test_bending_terms` energy/gradient vs finite differences; a strip curls to a prescribed rest angle; `k_bend = 0` is bitwise the M9 result |
| M12 | [Z25] Fig. 22 twisting cloth from the authors' OGC script: keyframed grips, $K_{\min}=2$, $\epsilon$ large | penetration-free for the whole run, and the sheet returns flat when the twist unwinds (the paper's own pass/fail); #CG and #contacts reported against Table 2 as information |
| M7 | Analytic sphere colliders with a prescribed radius (MS §4.1 PS, §10.1, §12.3) and the MEDIT `.mesh` reader; Fig. 21 trapped squishy balls | `test_sphere_contact` (rest clearance and no penetration inside a static inverted sphere; a shrinking sphere compresses a body without penetration), `test_mesh_load` MEDIT case |
| M6 | Performance plan of §12: microbenchmark gate first, then the adopted items in measured order; float screen and mixed-precision PCG behind their tests | per-stage timings within 3× of [Z25] on the RTX 4090 |

Merging policy: work on `dev`; `master` receives only milestones whose tests pass.

---

## 10. Risks and open questions

1. **Device SVD accuracy.** Warp's McAdams port runs a fixed 8 Jacobi sweeps on $F^\top F$ in
   double: expect orthogonality near $10^{-13}$ and reduced relative accuracy for singular
   values below $\sim10^{-8}\sigma_{\max}$. Both are harmless for the PSD-projected element
   Hessian (eigenvalues are clamped), but `test_svd3_double` must pin them down; libuipc's
   implicit-QR SVD is the fallback if a case needs more.
2. **Broad-phase cost.** [Z25] report CCD as the dominant cost in contact-heavy scenes; the
   LBVH refit + queries run once per outer iteration (not per line-search trial), which is
   already the paper's advantage over IPC. Keep pair buffers capacity-managed and measure the
   one-ring exclusion option.
3. **Conditioning at high stiffness.** Block-Jacobi PCG is [Z25]'s choice and matched their
   iteration counts; if stiff scenes (1 GPa) exceed a few hundred PCG iterations, a MAS-style
   preconditioner is the fallback (libuipc has one; out of v1 scope).
4. **Frozen-slack line search** can reject steps that the eliminated functional would accept
   (inactive pairs act as two-sided wells). If observed, evaluate $\mathcal L$ with $s^\star$
   recomputed per trial (cheap) and record the change in the math spec.
5. **Parallel edge pairs.** We keep them with a sub-gradient; if conflicting constraints
   appear in cloth-like configurations, switch `drop_parallel_edge_pairs` on (libuipc default).
6. **Kinematic colliders that intersect each other or the domain at $t=0$** are user errors;
   the initial DCD sweep (debug check) reports them.
7. **Assets.** The [Z25] scenes are not public; `coupled_solver/asset` (armadillo, funnel,
   animal_crossing, basic primitives) and libuipc's `animal_well.msh` cover M1–M5; larger
   benchmark scenes will need generated tet meshes (fTetWild is available locally).

---

## 11. Precision policy

Decided 2026-09-02 after review. `real` is `double` by default (`CS_USE_DOUBLE=ON`); a float
build is permitted only under the rules below, which also improve the double build.

1. **Always double, in every build:** closest-point queries, $\nabla d$, the anchor gap $g_i$,
   ACCD and plane TOIs, the TOI comparisons of the earliest-impact filter, the $\mu$ estimate,
   and every global reduction (energies, PCG dot products and norms, max-displacement checks).
   Reductions use fixed-order two-level block sums with double accumulators.
2. **Displacement form everywhere.** Constraints, slack, multiplier updates and friction use
   $\nabla d_i^\top(\hat x - x)$ or $J_i(\hat x - x^t)$; no kernel forms $\nabla d^\top x$ on absolute
   positions (MS §5.1). Distances and CCD kernels translate each pair to its own centroid
   before computing (differences taken in double, then optionally cast): the gap
   $10^{-3}$ m is then resolved relative to a $10^{-2}$ m local frame, not a $10$ m scene.
3. **Line search compares differences.** $\mathcal L_s(\hat x + rp) - \mathcal L_s(\hat x)$ is
   accumulated per element/pair as a difference and reduced in double, so the decrease test
   is not drowned by the magnitude of the total energy.
4. **Float-eligible in a float build:** the elasticity SVD/eigen-system (Warp
   `_svd_config<float>`, 4 sweeps), BSR block values and PCG vectors. Mixed precision is the
   intended float mode: matrix and vectors in float, residual and reductions in double, with
   automatic fallback to a double solve when the PCG residual stagnates (no decrease over 50
   iterations). The performance plan (§12) refines this with iterative refinement once the
   condition-number regime is measured.
5. **SVD:** Warp's templated McAdams port for both precisions (§3); `test_svd3_double` and a
   float twin assert $\det U=\det V=+1$, $\sigma_3<0\iff\det F<0$, reconstruction error and
   orthogonality bounds ($10^{-13}$ double, $10^{-6}$ float).
6. **Float broad phase by certificate.** AABBs are float with directed rounding (§3.2), so
   the float tree never drops a double overlap; the float anchor of a float build is *not*
   allowed — a float anchor at 10 m re-rounds by ~5·10⁻⁷ m per blend and would leave the
   CCD-certified segment (MS §10.2). Hence the state split of item 1: positions always double.

## 12. Performance plan (design review, 2026-09-02)

Outcome of a source-level review of [UIPC-AL] and of `coupled_solver`'s wheels, four
independent design passes, two scoring passes and adversarial verification of the top items.
Gains below are the *corrected* estimates after verification; every one is a hypothesis until
the M6 microbenchmarks (§12.3 item 0) have run.

### 12.1 Where the time goes

[Z25] Table 1, their implementation, per step (fractions of step time):

| Scene | Hess | PCG | CCD | LS+misc | Newton / CG iters |
| --- | --- | --- | --- | --- | --- |
| Fig 1 squishy balls (0.87 M V, 0.53 M pairs) | 14% | 23% | **58%** | 5% | 30 / 28 |
| Fig 4 animal well (0.43 M V, 38 k pairs) | 18% | 41% | 34% | 7% | 19 / 54 |
| Fig 23-easy stacked cloth | 23% | 20% | **53%** | 5% | 23 / 41 |

The fork's benchmark (animal well, 300 frames, mean ms per frame, from its
`component_breakdown.png`): original 218 / 402 / 384 / 88 (Hess / PCG / BVH&CCD / misc, total
1092 ms) versus libuipc 1078 / 624 / 1760 / 58 (total 3520 ms). Newton and PCG counts match,
so the 3.2× gap is per-iteration kernel cost: BVH&CCD 4.6× (57% of the gap), Hessian assembly
4.9× (35%), PCG 1.55× (9%).

### 12.2 Root causes in [UIPC-AL] (verified in source; none of them apply to our design)

CCD/BVH: three trees rebuilt from scratch every outer iteration (the code has no refit path);
every box padded by $\hat d$ + thickness on both query and tree side although the AL path
consumes hits only, so near convergence the candidate count is a proximity count; the query
set is Morton-sorted per query (points twice); the leaf broad-phase test is re-evaluated at
the top of the TOI kernel; TOIs are computed for all candidates including members of
$\mathcal C$; ACCD runs to $t = 1.1$ with a 1000-iteration cap (fork: $\eta = 10^{-3}$); every
vertex × plane pair is enumerated and merged each iteration; the merge sorts
$|\mathcal C| + |\text{all candidates}|$ 64-bit keys three times; overflow re-runs the whole
traversal; the traversal flushes 1024 shared slots per block synchronously; about nine host
syncs per outer iteration.

Assembly/PCG: generic 80-B triplets, $N_{\text{raw}} \approx n_v + 10n_{\text{tet}} + 10n_{\text{pairs}}$
(+10 per friction pair), i.e. ~7× the unique block count; a symmetrization pass that copies
all blocks twice for no effect; a full 64-bit radix sort + gather + run-length + segmented
reduce every Newton iteration although the FEM pattern is static and the contact pattern
changes only per outer iteration; SpMV launched over the raw capacity (~85% empty threads)
with three double atomics per off-diagonal block; buffers refilled and re-read for fixed
vertices; contact triplets copied twice; PCG on the plain launch path (graph replay disabled
for al-ipc) with ~10 launches per iteration and a sync every five; a numeric 9×9/12×12
eigendecomposition for the SPD projection ([Z25]'s analytic projection is listed as WIP).

### 12.3 Adopted items (in the order they enter the milestones)

| # | Item | Effect (corrected) | Condition / test |
| --- | --- | --- | --- |
| 0 | **Measure first**: `LBVHStacklessLite` build vs `refit_structure` vs `refit` and query time at 1 M and 2.4 M objects; ACCD hit/miss iteration histograms at $s=0.1$; BSR gather assembly vs the paper's warp-reduced scatter on a 2 M-tet fTetWild mesh; contact SpMV variants at 1.45 M pairs | orders the rest of M6 | all CCD numbers below are extrapolations from 150–185 k-object CSVs |
| 1 | Refit per outer iteration, rebuild per step + quality trigger (§6.10 step 2) | 20–38 ms per CCD at Fig 1 scale, i.e. 0.3–0.75 s per step; does **not** touch the query cost the paper names dominant | in-place box buffer; `__threadfence` fix; refit ground-truth test |
| 2 | Half-emission self-query, leaf-side admissibility, count-then-allocate capacity (§6.10 step 3) | ~1–2 ms per iteration plus two host syncs; ~10× smaller raw pair buffers | `>=` reject; mask-only fallback if per-leaf loads slow traversal |
| 3 | Collider-vertex vs body-triangle query (§6.10 step 2) | correctness | — |
| 4 | Hit-only narrow phase, device-side $\alpha$ and $T_v$, plane hits only, merge-path active-set union, skip logic (§6.10 steps 4–7). **The union was left on the host until 2026-09-06** (it was noise on the animal well and 29% of the step on the compressor); device version in `contact/active_set.cu`, measured on the squishy-press scenes | <1% of step time directly; removes four host syncs per outer iteration | integer atomics only (order-independent); skip logic invalidated by stall adaptation |
| 5 | Directed-rounding float AABBs (§3.2) | zero cost; certificate replaces a heuristic | — |
| 6 | Two-pass contact/friction SpMV with per-outer-iteration precompute (§6.5) | ≈0.5 ms per SpMV at 1.45 M pairs → ≈0.2–0.7 s per step at Fig 1 scale | exact identity; `test_bsr_pcg` parity |
| 7 | Upper-slot assembly with mirror writes (§6.4) | 37% fewer assembly flops, bitwise symmetry | prerequisite for the float matrix mirror |
| 8 | Fused three-kernel PCG iteration in a conditional CUDA graph (§6.6). **Built 2026-09-06** as a captured block of `check_interval` guarded iterations replayed with `cudaGraphExecUpdate` (§6.6 as built); measured in the PCG acceleration pass of 2026-09-06 | −13% per iteration at 0.87 M V, −35% at 50 k V | block-replay fallback; re-instantiate on capacity change |
| 9 | Spatial renumbering of free vertices and tets at init (§6.4) | assembly and every gather become L2-resident; measured | two-phase fallback if not |
| 10 | Batched line search: energies for $r\in\{1,\tfrac12,\tfrac14\}$ in one pass, decision on device | removes 2–3 host syncs per Newton step; enables whole-inner-step graph capture later | — |

Implementation-level plan of 2026-09-06 (all double; ordered by measured payoff).
**Status 2026-09-07: all built and measured, one by one, in that day's optimization pass.**
Items 11, 12 and 15 pay as planned; 10 and 13 are neutral (the cost they addressed, host
readbacks, was not where the time was); 14 became a fused cooperative CG kernel for small
systems rather than a whole-step graph; 16 was tried and is slower, not adopted.

| # | Item | Expected | Condition / test |
| --- | --- | --- | --- |
| 11 | Two-phase Hessian assembly: per-tet pass writes its ten blocks to a staging buffer (coalesced), per-slot gather sums 72 B per contribution instead of re-reading 360 B of cache | assembly stage halved on every tet scene (animal well 0.60 → ~0.3 s/step) | same per-contribution arithmetic and summation order as the direct kernel: agrees to rounding (one ulp from multiply-add contraction), bitwise symmetric, deterministic; `test_block_assembly` parity, single-chunk and chunked |
| 12 | Line-search energy reuse: the accepted trial's elastic energy is the next Newton step's start energy; only the contact and friction parts are re-evaluated | one energy SVD pass per tet per Newton step saved; compressor line search 0.59 → ~0.35 s/step | exact: the elastic energy at x_ls0 is the value the previous trial computed |
| 10' | Item 10 as above | | |
| 13 | Device-resident counts in the contact stage: candidate count, degenerate count and the union's sizes stay on the device; only $\alpha$ is read back | 4 fewer syncs per outer iteration; twisting cloth 15 → ~11 ms/step | integer counts only |
| 14 | Whole-inner-step graph after 10 and 13: element cache, gradient, assembly, PCG and line search of one Newton step captured as one graph | twisting cloth ~11 → ~8 ms/step | re-instantiate on capacity change |
| 15 | Chunked narrow phase: ACCD over the broad-phase output in chunks instead of materialising every candidate | compressor memory 18 → ~13 GB; the 300-frame run fits a shared GPU | identical hit set |
| 15b | Bounded hit storage: the CCD hit arrays grow only up to a cap (4 M entries, ~190 MB); a pass that overflows it is repeated with the earliest-impact tie filter applied at report time against the completed per-vertex earliest-impact times T_v, so only the hits the merge would keep are stored | compressor release phase: the >1e8 hits of a decompression pass no longer force a multi-GB allocation (the run went out of memory at frame ~205) | identical kept set: the filter is the predicate `k_keep_flag` applies; T_v is complete after any full pass and a repeated pass cannot lower it; `test_hit_capacity` compares the merged set against the unbounded path |
| 16 | Coalesced block loads for the SpMV: a warp reads a row's block range contiguously and shuffles the values to lanes | exploratory; up to 20% on the SpMV | measure before adopting |

### 12.4 Single and mixed precision (the fp32 question)

The AL formulation is what makes reduced precision possible at all: the contact Hessian is
$\gamma_i\mu\,\nabla d_i\nabla d_i^\top$ with $\|\nabla d_i\|\le\sqrt2$ and $\mu = 0.1\max\operatorname{diag}$,
so $\kappa(H)$ stays at the elasticity's level instead of diverging like a barrier's $1/d^2$.
What breaks in fp32 is geometry, not conditioning. The plan, in order of adoption:

1. **State split** (always, both builds): positions, velocities, anchor and targets are
   `double`; the compute type `real` applies to the element cache, matrix blocks, gradients,
   multipliers and PCG vectors. A float anchor would leave the CCD-certified segment (§11.6).
2. **Float BSR mirror, double vectors** (Tier A): `k_bsr_assemble` also writes `blocks_f`
   (36 B per block, mirror-rounded so the float matrix is exactly symmetric) and `pair_grad_f`;
   the SpMV reads float, accumulates in double. ≈1.5× less traffic per PCG iteration. The PCG
   then solves $H_f p = -G$ with $\|H_f - H\| \le 2^{-24}\|H\|$; this is Newton inexactness,
   not a loss of any AL guarantee (line search, CCD and multiplier update never read $H$).

   **Built, measured and removed (2026-09-06).** The mirror was implemented as described
   (48 B float4 blocks rounded from the double matrix after each assembly, SpMV accumulating
   in double, a breakdown guard re-solving in double) and measured in
   the PCG acceleration pass of 2026-09-06: on its own it was *slower* than the double SpMV, because the
   warp-per-row kernel was bound by the x gather and the per-row overhead rather than by block
   bytes; once the lane mapping (16 lanes per row) fixed that, the float mirror at 8 lanes was
   worth a further 15% on the SpMV and 26% per CG iteration on the animal well, 11% on the
   compressor. The project then decided to run the solver in double throughout, and the
   mirror, its guard (`stagnation_window`, which fired spuriously on CG's non-monotone
   residual) and the `matrix_precision` key were removed; the lane-mapped double SpMV stays.
   The numbers are kept in that document so the trade is on record: about 20% of the PCG
   stage on the tet scenes for a matrix rounded to 2^-24.

3. **Float vectors** (Tier B, float build only): ≈1.9× less traffic; needs iterative
   refinement (float inner PCG, double residual and correction) because CG in float loses
   orthogonality at $\kappa > 10^3$; converges while $\kappa\,2^{-24} \ll 1$.
4. **Float ACCD screen** (`ccd.float_screen`, off by default): translate the pair to its
   centroid in double, cast, run ACCD in float with the separation inflated by
   $\epsilon = c_f\,2^{-24}R$ ($R$ = local extent + displacement) and a slightly inflated
   motion bound; a float *miss* is a certified miss, everything else (hits, gaps below
   $50\epsilon$, ill-conditioned GTE sub-cases detected by a small 2×2 determinant) is
   re-run in double from $\theta=0$. Consumed TOIs are always double. Expected ≈4–8× cheaper
   narrow phase (fp32 runs 64× faster than fp64 on the RTX 4090). Adopted only after
   `test_accd` validates $c_f$ on random, sliver and near-parallel cases; the two reviewers
   split on whether the constant can be trusted, which is why it is gated.
5. **Float element cache and assembly** gated by $h^2E/(\rho l^2) < 10^4$: deferred; the SPD
   margin argument is crude and near-degenerate tets are where the float SVD is worst.
6. **Energies as differences** (§11.3) with closed-form $\Delta\Psi$ for SNH/NH (multilinear
   expansion of $\det$, `log1p`) so a float build can pass the decrease test at all.

Net: a mixed build (items 1–3) is expected to run the PCG ~1.5–1.9× faster with the same
guarantees; a float narrow phase (item 4) is the only way to cut CCD substantially below the
double cost and is a research item, not a milestone.

### 12.5 Rejected or deferred

* Merged-BSR contact blocks (paper design): loses to the two-pass path at 1.45 M pairs and
  ~30 CG iterations (§6.5); reconsider for ≥100–250 CG iterations per outer iteration.
* Modified Newton with a stale projected elastic Hessian: fragile break-even and a silent
  effect on the "full step accepted" rule; instead, a per-element exact skip (tets whose
  vertices did not move since the cached SVD) is the future optimization.
* Persistent candidate lists across outer iterations with per-vertex containment tests: the
  conservativeness proof holds, but the controller is large and a single moving vertex forces
  a refresh; revisit after item 0, together with sub-sweep boxes for the first, longest sweeps.
* $T_v$-bounded ACCD early-out: correct, but the gain depends on $T_v$ being small for most
  vertices, which the earliest-impact semantics of MS §9 do not guarantee.

### 12.6 Memory budget at [Z25] Fig 1 scale (0.87 M V, 2.25 M T, 1.45 M pairs)

| Buffers | Size |
| --- | --- |
| Vertex state (7 double arrays) + attributes | 0.25 GB |
| Element cache (45 reals) + DmInv/volume | 0.95 GB |
| Static BSR double (~14 M blocks) + float mirror + gather map | 0.95 + 0.48 + 0.35 GB |
| Pairs (SoA ≈ 150 B) + `vp_grad` + friction snapshot | 0.22 + 0.14 + 0.1 GB |
| Broad-phase pair output (high-water) + hit compaction | ≈ 0.5 GB |
| Two body trees + collider trees (~170 B per object) | ≈ 0.7 GB |
| PCG vectors, reductions, scratch | 0.15 GB |
| Total | ≈ 4.8 GB of 24 GB |

### 12.7 Open decisions

* $T_v$ domain: new hits only (paper, default) or all candidates (fork); `contact.toi_filter_domain`.
* Inexact-Newton tolerance schedule for PCG (loose while $\beta$ is far from $\varepsilon$,
  $10^{-4}$ near termination): must be validated against [Z25] iteration counts before
  becoming default.
* NH inversion guard restricted to tets near inversion (threshold on $\det D_s(\hat x)$).
* Every per-outer-iteration cache (`vp_grad`, `diag_contact`, gaps, skip flags) must be
  invalidated by the stall adaptation of MS §11.2 — stated here so no implementation forgets it.

## 13. Work breakdown (ordered)

1. M0: copy build skeleton + wheels; fix ACCD; vendor Warp `svd.h` as `svd3.cuh` with the
   `real3x3` adapter; tests 8.1 rows 1–3.
2. M1: `geometry/`, `scene/` (config schema), `gpu/SceneBuffers`; `fem/` cache, gradient,
   block assembly; `linsys/` BSR + PCG; `solver/` elastic-only stepper; `io/`; `cs_run`.
3. M2: `contact/planes.cu`, `linearize.cu`, `al_terms.cu`, `active_set.cu` (PH only), full
   stepper loop with β/stall; scenes `sphere_drop_plane`, `bar_drop_analytic`.
4. M3: `contact/broadphase.cu`, `narrowphase.cu`, PT/EE in `linearize`/`al_terms`, incidence
   CSR, matrix-free SpMV; scenes `two_cubes_momentum`, `armadillo_container`.
5. M4: `contact/friction.cu`; scenes `cube_on_slope`, `masonry_arch`.
6. M5: motion scripts/keyframes/colliders, NH inversion CCD, stall rule; `test_moving_boundary`.
7. M6: profiling, optional optimizations, implementation report `doc/al-ipc-implementation-report.md`.
