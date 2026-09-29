#pragma once
// The contact stage (doc/al-ipc-implementation-spec.md §4.4-4.5, §6.2, §6.3, §6.5, §6.8-6.10;
// math spec §4-§5, §8-§10, §13). Owns the active set C, the candidate/hit buffers, the
// per-outer-iteration precompute (incidence CSR, vp_grad, contact diagonal) and the friction
// snapshot. The stepper calls it in the order of math spec §6; every method launches on the
// default stream and checks errors.
//
// Pair layout (SoA, capacity-managed, permanently sorted by `key`):
//   type 0 = PT (p, t0, t1, t2), 1 = EE (a0, a1, b0, b1), 2 = PH (p, -, -, -; plane id in
//   `plane`). Gradient blocks are stored per vertex slot (4 x real3). Prescribed slots carry
//   their gradient too (needed for c_i) but never receive matrix/gradient contributions.
#include "core/typedef.cuh"
#include "contact/contact_spmv_view.cuh"
#include "core/device_buffer.cuh"
#include "gpu/scene_buffers.cuh"
#include "libculbvh/stacklessbvh_lite.cuh"
#include "scene/scene_desc.h"
#include <memory>
#include <vector>

namespace cs {

struct ContactParams {
    real d_hat = real(1e-3);            // δ̂ (adapted by the stall rule)
    real mu = real(0);                  // AL penalty stiffness (per step)
    real gamma_factor = real(0.9);      // Γ
    int remove_after = 44;              // n_i above which a pair is dropped (γ < γ_min)
    real accd_s = real(0.1);
    int accd_max_iter = 100;
    real toi_tie_tol = real(1e-6);
    bool toi_filter_all = false;        // MS §9: false = paper (new hits only), true = fork
    // MS §16. exclude_one_ring is not implemented (scene_desc rejects true). drop_parallel_ee
    // drops edge-edge pairs whose edges are within parallel_ee_eps of parallel, both when a hit
    // would admit them and when a member pair becomes parallel (it is then made inert until the
    // decay removes it): a folded sheet's parallel edge pairs have no separating direction and
    // lock the outer loop; their contact is carried by the point-triangle pairs (M12).
    bool exclude_one_ring = false;
    bool drop_parallel_ee = false;
    real parallel_ee_eps = real(1e-3);   // |ea x eb|^2 < eps |ea|^2 |eb|^2
    real rebuild_quality_ratio = real(1.5);
    bool friction_enable = true;
    real eps_v = real(1e-3);
    bool friction_normal_force_lambda = false; // false = paper (MS §13), true = λ_i
};

struct CcdResult {
    double alpha = 1.0;      // step bound (min over hits and the inversion guard supplied by the caller)
    long long candidates = 0;  // broad-phase pairs after leaf-side admissibility (64-bit: see StepStats)
    int hits = 0;            // ACCD hits (t < 1) not already in C
    int kept = 0;            // hits admitted by the earliest-impact filter
    int removed = 0;         // pairs removed by decay
    int filtered_passes = 0; // CCD passes repeated with the tie filter at report time (hit storage cap)
    long long prescribed_hits = 0; // hits between prescribed-only primitives: counted, never admitted (IS §5.x diagnostic)
    int n_pt = 0, n_ee = 0, n_ph = 0; // |C| by type after the update
};

class ContactSystem {
public:
    ContactSystem(const SceneBuffers& sb, const Scene& scene, const ContactDesc& desc);
    ~ContactSystem();
    ContactSystem(const ContactSystem&) = delete;
    ContactSystem& operator=(const ContactSystem&) = delete;

    ContactParams& params();
    const ContactParams& params() const;

    // ---- per step ------------------------------------------------------------------------
    // Called after the stepper set x_anchor = x_prev and x_hat (prescribed = targets), with the
    // step's μ. Rebuilds the trees from the first sweep [x_anchor, x_hat], relinearizes C at the
    // anchor, and takes the friction snapshot (MS §13) from the carried-over active set.
    void begin_step(const real3* x_anchor, const real3* x_hat, real mu);

    // ---- per outer iteration -------------------------------------------------------------
    // MS §5.1 line 6.1: (g_i, ∇d_i) at the anchor for every pair; then the per-outer-iteration
    // precompute (§6.5): incidence CSR, vp_grad, c_i = γ_i μ, contact diagonal blocks.
    void linearize(const real3* x_anchor);

    // ---- per Newton step -----------------------------------------------------------------
    // MS §7 first lines: s_i and the folded gap at the current x_hat. The slack is frozen for
    // the whole Newton step, line search included (MS §5.5), so call this once per step.
    void update_slack(const real3* x_hat, const real3* x_anchor);
    // Rebuilds the preconditioner diagonal (contact, plus the re-projected friction Hessian).
    // Also once per Newton step, before assembling.
    void update_hessian(const real3* x_hat);
    // G_v += contact gradient (folded gap model) for free vertices.
    void add_gradient(const real3* x_hat, const real3* x_anchor, real3* G);
    // 9 per free vertex, row-major: Σ γ_i μ (∇d_i)_v (∇d_i)_v^T (+ friction), valid after linearize().
    const real* diagonal_blocks() const;
    // y += H_contact x (two-pass rank-one product) and y += H_friction x.
    // Kernels launch on `stream`; `skip` is the PCG graph's early-exit flag (may be null).
    void spmv_add(const real3* x_in, real3* y, cudaStream_t stream = 0, const int* skip = nullptr) const;
    // POD view of the SpMV data for the fused CG kernel (IS §12.3 item 14); valid until the next merge/linearize.
    struct ContactSpmvView spmv_view() const;
    // Σ_i 1/2 γ_i μ (g̃_i + ∇d_i^T (x̂ - x))^2 with the frozen slack, plus the friction potential.
    double energy(const real3* x_hat, const real3* x_anchor);
    // The pair and friction sums written to two device doubles, no host synchronisation.
    void energy_to(const real3* x_hat, const real3* x_anchor, double* d_pairs, double* d_friction);

    // ---- after the subproblem ------------------------------------------------------------
    // MS §8 at the final x_hat.
    void update_multipliers(const real3* x_hat, const real3* x_anchor);
    // MS §9-§10: refit trees (or rebuild on the quality trigger), sweep [x_anchor, x_hat],
    // ACCD, earliest-impact filter, merge into C, decay removal. `alpha_cap` is the caller's
    // inversion-guard bound (1 if none). Returns alpha and the counts.
    CcdResult ccd_and_update(const real3* x_anchor, const real3* x_hat, double alpha_cap);

    // Stall rule support (MS §11.2): change δ̂ and μ; the caller must call linearize() again.
    void set_d_hat(real d_hat);
    void set_mu(real mu);

    int n_pairs() const;
    int n_free() const;
    bool has_friction_pairs() const;

    // Debug: exact minimum unsigned distance over all candidate pairs at x (DCD sweep), and the
    // number of pairs with d < ξ. Expensive; used by tests and the debug check.
    double debug_min_distance(const real3* x, int& n_violations);
    // Debug/test: cap of the CCD hit storage in entries (IS §12.3 item 15); passes past it are
    // repeated with the tie filter at report time. Default 4 M.
    void debug_set_hit_capacity(int entries);

    struct Impl;
    Impl& impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace cs
