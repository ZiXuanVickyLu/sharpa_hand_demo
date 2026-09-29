// Time stepper: math spec §6 (outer loop), §7 (subproblem), §8 (multipliers), §11 (penalty
// stiffness), §15 (termination). Host control flow plus the vertex-wise kernels; elasticity is
// in fem/, the linear system in linsys/, contact and friction in contact/.
#include "solver/stepper.h"
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include "core/cuda_check.h"
#include "core/timer.h"
#include "core/log.h"
#include "gpu/scene_buffers.cuh"
#include "fem/fem.cuh"
#include "linsys/bsr.cuh"
#include "linsys/pcg.cuh"
#include "linsys/reduce.cuh"
#include "contact/contact_system.cuh"
#include "fem/bending.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cs {

namespace {

constexpr int kBlock = 256;
inline unsigned grid_for(int n) { return unsigned(std::max(1, (n + kBlock - 1) / kBlock)); }

// x_tilde (free), x_hat (free: x_prev; prescribed: target), x_anchor = x_prev (all).
__global__ void k_predict(int n_vertices, int n_free, real h, real3 g,
                          const real3* __restrict__ x_prev, const real3* __restrict__ vel,
                          const real3* __restrict__ target, const int* __restrict__ held,
                          real3* __restrict__ x_tilde, real3* __restrict__ x_hat, real3* __restrict__ x_anchor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    const real3 xp = x_prev[i];
    x_anchor[i] = xp;
    // IS §5.x: a held free vertex is predicted onto its target like a prescribed one
    if (i < n_free && !(held != nullptr && held[i] != 0)) {
        x_tilde[i] = xp + h * vel[i] + (h * h) * g;
        x_hat[i] = xp;
    } else {
        x_tilde[i] = target[i];
        x_hat[i] = target[i];   // MS §12: every iterate satisfies the boundary condition exactly
    }
}

// MS §6 guarded Dirichlet advance: d = target - x_hat on prescribed and held vertices, 0 on free ones.
__global__ void k_bc_displacement(int n_vertices, int n_free, const int* __restrict__ held, const real3* __restrict__ target,
                                  const real3* __restrict__ x_hat, real3* __restrict__ d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    const bool driven = !(i < n_free && !(held != nullptr && held[i] != 0));
    d[i] = driven ? target[i] - x_hat[i] : make_real3(real(0), real(0), real(0));
}

// x_hat += sigma d on the driven vertices (sigma == 1 lands exactly on the target).
__global__ void k_bc_advance(int n_vertices, int n_free, const int* __restrict__ held, real sigma, const real3* __restrict__ target,
                             const real3* __restrict__ d, real3* __restrict__ x_hat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    const bool driven = !(i < n_free && !(held != nullptr && held[i] != 0));
    if (!driven) return;
    x_hat[i] = sigma >= real(1) ? target[i] : x_hat[i] + sigma * d[i];
}

// Joint start displacement: w (x_tilde - x_anchor), w = 1 on the driven vertices (x_tilde holds their targets) and
// the falloff weight on the free ones (1 next to a driven region, 0 beyond the falloff radius).
__global__ void k_joint_displacement(int n_vertices, const real* __restrict__ w, const real3* __restrict__ x_anchor,
                                     const real3* __restrict__ x_tilde, real3* __restrict__ d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    d[i] = w[i] * (x_tilde[i] - x_anchor[i]);
}

// x_hat = x_anchor + sigma d on all vertices.
__global__ void k_joint_start(int n_vertices, real sigma, const real3* __restrict__ x_anchor, const real3* __restrict__ d,
                              real3* __restrict__ x_hat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    x_hat[i] = x_anchor[i] + sigma * d[i];
}

// x_hat = x_anchor on the driven vertices (the advance starts from the anchor).
__global__ void k_bc_reset(int n_vertices, int n_free, const int* __restrict__ held, const real3* __restrict__ x_anchor,
                           real3* __restrict__ x_hat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    const bool driven = !(i < n_free && !(held != nullptr && held[i] != 0));
    if (driven) x_hat[i] = x_anchor[i];
}

__global__ void k_inertia_gradient(int n_free, const real* __restrict__ mass, const real3* __restrict__ x_hat,
                                   const real3* __restrict__ x_tilde, real3* __restrict__ G) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_free) return;
    G[i] = mass[i] * (x_hat[i] - x_tilde[i]);
}

__global__ void k_inertia_energy(int n_free, const real* __restrict__ mass, const real3* __restrict__ x_hat,
                                 const real3* __restrict__ x_tilde, real* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_free) return;
    const real3 d = x_hat[i] - x_tilde[i];
    out[i] = real(0.5) * mass[i] * dot(d, d);
}

__global__ void k_axpy_free(int n_free, real r, const real3* __restrict__ x0, const real3* __restrict__ p,
                            real3* __restrict__ x_hat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_free) return;
    x_hat[i] = x0[i] + r * p[i];
}

__global__ void k_mask_held(int n, const int* __restrict__ held, real3* __restrict__ g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (held[i] != 0) g[i] = make_real3(real(0), real(0), real(0));
}

// MS §6 line 6.6: x += α (x̂ - x) for every vertex (prescribed included, MS §12).
__global__ void k_blend(int n, real alpha, const real3* __restrict__ x_hat, real3* __restrict__ x_anchor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x_anchor[i] = x_anchor[i] + alpha * (x_hat[i] - x_anchor[i]);
}

__global__ void k_finish(int n, real inv_h, const real3* __restrict__ x_anchor, real3* __restrict__ x_prev,
                         real3* __restrict__ vel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real3 x = x_anchor[i];
    vel[i] = (x - x_prev[i]) * inv_h;
    x_prev[i] = x;
}

__global__ void k_negate(int n, real3* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = -p[i];
}

__global__ void k_displacement(int n, const real3* __restrict__ a, const real3* __restrict__ b, real3* __restrict__ d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = b[i] - a[i];
}

// A = BSR (mass + elasticity) + the matrix-free contact/friction terms (§6.5) + the
// Gauss-Newton bending of cloth (MS §3.7), which rides the same rank-one path.
struct ContactAugmentedOperator final : LinearOperator {
    const BsrMatrix* A = nullptr;
    ContactSystem* contact = nullptr;   // may be null (elastic-only path)
    BendingSystem* bending = nullptr;   // may be null (no cloth with k_bend > 0)
    real h2 = real(0);
    ContactAugmentedOperator(const BsrMatrix& m, ContactSystem* c, BendingSystem* b, real h2_)
        : A(&m), contact(c), bending(b), h2(h2_) {}
    void apply(const real3* x, real3* y, cudaStream_t stream, const int* skip) const override {
        bsr_spmv(*A, x, y, stream, skip);
        if (contact) contact->spmv_add(x, y, stream, skip);
        if (bending) bending->spmv_add(h2, x, y, stream, skip);
    }
    int rows() const override { return A->n_rows(); }
    bool fused_views(FusedViews& v) const override {
        v = FusedViews{};
        v.n_rows = A->n_rows();
        v.row_ptr = A->pattern->row_ptr.data();
        v.col_idx = A->pattern->col_idx.data();
        v.blocks = A->blocks.data();
        if (contact) { v.has_contact = true; v.contact = contact->spmv_view(); }
        if (bending) { v.has_bending = true; v.bending = bending->spmv_view(h2); }
        return true;
    }
};

}  // namespace

// ---------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------
struct Stepper::Impl {
    Scene scene;
    SceneBuffers sb;
    BsrPattern pattern;
    BsrMatrix A;
    ElementCache cache;
    Pcg pcg;
    RigidCoarse coarse;   // IS §6.6 two-level preconditioner (linear_solver.rigid_coarse)
    // contact.inversion_free: "on" guards every body, "auto" NH bodies only, "off" nothing.
    // Mirrors Scene::body_needs_inversion_guard, which is a pure function of this mode.
    bool guard_all_bodies = false;
    std::unique_ptr<ContactSystem> contact;
    // Sphere colliders (MS §12.3): radius at the step's target, and the ANCHOR radius, which
    // is blended toward the target with every accepted alpha exactly as k_blend moves the
    // vertices, so each CCD interpolates from where the wall actually is.
    std::vector<real> sphere_rt, sphere_ra;
    std::unique_ptr<BendingSystem> bending;   // cloth bending (MS §3.7), null when there is none
    DeviceArray<real> extra_diag;             // contact + bending diagonal blocks for the preconditioner

    DeviceArray<real3> G, p, dx;
    DeviceArray<real> scratch_r;
    DeviceArray<double> scratch_d;
    static constexpr int kBatchMaxRows = 100000;   // batched line-search trials below this many free vertices

    Eigen::Matrix3Xd target_host;
    std::vector<real3> target_r;
    std::vector<int> held_host;       // IS §5.x held free vertices of the current step
    const int* held_ptr = nullptr;

    int frame = 0;
    real h = 0, h2 = 0;
    real mu = 0;                 // AL penalty stiffness of the current step
    real d_hat = 0;              // current offset (the stall rule shrinks it)
    int stall_counter = 0;
    int stall_adaptations = 0;
    StepStats stats;

    explicit Impl(const Scene& s) : scene(s) {
        h = real(scene.desc.simulation.dt);
        h2 = h * h;
        d_hat = real(scene.desc.contact.d_hat);
        sb.upload(scene);
        pattern = BsrPattern::build(scene);
        A.init(pattern);
        e_parts.resize(size_t(kParts) * kSlots);
        e_parts.zero();
        h_parts.assign(size_t(kParts) * kSlots, 0.0);
        cache.resize(scene.n_tets);
        pcg.resize(scene.n_free);
        if (scene.desc.linear_solver.rigid_coarse && scene.n_free > 0) {
            // free rows are numbered body by body (scene.cpp), so each body's rows are one range
            std::vector<int> begin, count;
            int i = 0;
            while (i < scene.n_free) {
                int j = i;
                while (j < scene.n_free && scene.body_id[size_t(j)] == scene.body_id[size_t(i)]) ++j;
                begin.push_back(i);
                count.push_back(j - i);
                i = j;
            }
            coarse.setup(begin, count, scene.n_free, scene.desc.linear_solver.rigid_coarse_max_rows);
            log().info("Stepper: rigid-mode coarse correction on {} of {} bodies", coarse.bodies(), begin.size());
        }
        guard_all_bodies = (scene.desc.contact.inversion_free == "on");
        G.resize(scene.n_free);
        p.resize(scene.n_free);
        dx.resize(scene.n_vertices);
        scratch_r.resize(std::max({scene.n_free, scene.n_tets, 1}) + 1);
        // Prescribed vertices that no region drives simply hold their initial position, so the
        // target buffer starts as x0 and evaluate_targets only overwrites the driven columns.
        target_host = scene.x0;
        target_r.resize(scene.n_vertices);
        const bool want_contact = scene.desc.contact.enable &&
                                  (!scene.surf_tris.empty() || !scene.surf_edges.empty() || !scene.planes.empty() ||
                                   !scene.spheres.empty());
        if (want_contact) contact.reset(new ContactSystem(sb, scene, scene.desc.contact));
        {
            std::unique_ptr<BendingSystem> b(new BendingSystem(scene));
            if (!b->empty()) bending = std::move(b);
        }
        extra_diag.resize(9 * size_t(std::max(scene.n_free, 1)));
        log().info("Stepper: {} vertices ({} free), {} tets, {} BSR blocks, contact {}", scene.n_vertices,
                   scene.n_free, scene.n_tets, pattern.n_blocks, want_contact ? "on" : "off");
    }

    // ---- energies (MS §5.5: the line-search model with the slack frozen) ----------------
    // The model energy splits into a part that depends on the positions alone (inertia,
    // elasticity, bending) and the contact part (contact and friction), which also depends on
    // the anchor, the slack and the multipliers. The positional part of the last evaluated
    // line-search trial is the positional part at the current x_hat until the step ends, so it
    // is kept and reused as the next Newton step's start energy (IS §12.3 item 12): one fewer
    // elastic-energy pass, which for the corotated model is an SVD per tet, per Newton step.
    // Exact: the same kernels at the same positions give the same bits.
    double e_pos_cache = 0.0;
    bool e_pos_valid = false;

    // Energy parts on the device (IS §12.3 item 10): five doubles per slot -- inertia, elastic
    // (unscaled), bending (unscaled), contact pairs, friction -- written by the components'
    // device-output reductions and read back with one copy per evaluation (the M1 path did one
    // synchronising readback per component). The host combines them in the order and with the
    // scaling the M1 code used, so every energy is bitwise the same as before.
    static constexpr int kParts = 5;
    static constexpr int kSlots = 3;
    DeviceArray<double> e_parts;
    std::vector<double> h_parts;

    void launch_positional(const real3* x, int slot) {
        double* base = e_parts.data() + kParts * slot;
        const int n = sb.n_free;
        if (n) {
            k_inertia_energy<<<grid_for(n), kBlock>>>(n, sb.mass.data(), x, sb.x_tilde.data(), scratch_r.data());
            CS_CUDA_KERNEL_CHECK();
        }
        reduce_sum_to(scratch_r.data(), n, scratch_d, base + 0);
        fem_elastic_energy_to(sb, x, scratch_r, base + 1);
        if (bending) bending->energy_to(x, base + 2);
    }
    void launch_contact(const real3* x, int slot) {
        double* base = e_parts.data() + kParts * slot;
        if (contact) contact->energy_to(x, sb.x_anchor.data(), base + 3, base + 4);
    }
    void read_parts(int slots) {
        CS_CUDA_CHECK(cudaMemcpy(h_parts.data(), e_parts.data(), sizeof(double) * kParts * slots, cudaMemcpyDeviceToHost));
    }
    double combine_positional(int slot) const {
        const double* h = h_parts.data() + kParts * slot;
        double e = h[0];
        e += double(h2) * h[1];
        if (bending) e += double(h2) * h[2];
        return e;
    }
    double combine_contact(int slot) const {
        const double* h = h_parts.data() + kParts * slot;
        double e = 0.0;
        if (contact) { e += h[3]; e += h[4]; }
        return e;
    }
    // Both parts at x, no cache (the public incremental potential).
    double model_energy(const real3* x) {
        launch_positional(x, 0);
        launch_contact(x, 0);
        read_parts(1);
        return combine_positional(0) + combine_contact(0);
    }
    // The model energy at the current x_hat, the positional part from the cache when the line
    // search left it valid; refreshes the cache otherwise.
    double model_energy_at_x_hat() {
        if (!e_pos_valid) launch_positional(sb.x_hat.data(), 0);
        launch_contact(sb.x_hat.data(), 0);
        read_parts(1);
        if (!e_pos_valid) {
            e_pos_cache = combine_positional(0);
            e_pos_valid = true;
        }
        return e_pos_cache + combine_contact(0);
    }

    void gradient(const real3* x_hat) {
        const int n = sb.n_free;
        if (!n) return;
        k_inertia_gradient<<<grid_for(n), kBlock>>>(n, sb.mass.data(), x_hat, sb.x_tilde.data(), G.data());
        CS_CUDA_KERNEL_CHECK();
        fem_add_gradient(sb, cache, h2, G.data());
        if (bending) bending->add_gradient(h2, G.data());
        if (contact) contact->add_gradient(x_hat, sb.x_anchor.data(), G.data());
    }

    void assemble_matrix() {
        A.zero();
        bsr_add_scaled_identity(A, sb.mass.data());
        fem_assemble_bsr(sb, cache, h2, A);
        // Contact and bending blocks stay matrix-free; only their diagonals enter the
        // preconditioner (§6.5, MS §3.7).
        const real* extra = contact ? contact->diagonal_blocks() : nullptr;
        if (bending) {
            const size_t bytes = 9 * size_t(sb.n_free) * sizeof(real);
            if (extra) CS_CUDA_CHECK(cudaMemcpy(extra_diag.data(), extra, bytes, cudaMemcpyDeviceToDevice));
            else CS_CUDA_CHECK(cudaMemset(extra_diag.data(), 0, bytes));
            bending->add_diagonal_blocks(h2, extra_diag.data());
            extra = extra_diag.data();
        }
        bsr_build_block_jacobi(A, extra);
    }

    // ---- one Newton step (MS §7); returns the accepted line-search factor -----------------
    real newton_step(int& pcg_iters, int& halvings) {
        const int n = sb.n_free;
        if (contact) {
            // MS §5.5: freeze the slack (and the friction Hessian) for this Newton step,
            // including its line search.
            ScopedTimer t("contact");
            contact->update_slack(sb.x_hat.data(), sb.x_anchor.data());
            contact->update_hessian(sb.x_hat.data());
        }
        {
            ScopedTimer t("elem_cache");
            fem_element_cache(sb, sb.x_hat.data(), cache, /*need_hessian=*/true);
            if (bending) bending->cache(sb.x_hat.data());
        }
        {
            ScopedTimer t("gradient");
            gradient(sb.x_hat.data());
            if (held_ptr != nullptr) {   // IS §5.x: held rows carry no residual
                k_mask_held<<<grid_for(n), kBlock>>>(n, held_ptr, G.data());
                CS_CUDA_KERNEL_CHECK();
            }
        }
        {
            ScopedTimer t("assemble");
            assemble_matrix();
        }
        PcgOptions opt;
        opt.rel_tol = real(scene.desc.linear_solver.rel_tol);
        opt.max_iters = scene.desc.linear_solver.max_iters;
        opt.check_interval = scene.desc.linear_solver.check_interval;
        opt.use_graph = scene.desc.linear_solver.graph;
        opt.use_fused = scene.desc.linear_solver.fused;
        opt.held = held_ptr;
        opt.fused_max_rows = scene.desc.linear_solver.fused_max_rows;
        // The spectrum estimate is a diagnostic for debugging conditioning; it rides on the
        // debug log level so that production runs pay nothing for it.
        opt.estimate_spectrum = log().should_log(spdlog::level::debug);
        PcgStats ps;
        {
            ScopedTimer t("pcg");
            if (coarse.active()) {
                // body-diagonal part exactly (BSR + bending never span bodies), contact by its diagonal blocks
                ContactAugmentedOperator op_nc(A, nullptr, bending.get(), h2);
                coarse.build(op_nc, contact ? contact->diagonal_blocks() : nullptr, sb.x_hat.data(), held_ptr);
                opt.coarse = &coarse;
            }
            ContactAugmentedOperator op(A, contact.get(), bending.get(), h2);
            ps = pcg.solve(op, A.diag_inv.data(), G.data(), p.data(), opt);
        }
        pcg_iters += ps.iterations;
        log().debug("  pcg n={} iters={} conv={} |r0|={:.4e} |r|={:.4e}  M^-1A spectrum ~[{:.3e}, {:.3e}] kappa>={:.3e}",
                    n, ps.iterations, ps.converged, ps.initial_residual, ps.final_residual, ps.lambda_min,
                    ps.lambda_max, ps.kappa);
        if (!ps.converged) {
            // A truncated CG iterate is still a descent direction (Galerkin: p^T G = -p^T H p),
            // so a cap hit is inexactness, not failure; report the preconditioned ratio, which
            // is the forcing term actually achieved, alongside the Euclidean one. A breakdown
            // that produced NO iterate is different: p = 0 makes the line search accept r = 1
            // with E == E0, the CCD of a zero motion returns alpha = 1, and the step silently
            // ends with x^{t+1} = x^t. That must not pass as a step.
            if (ps.iterations == 0) {
                throw std::runtime_error(fmt::format(
                    "frame {}: PCG produced no iterate (non-finite right-hand side or non-SPD operator); "
                    "the step cannot proceed", frame));
            }
            log().warn("frame {}: PCG hit the cap ({} iters): preconditioned ratio {:.3e} (criterion {:.1e}), "
                       "Euclidean ||r||/||b|| {:.3e}",
                       frame, ps.iterations, ps.preconditioned_ratio, std::sqrt(double(opt.rel_tol)),
                       ps.final_residual / std::max(ps.initial_residual, 1e-300));
        }
        k_negate<<<grid_for(n), kBlock>>>(n, p.data());   // p = -H^{-1} G
        CS_CUDA_KERNEL_CHECK();

        ScopedTimer t("line_search");
        sb.x_ls0.copy_from(sb.x_hat);
        // x_ls0 == x_hat: the positional part is the previous trial's when one was evaluated
        const double E0 = model_energy_at_x_hat();
        const double tol = scene.desc.newton.line_search.energy_tolerance;
        const auto accepts = [&](double E) { return E <= E0 + tol * std::max(1.0, std::abs(E0)); };
        // MS §10.4: for non-invertible materials the Newton step itself must stay inversion free.
        real r = real(1);
        if (scene.desc.contact.inversion_free != "off") {
            const real t_inv = fem_inversion_toi(sb, sb.x_ls0.data(), p.data(), scratch_r, guard_all_bodies);
            r = std::min(r, t_inv);
        }
        int it = 0;
        // IS §12.3 item 10: on a small system the three trials r, r/2, r/4 are evaluated back to
        // back and read with one copy; the extra kernels cost less than the two readbacks they
        // replace there, and would cost more than that on a large one.
        const int max_halvings = scene.desc.newton.line_search.max_halvings;
        const bool batched = scene.desc.newton.line_search.batched && n <= kBatchMaxRows && max_halvings >= 2;
        if (batched) {
            real rk[kSlots];
            for (int k = 0; k < kSlots; ++k) {
                rk[k] = r * real(k == 0 ? 1.0 : k == 1 ? 0.5 : 0.25);
                k_axpy_free<<<grid_for(n), kBlock>>>(n, rk[k], sb.x_ls0.data(), p.data(), sb.x_hat.data());
                CS_CUDA_KERNEL_CHECK();
                launch_positional(sb.x_hat.data(), k);
                launch_contact(sb.x_hat.data(), k);
            }
            read_parts(kSlots);
            for (int k = 0; k < kSlots; ++k) {
                const double pos = combine_positional(k);
                if (accepts(pos + combine_contact(k))) {
                    if (k != kSlots - 1) {   // x_hat holds the last trial: restore trial k (same bits)
                        k_axpy_free<<<grid_for(n), kBlock>>>(n, rk[k], sb.x_ls0.data(), p.data(), sb.x_hat.data());
                        CS_CUDA_KERNEL_CHECK();
                    }
                    e_pos_cache = pos;
                    e_pos_valid = true;
                    halvings += k;
                    return rk[k];
                }
            }
            // none accepted: x_hat holds the last trial and so must the cache
            e_pos_cache = combine_positional(kSlots - 1);
            e_pos_valid = true;
            r = rk[kSlots - 1] * real(0.5);
            halvings += kSlots;
            it = kSlots;
        }
        for (; it <= max_halvings; ++it) {
            k_axpy_free<<<grid_for(n), kBlock>>>(n, r, sb.x_ls0.data(), p.data(), sb.x_hat.data());
            CS_CUDA_KERNEL_CHECK();
            // the slack stays frozen at its start-of-step value (MS §5.5)
            launch_positional(sb.x_hat.data(), 0);
            launch_contact(sb.x_hat.data(), 0);
            read_parts(1);
            e_pos_cache = combine_positional(0);
            e_pos_valid = true;   // x_hat now holds this trial, accepted or not
            if (accepts(e_pos_cache + combine_contact(0))) return r;
            r *= real(0.5);
            ++halvings;
        }
        log().warn("frame {}: line search exhausted; keeping r = {:.3e}", frame, double(r));
        return r;
    }

    // MS §7: Newton steps until a full step is accepted (or the inner cap).
    // MS §10.4 pull-back: the next subproblem solve is one guarded Newton step from the anchor.
    bool single_newton_next = false;
    // MS §6 guarded Dirichlet advance: the iterate's prescribed vertices are short of their targets.
    bool bc_pending = false;
    bool bc_started_this_iter = false;   // the joint start already advanced the boundary: solve before advancing again

    // Joint-start weights (MS §6): 1 on prescribed vertices, and on a free vertex a smooth falloff of its REST
    // distance to the nearest prescribed vertex of the same body (1 at the region, 0 beyond 5 % of the body's
    // bounding-box diagonal). Built once, on first use. Held free regions are not considered (weight by distance
    // to prescribed vertices only); bodies without prescribed vertices get 0.
    DeviceArray<real> joint_w;
    bool joint_w_ready = false;
    void ensure_joint_weights() {
        if (joint_w_ready) return;
        const int n = scene.n_vertices, nf = scene.n_free;
        std::vector<real> w(static_cast<size_t>(n), real(0));
        for (int i = nf; i < n; ++i) w[size_t(i)] = real(1);
        const int nb = int(scene.desc.bodies.size());
        std::vector<std::vector<int>> driven(static_cast<size_t>(nb)), free_v(static_cast<size_t>(nb));
        std::vector<Eigen::Vector3d> lo(size_t(nb), Eigen::Vector3d::Constant(1e300)), hi(size_t(nb), Eigen::Vector3d::Constant(-1e300));
        for (int i = 0; i < scene.n_body_vertices; ++i) {
            const int b = scene.body_id[size_t(i)];
            if (b < 0 || b >= nb) continue;
            (i < nf ? free_v : driven)[size_t(b)].push_back(i);
            lo[size_t(b)] = lo[size_t(b)].cwiseMin(scene.x0.col(i));
            hi[size_t(b)] = hi[size_t(b)].cwiseMax(scene.x0.col(i));
        }
        for (int b = 0; b < nb; ++b) {
            const auto& D = driven[size_t(b)];
            const auto& Fv = free_v[size_t(b)];
            if (D.empty() || Fv.empty()) continue;
            const double rho = 0.05 * (hi[size_t(b)] - lo[size_t(b)]).norm();
            if (double(D.size()) * double(Fv.size()) > 4e8) {   // too many pairs for the brute-force search: plain joint start
                for (int i : Fv) w[size_t(i)] = real(1);
                continue;
            }
            for (int i : Fv) {
                // cheap reject by bounding box of the driven set is not worth it at these sizes
                double best = 1e300;
                const Eigen::Vector3d p = scene.x0.col(i);
                for (int j : D) best = std::min(best, (scene.x0.col(j) - p).squaredNorm());
                const double t = std::clamp(1.0 - std::sqrt(best) / rho, 0.0, 1.0);
                w[size_t(i)] = real(t * t * (3.0 - 2.0 * t));   // smoothstep
            }
        }
        joint_w.upload(w);
        joint_w_ready = true;
    }

    // Moves the iterate's driven vertices toward their targets as far as no guarded element inverts
    // (free vertices fixed). Returns true when they are on the targets afterwards.
    bool advance_prescribed() {
        const int n = sb.n_vertices;
        k_bc_displacement<<<grid_for(n), kBlock>>>(n, sb.n_free, held_ptr, sb.x_target.data(), sb.x_hat.data(), dx.data());
        CS_CUDA_KERNEL_CHECK();
        // one advance may at most halve any guarded element's volume (0.9 of the way to inversion outran the
        // relaxation of the free vertices and collapsed the elements geometrically)
        const real t = fem_volume_loss_toi(sb, sb.x_hat.data(), dx.data(), 0.5, scratch_r, guard_all_bodies, sb.x_prev.data());
        k_bc_advance<<<grid_for(n), kBlock>>>(n, sb.n_free, held_ptr, t, sb.x_target.data(), dx.data(), sb.x_hat.data());
        CS_CUDA_KERNEL_CHECK();
        e_pos_valid = false;
        log().debug("  guarded Dirichlet advance: sigma {:.4e}", double(t));
        return t >= real(1);
    }

    void solve_subproblem() {
        const double vtol = scene.desc.newton.increment_velocity_tol;
        const bool single = single_newton_next;
        for (int it = 0; it < scene.desc.newton.inner_max_iters; ++it) {
            int pcg_iters = 0, halvings = 0;
            const real r = newton_step(pcg_iters, halvings);
            stats.inner_newton += 1;
            stats.pcg_iters_total += pcg_iters;
            stats.line_search_halvings += halvings;
            if (log().should_log(spdlog::level::debug) && sb.n_free > 0) {
                const double v = std::sqrt(reduce_sum_sq(p.data(), sb.n_free, scratch_d) / double(sb.n_free)) / double(h);
                log().debug("    newton {}: line-search factor {:.3g}, RMS increment velocity {:.4e} m/s (tol {:.3e}), CG {}", it,
                            double(r), double(r) * v, vtol, pcg_iters);
            }
            if (single) break;   // the chord anchor -> x_hat must be this one step (inversion free by its line search)
            // MS §7: the loop repeats only when the ENERGY line search had to halve its first trial. A
            // first trial shortened by the NH inversion bound counts as accepted (it could never be 1
            // while the Newton step inverts an element, and the repeats did not converge), unless the
            // bound left less than a tenth of the step.
            const bool accepted = (halvings == 0) && double(r) >= 0.1;
            if (!accepted) continue;
            if (vtol <= 0.0 || sb.n_free == 0) break;
            // MS §7 optional rule (C-IPC's PNTol): the accepted increment is r p; keep iterating
            // while its RMS velocity over the free vertices is above the tolerance.
            const double rms_v =
                double(r) * std::sqrt(reduce_sum_sq(p.data(), sb.n_free, scratch_d) / double(sb.n_free)) / double(h);
            if (rms_v < vtol) break;
        }
    }

    // ---- penalty stiffness (MS §11.1) -----------------------------------------------------
    real estimate_mu() {
        const auto& cd = scene.desc.contact;
        if (cd.mu_mode == "fixed" && cd.mu_fixed) return real(*cd.mu_fixed);
        fem_element_cache(sb, sb.x_prev.data(), cache, /*need_hessian=*/true);
        const real dmax = fem_max_diagonal(sb, cache, h2, scratch_r);
        real m = real(cd.mu_scale) * dmax;
        // MS §11.1 upper bound: one nearly flat NH element must not set the whole scene's penalty
        if (cd.mu_max && double(m) > *cd.mu_max) m = real(*cd.mu_max);
        return m;
    }

    // ---- the step (MS §6) -------------------------------------------------------------------
    void step() {
        const auto t_start = std::chrono::steady_clock::now();
        TimerRegistry::reset();
        stats = StepStats{};
        stats.frame = frame;

        const int n = sb.n_vertices, nf = sb.n_free;
        const auto& cd = scene.desc.contact;
        const double vtol = scene.desc.newton.increment_velocity_tol;   // MS §7 / §15.4

        // 1-3. targets, prediction, μ
        scene.evaluate_targets(frame, target_host);
        if (!scene.spheres.empty()) {
            // MS §12.3: the sphere's radius is prescribed like a vertex target -- r0 at the
            // anchor, r1 at the end of the step; constraints see r1, the CCD interpolates.
            const double t0 = double(frame) * double(h), t1 = t0 + double(h);
            sphere_ra.resize(scene.spheres.size());
            sphere_rt.resize(scene.spheres.size());
            for (size_t q = 0; q < scene.spheres.size(); ++q) {
                sphere_ra[q] = real(scene.sphere_radius(int(q), t0));
                sphere_rt[q] = real(scene.sphere_radius(int(q), t1));
            }
            sb.set_sphere_radii(sphere_ra, sphere_rt);
        }
        for (int i = 0; i < n; ++i) target_r[i] = to_real3(target_host.col(i));
        sb.upload_targets(target_r);
        // IS §5.x held free regions: the mask of this step (empty pointer when the scene has none)
        held_ptr = nullptr;
        if (scene.has_held_regions()) {
            scene.held_mask(frame, held_host);
            sb.upload_held(held_host);
            held_ptr = sb.held.data();
        }
        k_predict<<<grid_for(n), kBlock>>>(n, nf, h, sb.gravity, sb.x_prev.data(), sb.v.data(), sb.x_target.data(),
                                           held_ptr, sb.x_tilde.data(), sb.x_hat.data(), sb.x_anchor.data());
        CS_CUDA_KERNEL_CHECK();
        e_pos_valid = false;   // x_hat re-initialised for the step
        mu = estimate_mu();
        stats.mu = double(mu);
        stall_counter = 0;
        single_newton_next = false;
        // MS §6 guarded Dirichlet advance: if the whole prescribed increment would invert a guarded
        // element with the free vertices where they are, start the iterate's driven vertices at the
        // anchor and apply the increment over the outer iterations. Otherwise nothing changes.
        bc_pending = false;
        bc_started_this_iter = false;
        if (scene.desc.contact.inversion_free != "off" && (n > nf || held_ptr != nullptr) && sb.n_tets > 0) {
            k_displacement<<<grid_for(n), kBlock>>>(n, sb.x_anchor.data(), sb.x_hat.data(), dx.data());   // = target - x^t on driven, 0 on free
            CS_CUDA_KERNEL_CHECK();
            if (fem_volume_loss_toi(sb, sb.x_anchor.data(), dx.data(), 0.5, scratch_r, guard_all_bodies) < real(1)) {
                // First try the joint start: the free vertices at their inertial prediction x_tilde, the driven
                // ones on their targets. In a steady scripted motion the material next to the driven region
                // already moves with it, so this displacement is nearly rigid there and harmless to the
                // elements; the iterate then satisfies the boundary condition from the start.
                ensure_joint_weights();
                k_joint_displacement<<<grid_for(n), kBlock>>>(n, joint_w.data(), sb.x_anchor.data(), sb.x_tilde.data(), dx.data());
                CS_CUDA_KERNEL_CHECK();
                const real sj = fem_volume_loss_toi(sb, sb.x_anchor.data(), dx.data(), 0.5, scratch_r, guard_all_bodies);
                k_joint_start<<<grid_for(n), kBlock>>>(n, sj, sb.x_anchor.data(), dx.data(), sb.x_hat.data());
                CS_CUDA_KERNEL_CHECK();
                bc_pending = sj < real(1);
                bc_started_this_iter = bc_pending;
                log().debug("  guarded Dirichlet start: joint fraction {:.4e}", double(sj));
            }
        }

        // 4. contact: trees from the first (largest) sweep, friction snapshot from C^t
        if (contact) {
            contact->set_d_hat(d_hat);
            contact->begin_step(sb.x_anchor.data(), sb.x_hat.data(), mu);
        }

        // 5. outer loop
        double beta = 1.0;
        int k = 0;
        double alpha_sum = 0.0;
        while (true) {
            if (contact) contact->linearize(sb.x_anchor.data());   // MS §6 line 6.1
            bool bc_partial = false;
            if (bc_pending) {                                       // MS §6 guarded Dirichlet advance
                if (bc_started_this_iter) bc_started_this_iter = false;   // one advance per solve
                else bc_pending = !advance_prescribed();
                bc_partial = bc_pending;
                if (bc_partial) ++stats.bc_partial_iters;
            }
            solve_subproblem();                                     // §6.2
            if (contact) contact->update_multipliers(sb.x_hat.data(), sb.x_anchor.data());  // §6.3

            double alpha = 1.0;
            if (contact) {
                // §6.4-6.5: one CCD from the anchor to x̂ gives α and the new candidates.
                const CcdResult r = contact->ccd_and_update(sb.x_anchor.data(), sb.x_hat.data(), 1.0);
                alpha = r.alpha;
                if (scene.desc.contact.inversion_free != "off" && alpha > 0.0) {
                    // MS §10.4, end-state rule: the blended STATE must be free of (nearly) inverted
                    // elements, its straight path need not be. Take the largest of eight fractions of
                    // alpha_ccd whose end state keeps every guarded element above eta_J of the smaller
                    // of its two end values; the first-root rule is only the last resort (it walks the
                    // anchor into the singular configuration when the path crosses J = 0).
                    k_displacement<<<grid_for(n), kBlock>>>(n, sb.x_anchor.data(), sb.x_hat.data(), dx.data());
                    CS_CUDA_KERNEL_CHECK();
                    constexpr double kEtaJ = 0.1;
                    const double alpha_ccd = alpha;
                    bool found = false;
                    const char* how = "end state ok";
                    if (single_newton_next) {
                        // the chord is one guarded Newton step from the anchor: valid all along by construction
                        found = fem_inversion_margin(sb, sb.x_anchor.data(), dx.data(), alpha_ccd, scratch_r, guard_all_bodies) > 0.0;
                        if (!found) {   // numerical last resort
                            alpha = std::min(alpha_ccd, double(fem_inversion_toi(sb, sb.x_anchor.data(), dx.data(), scratch_r, guard_all_bodies)));
                            found = true;
                            how = "first root (last resort)";
                        }
                    }
                    for (int j = 0; j < 8 && !found; ++j) {
                        const double th = alpha_ccd * (1.0 - double(j) / 8.0);
                        if (fem_inversion_margin(sb, sb.x_anchor.data(), dx.data(), th, scratch_r, guard_all_bodies) >= kEtaJ) {
                            alpha = th;
                            found = true;
                        }
                    }
                    single_newton_next = false;
                    if (!found) {
                        // MS §10.4 pull-back: the valid states beyond the inverted interval cannot be reached on
                        // this chord. Do not creep toward its near end: keep the anchor, reset the iterate to it
                        // and let the next solve be a single guarded Newton step from there.
                        alpha = 0.0;
                        sb.x_hat.copy_from(sb.x_anchor);
                        e_pos_valid = false;
                        single_newton_next = true;
                        bc_pending = n > nf || held_ptr != nullptr;   // the copy took the driven vertices off their targets: re-arm the advance
                        ++stats.inversion_pullbacks;
                        how = "pull-back";
                    }
                    if (alpha < alpha_ccd) ++stats.inversion_limited;
                    log().debug("  outer {}: alpha {:.4e} (CCD {:.4e}, {}), CCD hits {}, |C| {}", k, alpha, alpha_ccd, how, r.hits,
                                r.n_pt + r.n_ee + r.n_ph);
                } else {
                    log().debug("  outer {}: alpha {:.4e}, CCD hits {}, |C| {}", k, alpha, r.hits, r.n_pt + r.n_ee + r.n_ph);
                }
                stats.candidates += r.candidates;
                stats.hits += r.hits;
                stats.prescribed_hits += r.prescribed_hits;
                stats.ccd_filtered += r.filtered_passes;
                stats.kept += r.kept;
                stats.removed += r.removed;
                stats.pairs_pt = r.n_pt;
                stats.pairs_ee = r.n_ee;
                stats.pairs_ph = r.n_ph;
            }

            if (vtol > 0.0) {   // the accepted step of this outer iteration (before the blend), MS §15.4
                k_displacement<<<grid_for(n), kBlock>>>(n, sb.x_anchor.data(), sb.x_hat.data(), dx.data());
                CS_CUDA_KERNEL_CHECK();
            }
            // §6.6: advance the anchor only when the step is meaningful.
            if (alpha > cd.alpha_lower_bound) {
                k_blend<<<grid_for(n), kBlock>>>(n, real(alpha), sb.x_hat.data(), sb.x_anchor.data());
                CS_CUDA_KERNEL_CHECK();
                if (!sphere_ra.empty()) {
                    // MS §12.3: the wall's anchor radius advances with the same alpha.
                    for (size_t q = 0; q < sphere_ra.size(); ++q)
                        sphere_ra[q] += real(alpha) * (sphere_rt[q] - sphere_ra[q]);
                    sb.set_sphere_radii(sphere_ra, sphere_rt);
                }
                stall_counter = 0;
            } else {
                alpha = 0.0;
                ++stall_counter;
            }
            ++k;
            alpha_sum += alpha;
            stats.alpha_min = std::min(stats.alpha_min, alpha);

            if (k >= cd.K_min) beta *= (1.0 - alpha);            // §6.7 / §15.1
            if (bc_partial || bc_pending) beta = 1.0;             // MS §6: only iterates on the full boundary condition count

            // §6.8 / §11.2: stall adaptation (never triggered in the paper's experiments).
            if (contact && alpha < cd.stall.alpha && stall_counter >= cd.stall.iters &&
                stall_adaptations < cd.stall.max_adaptations) {
                mu *= real(cd.stall.mu_factor);
                d_hat *= real(cd.stall.d_hat_factor);
                contact->set_mu(mu);
                contact->set_d_hat(d_hat);
                stall_counter = 0;
                ++stall_adaptations;
                stats.mu = double(mu);
                log().warn("frame {}: stall adaptation {} -> mu {:.3e}, d_hat {:.3e}", frame, stall_adaptations,
                           double(mu), double(d_hat));
            }

            // MS §15.4 (optional, newton.increment_velocity_tol > 0): a step accepted in full is
            // not converged while the subproblem's solution still moved the free vertices faster
            // than the tolerance; another outer iteration re-linearizes the contact set at the
            // new anchor. Needed where a pair's normal rotates within the step (a card's row
            // sliding past a held neighbour's row); the β rule alone stops after K_min iterations.
            // Bounded to K_min extra iterations per step: a scene that keeps creeping or
            // oscillating (the physical card packets) would otherwise run to max_outer_iters.
            bool moving = false;
            if (vtol > 0.0 && nf > 0 && alpha > cd.alpha_lower_bound && stats.outer_continued < cd.K_min) {
                const double rms_v = std::sqrt(reduce_sum_sq(dx.data(), nf, scratch_d) / double(nf)) / double(h);
                moving = rms_v >= vtol;
                if (moving && beta <= cd.epsilon) ++stats.outer_continued;
            }
            if (beta <= cd.epsilon && !moving) break;
            if (k >= cd.max_outer_iters) {
                stats.hit_iteration_cap = true;
                log().warn("frame {}: outer iteration cap {} reached (beta {:.3e})", frame, cd.max_outer_iters, beta);
                break;
            }
        }
        stats.outer_iters = k;
        if (stats.prescribed_hits)
            log().warn("frame {}: {} CCD hits between prescribed primitives only (a prescribed motion is driven into a "
                       "collider or another prescribed region; such pairs have no free DOF, are not contact candidates "
                       "and pass through; IS §5.x)", frame, stats.prescribed_hits);
        stats.alpha_mean = alpha_sum / std::max(1, k);
        stats.beta_final = beta;
        stats.pcg_iters_avg = stats.inner_newton ? double(stats.pcg_iters_total) / stats.inner_newton : 0.0;
        stats.stall_adaptations = stall_adaptations;
        {
            std::size_t mem_free = 0, mem_total = 0;
            if (cudaMemGetInfo(&mem_free, &mem_total) == cudaSuccess)
                stats.gpu_mem_mb = double(mem_total - mem_free) / (1024.0 * 1024.0);
        }
        // The reported energy is the incremental potential of the step's final state under the
        // final constraint set. The last CCD merged new pairs into C after the last
        // linearization, so the per-pair gap, gradient, coefficient and slack arrays are re-derived
        // at the final anchor first (the same two calls the next outer iteration would open with);
        // without them the energy kernel would read the previous set's arrays against the new
        // pair indices. Also keeps `incremental_potential()` consistent after step().
        if (contact) {
            contact->linearize(sb.x_anchor.data());
            contact->update_slack(sb.x_hat.data(), sb.x_anchor.data());
        }
        stats.energy = model_energy_at_x_hat();

        // 7. close the step (MS §14)
        k_finish<<<grid_for(n), kBlock>>>(n, real(1) / h, sb.x_anchor.data(), sb.x_prev.data(), sb.v.data());
        CS_CUDA_KERNEL_CHECK();

        if (contact && scene.desc.logging.debug_check_penetration) {
            int violations = 0;
            stats.max_penetration = -contact->debug_min_distance(sb.x_prev.data(), violations);
            if (violations) log().error("frame {}: {} pairs violate the minimum separation", frame, violations);
        }
        CS_CUDA_CHECK(cudaDeviceSynchronize());

        for (const auto& [name, e] : TimerRegistry::report()) stats.stage_ms[name] = e.total_ms;
        stats.time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        ++frame;
    }
};

// ---------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------
Stepper::Stepper(const Scene& scene) : impl_(new Impl(scene)) {}
Stepper::~Stepper() = default;

void Stepper::step() { impl_->step(); }
int Stepper::frame() const { return impl_->frame; }
const StepStats& Stepper::last_stats() const { return impl_->stats; }

ContactSystem* Stepper::contact_system() { return impl_->contact.get(); }
const Scene& Stepper::scene() const { return impl_->scene; }

void Stepper::positions(Eigen::Matrix3Xd& X) const {
    std::vector<real3> x;
    impl_->sb.download_positions(x);
    X.resize(3, int(x.size()));
    for (size_t i = 0; i < x.size(); ++i)
        X.col(int(i)) = Eigen::Vector3d(double(x[i].x), double(x[i].y), double(x[i].z));
}

double Stepper::set_state(const Eigen::Matrix3Xd& X, const Eigen::Matrix3Xd& V, int frame) {
    Impl& m = *impl_;
    const int n = m.sb.n_vertices, nf = m.sb.n_free;
    if (frame < 1) throw std::runtime_error("Stepper::set_state: frame must be >= 1");
    if (X.rows() != 3 || X.cols() != n || V.rows() != 3 || V.cols() != n)
        throw std::runtime_error("Stepper::set_state: state must be 3 x " + std::to_string(n) + " (all vertices), got " +
                                 std::to_string(X.cols()) + " positions and " + std::to_string(V.cols()) + " velocities");
    // Prescribed vertices: where the scene's motions put them at the end of step frame-1, and the
    // velocity k_finish would have left there (their displacement over that step).
    Eigen::Matrix3Xd T1 = m.scene.x0, T0 = m.scene.x0;
    m.scene.evaluate_targets(frame - 1, T1);
    if (frame >= 2) m.scene.evaluate_targets(frame - 2, T0);
    std::vector<real3> hx(static_cast<size_t>(n)), hv(static_cast<size_t>(n));
    double worst = 0.0;
    const double inv_h = 1.0 / double(m.h);
    for (int i = 0; i < n; ++i) {
        if (i < nf) {
            hx[size_t(i)] = to_real3(X.col(i));
            hv[size_t(i)] = to_real3(V.col(i));
        } else {
            worst = std::max(worst, (T1.col(i) - X.col(i)).norm());
            hx[size_t(i)] = to_real3(T1.col(i));
            const Eigen::Vector3d vp = (T1.col(i) - T0.col(i)) * inv_h;
            hv[size_t(i)] = to_real3(vp);
        }
    }
    m.sb.x_prev.upload(hx);
    m.sb.x_anchor.upload(hx);
    m.sb.x_hat.upload(hx);
    m.sb.x_tilde.upload(hx);
    m.sb.x_ls0.upload(hx);
    m.sb.v.upload(hv);
    m.frame = frame;
    m.e_pos_valid = false;
    m.stall_counter = 0;
    return worst;
}

void Stepper::velocities(Eigen::Matrix3Xd& V) const {
    std::vector<real3> v;
    impl_->sb.download_velocities(v);
    V.resize(3, int(v.size()));
    for (size_t i = 0; i < v.size(); ++i)
        V.col(int(i)) = Eigen::Vector3d(double(v[i].x), double(v[i].y), double(v[i].z));
}

double Stepper::incremental_potential() const { return impl_->model_energy(impl_->sb.x_hat.data()); }
double Stepper::elastic_energy() const {
    return double(fem_elastic_energy(impl_->sb, impl_->sb.x_hat.data(), impl_->scratch_r));
}

const std::vector<std::string>& step_stats_columns() {
    static const std::vector<std::string> cols = {
        "frame", "outer_iters", "inner_newton", "ls_halvings", "pcg_iters_total", "pcg_iters_avg",
        "pairs_pt", "pairs_ee", "pairs_ph", "candidates", "hits", "kept", "removed",
        "alpha_min", "alpha_mean", "beta_final", "mu", "stall_adaptations", "energy",
        "max_penetration", "time_ms", "ms_elem_cache", "ms_gradient", "ms_assemble", "ms_pcg",
        "ms_line_search", "ms_ccd", "ms_active_set", "ms_contact", "hit_cap", "gpu_mem_mb", "ccd_filtered", "outer_continued",
        "prescribed_hits", "inversion_limited", "inversion_pullbacks", "bc_partial_iters"};
    return cols;
}

std::vector<double> step_stats_row(const StepStats& s) {
    auto ms = [&](const char* k) {
        auto it = s.stage_ms.find(k);
        return it == s.stage_ms.end() ? 0.0 : it->second;
    };
    return {double(s.frame), double(s.outer_iters), double(s.inner_newton), double(s.line_search_halvings),
            double(s.pcg_iters_total), s.pcg_iters_avg, double(s.pairs_pt), double(s.pairs_ee), double(s.pairs_ph),
            double(s.candidates), double(s.hits), double(s.kept), double(s.removed), s.alpha_min, s.alpha_mean,
            s.beta_final, s.mu, double(s.stall_adaptations), s.energy, s.max_penetration, s.time_ms,
            ms("elem_cache"), ms("gradient"), ms("assemble"), ms("pcg"), ms("line_search"), ms("ccd"),
            ms("active_set"), ms("contact"), s.hit_iteration_cap ? 1.0 : 0.0, s.gpu_mem_mb, double(s.ccd_filtered), double(s.outer_continued),
            double(s.prescribed_hits), double(s.inversion_limited), double(s.inversion_pullbacks), double(s.bc_partial_iters)};
}

}  // namespace cs
