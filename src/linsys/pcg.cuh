#pragma once
// Preconditioned conjugate gradient on the coupled 3·n_free vector
// (doc/al-ipc-implementation-spec.md §6.6). The operator is abstract so the contact and
// friction terms can be applied matrix-free on top of the BSR matrix (§6.5).
//
// Two paths. The graph path (default) keeps the whole recurrence on the device: the CG scalars
// live in a device state struct, every kernel returns at once when the state's `done` flag is
// set, and `check_interval` iterations are captured into a CUDA graph that is replayed until
// the flag is set, with one readback per replay (§6.6 as built). The plain path is the M1
// launch loop with host reductions, kept for comparison (`graph: false`).
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include "linsys/bsr.cuh"
#include "linsys/pcg_fused.cuh"
#include "linsys/rigid_coarse.cuh"

#include <cuda_runtime.h>
#include <vector>

namespace cs {

struct LinearOperator {
    virtual ~LinearOperator() = default;
    // y = A x (overwrite), x and y have n_rows entries. Every kernel of the apply must launch on
    // `stream` (the graph path captures it) and must return at once when `skip` is non-null and
    // *skip != 0 (the graph's early exit); no host synchronisation of any kind inside.
    virtual void apply(const real3* x, real3* y, cudaStream_t stream, const int* skip) const = 0;
    virtual int rows() const = 0;
    // The operator's data as POD views for the fused solve (§12.3 item 14); false when the
    // operator has no fused form, in which case the graph path is used.
    virtual bool fused_views(FusedViews&) const { return false; }
};

// The static BSR matrix alone.
struct BsrOperator final : LinearOperator {
    const BsrMatrix* A = nullptr;
    explicit BsrOperator(const BsrMatrix& m) : A(&m) {}
    void apply(const real3* x, real3* y, cudaStream_t stream, const int* skip) const override;
    int rows() const override { return A->n_rows(); }
    bool fused_views(FusedViews& v) const override;
};

// Which residual the relative tolerance applies to.
enum class PcgCriterion {
    // r^T M^-1 r <= rel_tol * r0^T M^-1 r0. This is what [Z25] and [UIPC-AL] use ("CG relative
    // error tolerance"), and it is free: the quantity is already formed every iteration for
    // beta, so no extra reduction or host readback is needed. Default, so that iteration counts
    // are comparable with the paper's.
    PreconditionedResidual = 0,
    // ||r|| <= rel_tol * ||b||. Roughly rel_tol^2 in the preconditioned metric, so it costs
    // about twice as many iterations; kept for tests that want a tightly solved system.
    AbsoluteResidual = 1,
};

struct PcgOptions {
    real rel_tol = real(1e-4);
    int max_iters = 2000;
    // Graph path: iterations per captured replay (one host readback each). Plain path: host
    // reads ||r||^2 every k iterations (AbsoluteResidual only).
    int check_interval = 8;
    PcgCriterion criterion = PcgCriterion::PreconditionedResidual;
    bool use_graph = true;
    // Fused single-kernel solve (§12.3 item 14) for systems of at most fused_max_rows rows whose
    // operator provides views; falls back to the graph path when the grid cannot be co-resident.
    bool use_fused = true;
    int fused_max_rows = 65536;
    // IS §5.x held free vertices: when non-null (n rows), the preconditioned residual z and the
    // residual r are zeroed at rows with held[i] != 0, so the solve is the projected system and
    // the increment there is exactly zero (the right-hand side must be zeroed there too).
    const int* held = nullptr;
    // IS §6.6 two-level preconditioner: when non-null and active, z = D^-1 r + P G^+ P^T r (per-body
    // rigid modes), built by the caller for this system. The fused path is bypassed.
    const RigidCoarse* coarse = nullptr;
    // Diagnostic: estimate the extreme eigenvalues of the preconditioned operator M^-1 A from the
    // CG coefficients (the Lanczos tridiagonal built from alpha_k, beta_k; Saad, Iterative
    // Methods, §6.7.3). Plain path: two host doubles per iteration; graph path: two device
    // histories read once at the end. The Ritz values lie inside the true spectrum, so the
    // reported kappa is a lower bound that tightens with the iteration count.
    bool estimate_spectrum = false;
};

struct PcgStats {
    int iterations = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
    bool converged = false;
    // The solve stopped because the operator was not SPD on the vector it saw (p^T A p <= 0 or
    // r^T M^-1 r <= 0). x is the last iterate.
    bool breakdown = false;
    // sqrt(r^T M^-1 r / r0^T M^-1 r0) at exit: the quantity the PreconditionedResidual criterion
    // actually tests, and the inexact-Newton forcing term eta achieved by this solve. The
    // Euclidean initial/final_residual above are a different norm and differ from it by a
    // geometry-dependent factor, so a cap hit can only be judged by this number.
    double preconditioned_ratio = 0.0;
    bool fused = false;   // solved by the fused cooperative kernel (§12.3 item 14)
    // Filled only with PcgOptions::estimate_spectrum: extreme Ritz values of M^-1 A and their
    // ratio (0 when not estimated or when fewer than two iterations ran).
    double lambda_min = 0.0, lambda_max = 0.0, kappa = 0.0;
};

// Device-resident CG state of the graph path (one struct in device memory, mirrored to pinned
// host memory once per replay).
struct PcgDeviceState {
    double rz, rz0, rz_tol, tol2, bb, pAp, alpha, beta, rr;
    int iter, done, converged, breakdown, use_rz, max_iters;
};

// CG coefficients -> extreme Ritz values of M^-1 A (pcg.cu); shared with the fused path.
void lanczos_extremes_public(const std::vector<double>& alphas, const std::vector<double>& betas, PcgStats& st);

class Pcg {
public:
    Pcg();
    ~Pcg();
    Pcg(const Pcg&) = delete;
    Pcg& operator=(const Pcg&) = delete;

    void resize(int n_rows);
    // Solves A x = b with the block-Jacobi preconditioner diag_inv (9 per row, row-major).
    // x is overwritten (initial guess 0). All reductions are fixed-order double sums, so a
    // given input solves to a bitwise identical x on every run (on either path; the two paths
    // differ from each other in the last bits because their partial sums fold differently).
    PcgStats solve(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x, const PcgOptions& opt);

private:
    // The path selection (fused -> graph -> plain). solve() wraps it with the optional
    // CS_PCG_VERIFY cross-check against the plain loop.
    PcgStats solve_dispatch(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x,
                            const PcgOptions& opt);
    PcgStats solve_plain(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x,
                         const PcgOptions& opt);
    PcgStats solve_graph(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x,
                         const PcgOptions& opt);
    // Returns false (without touching x) when the cooperative grid cannot be resident.
    bool solve_fused(const FusedViews& V, const real* diag_inv, const real3* b, real3* x, const PcgOptions& opt,
                     PcgStats& st);
    void capture_block(const LinearOperator& A, const real* diag_inv, real3* x, int n, int k_iters);
    const RigidCoarse* coarse_ = nullptr;   // of the current solve (captured by the graph block)

    int n_ = 0;
    DeviceArray<real3> r_, z_, p_, Ap_;
    DeviceArray<real3> verify_x_, verify_d_;   // CS_PCG_VERIFY scratch
    DeviceArray<double> scratch_;
    // graph path
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    PcgDeviceState* d_state_ = nullptr;
    PcgDeviceState* h_state_ = nullptr;   // pinned
    DeviceArray<double> part_a_, part_b_, part_c_;  // block partials (the fused kernel needs a third: see pcg_fused.cu)
    DeviceArray<double> alpha_hist_, beta_hist_;
    int part_blocks_ = 0, part_chunk_ = 0;
    const int* held_ = nullptr;   // the held mask of the solve being captured
};

}  // namespace cs
