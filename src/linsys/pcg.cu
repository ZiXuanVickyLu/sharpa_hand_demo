// pcg.cu - block-Jacobi preconditioned conjugate gradient (doc/al-ipc-implementation-spec.md
// §6.6, math spec §7).
//
// Graph path (default): the CG scalars live in a PcgDeviceState on the device; one iteration is
// the operator apply, a block-partial p.Ap, a one-block finisher (alpha, breakdown), the fused
// x/r/z update with block partials of r.z and r.r, a one-block finisher (convergence, beta)
// and the p update. Every kernel returns at once when the state's `done` flag is
// set. `check_interval` iterations are captured into a CUDA graph per solve and replayed until
// the flag is set, with one pinned readback per replay. Plain path: the M1 launch loop with
// host reductions, kept for comparison.
#include "linsys/pcg.cuh"
#include "linsys/reduce.cuh"
#include "linsys/linsys_detail.cuh"
#include "core/cuda_check.h"
#include "core/log.h"
#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cs {

using linsys_detail::block_apply;
using linsys_detail::grid_for;
using linsys_detail::kBlock;

namespace {

// ---- reductions inside the graph: the reduce.cu layout (a pure function of n) ----------------
constexpr int kMaxPartials = 1024;
constexpr int kMinChunk = kBlock * 8;

void partial_layout(int n, int& blocks, int& chunk)
{
    long long b = (static_cast<long long>(n) + kMinChunk - 1) / kMinChunk;
    if (b < 1) b = 1;
    if (b > kMaxPartials) b = kMaxPartials;
    long long c = (static_cast<long long>(n) + b - 1) / b;
    c = (c + kBlock - 1) / kBlock * kBlock;
    b = (static_cast<long long>(n) + c - 1) / c;
    blocks = static_cast<int>(b);
    chunk = static_cast<int>(c);
}

// Fixed shared-memory tree over the block's kBlock values (every thread must call it).
__device__ __forceinline__ double block_fold_sum(double acc, double* sh)
{
    sh[threadIdx.x] = acc;
    __syncthreads();
#pragma unroll
    for (int w = kBlock / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
        __syncthreads();
    }
    const double v = sh[0];
    __syncthreads();   // sh may be reused by the caller
    return v;
}

// One block folds m partials (m <= kMaxPartials) in a fixed order.
__device__ __forceinline__ double fold_partials(const double* __restrict__ partials, int m, double* sh)
{
    double acc = 0.0;
    for (int i = threadIdx.x; i < m; i += kBlock) acc += partials[i];
    return block_fold_sum(acc, sh);
}

__device__ __forceinline__ double dot3(const real3& a, const real3& b)
{
    return static_cast<double>(a.x) * static_cast<double>(b.x) + static_cast<double>(a.y) * static_cast<double>(b.y) +
           static_cast<double>(a.z) * static_cast<double>(b.z);
}

// ---- plain-path kernels ----------------------------------------------------------------------
__device__ __forceinline__ bool is_held(const int* __restrict__ held, int i) { return held != nullptr && held[i] != 0; }

__global__ void k_precondition(int n, const real* __restrict__ D, const int* __restrict__ held, const real3* __restrict__ r,
                               real3* __restrict__ z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = is_held(held, i) ? make_real3(real(0), real(0), real(0)) : block_apply(D + 9 * static_cast<std::size_t>(i), r[i]);
}

// x += alpha p; r -= alpha Ap; z = D r (from the stored r, so z = M^-1 r holds exactly for
// the vector PCG sees in every precision).
__global__ void k_update_xrz(int n, double alpha, const real3* __restrict__ p, const real3* __restrict__ Ap,
                             const real* __restrict__ D, const int* __restrict__ held, real3* __restrict__ x,
                             real3* __restrict__ r, real3* __restrict__ z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (is_held(held, i)) {   // projected row: no motion, no residual
        r[i] = make_real3(real(0), real(0), real(0));
        z[i] = make_real3(real(0), real(0), real(0));
        return;
    }
    const real3 xi = x[i], pi = p[i], ri = r[i], api = Ap[i];
    x[i] = make_real3(static_cast<real>(static_cast<double>(xi.x) + alpha * static_cast<double>(pi.x)),
                      static_cast<real>(static_cast<double>(xi.y) + alpha * static_cast<double>(pi.y)),
                      static_cast<real>(static_cast<double>(xi.z) + alpha * static_cast<double>(pi.z)));
    const real3 rn = make_real3(static_cast<real>(static_cast<double>(ri.x) - alpha * static_cast<double>(api.x)),
                                static_cast<real>(static_cast<double>(ri.y) - alpha * static_cast<double>(api.y)),
                                static_cast<real>(static_cast<double>(ri.z) - alpha * static_cast<double>(api.z)));
    r[i] = rn;
    z[i] = block_apply(D + 9 * static_cast<std::size_t>(i), rn);
}

// p = z + beta p
__global__ void k_update_p(int n, double beta, const real3* __restrict__ z, real3* __restrict__ p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real3 zi = z[i], pi = p[i];
    p[i] = make_real3(static_cast<real>(static_cast<double>(zi.x) + beta * static_cast<double>(pi.x)),
                      static_cast<real>(static_cast<double>(zi.y) + beta * static_cast<double>(pi.y)),
                      static_cast<real>(static_cast<double>(zi.z) + beta * static_cast<double>(pi.z)));
}

// ---- graph-path kernels ----------------------------------------------------------------------
// Block partials of a.b (and, when bb != nullptr, of b.b) in the fixed layout.
__global__ void __launch_bounds__(kBlock) k_partials(const int* __restrict__ done, int n, int chunk,
                                                     const real3* __restrict__ a, const real3* __restrict__ b,
                                                     double* __restrict__ part_ab, double* __restrict__ part_bb)
{
    __shared__ double sh[kBlock];
    if (done != nullptr && *done) return;
    const long long begin = static_cast<long long>(blockIdx.x) * chunk;
    long long end = begin + chunk;
    if (end > n) end = n;
    double ab = 0.0, bb = 0.0;
    for (long long i = begin + threadIdx.x; i < end; i += kBlock) {
        const real3 ai = a[static_cast<int>(i)], bi = b[static_cast<int>(i)];
        ab += dot3(ai, bi);
        if (part_bb != nullptr) bb += dot3(bi, bi);
    }
    const double vab = block_fold_sum(ab, sh);
    if (threadIdx.x == 0) part_ab[blockIdx.x] = vab;
    if (part_bb != nullptr) {
        const double vbb = block_fold_sum(bb, sh);
        if (threadIdx.x == 0) part_bb[blockIdx.x] = vbb;
    }
}

// Prologue: rz0 = r.z, bb = b.b; thresholds, counters and the trivial exits.
__global__ void k_init_state(PcgDeviceState* S, const double* __restrict__ part_rz, const double* __restrict__ part_bb,
                             int m, double rel, int use_rz, int max_iters)
{
    __shared__ double sh[kBlock];
    const double rz0 = fold_partials(part_rz, m, sh);
    const double bb = fold_partials(part_bb, m, sh);
    if (threadIdx.x != 0) return;
    S->rz = S->rz0 = rz0;
    S->bb = bb;
    S->rr = bb;
    S->rz_tol = rel * rz0;
    S->tol2 = rel * rel * bb;
    S->pAp = S->alpha = S->beta = 0.0;
    S->iter = 0;
    S->done = S->converged = S->breakdown = 0;
    S->use_rz = use_rz;
    S->max_iters = max_iters;
    if (!isfinite(bb)) {              // garbage in: x = 0, not converged, no iterate
        S->done = 1; S->breakdown = 1;
    } else if (bb == 0.0) {           // b = 0: x = 0 is the exact solution
        S->done = 1; S->converged = 1;
    } else if (!(rz0 > 0.0) || !isfinite(rz0)) {   // preconditioner not SPD on r
        S->done = 1; S->breakdown = 1;
    }
}

// alpha = rz / pAp, or a breakdown when the operator is not SPD on p.
__global__ void k_finish_pAp(PcgDeviceState* S, const double* __restrict__ partials, int m,
                             double* __restrict__ alpha_hist)
{
    __shared__ double sh[kBlock];
    if (S->done) return;
    const double pAp = fold_partials(partials, m, sh);
    if (threadIdx.x != 0) return;
    S->pAp = pAp;
    if (!(pAp > 0.0) || !isfinite(pAp)) {
        S->done = 1;
        S->breakdown = 1;
        return;
    }
    S->alpha = S->rz / pAp;
    alpha_hist[S->iter] = S->alpha;
}

// x += alpha p; r -= alpha Ap; z = D r; block partials of r.z and r.r.
__global__ void __launch_bounds__(kBlock) k_update_xrz_g(const PcgDeviceState* S, int n, int chunk,
                                                         const real3* __restrict__ p, const real3* __restrict__ Ap,
                                                         const real* __restrict__ D, const int* __restrict__ held,
                                                         real3* __restrict__ x, real3* __restrict__ r,
                                                         real3* __restrict__ z, double* __restrict__ part_rz,
                                                         double* __restrict__ part_rr)
{
    __shared__ double sh[kBlock];
    if (S->done) return;
    const double alpha = S->alpha;
    const long long begin = static_cast<long long>(blockIdx.x) * chunk;
    long long end = begin + chunk;
    if (end > n) end = n;
    double rz = 0.0, rr = 0.0;
    for (long long ii = begin + threadIdx.x; ii < end; ii += kBlock) {
        const int i = static_cast<int>(ii);
        if (is_held(held, i)) {
            r[i] = make_real3(real(0), real(0), real(0));
            z[i] = make_real3(real(0), real(0), real(0));
            continue;
        }
        const real3 xi = x[i], pi = p[i], ri = r[i], api = Ap[i];
        x[i] = make_real3(static_cast<real>(static_cast<double>(xi.x) + alpha * static_cast<double>(pi.x)),
                          static_cast<real>(static_cast<double>(xi.y) + alpha * static_cast<double>(pi.y)),
                          static_cast<real>(static_cast<double>(xi.z) + alpha * static_cast<double>(pi.z)));
        const real3 rn = make_real3(static_cast<real>(static_cast<double>(ri.x) - alpha * static_cast<double>(api.x)),
                                    static_cast<real>(static_cast<double>(ri.y) - alpha * static_cast<double>(api.y)),
                                    static_cast<real>(static_cast<double>(ri.z) - alpha * static_cast<double>(api.z)));
        r[i] = rn;
        const real3 zn = block_apply(D + 9 * static_cast<std::size_t>(i), rn);
        z[i] = zn;
        rz += dot3(rn, zn);
        rr += dot3(rn, rn);
    }
    const double vrz = block_fold_sum(rz, sh);
    if (threadIdx.x == 0) part_rz[blockIdx.x] = vrz;
    const double vrr = block_fold_sum(rr, sh);
    if (threadIdx.x == 0) part_rr[blockIdx.x] = vrr;
}

// Convergence in either criterion, beta and the iteration count.
__global__ void k_finish_beta(PcgDeviceState* S, const double* __restrict__ part_rz,
                              const double* __restrict__ part_rr, int m, double* __restrict__ beta_hist)
{
    __shared__ double sh[kBlock];
    if (S->done) return;
    const double rz_new = fold_partials(part_rz, m, sh);
    const double rr = fold_partials(part_rr, m, sh);
    if (threadIdx.x != 0) return;
    S->rr = rr;
    S->iter += 1;                       // x was updated this iteration
    if (!(rz_new > 0.0) || !isfinite(rz_new)) {   // r = 0 or breakdown: keep x
        S->rz = (rz_new == 0.0) ? 0.0 : S->rz;
        S->done = 1;
        S->breakdown = (rz_new == 0.0) ? 0 : 1;
        S->converged = (rz_new == 0.0) ? 1 : 0;
        return;
    }
    const bool conv = S->use_rz ? (rz_new <= S->rz_tol) : (rr <= S->tol2);
    S->beta = rz_new / S->rz;
    S->rz = rz_new;
    beta_hist[S->iter - 1] = S->beta;
    if (conv) {
        S->converged = 1;
        S->done = 1;
        return;
    }
    if (S->iter >= S->max_iters) S->done = 1;
}

// p = z + beta p
__global__ void k_update_p_g(const PcgDeviceState* S, int n, const real3* __restrict__ z, real3* __restrict__ p)
{
    if (S->done) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double beta = S->beta;
    const real3 zi = z[i], pi = p[i];
    p[i] = make_real3(static_cast<real>(static_cast<double>(zi.x) + beta * static_cast<double>(pi.x)),
                      static_cast<real>(static_cast<double>(zi.y) + beta * static_cast<double>(pi.y)),
                      static_cast<real>(static_cast<double>(zi.z) + beta * static_cast<double>(pi.z)));
}

// ---- Lanczos spectrum estimate (diagnostic) --------------------------------------------------
// Number of eigenvalues of the symmetric tridiagonal (d, e) that are < sigma (Sturm sequence).
int sturm_count(const std::vector<double>& d, const std::vector<double>& e, double sigma)
{
    int count = 0;
    double q = d[0] - sigma;
    if (q < 0.0) ++count;
    for (std::size_t j = 1; j < d.size(); ++j) {
        if (q == 0.0) q = 1e-300;  // exact zero pivot: nudge, the standard remedy
        q = (d[j] - sigma) - e[j - 1] * e[j - 1] / q;
        if (q < 0.0) ++count;
    }
    return count;
}

// The k-th smallest eigenvalue (k in [1, m]) of the symmetric tridiagonal (d, e) by bisection on
// the Sturm count, bracketed by Gershgorin discs.
double tridiag_eigenvalue(const std::vector<double>& d, const std::vector<double>& e, int k)
{
    const std::size_t m = d.size();
    double lo = 1e300, hi = -1e300;
    for (std::size_t j = 0; j < m; ++j) {
        const double r = (j > 0 ? std::fabs(e[j - 1]) : 0.0) + (j + 1 < m ? std::fabs(e[j]) : 0.0);
        lo = std::min(lo, d[j] - r);
        hi = std::max(hi, d[j] + r);
    }
    for (int it = 0; it < 200 && hi - lo > 1e-14 * std::max(1.0, std::fabs(hi)); ++it) {
        const double mid = 0.5 * (lo + hi);
        if (sturm_count(d, e, mid) >= k) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

// CG coefficients -> Lanczos tridiagonal of the preconditioned operator:
//   T_jj = 1/alpha_j + beta_{j-1}/alpha_{j-1},  T_{j,j+1} = sqrt(beta_j)/alpha_j.
// Writes the extreme Ritz values into st. alphas has m entries, betas has at least m-1.
void lanczos_extremes(const std::vector<double>& alphas, const std::vector<double>& betas, PcgStats& st)
{
    const std::size_t m = alphas.size();
    if (m < 2) return;
    std::vector<double> d(m), e(m - 1);
    for (std::size_t j = 0; j < m; ++j) {
        d[j] = 1.0 / alphas[j] + (j > 0 ? betas[j - 1] / alphas[j - 1] : 0.0);
        if (j + 1 < m) e[j] = std::sqrt(std::max(betas[j], 0.0)) / alphas[j];
    }
    st.lambda_min = tridiag_eigenvalue(d, e, 1);
    st.lambda_max = tridiag_eigenvalue(d, e, static_cast<int>(m));
    st.kappa = (st.lambda_min > 0.0) ? st.lambda_max / st.lambda_min : 0.0;
}

}  // namespace

void BsrOperator::apply(const real3* x, real3* y, cudaStream_t stream, const int* skip) const
{
    bsr_spmv(*A, x, y, stream, skip);
}

// ---------------------------------------------------------------------------------------------
// Pcg
// ---------------------------------------------------------------------------------------------
Pcg::Pcg()
{
    // A blocking stream: the legacy default stream, on which the rest of the solver runs,
    // orders itself around this stream's work without events.
    CS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamDefault));
    CS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_state_), sizeof(PcgDeviceState)));
    CS_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_state_), sizeof(PcgDeviceState)));
}

Pcg::~Pcg()
{
    if (exec_) (void)cudaGraphExecDestroy(exec_);
    if (d_state_) (void)cudaFree(d_state_);
    if (h_state_) (void)cudaFreeHost(h_state_);
    if (stream_) (void)cudaStreamDestroy(stream_);
}

void Pcg::resize(int n_rows)
{
    n_ = n_rows;
    const std::size_t n = static_cast<std::size_t>(std::max(n_rows, 0));
    r_.resize(n);
    z_.resize(n);
    p_.resize(n);
    Ap_.resize(n);
    partial_layout(std::max(n_rows, 1), part_blocks_, part_chunk_);
    part_a_.resize(static_cast<std::size_t>(part_blocks_));
    part_b_.resize(static_cast<std::size_t>(part_blocks_));
}

void lanczos_extremes_public(const std::vector<double>& alphas, const std::vector<double>& betas, PcgStats& st)
{
    lanczos_extremes(alphas, betas, st);
}

namespace {
__global__ void k_diff3(int n, const real3* __restrict__ a, const real3* __restrict__ b, real3* __restrict__ d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = make_real3(a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z);
}
}  // namespace

// CS_PCG_VERIFY=1: every solve also runs the plain loop on the same system and the two
// solutions are compared (diagnostic; the plain result is discarded).
PcgStats Pcg::solve(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x, const PcgOptions& opt)
{
    static const bool verify = std::getenv("CS_PCG_VERIFY") != nullptr;
    if (!verify || !(opt.use_graph || opt.use_fused) || A.rows() <= 0) return solve_dispatch(A, diag_inv, b, x, opt);
    const int n = A.rows();
    verify_x_.resize(static_cast<std::size_t>(n));
    verify_d_.resize(static_cast<std::size_t>(n));
    PcgOptions po = opt;
    po.use_graph = false;
    po.use_fused = false;
    const PcgStats ps = solve_plain(A, diag_inv, b, verify_x_.data(), po);
    const PcgStats st = solve_dispatch(A, diag_inv, b, x, opt);
    CS_CUDA_CHECK(cudaDeviceSynchronize());
    k_diff3<<<grid_for(n), kBlock>>>(n, x, verify_x_.data(), verify_d_.data());
    CS_CUDA_KERNEL_CHECK();
    const double dd = std::sqrt(reduce_sum_sq(verify_d_.data(), n, scratch_));
    const double xx = std::sqrt(reduce_sum_sq(verify_x_.data(), n, scratch_));
    const double rel = xx > 0.0 ? dd / xx : dd;
    auto& lg = log();
    lg.log(spdlog::level::warn,   // always visible: the mode is opt-in
           "pcg verify: n={} {}: iters {} conv {} bd {} r0 {:.3e} rN {:.3e} | plain: iters {} conv {} bd {} r0 {:.3e} rN {:.3e} | rel diff {:.3e}",
           n, st.fused ? "fused" : "graph", st.iterations, st.converged, st.breakdown, st.initial_residual,
           st.final_residual, ps.iterations, ps.converged, ps.breakdown, ps.initial_residual, ps.final_residual, rel);
    return st;
}

PcgStats Pcg::solve_dispatch(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x, const PcgOptions& opt)
{
    const int n = A.rows();
    const bool two_level = opt.coarse != nullptr && opt.coarse->active();
    if (opt.use_fused && !two_level && n > 0 && n <= opt.fused_max_rows) {
        FusedViews V;
        if (A.fused_views(V)) {
            if (n != n_) resize(n);
            PcgStats st;
            if (solve_fused(V, diag_inv, b, x, opt, st)) return st;
        }
    }
    return opt.use_graph ? solve_graph(A, diag_inv, b, x, opt) : solve_plain(A, diag_inv, b, x, opt);
}

// ---- plain path (M1 loop) --------------------------------------------------------------------
PcgStats Pcg::solve_plain(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x,
                          const PcgOptions& opt)
{
    PcgStats st;
    const int n = A.rows();
    if (n != n_) resize(n);
    if (n <= 0) {
        st.converged = true;
        return st;
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(real3);
    CS_CUDA_CHECK(cudaMemset(x, 0, bytes));

    const double bb = reduce_sum_sq(b, n, scratch_);
    st.initial_residual = std::sqrt(bb);
    if (!std::isfinite(bb)) {  // garbage in: x = 0, not converged
        st.final_residual = st.initial_residual;
        return st;
    }
    if (bb == 0.0) {  // b = 0: x = 0 is the exact solution
        st.converged = true;
        return st;
    }
    const double rel = static_cast<double>(opt.rel_tol);
    const double tol2 = rel * rel * bb;
    const int check = std::max(opt.check_interval, 1);

    real3* r = r_.data();
    real3* z = z_.data();
    real3* p = p_.data();
    real3* Ap = Ap_.data();
    const unsigned grid = grid_for(n);

    // r = b, z = M^-1 r, p = z
    CS_CUDA_CHECK(cudaMemcpy(r, b, bytes, cudaMemcpyDeviceToDevice));
    const bool two_level = opt.coarse != nullptr && opt.coarse->active();
    k_precondition<<<grid, kBlock>>>(n, diag_inv, opt.held, r, z);
    CS_CUDA_KERNEL_CHECK();
    if (two_level) opt.coarse->apply(r, z, 0, nullptr);
    CS_CUDA_CHECK(cudaMemcpy(p, z, bytes, cudaMemcpyDeviceToDevice));
    double rz = reduce_dot(r, z, n, scratch_);
    const double rz0 = rz;
    const double rz_tol = rel * rz;   // preconditioned-residual criterion
    const bool use_rz = (opt.criterion == PcgCriterion::PreconditionedResidual);

    std::vector<double> alphas, betas;   // for the Lanczos spectrum estimate (diagnostic)
    if (opt.estimate_spectrum) {
        alphas.reserve(static_cast<std::size_t>(opt.max_iters));
        betas.reserve(static_cast<std::size_t>(opt.max_iters));
    }

    double rr = bb;
    bool rr_current = true;  // rr describes the current r
    for (int it = 1; it <= opt.max_iters; ++it) {
        if (!(rz > 0.0) || !std::isfinite(rz)) {  // breakdown (or r = 0): keep x
            st.breakdown = (rz != 0.0);
            break;
        }
        A.apply(p, Ap, 0, nullptr);
        const double pAp = reduce_dot(p, Ap, n, scratch_);
        if (!(pAp > 0.0) || !std::isfinite(pAp)) {  // not SPD / breakdown: keep x
            st.breakdown = true;
            break;
        }
        const double alpha = rz / pAp;
        if (opt.estimate_spectrum) alphas.push_back(alpha);
        k_update_xrz<<<grid, kBlock>>>(n, alpha, p, Ap, diag_inv, opt.held, x, r, z);
        CS_CUDA_KERNEL_CHECK();
        if (two_level) opt.coarse->apply(r, z, 0, nullptr);
        st.iterations = it;
        rr_current = false;
        // ||r||^2 costs an extra reduction and host readback, so it is only formed for the
        // absolute criterion; the preconditioned one reuses the r^T z below.
        if (!use_rz && (it % check == 0 || it == opt.max_iters)) {
            rr = reduce_sum_sq(r, n, scratch_);
            rr_current = true;
            if (rr <= tol2) {
                st.converged = true;
                break;
            }
        }
        const double rz_new = reduce_dot(r, z, n, scratch_);
        if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {  // r = 0 or breakdown: keep x
            st.breakdown = (rz_new != 0.0);
            if (rz_new == 0.0) {
                rz = 0.0;
                st.converged = true;
            }
            break;
        }
        if (use_rz && rz_new <= rz_tol) {
            rz = rz_new;
            st.converged = true;
            break;
        }
        const double beta = rz_new / rz;
        if (opt.estimate_spectrum) betas.push_back(beta);
        rz = rz_new;
        k_update_p<<<grid, kBlock>>>(n, beta, z, p);
        CS_CUDA_KERNEL_CHECK();
    }
    if (!rr_current) rr = reduce_sum_sq(r, n, scratch_);
    st.final_residual = std::sqrt(rr);
    if (!use_rz && rr <= tol2) st.converged = true;
    // rz is the current r^T z: the loop keeps it in step with r on every exit path.
    st.preconditioned_ratio = (rz0 > 0.0 && rz >= 0.0) ? std::sqrt(rz / rz0) : 0.0;
    if (opt.estimate_spectrum) lanczos_extremes(alphas, betas, st);
    return st;
}

// ---- graph path ------------------------------------------------------------------------------
// One captured block: k_iters guarded iterations on stream_.
void Pcg::capture_block(const LinearOperator& A, const real* diag_inv, real3* x, int n, int k_iters)
{
    real3* r = r_.data();
    real3* z = z_.data();
    real3* p = p_.data();
    real3* Ap = Ap_.data();
    const int* done = &d_state_->done;
    const int m = part_blocks_;
    const unsigned grid = grid_for(n);
    cudaGraph_t graph = nullptr;
    CS_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    ++detail::g_stream_capture_depth;
    try {
        for (int k = 0; k < k_iters; ++k) {
            A.apply(p, Ap, stream_, done);
            k_partials<<<static_cast<unsigned>(m), kBlock, 0, stream_>>>(done, n, part_chunk_, p, Ap, part_a_.data(),
                                                                         nullptr);
            k_finish_pAp<<<1, kBlock, 0, stream_>>>(d_state_, part_a_.data(), m, alpha_hist_.data());
            k_update_xrz_g<<<static_cast<unsigned>(m), kBlock, 0, stream_>>>(d_state_, n, part_chunk_, p, Ap, diag_inv,
                                                                            held_, x, r, z, part_a_.data(), part_b_.data());
            if (coarse_ != nullptr) {
                // two-level: z gains the coarse correction, so r.z is formed again (r.r stays in part_b_)
                coarse_->apply(r, z, stream_, done);
                k_partials<<<static_cast<unsigned>(m), kBlock, 0, stream_>>>(done, n, part_chunk_, r, z, part_a_.data(),
                                                                             nullptr);
            }
            k_finish_beta<<<1, kBlock, 0, stream_>>>(d_state_, part_a_.data(), part_b_.data(), m, beta_hist_.data());
            k_update_p_g<<<grid, kBlock, 0, stream_>>>(d_state_, n, z, p);
        }
    } catch (...) {
        // Leave the stream out of capture mode before propagating, so the object stays usable
        // and the original error is what the caller sees.
        --detail::g_stream_capture_depth;
        cudaGraph_t partial = nullptr;
        (void)cudaStreamEndCapture(stream_, &partial);
        if (partial) (void)cudaGraphDestroy(partial);
        (void)cudaGetLastError();
        throw;
    }
    --detail::g_stream_capture_depth;
    CS_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    CS_CUDA_CHECK(cudaGetLastError());
    if (exec_ != nullptr) {
        // Same topology as the last solve (the common case): refresh the kernel parameters in
        // place. A topology change (contact or bending appearing) fails the update and the
        // graph is instantiated afresh.
        cudaGraphExecUpdateResultInfo info{};
        if (cudaGraphExecUpdate(exec_, graph, &info) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
    }
    if (exec_ == nullptr) CS_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph, 0));
    CS_CUDA_CHECK(cudaGraphDestroy(graph));
}

PcgStats Pcg::solve_graph(const LinearOperator& A, const real* diag_inv, const real3* b, real3* x,
                          const PcgOptions& opt)
{
    PcgStats st;
    const int n = A.rows();
    if (n != n_) resize(n);
    if (n <= 0) {
        st.converged = true;
        return st;
    }
    const int max_iters = std::max(opt.max_iters, 1);
    if (alpha_hist_.size() < static_cast<std::size_t>(max_iters) + 1) {
        alpha_hist_.resize(static_cast<std::size_t>(max_iters) + 1);
        beta_hist_.resize(static_cast<std::size_t>(max_iters) + 1);
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(real3);
    real3* r = r_.data();
    real3* z = z_.data();
    real3* p = p_.data();
    const int m = part_blocks_;
    const unsigned grid = grid_for(n);
    const bool use_rz = (opt.criterion == PcgCriterion::PreconditionedResidual);
    held_ = opt.held;   // captured by k_update_xrz_g; unset it stayed null and held rows were solved as free
    coarse_ = (opt.coarse != nullptr && opt.coarse->active()) ? opt.coarse : nullptr;

    // Prologue on stream_: x = 0, r = b, z = M^-1 r, p = z, rz0 and bb, state.
    CS_CUDA_CHECK(cudaMemsetAsync(x, 0, bytes, stream_));
    CS_CUDA_CHECK(cudaMemcpyAsync(r, b, bytes, cudaMemcpyDeviceToDevice, stream_));
    k_precondition<<<grid, kBlock, 0, stream_>>>(n, diag_inv, opt.held, r, z);
    CS_CUDA_KERNEL_CHECK();
    if (coarse_ != nullptr) coarse_->apply(r, z, stream_, nullptr);
    CS_CUDA_CHECK(cudaMemcpyAsync(p, z, bytes, cudaMemcpyDeviceToDevice, stream_));
    k_partials<<<static_cast<unsigned>(m), kBlock, 0, stream_>>>(nullptr, n, part_chunk_, z, r, part_a_.data(),
                                                                 part_b_.data());
    CS_CUDA_KERNEL_CHECK();
    k_init_state<<<1, kBlock, 0, stream_>>>(d_state_, part_a_.data(), part_b_.data(), m,
                                            static_cast<double>(opt.rel_tol), use_rz ? 1 : 0, max_iters);
    CS_CUDA_KERNEL_CHECK();
    CS_CUDA_CHECK(cudaMemcpyAsync(h_state_, d_state_, sizeof(PcgDeviceState), cudaMemcpyDeviceToHost, stream_));
    CS_CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (!h_state_->done) {
        const int k_iters = std::max(1, std::min(opt.check_interval, max_iters));
        capture_block(A, diag_inv, x, n, k_iters);
        while (!h_state_->done) {
            CS_CUDA_CHECK(cudaGraphLaunch(exec_, stream_));
            CS_CUDA_CHECK(cudaMemcpyAsync(h_state_, d_state_, sizeof(PcgDeviceState), cudaMemcpyDeviceToHost, stream_));
            CS_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    const PcgDeviceState& S = *h_state_;
    st.iterations = S.iter;
    st.converged = S.converged != 0;
    st.breakdown = S.breakdown != 0;
    st.initial_residual = std::sqrt(S.bb);
    st.final_residual = std::sqrt(S.rr);
    st.preconditioned_ratio = (S.rz0 > 0.0 && S.rz >= 0.0) ? std::sqrt(S.rz / S.rz0) : 0.0;
    if (opt.estimate_spectrum && S.iter >= 2) {
        std::vector<double> alphas(static_cast<std::size_t>(S.iter)), betas(static_cast<std::size_t>(S.iter));
        alpha_hist_.download(alphas.data(), alphas.size());
        beta_hist_.download(betas.data(), betas.size());
        lanczos_extremes(alphas, betas, st);
    }
    return st;
}

}  // namespace cs
