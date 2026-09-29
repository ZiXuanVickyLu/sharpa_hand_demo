// pcg_fused.cu - the fused cooperative CG solve for small systems (IS §12.3 item 14). See
// pcg_fused.cuh. Every reduction is a fixed-order block fold written to per-block partials that
// every block then folds in the same order, so all blocks hold the same alpha, beta and
// decisions and the result is bitwise reproducible; there are no atomics.
#include "linsys/pcg.cuh"
#include "linsys/linsys_detail.cuh"
#include "core/cuda_check.h"

#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>

namespace cg = cooperative_groups;

namespace cs {

using linsys_detail::block_apply;
using linsys_detail::block_mul_acc;

namespace {

constexpr int kBlock = 256;
constexpr int kLanes = 16;   // as the BSR SpMV kernel (bsr.cu)

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
    __syncthreads();
    return v;
}

// Every block folds the m per-block partials in the same order: the same double everywhere.
__device__ __forceinline__ double fold_all(const double* __restrict__ part, int m, double* sh)
{
    double acc = 0.0;
    for (int i = threadIdx.x; i < m; i += kBlock) acc += part[i];
    return block_fold_sum(acc, sh);
}

__device__ __forceinline__ double dot3d(const real3& a, const real3& b)
{
    return static_cast<double>(a.x) * static_cast<double>(b.x) + static_cast<double>(a.y) * static_cast<double>(b.y) +
           static_cast<double>(a.z) * static_cast<double>(b.z);
}

__global__ void __launch_bounds__(kBlock) k_pcg_fused(FusedViews V, const real* __restrict__ diag_inv,
                                                      const real3* __restrict__ b, real3* __restrict__ x,
                                                      real3* __restrict__ r, real3* __restrict__ z,
                                                      real3* __restrict__ p, real3* __restrict__ Ap,
                                                      PcgDeviceState* __restrict__ S, double* __restrict__ part_a,
                                                      double* __restrict__ part_b, double* __restrict__ part_c,
                                                      double* __restrict__ alpha_hist,
                                                      double* __restrict__ beta_hist, double rel, int use_rz,
                                                      int max_iters, const int* __restrict__ held)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ double sh[kBlock];
    const int n = V.n_rows;
    const int tid = static_cast<int>(grid.thread_rank());
    const int nt = static_cast<int>(grid.num_threads());
    const int nb = static_cast<int>(gridDim.x);
    const bool leader = (blockIdx.x == 0 && threadIdx.x == 0);

    // ---- phase 0: x = 0, r = b, z = M^-1 r, p = z; rz0 and bb ------------------------------
    double acc_a = 0.0, acc_b = 0.0;
    for (int i = tid; i < n; i += nt) {
        const real3 bi = b[i];
        x[i] = make_real3(real(0), real(0), real(0));
        r[i] = bi;
        const bool h = (held != nullptr && held[i] != 0);
        const real3 zi = h ? make_real3(real(0), real(0), real(0)) : block_apply(diag_inv + 9 * static_cast<std::size_t>(i), bi);
        z[i] = zi;
        p[i] = zi;
        acc_a += dot3d(bi, zi);
        acc_b += dot3d(bi, bi);
    }
    {
        const double va = block_fold_sum(acc_a, sh);
        if (threadIdx.x == 0) part_a[blockIdx.x] = va;
        const double vb = block_fold_sum(acc_b, sh);
        if (threadIdx.x == 0) part_b[blockIdx.x] = vb;
    }
    grid.sync();
    const double rz0 = fold_all(part_a, nb, sh);
    const double bb = fold_all(part_b, nb, sh);
    double rz = rz0, rr = bb;
    int iter = 0;
    bool done = false, converged = false, breakdown = false;
    if (!isfinite(bb)) { done = true; breakdown = true; }
    else if (bb == 0.0) { done = true; converged = true; }
    else if (!(rz0 > 0.0) || !isfinite(rz0)) { done = true; breakdown = true; }
    const double rz_tol = rel * rz0;
    const double tol2 = rel * rel * bb;
    grid.sync();   // the partials are free again

    while (!done) {
        // ---- phase 1: Ap = A p: BSR rows, then the pair-side passes of contact, friction, bending
        const unsigned lane_mask = (threadIdx.x & kLanes) ? 0xffff0000u : 0x0000ffffu;   // this 16-lane group
        for (int g = tid / kLanes; g < n; g += nt / kLanes) {
            const int lane = tid % kLanes;
            const int begin = V.row_ptr[g], end = V.row_ptr[g + 1];
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int s = begin + lane; s < end; s += kLanes) {
                const real3 xv = p[V.col_idx[s]];
                block_mul_acc(V.blocks + 9 * static_cast<std::size_t>(s), static_cast<double>(xv.x),
                              static_cast<double>(xv.y), static_cast<double>(xv.z), sx, sy, sz);
            }
#pragma unroll
            for (int off = kLanes / 2; off > 0; off >>= 1) {
                sx += __shfl_down_sync(lane_mask, sx, off, kLanes);
                sy += __shfl_down_sync(lane_mask, sy, off, kLanes);
                sz += __shfl_down_sync(lane_mask, sz, off, kLanes);
            }
            if (lane == 0) Ap[g] = make_real3(static_cast<real>(sx), static_cast<real>(sy), static_cast<real>(sz));
        }
        if (V.has_contact) {
            for (int i = tid; i < V.contact.n_pairs; i += nt) contact_pair_sigma(V.contact, i, p);
            for (int i = tid; i < V.contact.n_fr; i += nt) friction_pair_scalar(V.contact, i, p);
        }
        if (V.has_bending) {
            for (int h = tid; h < V.bending.n; h += nt) bending_hinge_sigma(V.bending, h, p);
        }
        grid.sync();
        // ---- phase 2: the vertex-side gathers into Ap in the kernel path's order; p . Ap
        acc_a = 0.0;
        for (int v = tid; v < n; v += nt) {
            if (V.has_contact) {
                if (V.contact.n_pairs) contact_vertex_gather(V.contact, v, Ap);
                if (V.contact.n_fr) friction_vertex_gather(V.contact, v, Ap);
            }
            if (V.has_bending && V.bending.n) bending_vertex_gather(V.bending, v, Ap);
            acc_a += dot3d(p[v], Ap[v]);
        }
        // p . Ap goes to its own partial array: a block that has folded the phase-2 partials
        // and moved on must not overwrite an entry another block is still reading; with three
        // arrays every array is read and rewritten with two barriers in between.
        {
            const double vc = block_fold_sum(acc_a, sh);
            if (threadIdx.x == 0) part_c[blockIdx.x] = vc;
        }
        grid.sync();
        const double pAp = fold_all(part_c, nb, sh);
        if (!(pAp > 0.0) || !isfinite(pAp)) {
            done = true;
            breakdown = true;
        } else {
            const double alpha = rz / pAp;
            if (leader) alpha_hist[iter] = alpha;
            // ---- phase 3: x += alpha p, r -= alpha Ap, z = M^-1 r; r . z and r . r
            acc_a = 0.0;
            acc_b = 0.0;
            for (int i = tid; i < n; i += nt) {
                if (held != nullptr && held[i] != 0) {
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
                const real3 zn = block_apply(diag_inv + 9 * static_cast<std::size_t>(i), rn);
                z[i] = zn;
                acc_a += dot3d(rn, zn);
                acc_b += dot3d(rn, rn);
            }
            const double va = block_fold_sum(acc_a, sh);
            if (threadIdx.x == 0) part_a[blockIdx.x] = va;
            const double vb = block_fold_sum(acc_b, sh);
            if (threadIdx.x == 0) part_b[blockIdx.x] = vb;
        }
        grid.sync();
        if (done) break;   // uniform: every thread folded the same pAp
        const double rz_new = fold_all(part_a, nb, sh);
        rr = fold_all(part_b, nb, sh);
        iter += 1;
        if (!(rz_new > 0.0) || !isfinite(rz_new)) {   // r = 0 or breakdown: keep x
            if (rz_new == 0.0) { rz = 0.0; converged = true; }
            else breakdown = true;
            done = true;
        } else {
            const bool conv = use_rz ? (rz_new <= rz_tol) : (rr <= tol2);
            const double beta = rz_new / rz;
            rz = rz_new;
            if (leader) beta_hist[iter - 1] = beta;
            if (conv) { converged = true; done = true; }
            else if (iter >= max_iters) done = true;
            else {
                // ---- phase 4: p = z + beta p
                for (int i = tid; i < n; i += nt) {
                    const real3 zi = z[i], pi = p[i];
                    p[i] = make_real3(static_cast<real>(static_cast<double>(zi.x) + beta * static_cast<double>(pi.x)),
                                      static_cast<real>(static_cast<double>(zi.y) + beta * static_cast<double>(pi.y)),
                                      static_cast<real>(static_cast<double>(zi.z) + beta * static_cast<double>(pi.z)));
                }
            }
        }
        grid.sync();   // p complete, partials free
    }
    if (leader) {
        S->iter = iter;
        S->done = 1;
        S->converged = converged ? 1 : 0;
        S->breakdown = breakdown ? 1 : 0;
        S->rz = rz;
        S->rz0 = rz0;
        S->bb = bb;
        S->rr = rr;
        S->use_rz = use_rz;
        S->max_iters = max_iters;
    }
}

}  // namespace

bool BsrOperator::fused_views(FusedViews& v) const
{
    v = FusedViews{};
    v.n_rows = A->n_rows();
    v.row_ptr = A->pattern->row_ptr.data();
    v.col_idx = A->pattern->col_idx.data();
    v.blocks = A->blocks.data();
    return true;
}

bool Pcg::solve_fused(const FusedViews& V, const real* diag_inv, const real3* b, real3* x, const PcgOptions& opt,
                      PcgStats& st)
{
    const int n = V.n_rows;
    long long work = static_cast<long long>(kLanes) * n;
    if (V.has_contact) work = std::max(work, static_cast<long long>(std::max(V.contact.n_pairs, V.contact.n_fr)));
    if (V.has_bending) work = std::max(work, static_cast<long long>(V.bending.n));
    int blocks = static_cast<int>((work + kBlock - 1) / kBlock);
    blocks = std::max(blocks, 1);
    int per_sm = 0;
    CS_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_pcg_fused, kBlock, 0));
    int device = 0, sms = 0;
    CS_CUDA_CHECK(cudaGetDevice(&device));
    CS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int max_blocks = per_sm * sms;
    if (blocks > max_blocks) return false;   // the grid would not be co-resident: graph path

    const int max_iters = std::max(opt.max_iters, 1);
    if (alpha_hist_.size() < static_cast<std::size_t>(max_iters) + 1) {
        alpha_hist_.resize(static_cast<std::size_t>(max_iters) + 1);
        beta_hist_.resize(static_cast<std::size_t>(max_iters) + 1);
    }
    if (part_a_.size() < static_cast<std::size_t>(blocks)) {
        part_a_.resize(static_cast<std::size_t>(blocks));
        part_b_.resize(static_cast<std::size_t>(blocks));
    }
    if (part_c_.size() < static_cast<std::size_t>(blocks)) part_c_.resize(static_cast<std::size_t>(blocks));
    double rel = static_cast<double>(opt.rel_tol);
    int use_rz = (opt.criterion == PcgCriterion::PreconditionedResidual) ? 1 : 0;
    int max_it = max_iters;
    const int* held = opt.held;
    FusedViews Vv = V;
    const real* d_inv = diag_inv;
    const real3* bp = b;
    real3* xp = x;
    real3* rp = r_.data();
    real3* zp = z_.data();
    real3* pp = p_.data();
    real3* app = Ap_.data();
    PcgDeviceState* sp = d_state_;
    double* pa = part_a_.data();
    double* pb = part_b_.data();
    double* pc = part_c_.data();
    double* ah = alpha_hist_.data();
    double* bh = beta_hist_.data();
    void* args[] = {&Vv, &d_inv, &bp, &xp, &rp, &zp, &pp, &app, &sp, &pa, &pb, &pc, &ah, &bh, &rel, &use_rz, &max_it, &held};
    CS_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(k_pcg_fused), dim3(blocks), dim3(kBlock),
                                              args, 0, stream_));
    CS_CUDA_KERNEL_CHECK();
    CS_CUDA_CHECK(cudaMemcpyAsync(h_state_, d_state_, sizeof(PcgDeviceState), cudaMemcpyDeviceToHost, stream_));
    CS_CUDA_CHECK(cudaStreamSynchronize(stream_));
    const PcgDeviceState& S = *h_state_;
    st = PcgStats{};
    st.iterations = S.iter;
    st.converged = S.converged != 0;
    st.breakdown = S.breakdown != 0;
    st.initial_residual = std::sqrt(S.bb);
    st.final_residual = std::sqrt(S.rr);
    st.preconditioned_ratio = (S.rz0 > 0.0 && S.rz >= 0.0) ? std::sqrt(S.rz / S.rz0) : 0.0;
    st.fused = true;
    if (opt.estimate_spectrum && S.iter >= 2) {
        std::vector<double> alphas(static_cast<std::size_t>(S.iter)), betas(static_cast<std::size_t>(S.iter));
        alpha_hist_.download(alphas.data(), alphas.size());
        beta_hist_.download(betas.data(), betas.size());
        lanczos_extremes_public(alphas, betas, st);
    }
    return true;
}

}  // namespace cs
