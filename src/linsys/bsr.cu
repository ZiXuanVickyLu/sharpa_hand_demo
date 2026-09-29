// bsr.cu - BSR pattern construction (host), diagonal assembly helpers, the gather SpMV and the
// block-Jacobi preconditioner (doc/al-ipc-implementation-spec.md §4.3, §6.4, §6.6).
#include "linsys/bsr.cuh"
#include "linsys/reduce.cuh"
#include "linsys/linsys_detail.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace cs {

using linsys_detail::block_apply;
using linsys_detail::block_mul_acc;
using linsys_detail::grid_for;
using linsys_detail::kBlock;

namespace {

// Slot of (row, col) in a host CSR, or -1.
int find_slot(const std::vector<int>& row_ptr, const std::vector<int>& col_idx, int row, int col)
{
    const int* b = col_idx.data() + row_ptr[row];
    const int* e = col_idx.data() + row_ptr[row + 1];
    const int* it = std::lower_bound(b, e, col);
    return (it != e && *it == col) ? static_cast<int>(it - col_idx.data()) : -1;
}

// Same, but a miss is a bug in the construction above (the pattern contains both triangles).
int slot_or_throw(const std::vector<int>& row_ptr, const std::vector<int>& col_idx, int row, int col)
{
    const int s = find_slot(row_ptr, col_idx, row, col);
    if (s < 0)
        throw std::runtime_error("BsrPattern::build: missing slot (" + std::to_string(row) + ", " +
                                 std::to_string(col) + ")");
    return s;
}

// ------------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------------
__global__ void k_add_scaled_identity(int n, const int* __restrict__ diag_slot, const real* __restrict__ s,
                                      real* __restrict__ blocks)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    real* b = blocks + 9 * static_cast<std::size_t>(diag_slot[r]);
    const real v = s[r];
    b[0] += v;
    b[4] += v;
    b[8] += v;
}

__global__ void k_add_diagonal_blocks(int n, const int* __restrict__ diag_slot, const real* __restrict__ D,
                                      real* __restrict__ blocks)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    real* b = blocks + 9 * static_cast<std::size_t>(diag_slot[r]);
    const real* d = D + 9 * static_cast<std::size_t>(r);
#pragma unroll
    for (int k = 0; k < 9; ++k) b[k] += d[k];
}

// L lanes per block-row (a warp handles 32/L rows); the lanes stride over the row's slots,
// accumulate in double, then a fixed shuffle tree of width L folds the lane partials into the
// group's first lane. With ~11 blocks per row a whole warp per row leaves two thirds of the
// lanes idle and the kernel is bound by the x gather and the per-row overhead, not by the
// block bytes: on the animal-well pattern 16 lanes per row run the SpMV in 0.42 ms against
// 0.65 ms for 32 (doc/pcg-acceleration.md §2). `skip` is the PCG graph's early-exit flag.
constexpr int kLanes = 16;

template <int L>
__global__ void __launch_bounds__(kBlock) k_spmv(const int* __restrict__ skip, int n_rows,
                                                 const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                                                 const real* __restrict__ blocks, const real3* __restrict__ x,
                                                 real3* __restrict__ y)
{
    if (skip != nullptr && *skip) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid / L;
    const int lane = tid % L;
    if (row >= n_rows) return;  // uniform over the L-lane group
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int s = begin + lane; s < end; s += L) {
        const real3 xv = x[col_idx[s]];
        block_mul_acc(blocks + 9 * static_cast<std::size_t>(s), static_cast<double>(xv.x), static_cast<double>(xv.y),
                      static_cast<double>(xv.z), sx, sy, sz);
    }
    // The shuffle mask is this group's lanes only: the warp's other groups may have returned
    // above (rows past n_rows) or be mid-loop, and a full-warp mask would then be undefined.
    const unsigned lane_mask = ((1u << L) - 1u) << ((threadIdx.x % 32) / L * L);
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) {
        sx += __shfl_down_sync(lane_mask, sx, off, L);
        sy += __shfl_down_sync(lane_mask, sy, off, L);
        sz += __shfl_down_sync(lane_mask, sz, off, L);
    }
    if (lane == 0) y[row] = make_real3(static_cast<real>(sx), static_cast<real>(sy), static_cast<real>(sz));
}

// diag_inv[r] = sym(inverse(blocks[diag(r)] + extra[r])); singular or non-finite -> identity.
__global__ void k_block_jacobi_build(int n, const int* __restrict__ diag_slot, const real* __restrict__ blocks,
                                     const real* __restrict__ extra, real* __restrict__ diag_inv)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const real* b = blocks + 9 * static_cast<std::size_t>(diag_slot[r]);
    double m[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) m[k] = static_cast<double>(b[k]);
    if (extra != nullptr) {
        const real* e = extra + 9 * static_cast<std::size_t>(r);
#pragma unroll
        for (int k = 0; k < 9; ++k) m[k] += static_cast<double>(e[k]);
    }
    const double c00 = m[4] * m[8] - m[5] * m[7];
    const double c01 = m[2] * m[7] - m[1] * m[8];
    const double c02 = m[1] * m[5] - m[2] * m[4];
    const double c10 = m[5] * m[6] - m[3] * m[8];
    const double c11 = m[0] * m[8] - m[2] * m[6];
    const double c12 = m[2] * m[3] - m[0] * m[5];
    const double c20 = m[3] * m[7] - m[4] * m[6];
    const double c21 = m[1] * m[6] - m[0] * m[7];
    const double c22 = m[0] * m[4] - m[1] * m[3];
    const double det = m[0] * c00 + m[1] * c10 + m[2] * c20;
    // Numerically singular when |det| is below the scale of the block cubed: the identity is
    // a safe preconditioner, a 10^15 inverse is not.
    double scale = 0.0;
#pragma unroll
    for (int k = 0; k < 9; ++k) scale = fmax(scale, fabs(m[k]));
    constexpr double kDetEps = (sizeof(real) == 4) ? 1e-6 : 1e-14;
    double inv[9];
    bool ok = isfinite(det) && scale > 0.0 && fabs(det) > kDetEps * scale * scale * scale;
    if (ok) {
        const double id = 1.0 / det;
        inv[0] = c00 * id; inv[1] = c01 * id; inv[2] = c02 * id;
        inv[3] = c10 * id; inv[4] = c11 * id; inv[5] = c12 * id;
        inv[6] = c20 * id; inv[7] = c21 * id; inv[8] = c22 * id;
#pragma unroll
        for (int k = 0; k < 9; ++k) ok = ok && isfinite(inv[k]);
    }
    real* out = diag_inv + 9 * static_cast<std::size_t>(r);
    if (!ok) {
        out[0] = real(1); out[1] = real(0); out[2] = real(0);
        out[3] = real(0); out[4] = real(1); out[5] = real(0);
        out[6] = real(0); out[7] = real(0); out[8] = real(1);
        return;
    }
    const double s01 = 0.5 * (inv[1] + inv[3]);
    const double s02 = 0.5 * (inv[2] + inv[6]);
    const double s12 = 0.5 * (inv[5] + inv[7]);
    out[0] = static_cast<real>(inv[0]); out[1] = static_cast<real>(s01); out[2] = static_cast<real>(s02);
    out[3] = static_cast<real>(s01);    out[4] = static_cast<real>(inv[4]); out[5] = static_cast<real>(s12);
    out[6] = static_cast<real>(s02);    out[7] = static_cast<real>(s12);    out[8] = static_cast<real>(inv[8]);
}

__global__ void k_block_jacobi_apply(int n, const real* __restrict__ diag_inv, const real3* __restrict__ r,
                                     real3* __restrict__ z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = block_apply(diag_inv + 9 * static_cast<std::size_t>(i), r[i]);
}

__global__ void k_diag_max(int n, const int* __restrict__ diag_slot, const real* __restrict__ blocks,
                           real* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const real* b = blocks + 9 * static_cast<std::size_t>(diag_slot[r]);
    out[r] = fmax(fmax(b[0], b[4]), b[8]);
}

}  // namespace

// ------------------------------------------------------------------------------------------
// BsrPattern
// ------------------------------------------------------------------------------------------
BsrPattern BsrPattern::build(const Scene& scene)
{
    BsrPattern P;
    const int n = scene.n_free;
    if (n < 0 || n > scene.n_vertices)
        throw std::runtime_error("BsrPattern::build: n_free out of range (" + std::to_string(n) + ")");
    P.n_rows = n;
    const std::size_t n_tets = scene.tets.size();
    const auto is_free = [n](int g) { return g >= 0 && g < n; };
    for (std::size_t t = 0; t < n_tets; ++t)
        for (int a = 0; a < 4; ++a)
            if ((a == 3 && scene.tets[t][3] == -1) ? false : (scene.tets[t][a] < 0 || scene.tets[t][a] >= scene.n_vertices))
                throw std::runtime_error("BsrPattern::build: tet " + std::to_string(t) + " has vertex " +
                                         std::to_string(scene.tets[t][a]) + " outside [0, n_vertices)");

    // 1. Candidate (row, col) pairs: the diagonal of every free row plus every ordered pair of
    //    free vertices of every tet. Counted, then written into per-row segments.
    std::vector<std::size_t> off(static_cast<std::size_t>(n) + 1, 0);
    for (int r = 0; r < n; ++r) off[r + 1] = 1;
    for (std::size_t t = 0; t < n_tets; ++t) {
        const auto& g = scene.tets[t];
        for (int a = 0; a < 4; ++a) {
            if (!is_free(g[a])) continue;
            for (int b = 0; b < 4; ++b)
                if (is_free(g[b])) ++off[g[a] + 1];
        }
    }
    for (int r = 0; r < n; ++r) off[r + 1] += off[r];
    std::vector<int> cols(off[n]);
    std::vector<std::size_t> fill(off.begin(), off.end() - 1);
    for (int r = 0; r < n; ++r) cols[fill[r]++] = r;
    for (std::size_t t = 0; t < n_tets; ++t) {
        const auto& g = scene.tets[t];
        for (int a = 0; a < 4; ++a) {
            if (!is_free(g[a])) continue;
            for (int b = 0; b < 4; ++b)
                if (is_free(g[b])) cols[fill[g[a]]++] = g[b];
        }
    }

    // 2. Sort and unique per row -> CSR (rows ascending, cols ascending).
    P.h_row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    P.h_col_idx.clear();
    for (int r = 0; r < n; ++r) {
        int* b = cols.data() + off[r];
        int* e = cols.data() + off[r + 1];
        std::sort(b, e);
        e = std::unique(b, e);
        P.h_col_idx.insert(P.h_col_idx.end(), b, e);
        P.h_row_ptr[r + 1] = static_cast<int>(P.h_col_idx.size());
    }
    cols.clear();
    cols.shrink_to_fit();
    P.n_blocks = static_cast<int>(P.h_col_idx.size());

    // 3. Diagonal slot and transpose mirror.
    std::vector<int> h_diag(n), h_mirror(P.n_blocks);
    for (int r = 0; r < n; ++r) {
        h_diag[r] = slot_or_throw(P.h_row_ptr, P.h_col_idx, r, r);
        for (int s = P.h_row_ptr[r]; s < P.h_row_ptr[r + 1]; ++s) {
            const int c = P.h_col_idx[s];
            h_mirror[s] = (c == r) ? s : slot_or_throw(P.h_row_ptr, P.h_col_idx, c, r);
        }
    }

    // 4. Upper slots (col >= row) in slot order, and the (tet, a, b) gather map: tets ascending,
    //    a ascending, b ascending, stably bucketed by slot -> every slot's list is (tet, a, b) sorted.
    std::vector<int> h_upper;
    std::vector<int> slot_to_upper(P.n_blocks, -1);
    for (int r = 0; r < n; ++r)
        for (int s = P.h_row_ptr[r]; s < P.h_row_ptr[r + 1]; ++s)
            if (P.h_col_idx[s] >= r) {
                slot_to_upper[s] = static_cast<int>(h_upper.size());
                h_upper.push_back(s);
            }
    P.n_upper = static_cast<int>(h_upper.size());
    std::vector<int> h_contrib_ptr(static_cast<std::size_t>(P.n_upper) + 1, 0);
    std::vector<int> seq_upper;  // upper index of every contribution, in (tet, a, b) order
    for (std::size_t t = 0; t < n_tets; ++t) {
        const auto& g = scene.tets[t];
        for (int a = 0; a < 4; ++a) {
            if (!is_free(g[a])) continue;
            for (int b = 0; b < 4; ++b) {
                if (!is_free(g[b]) || g[b] < g[a]) continue;
                const int u = slot_to_upper[slot_or_throw(P.h_row_ptr, P.h_col_idx, g[a], g[b])];
                seq_upper.push_back(u);
                ++h_contrib_ptr[u + 1];
            }
        }
    }
    for (int u = 0; u < P.n_upper; ++u) h_contrib_ptr[u + 1] += h_contrib_ptr[u];
    const int n_contrib = h_contrib_ptr[P.n_upper];
    std::vector<int> h_contrib_tet(n_contrib);
    std::vector<unsigned char> h_contrib_ab(n_contrib);
    std::vector<int> cfill(h_contrib_ptr.begin(), h_contrib_ptr.end() - 1);
    std::size_t k_seq = 0;
    for (std::size_t t = 0; t < n_tets; ++t) {
        const auto& g = scene.tets[t];
        for (int a = 0; a < 4; ++a) {
            if (!is_free(g[a])) continue;
            for (int b = 0; b < 4; ++b) {
                if (!is_free(g[b]) || g[b] < g[a]) continue;
                const int u = seq_upper[k_seq++];
                const int k = cfill[u]++;
                h_contrib_tet[k] = static_cast<int>(t);
                h_contrib_ab[k] = static_cast<unsigned char>(a * 4 + b);
            }
        }
    }
    seq_upper.clear();
    seq_upper.shrink_to_fit();

    // 5. Upload.
    P.row_ptr.upload(P.h_row_ptr);
    P.col_idx.upload(P.h_col_idx);
    P.diag_slot.upload(h_diag);
    P.slot_mirror.upload(h_mirror);
    P.upper_slot.upload(h_upper);
    P.contrib_ptr.upload(h_contrib_ptr);
    P.contrib_tet.upload(h_contrib_tet);
    P.contrib_ab.upload(h_contrib_ab);
    return P;
}

// ------------------------------------------------------------------------------------------
// BsrMatrix
// ------------------------------------------------------------------------------------------
void BsrMatrix::init(const BsrPattern& p)
{
    pattern = &p;
    blocks.resize(9 * static_cast<std::size_t>(p.n_blocks));
    diag_inv.resize(9 * static_cast<std::size_t>(p.n_rows));
    blocks.zero();
    diag_inv.zero();
}

void BsrMatrix::zero()
{
    blocks.zero();
}

// ------------------------------------------------------------------------------------------
// Operations
// ------------------------------------------------------------------------------------------
void bsr_add_scaled_identity(BsrMatrix& A, const real* per_row_scalar)
{
    const int n = A.n_rows();
    if (n == 0) return;
    k_add_scaled_identity<<<grid_for(n), kBlock>>>(n, A.pattern->diag_slot.data(), per_row_scalar, A.blocks.data());
    CS_CUDA_KERNEL_CHECK();
}

void bsr_add_diagonal_blocks(BsrMatrix& A, const real* per_row_block9)
{
    const int n = A.n_rows();
    if (n == 0) return;
    k_add_diagonal_blocks<<<grid_for(n), kBlock>>>(n, A.pattern->diag_slot.data(), per_row_block9, A.blocks.data());
    CS_CUDA_KERNEL_CHECK();
}

void bsr_spmv(const BsrMatrix& A, const real3* x, real3* y, cudaStream_t stream, const int* skip)
{
    const int n = A.n_rows();
    if (n == 0) return;
    const unsigned grid = static_cast<unsigned>((static_cast<long long>(kLanes) * n + kBlock - 1) / kBlock);
    k_spmv<kLanes><<<grid, kBlock, 0, stream>>>(skip, n, A.pattern->row_ptr.data(), A.pattern->col_idx.data(),
                                                A.blocks.data(), x, y);
    CS_CUDA_KERNEL_CHECK();
}


void bsr_build_block_jacobi(BsrMatrix& A, const real* extra_diag_block9)
{
    const int n = A.n_rows();
    if (n == 0) return;
    k_block_jacobi_build<<<grid_for(n), kBlock>>>(n, A.pattern->diag_slot.data(), A.blocks.data(), extra_diag_block9,
                                                  A.diag_inv.data());
    CS_CUDA_KERNEL_CHECK();
}

void bsr_apply_block_jacobi(const BsrMatrix& A, const real3* r, real3* z)
{
    const int n = A.n_rows();
    if (n == 0) return;
    k_block_jacobi_apply<<<grid_for(n), kBlock>>>(n, A.diag_inv.data(), r, z);
    CS_CUDA_KERNEL_CHECK();
}

double bsr_max_diagonal_entry(const BsrMatrix& A, DeviceArray<real>& scratch)
{
    const int n = A.n_rows();
    if (n == 0) return 0.0;
    if (scratch.size() < static_cast<std::size_t>(n)) scratch.resize(n);
    k_diag_max<<<grid_for(n), kBlock>>>(n, A.pattern->diag_slot.data(), A.blocks.data(), scratch.data());
    CS_CUDA_KERNEL_CHECK();
    // The reduction cannot share `scratch` (it would overwrite its own input); its workspace is
    // at most kMaxBlocks + 1 doubles, allocated for the duration of the call.
    DeviceArray<double> partials;
    return reduce_max(scratch.data(), n, partials);
}

void bsr_to_dense(const BsrMatrix& A, std::vector<double>& dense)
{
    if (A.pattern == nullptr) {
        dense.clear();
        return;
    }
    const BsrPattern& P = *A.pattern;
    const int n = P.n_rows;
    const std::size_t N = 3 * static_cast<std::size_t>(n);
    dense.assign(N * N, 0.0);
    std::vector<real> blocks;
    A.blocks.download(blocks);
    for (int r = 0; r < n; ++r) {
        for (int s = P.h_row_ptr[r]; s < P.h_row_ptr[r + 1]; ++s) {
            const int c = P.h_col_idx[s];
            const real* b = blocks.data() + 9 * static_cast<std::size_t>(s);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    dense[(3 * static_cast<std::size_t>(r) + i) * N + 3 * static_cast<std::size_t>(c) + j] =
                        static_cast<double>(b[3 * i + j]);
        }
    }
}

}  // namespace cs
