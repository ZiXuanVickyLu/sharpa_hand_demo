// fem.cu - elasticity kernels (doc/al-ipc-implementation-spec.md §6.4, math spec §3 and
// Appendix B): per-tet cache, energy, gradient gather, diagonal blocks, inversion guard and the
// per-upper-slot BSR assembly. No atomics anywhere: every output element is owned by exactly
// one thread and every reduction is a fixed-order two-level block sum in double (§11.1).
#include "fem/fem.cuh"
#include "fem/elastic_models.cuh"
#include "linsys/bsr.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace cs {

void ElementCache::resize(int n_tets)
{
    const std::size_t n = static_cast<std::size_t>(n_tets < 0 ? 0 : n_tets);
    U.resize(9 * n);
    V.resize(9 * n);
    eig.resize(9 * n);
    q.resize(9 * n);
    P.resize(9 * n);
    energy.resize(n);
}

namespace {

constexpr int kBlock = 256;
constexpr int kReduceItems = 8;                       // elements per thread in the first level
constexpr int kReduceChunk = kBlock * kReduceItems;   // elements per block in the first level

inline unsigned grid_for(int n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }
inline int reduce_blocks_for(int n) { return (n + kReduceChunk - 1) / kReduceChunk; }

// ---------------------------------------------------------------------------------------------
// Views of the device arrays (plain pointers for the kernels)
// ---------------------------------------------------------------------------------------------
struct TetView {
    const int4* tet;
    const real* DmInv;
    const real* volume;
    const int* tet_material;
    const MaterialParams* materials;
    int n_tets;
};

struct VtView {
    const int* ptr;
    const int* tet;
    const int* slot;
    int n_free;
};

struct CacheView {
    const real* U;
    const real* V;
    const real* eig;
    const real* q;
    const real* P;
};

TetView tet_view(const SceneBuffers& sb)
{
    return {sb.tet.data(), sb.DmInv.data(), sb.volume.data(), sb.tet_material.data(), sb.materials.data(), sb.n_tets};
}
VtView vt_view(const SceneBuffers& sb) { return {sb.vt_ptr.data(), sb.vt_tet.data(), sb.vt_slot.data(), sb.n_free}; }
CacheView cache_view(const ElementCache& c) { return {c.U.data(), c.V.data(), c.eig.data(), c.q.data(), c.P.data()}; }

// ---------------------------------------------------------------------------------------------
// Deterministic reductions in double: level 1 = one partial per block of kReduceChunk inputs
// (each thread walks its kReduceItems inputs in a fixed order, then a shared-memory tree);
// level 2 = one block over the partials, same scheme. The grid is a function of n only, so the
// result is bitwise reproducible from run to run.
// ---------------------------------------------------------------------------------------------
struct OpSum {
    __device__ static double init() { return 0.0; }
    __device__ static double apply(double a, double b) { return a + b; }
};
struct OpMax {
    __device__ static double init() { return -HUGE_VAL; }
    __device__ static double apply(double a, double b) { return fmax(a, b); }
};
struct OpMin {
    __device__ static double init() { return HUGE_VAL; }
    __device__ static double apply(double a, double b) { return fmin(a, b); }
};

template <class Op> __device__ __forceinline__ double block_tree(double v, double* sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int w = kBlock / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] = Op::apply(sh[threadIdx.x], sh[threadIdx.x + w]);
        __syncthreads();
    }
    return sh[0];
}

template <class Op> __global__ void k_reduce_level1(const double* __restrict__ in, int n, double* __restrict__ out)
{
    __shared__ double sh[kBlock];
    double acc = Op::init();
    const int base = blockIdx.x * kReduceChunk;
    for (int i = 0; i < kReduceItems; ++i) {
        const int idx = base + i * kBlock + threadIdx.x;
        if (idx < n) acc = Op::apply(acc, in[idx]);
    }
    const double r = block_tree<Op>(acc, sh);
    if (threadIdx.x == 0) out[blockIdx.x] = r;
}

template <class Op> __global__ void k_reduce_level2(const double* __restrict__ in, int n, double* __restrict__ out)
{
    __shared__ double sh[kBlock];
    double acc = Op::init();
    for (int idx = threadIdx.x; idx < n; idx += kBlock) acc = Op::apply(acc, in[idx]);
    const double r = block_tree<Op>(acc, sh);
    if (threadIdx.x == 0) *out = r;
}

// The scratch array is viewed as raw double storage (cudaMalloc is 256-byte aligned, so the
// reinterpretation is valid for float and double `real`). Layout: n values | nb partials | result.
double* scratch_doubles(DeviceArray<real>& scratch, std::size_t n_doubles)
{
    const std::size_t bytes = n_doubles * sizeof(double);
    const std::size_t n_real = (bytes + sizeof(real) - 1) / sizeof(real);
    if (scratch.size() < n_real) scratch.resize(n_real);
    return reinterpret_cast<double*>(scratch.data());
}

template <class Op> double reduce_doubles(const double* d_vals, int n, double* d_work)
{
    const int nb = reduce_blocks_for(n);
    k_reduce_level1<Op><<<static_cast<unsigned>(nb), kBlock>>>(d_vals, n, d_work);
    CS_CUDA_KERNEL_CHECK();
    k_reduce_level2<Op><<<1, kBlock>>>(d_work, nb, d_work + nb);
    CS_CUDA_KERNEL_CHECK();
    double r = 0.0;
    CS_CUDA_CHECK(cudaMemcpy(&r, d_work + nb, sizeof(double), cudaMemcpyDeviceToHost));
    return r;
}

// ---------------------------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void load3(const real3& v, real* out)
{
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

// Padded F of a shell element (tet slot w = -1): D_s = [x1 - x0, x2 - x0, 0] times the padded
// DmInv, so the third column of F is zero (MS §3.6).
__device__ __forceinline__ void shell_F(const TetView& tv, const real3* x, int e, real* F)
{
    const int4 t = tv.tet[e];
    real x0[3], x1[3], x2[3];
    load3(x[t.x], x0);
    load3(x[t.y], x1);
    load3(x[t.z], x2);
    real Ds[9];
    for (int r = 0; r < 3; ++r) {
        Ds[r] = x1[r] - x0[r];
        Ds[3 + r] = x2[r] - x0[r];
        Ds[6 + r] = real(0);
    }
    fem::m3_mul(Ds, tv.DmInv + 9 * e, F);
}

__device__ __forceinline__ void tet_F(const TetView& tv, const real3* x, int e, real* F)
{
    const int4 t = tv.tet[e];
    real x0[3], x1[3], x2[3], x3[3];
    load3(x[t.x], x0);
    load3(x[t.y], x1);
    load3(x[t.z], x2);
    load3(x[t.w], x3);
    fem::deformation_gradient(x0, x1, x2, x3, tv.DmInv + 9 * e, F);
}

// Rounds a mathematically symmetric 3x3 (row-major) to its exact symmetric part. U S U^T
// evaluates the (r, c) and (c, r) entries in different summation orders, so they can differ in
// the last bit; the diagonal blocks are symmetric by construction and the spec asks for a
// bitwise symmetric matrix (implementation spec §6.4), so the pair is replaced by its mean.
__device__ __forceinline__ void symmetrize3(real* m)
{
    const real m01 = real(0.5) * (m[1] + m[3]);
    const real m02 = real(0.5) * (m[2] + m[6]);
    const real m12 = real(0.5) * (m[5] + m[7]);
    m[1] = m[3] = m01;
    m[2] = m[6] = m02;
    m[5] = m[7] = m12;
}

// h2 * Σ_{e ∋ v} V_e (H_e^+)_{aa} of free vertex v (row-major 9), gathered in CSR order.
__device__ __forceinline__ void vertex_diag_block(const VtView& vt, const TetView& tv, const CacheView& cache, real h2,
                                                  int v, real* out)
{
    for (int i = 0; i < 9; ++i) out[i] = real(0);
    const int k0 = vt.ptr[v], k1 = vt.ptr[v + 1];
    for (int k = k0; k < k1; ++k) {
        const int e = vt.tet[k];
        const int a = vt.slot[k];
        real aa[3], ba[3];
        fem::shape_gradient(tv.DmInv + 9 * e, a, aa);
        fem::m3_mulvec_t(cache.V + 9 * e, aa, ba);
        real blk[9];
        fem::elem_block(cache.U + 9 * e, cache.eig + 9 * e, cache.q + 9 * e, ba, ba, h2 * tv.volume[e], blk);
        for (int i = 0; i < 9; ++i) out[i] += blk[i];
    }
    symmetrize3(out);
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------
__global__ void k_element_cache(TetView tv, const real3* __restrict__ x, real* __restrict__ U, real* __restrict__ V,
                                real* __restrict__ eig, real* __restrict__ q, real* __restrict__ P,
                                real* __restrict__ energy, bool need_hessian)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tv.n_tets) return;

    real F[9];
    const fem::ModelParams<real> mp = fem::model_params<real>(tv.materials[tv.tet_material[e]]);
    const real Ve = tv.volume[e];
    if (fem::is_cloth(mp.model)) {
        // MS §3.5-3.6: thin SVD, membrane derivatives, everything written in the padded layout.
        shell_F(tv, x, e, F);
        real Um[9], Vm[9], S[3], ev[9], Q[9], dpsi[2];
        fem::shell_svd(F, Um, S, Vm);
        energy[e] = Ve * fem::psi2_sigma(mp, S[0], S[1]);
        fem::shell_eigen_system(mp, S[0], S[1], ev, Q, dpsi);
        real Pm[9];
        fem::shell_pk1(Um, dpsi, Vm, Pm);
        for (int i = 0; i < 9; ++i) P[9 * e + i] = Pm[i];
        if (need_hessian) {
            fem::project_eig(ev);
            for (int i = 0; i < 9; ++i) {
                U[9 * e + i] = Um[i];
                V[9 * e + i] = Vm[i];
                eig[9 * e + i] = ev[i];
                q[9 * e + i] = Q[i];
            }
        }
        return;
    }
    tet_F(tv, x, e, F);
    const bool is_cor = (mp.model == fem::kCOR);
    const bool need_svd = need_hessian || is_cor;

    real Um[9] = {real(1), real(0), real(0), real(0), real(1), real(0), real(0), real(0), real(1)};
    real Vm[9] = {real(1), real(0), real(0), real(0), real(1), real(0), real(0), real(0), real(1)};
    real S[3] = {real(1), real(1), real(1)};
    if (need_svd) svd3(F, Um, S, Vm);

    // Energy: SNH/NH straight from F (no SVD error), COR from the singular values.
    energy[e] = Ve * (is_cor ? fem::psi_sigma(mp, S) : fem::psi_F_nosvd(mp, F));

    real Pm[9];
    fem::pk1(mp, F, Um, Vm, Pm);
    for (int i = 0; i < 9; ++i) P[9 * e + i] = Pm[i];

    if (need_hessian) {
        real ev[9], Q[9];
        fem::eigen_system(mp, S, ev, Q);
        fem::project_eig(ev);
        for (int i = 0; i < 9; ++i) {
            U[9 * e + i] = Um[i];
            V[9 * e + i] = Vm[i];
            eig[9 * e + i] = ev[i];
            q[9 * e + i] = Q[i];
        }
    }
}

__global__ void k_element_energy(TetView tv, const real3* __restrict__ x, double* __restrict__ out)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tv.n_tets) return;
    real F[9];
    const fem::ModelParams<real> mp = fem::model_params<real>(tv.materials[tv.tet_material[e]]);
    if (fem::is_cloth(mp.model)) {
        shell_F(tv, x, e, F);
        real Um[9], Vm[9], S[3];
        fem::shell_svd(F, Um, S, Vm);
        out[e] = static_cast<double>(tv.volume[e]) * static_cast<double>(fem::psi2_sigma(mp, S[0], S[1]));
        return;
    }
    tet_F(tv, x, e, F);
    out[e] = static_cast<double>(tv.volume[e]) * static_cast<double>(fem::psi_F(mp, F));
}

__global__ void k_add_gradient(VtView vt, TetView tv, const real* __restrict__ P, real h2, real3* __restrict__ G)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vt.n_free) return;
    real acc[3] = {real(0), real(0), real(0)};
    const int k0 = vt.ptr[v], k1 = vt.ptr[v + 1];
    for (int k = k0; k < k1; ++k) {
        const int e = vt.tet[k];
        const int a = vt.slot[k];
        real aa[3], Pa[3];
        fem::shape_gradient(tv.DmInv + 9 * e, a, aa);
        fem::m3_mulvec(P + 9 * e, aa, Pa);
        const real s = h2 * tv.volume[e];
        acc[0] += s * Pa[0];
        acc[1] += s * Pa[1];
        acc[2] += s * Pa[2];
    }
    real3 g = G[v];
    g.x += acc[0];
    g.y += acc[1];
    g.z += acc[2];
    G[v] = g;
}

__global__ void k_diagonal_blocks(VtView vt, TetView tv, CacheView cache, real h2, real* __restrict__ out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vt.n_free) return;
    real blk[9];
    vertex_diag_block(vt, tv, cache, h2, v, blk);
    for (int i = 0; i < 9; ++i) out[9 * v + i] = blk[i];
}

__global__ void k_max_diagonal_entry(VtView vt, TetView tv, CacheView cache, const real* __restrict__ mass, real h2,
                                     double* __restrict__ out)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vt.n_free) return;
    real blk[9];
    vertex_diag_block(vt, tv, cache, h2, v, blk);
    const double m = static_cast<double>(mass[v]);
    double best = m + static_cast<double>(blk[0]);
    best = fmax(best, m + static_cast<double>(blk[4]));
    best = fmax(best, m + static_cast<double>(blk[8]));
    out[v] = best;
}

// --- inversion guard: first root in (0, 1] of the cubic det D_s(x + θ dx) ---------------------
__device__ __forceinline__ double cubic_eval(double c0, double c1, double c2, double c3, double t)
{
    return c0 + t * (c1 + t * (c2 + t * c3));
}

// Requires f(0) = c0 > 0 (the element is not inverted at θ = 0, the invariant the guard
// maintains). Splits [0, 1] at the critical points of the cubic so that f is monotone on every
// piece, and bisects the first piece with a sign change (64 steps, well below double
// resolution). Returns 2 when there is no root, and also when c0 <= 0: an element that is
// already inverted at θ = 0 cannot be rescued by shortening the step, so it does not bound it
// (the caller would otherwise deadlock the line search at α = 0).
__device__ double first_root_unit(double c0, double c1, double c2, double c3)
{
    if (!(c0 > 0.0)) return 2.0;
    double ts[4];
    int nt = 0;
    ts[nt++] = 0.0;
    // f'(t) = c1 + 2 c2 t + 3 c3 t^2
    const double qa = 3.0 * c3, qb = 2.0 * c2, qc = c1;
    if (qa != 0.0) {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc > 0.0) {
            const double sq = sqrt(disc);
            const double qq = -0.5 * (qb + (qb < 0.0 ? -sq : sq));
            const double r1 = qq / qa;
            const double r2 = (qq != 0.0) ? qc / qq : r1;
            if (r1 > 0.0 && r1 < 1.0) ts[nt++] = r1;
            if (r2 > 0.0 && r2 < 1.0 && r2 != r1) ts[nt++] = r2;
        }
    } else if (qb != 0.0) {
        const double r = -qc / qb;
        if (r > 0.0 && r < 1.0) ts[nt++] = r;
    }
    ts[nt++] = 1.0;
    // insertion sort of at most 4 entries (0 and 1 are already in place)
    for (int i = 1; i < nt; ++i) {
        const double v = ts[i];
        int j = i - 1;
        while (j >= 0 && ts[j] > v) {
            ts[j + 1] = ts[j];
            --j;
        }
        ts[j + 1] = v;
    }
    for (int i = 0; i + 1 < nt; ++i) {
        double lo = ts[i], hi = ts[i + 1];
        const double flo = cubic_eval(c0, c1, c2, c3, lo);
        if (flo <= 0.0) return lo;  // only reachable for lo > 0 (c0 > 0)
        const double fhi = cubic_eval(c0, c1, c2, c3, hi);
        if (fhi > 0.0) continue;
        for (int it = 0; it < 64; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (cubic_eval(c0, c1, c2, c3, mid) > 0.0) lo = mid;
            else hi = mid;
        }
        return hi;
    }
    return 2.0;
}

// Oriented cubic f(t) = sign(det D_m) det D_s(x + t dx) of tet e; false when the tet is not guarded
// (cloth, or not NH unless all_bodies) or none of its vertices moves.
__device__ bool inversion_cubic(const TetView& tv, const real3* __restrict__ x, const real3* __restrict__ dx, int all_bodies,
                                int e, double& c0, double& c1, double& c2, double& c3)
{
    const int model = tv.materials[tv.tet_material[e]].model;
    if (fem::is_cloth(model) || !(all_bodies || model == fem::kNH)) return false;
    const int4 t = tv.tet[e];
    const int id[4] = {t.x, t.y, t.z, t.w};
    double p[4][3], d[4][3];
    bool moving = false;
    for (int a = 0; a < 4; ++a) {
        const real3 xa = x[id[a]], da = dx[id[a]];
        p[a][0] = xa.x; p[a][1] = xa.y; p[a][2] = xa.z;
        d[a][0] = da.x; d[a][1] = da.y; d[a][2] = da.z;
        moving = moving || (da.x != real(0)) || (da.y != real(0)) || (da.z != real(0));
    }
    if (!moving) return false;
    double ec[3][3], fc[3][3];
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) {
            ec[c][r] = p[c + 1][r] - p[0][r];
            fc[c][r] = d[c + 1][r] - d[0][r];
        }
    double e1e2[3], f1e2[3], e1f2[3], f1f2[3];
    fem::v3_cross(ec[1], ec[2], e1e2);
    fem::v3_cross(fc[1], ec[2], f1e2);
    fem::v3_cross(ec[1], fc[2], e1f2);
    fem::v3_cross(fc[1], fc[2], f1f2);
    auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    c0 = dot(ec[0], e1e2);
    c1 = dot(fc[0], e1e2) + dot(ec[0], f1e2) + dot(ec[0], e1f2);
    c2 = dot(ec[0], f1f2) + dot(fc[0], f1e2) + dot(fc[0], e1f2);
    c3 = dot(fc[0], f1f2);
    if (fem::m3_det(tv.DmInv + 9 * e) < real(0)) { c0 = -c0; c1 = -c1; c2 = -c2; c3 = -c3; }
    return true;
}

// MS §10.4 end-state rule: f(theta) / min(f(0), f(1)) per tet (a large value when not tested).
__global__ void k_inversion_margin(TetView tv, const real3* __restrict__ x, const real3* __restrict__ dx,
                                   int all_bodies, double theta, double* __restrict__ out)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tv.n_tets) return;
    double result = 1e300;
    double c0, c1, c2, c3;
    if (inversion_cubic(tv, x, dx, all_bodies, e, c0, c1, c2, c3) && c0 > 0.0) {
        const double f1 = cubic_eval(c0, c1, c2, c3, 1.0);
        const double ref = (f1 > 0.0 && f1 < c0) ? f1 : c0;
        result = cubic_eval(c0, c1, c2, c3, theta) / ref;
    }
    out[e] = result;
}

// MS §6 guarded Dirichlet advance: first theta in (0, 1] with f(theta) = eta f(0), 2 when there is none.
__global__ void k_volume_loss_toi(TetView tv, const real3* __restrict__ x, const real3* __restrict__ dx, const real3* x_ref,
                                  int all_bodies, double eta, double* __restrict__ out)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tv.n_tets) return;
    double result = 2.0;
    double c0, c1, c2, c3;
    if (inversion_cubic(tv, x, dx, all_bodies, e, c0, c1, c2, c3) && c0 > 0.0) {
        double bound = eta * c0;
        if (x_ref != nullptr) {   // oriented det D_s at the reference state
            const int4 t = tv.tet[e];
            const int id[4] = {t.x, t.y, t.z, t.w};
            double q[3][3];
            for (int c = 0; c < 3; ++c) {
                q[c][0] = double(x_ref[id[c + 1]].x) - double(x_ref[id[0]].x);
                q[c][1] = double(x_ref[id[c + 1]].y) - double(x_ref[id[0]].y);
                q[c][2] = double(x_ref[id[c + 1]].z) - double(x_ref[id[0]].z);
            }
            double cr[3];
            fem::v3_cross(q[1], q[2], cr);
            double fr = q[0][0] * cr[0] + q[0][1] * cr[1] + q[0][2] * cr[2];
            if (fem::m3_det(tv.DmInv + 9 * e) < real(0)) fr = -fr;
            bound = eta * fr;
        }
        // first_root_unit needs a positive start value: a tet already at or below the bound does not constrain
        if (c0 - bound > 0.0) result = first_root_unit(c0 - bound, c1, c2, c3);
    }
    out[e] = result;
}

__global__ void k_inversion_toi(TetView tv, const real3* __restrict__ x, const real3* __restrict__ dx,
                                int all_bodies, double* __restrict__ out)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= tv.n_tets) return;
    // Shells cannot invert (MS §3.8) and have no fourth vertex: never guarded. Inversion is
    // J = det D_s / det D_m < 0, so the cubic is oriented by sign(det D_m) and f(0) > 0 always
    // means "not inverted" (first_root_unit's precondition).
    double result = 2.0;
    double c0, c1, c2, c3;
    if (inversion_cubic(tv, x, dx, all_bodies, e, c0, c1, c2, c3)) result = first_root_unit(c0, c1, c2, c3);
    out[e] = result;
}

__global__ void k_assemble_bsr(TetView tv, CacheView cache, real h2, const int* __restrict__ upper_slot,
                               const int* __restrict__ contrib_ptr, const int* __restrict__ contrib_tet,
                               const unsigned char* __restrict__ contrib_ab, const int* __restrict__ slot_mirror,
                               int n_upper, real* __restrict__ blocks)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_upper) return;
    real acc[9];
    for (int i = 0; i < 9; ++i) acc[i] = real(0);
    const int k0 = contrib_ptr[u], k1 = contrib_ptr[u + 1];
    for (int k = k0; k < k1; ++k) {
        const int e = contrib_tet[k];
        const int ab = contrib_ab[k];
        const int a = ab >> 2, b = ab & 3;
        const real* DmInv = tv.DmInv + 9 * e;
        const real* V = cache.V + 9 * e;
        real aa[3], ab_[3], ba[3], bb[3];
        fem::shape_gradient(DmInv, a, aa);
        fem::shape_gradient(DmInv, b, ab_);
        fem::m3_mulvec_t(V, aa, ba);
        fem::m3_mulvec_t(V, ab_, bb);
        real blk[9];
        fem::elem_block(cache.U + 9 * e, cache.eig + 9 * e, cache.q + 9 * e, ba, bb, h2 * tv.volume[e], blk);
        for (int i = 0; i < 9; ++i) acc[i] += blk[i];
    }
    const int slot = upper_slot[u];
    const int mirror = slot_mirror[slot];
    if (mirror == slot) symmetrize3(acc);  // diagonal slot: every contribution has a == b
    real* dst = blocks + 9 * slot;
    for (int i = 0; i < 9; ++i) dst[i] += acc[i];
    if (mirror != slot) {
        real* dstT = blocks + 9 * mirror;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) dstT[3 * c + r] += acc[3 * r + c];
    }
}

// ---- two-phase assembly (IS §12.3 item 11) --------------------------------------------------
// Index of the unordered local pair {i, j}, i <= j, among the ten of a tet:
// (0,0)0 (0,1)1 (0,2)2 (0,3)3 (1,1)4 (1,2)5 (1,3)6 (2,2)7 (2,3)8 (3,3)9.
__device__ __forceinline__ int pair_index(int i, int j) { return i * 4 - (i * (i - 1)) / 2 + (j - i); }

// Phase A: one thread per tet writes its ten blocks (720 B, contiguous per tet) into the
// staging buffer, each in the orientation its contribution uses (row = the vertex with the
// smaller global index), so that the gather below reads it without a transpose. The block is
// the one the direct kernel forms for that contribution up to the compiler's multiply-add
// contraction across the stored (hence rounded) value: one ulp, see test_block_assembly.
__global__ void k_element_blocks(TetView tv, CacheView cache, real h2, int t0, int t1, real* __restrict__ stage)
{
    const int e = t0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t1) return;
    const int4 g = tv.tet[e];
    const int gi[4] = {g.x, g.y, g.z, g.w};
    const real* DmInv = tv.DmInv + 9 * e;
    const real* V = cache.V + 9 * e;
    real b[4][3];
    for (int a = 0; a < 4; ++a) {
        real aa[3];
        fem::shape_gradient(DmInv, a, aa);
        fem::m3_mulvec_t(V, aa, b[a]);
    }
    const real scale = h2 * tv.volume[e];
    real* out = stage + 90 * static_cast<std::size_t>(e - t0);
    for (int i = 0; i < 4; ++i)
        for (int j = i; j < 4; ++j) {
            int a = i, c = j;
            if (gi[j] < gi[i]) { a = j; c = i; }
            fem::elem_block(cache.U + 9 * e, cache.eig + 9 * e, cache.q + 9 * e, b[a], b[c], scale,
                            out + 9 * pair_index(i, j));
        }
}

// Phase B: one thread per upper slot sums the staged blocks of its contributions whose tet lies
// in the chunk (the contribution list of a slot is sorted by tet) and adds the sum to the slot
// and its mirror, exactly as the direct kernel does.
__global__ void k_gather_blocks(const real* __restrict__ stage, int t0, int t1, const int* __restrict__ upper_slot,
                                const int* __restrict__ contrib_ptr, const int* __restrict__ contrib_tet,
                                const unsigned char* __restrict__ contrib_ab, const int* __restrict__ slot_mirror,
                                int n_upper, bool whole, real* __restrict__ blocks)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_upper) return;
    const int k0 = contrib_ptr[u], k1 = contrib_ptr[u + 1];
    int kb = k0, ke = k1;
    if (!whole) {   // chunked: the contribution list of a slot is sorted by tet
        while (kb < k1 && contrib_tet[kb] < t0) ++kb;
        ke = kb;
        while (ke < k1 && contrib_tet[ke] < t1) ++ke;
    }
    if (kb == ke) return;
    real acc[9];
    for (int i = 0; i < 9; ++i) acc[i] = real(0);
    for (int k = kb; k < ke; ++k) {
        const int e = contrib_tet[k];
        const int ab = contrib_ab[k];
        const int a = ab >> 2, c = ab & 3;
        const int i = a < c ? a : c, j = a < c ? c : a;
        const real* blk = stage + 90 * static_cast<std::size_t>(e - t0) + 9 * pair_index(i, j);
        for (int q = 0; q < 9; ++q) acc[q] += blk[q];
    }
    const int slot = upper_slot[u];
    const int mirror = slot_mirror[slot];
    if (mirror == slot) symmetrize3(acc);  // diagonal slot: every contribution has a == b
    real* dst = blocks + 9 * slot;
    for (int i = 0; i < 9; ++i) dst[i] += acc[i];
    if (mirror != slot) {
        real* dstT = blocks + 9 * mirror;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) dstT[3 * c + r] += acc[3 * r + c];
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------------------------
void fem_element_cache(const SceneBuffers& sb, const real3* x, ElementCache& cache, bool need_hessian)
{
    const int n = sb.n_tets;
    if (n <= 0) return;
    if (cache.energy.size() < static_cast<std::size_t>(n)) cache.resize(n);
    k_element_cache<<<grid_for(n), kBlock>>>(tet_view(sb), x, cache.U.data(), cache.V.data(), cache.eig.data(),
                                             cache.q.data(), cache.P.data(), cache.energy.data(), need_hessian);
    CS_CUDA_KERNEL_CHECK();
}

void fem_elastic_energy_to(const SceneBuffers& sb, const real3* x, DeviceArray<real>& scratch, double* d_out)
{
    const int n = sb.n_tets;
    if (n <= 0) {
        double* d = scratch_doubles(scratch, 1);
        k_reduce_level2<OpSum><<<1, kBlock>>>(d, 0, d_out);   // folds nothing: 0
        CS_CUDA_KERNEL_CHECK();
        return;
    }
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_element_energy<<<grid_for(n), kBlock>>>(tet_view(sb), x, d);
    CS_CUDA_KERNEL_CHECK();
    k_reduce_level1<OpSum><<<static_cast<unsigned>(nb), kBlock>>>(d, n, d + n);
    CS_CUDA_KERNEL_CHECK();
    k_reduce_level2<OpSum><<<1, kBlock>>>(d + n, nb, d_out);
    CS_CUDA_KERNEL_CHECK();
}

real fem_elastic_energy(const SceneBuffers& sb, const real3* x, DeviceArray<real>& scratch)
{
    const int n = sb.n_tets;
    if (n <= 0) return real(0);
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_element_energy<<<grid_for(n), kBlock>>>(tet_view(sb), x, d);
    CS_CUDA_KERNEL_CHECK();
    return static_cast<real>(reduce_doubles<OpSum>(d, n, d + n));
}

void fem_add_gradient(const SceneBuffers& sb, const ElementCache& cache, real h2, real3* G)
{
    const int n = sb.n_free;
    if (n <= 0) return;
    k_add_gradient<<<grid_for(n), kBlock>>>(vt_view(sb), tet_view(sb), cache.P.data(), h2, G);
    CS_CUDA_KERNEL_CHECK();
}

void fem_diagonal_blocks(const SceneBuffers& sb, const ElementCache& cache, real h2, real* out)
{
    const int n = sb.n_free;
    if (n <= 0) return;
    k_diagonal_blocks<<<grid_for(n), kBlock>>>(vt_view(sb), tet_view(sb), cache_view(cache), h2, out);
    CS_CUDA_KERNEL_CHECK();
}

real fem_max_diagonal(const SceneBuffers& sb, const ElementCache& cache, real h2, DeviceArray<real>& scratch)
{
    const int n = sb.n_free;
    if (n <= 0) return real(0);
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_max_diagonal_entry<<<grid_for(n), kBlock>>>(vt_view(sb), tet_view(sb), cache_view(cache), sb.mass.data(), h2, d);
    CS_CUDA_KERNEL_CHECK();
    return static_cast<real>(reduce_doubles<OpMax>(d, n, d + n));
}

real fem_volume_loss_toi(const SceneBuffers& sb, const real3* x, const real3* dx, double eta, DeviceArray<real>& scratch,
                         bool all_bodies, const real3* x_ref)
{
    const int n = sb.n_tets;
    if (n <= 0) return real(1);
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_volume_loss_toi<<<grid_for(n), kBlock>>>(tet_view(sb), x, dx, x_ref, all_bodies ? 1 : 0, eta, d);
    CS_CUDA_KERNEL_CHECK();
    const double m = reduce_doubles<OpMin>(d, n, d + n);
    return (m < 1.0) ? static_cast<real>(m) : real(1);
}

double fem_inversion_margin(const SceneBuffers& sb, const real3* x, const real3* dx, double theta,
                            DeviceArray<real>& scratch, bool all_bodies)
{
    const int n = sb.n_tets;
    if (n <= 0) return 1e300;
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_inversion_margin<<<grid_for(n), kBlock>>>(tet_view(sb), x, dx, all_bodies ? 1 : 0, theta, d);
    CS_CUDA_KERNEL_CHECK();
    return reduce_doubles<OpMin>(d, n, d + n);
}

real fem_inversion_toi(const SceneBuffers& sb, const real3* x, const real3* dx, DeviceArray<real>& scratch,
                       bool all_bodies)
{
    const int n = sb.n_tets;
    if (n <= 0) return real(1);
    const int nb = reduce_blocks_for(n);
    double* d = scratch_doubles(scratch, static_cast<std::size_t>(n) + nb + 1);
    k_inversion_toi<<<grid_for(n), kBlock>>>(tet_view(sb), x, dx, all_bodies ? 1 : 0, d);
    CS_CUDA_KERNEL_CHECK();
    const double m = reduce_doubles<OpMin>(d, n, d + n);
    return (m <= 1.0) ? static_cast<real>(0.9 * m) : real(1);
}

void fem_assemble_bsr_direct(const SceneBuffers& sb, const ElementCache& cache, real h2, BsrMatrix& A)
{
    const BsrPattern* p = A.pattern;
    if (p == nullptr || p->n_upper <= 0) return;
    const int n = p->n_upper;
    k_assemble_bsr<<<grid_for(n), kBlock>>>(tet_view(sb), cache_view(cache), h2, p->upper_slot.data(),
                                            p->contrib_ptr.data(), p->contrib_tet.data(), p->contrib_ab.data(),
                                            p->slot_mirror.data(), n, A.blocks.data());
    CS_CUDA_KERNEL_CHECK();
}

// Staging budget of the two-phase assembly: 720 B per tet, so 256 MB holds 373 k tets; larger
// meshes are assembled in chunks of that size (the per-slot sum is then folded into the matrix
// once per chunk, which changes the last bits against a single pass but stays deterministic).
// The budget is deliberately small: the compressor scene already sits near the card's memory.
int fem_assembly_chunk_tets(int n_tets)
{
    constexpr std::size_t kBudget = std::size_t(256) << 20;
    const int per_chunk = static_cast<int>(kBudget / (90 * sizeof(real)));
    return n_tets < per_chunk ? (n_tets < 1 ? 1 : n_tets) : per_chunk;
}

// Two-phase assembly (IS §12.3 item 11): the direct kernel re-derived every contribution's
// block from 360 B of per-tet cache, 32 different tets per warp; here each tet forms its ten
// blocks once with coalesced reads of its cache, and the gather per slot reads 72 B per
// contribution. Same per-contribution arithmetic, same summation order within a chunk.
namespace {
void assemble_two_phase(const SceneBuffers& sb, const ElementCache& cache, real h2, BsrMatrix& A, float* ms_a,
                        float* ms_b, int chunk_override = 0)
{
    if (ms_a != nullptr) *ms_a = 0.0f;
    if (ms_b != nullptr) *ms_b = 0.0f;
    const BsrPattern* p = A.pattern;
    if (p == nullptr || p->n_upper <= 0 || sb.n_tets <= 0) return;
    const int n_upper = p->n_upper;
    const int chunk = chunk_override > 0 ? std::min(chunk_override, sb.n_tets) : fem_assembly_chunk_tets(sb.n_tets);
    A.assembly_stage.loose_resize(90 * static_cast<std::size_t>(chunk));
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
    const bool timed = ms_a != nullptr && ms_b != nullptr;
    if (timed) {
        CS_CUDA_CHECK(cudaEventCreate(&e0)); CS_CUDA_CHECK(cudaEventCreate(&e1)); CS_CUDA_CHECK(cudaEventCreate(&e2));
    }
    for (int t0 = 0; t0 < sb.n_tets; t0 += chunk) {
        const int t1 = (t0 + chunk < sb.n_tets) ? t0 + chunk : sb.n_tets;
        if (timed) CS_CUDA_CHECK(cudaEventRecord(e0));
        k_element_blocks<<<grid_for(t1 - t0), kBlock>>>(tet_view(sb), cache_view(cache), h2, t0, t1,
                                                        A.assembly_stage.data());
        CS_CUDA_KERNEL_CHECK();
        if (timed) CS_CUDA_CHECK(cudaEventRecord(e1));
        k_gather_blocks<<<grid_for(n_upper), kBlock>>>(A.assembly_stage.data(), t0, t1, p->upper_slot.data(),
                                                       p->contrib_ptr.data(), p->contrib_tet.data(),
                                                       p->contrib_ab.data(), p->slot_mirror.data(), n_upper,
                                                       /*whole=*/t0 == 0 && t1 == sb.n_tets, A.blocks.data());
        CS_CUDA_KERNEL_CHECK();
        if (timed) {
            CS_CUDA_CHECK(cudaEventRecord(e2));
            CS_CUDA_CHECK(cudaEventSynchronize(e2));
            float a = 0.0f, b = 0.0f;
            CS_CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
            CS_CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
            *ms_a += a; *ms_b += b;
        }
    }
    if (timed) { cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); }
}
}  // namespace

void fem_assemble_bsr(const SceneBuffers& sb, const ElementCache& cache, real h2, BsrMatrix& A)
{
    assemble_two_phase(sb, cache, h2, A, nullptr, nullptr);
}

void fem_assemble_bsr_timed(const SceneBuffers& sb, const ElementCache& cache, real h2, BsrMatrix& A, float* ms_a,
                            float* ms_b)
{
    assemble_two_phase(sb, cache, h2, A, ms_a, ms_b);
}

void fem_assemble_bsr_chunked(const SceneBuffers& sb, const ElementCache& cache, real h2, BsrMatrix& A, int chunk_tets)
{
    assemble_two_phase(sb, cache, h2, A, nullptr, nullptr, chunk_tets);
}

}  // namespace cs
