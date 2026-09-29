// Contact stage (doc/al-ipc-implementation-spec.md §6.2, §6.3, §6.5, §6.8, §6.9, §6.10;
// math spec §4, §5, §8, §9, §10, §13). One translation unit: the pair machinery, the CCD stage
// and the semi-implicit friction all share the same SoA layout and device helpers.
//
// Determinism: the pair arrays are always sorted by key; every per-vertex accumulation is a
// gather in CSR order; the only atomics are integer minima on non-negative double bit patterns
// (order-independent) and the hit-compaction counter, whose order the key sort erases.
#include "contact/contact_system.cuh"
#include "core/cuda_check.h"
#include "core/log.h"
#include "core/timer.h"
#include "ccd/distance_gradient.cuh"
#include "ccd/plane_ccd.cuh"
#include "ccd/sphere_ccd.cuh"
#include "ccd/accd.cuh"
#include "linsys/reduce.cuh"
#include "contact/active_set.cuh"
#ifdef CCCL_VERSION_GREATER_EQUAL_13_0
#include <cccl/thrust/device_ptr.h>
#include <cccl/thrust/scan.h>
#include <cccl/thrust/execution_policy.h>
#else
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs {

namespace {

constexpr int kBlock = 256;
inline unsigned grid_for(int n) { return unsigned(std::max(1, (n + kBlock - 1) / kBlock)); }

enum PairType : int { kPT = 0, kEE = 1, kPH = 2, kPS = 3 };   // kPS: vertex vs analytic sphere (MS §4.1)
using key_t = unsigned long long;

CUDA_INLINE_CALLABLE key_t make_key(int type, int a, int b) {
    return (key_t(unsigned(type)) << 60) | (key_t(unsigned(a)) << 30) | key_t(unsigned(b));
}
inline int key_type(key_t k) { return int(k >> 60); }

// MS §16 / M12: edges within eps of parallel. |ea x eb|^2 < eps |ea|^2 |eb|^2.
CUDA_INLINE_CALLABLE bool edges_parallel(const real3& a0, const real3& a1, const real3& b0, const real3& b1, real eps) {
    const real3 ea = a1 - a0, eb = b1 - b0;
    const real3 cr = cross(ea, eb);
    return dot(cr, cr) < eps * dot(ea, ea) * dot(eb, eb);
}

// Non-negative doubles have a monotone unsigned bit pattern, so an integer atomicMin on the
// bits is an order-independent minimum (implementation spec §1).
CUDA_INLINE_CALLABLE unsigned long long d2bits(double v) {
#if defined(__CUDA_ARCH__)
    return (unsigned long long)__double_as_longlong(v);
#else
    unsigned long long b;
    __builtin_memcpy(&b, &v, sizeof(b));
    return b;
#endif
}
CUDA_INLINE_CALLABLE double bits2d(unsigned long long b) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double((long long)b);
#else
    double v;
    __builtin_memcpy(&v, &b, sizeof(v));
    return v;
#endif
}

DEVICE_INLINE_CALLABLE bool key_member(const key_t* keys, int n, key_t k) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const key_t v = keys[mid];
        if (v == k) return true;
        if (v < k) lo = mid + 1; else hi = mid - 1;
    }
    return false;
}

// Swept AABB with directed rounding (implementation spec §3.2): every double point of the
// swept primitive lies inside the returned float box, so the broad phase cannot lose a pair.
CUDA_INLINE_CALLABLE culbvh::Bound<float> swept_box(const real3& a, const real3& b, real pad) {
    const double lo_x = fmin(double(a.x), double(b.x)) - double(pad);
    const double lo_y = fmin(double(a.y), double(b.y)) - double(pad);
    const double lo_z = fmin(double(a.z), double(b.z)) - double(pad);
    const double hi_x = fmax(double(a.x), double(b.x)) + double(pad);
    const double hi_y = fmax(double(a.y), double(b.y)) + double(pad);
    const double hi_z = fmax(double(a.z), double(b.z)) + double(pad);
    culbvh::Bound<float> bx;
#if defined(__CUDA_ARCH__)
    bx.min = make_float3(__double2float_rd(lo_x), __double2float_rd(lo_y), __double2float_rd(lo_z));
    bx.max = make_float3(__double2float_ru(hi_x), __double2float_ru(hi_y), __double2float_ru(hi_z));
#else
    bx.min = make_float3(nextafterf(float(lo_x), -1e30f), nextafterf(float(lo_y), -1e30f), nextafterf(float(lo_z), -1e30f));
    bx.max = make_float3(nextafterf(float(hi_x), 1e30f), nextafterf(float(hi_y), 1e30f), nextafterf(float(hi_z), 1e30f));
#endif
    return bx;
}

// The body and contact-table rules alone (MS §4.1). Prescribed-only pairs pass this test so
// the CCD can count their hits (IS §5.x diagnostic); they are never admitted as candidates.
CUDA_INLINE_CALLABLE bool admissible_by_scene(const PrimInfo& a, const PrimInfo& b, const unsigned char* table, int n_cid) {
    if (a.body_id() == b.body_id() && !a.self_collision()) return false;
    return table[a.contact_id() * n_cid + b.contact_id()] != 0;
}
CUDA_INLINE_CALLABLE bool admissible(const PrimInfo& a, const PrimInfo& b, const unsigned char* table, int n_cid) {
    if (a.all_prescribed() && b.all_prescribed()) return false;          // no free DOF can resolve it (MS §4.1)
    return admissible_by_scene(a, b, table, n_cid);
}

// =======================================================================================
// Normal contact: linearization, slack, gradient, SpMV, multipliers
// =======================================================================================

__global__ void k_linearize(int n, const unsigned char* __restrict__ type, const int4* __restrict__ idx,
                            const real* __restrict__ xi, real d_hat, const real3* __restrict__ x,
                            const real3* __restrict__ plane_o, const real3* __restrict__ plane_n,
                            const real3* __restrict__ sphere_c, const real* __restrict__ sphere_r1,
                            const unsigned char* __restrict__ sphere_inv, bool drop_parallel, real par_eps,
                            real* __restrict__ gap, real3* __restrict__ grad4, int* __restrict__ n_degenerate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    const real delta = d_hat + xi[i];
    if (type[i] == kPH || type[i] == kPS) {
        real d;
        // MS §12.3: a sphere pair is linearized against the end-of-step radius r1.
        const real3 g = (type[i] == kPH)
                            ? plane_distance_gradient(x[id.x], plane_o[id.y], plane_n[id.y], d)
                            : sphere_distance_gradient(x[id.x], sphere_c[id.y], sphere_r1[id.y], sphere_inv[id.y] != 0, d);
        gap[i] = d - delta;
        grad4[4 * i + 0] = g;
        grad4[4 * i + 1] = make_real3(real(0), real(0), real(0));
        grad4[4 * i + 2] = make_real3(real(0), real(0), real(0));
        grad4[4 * i + 3] = make_real3(real(0), real(0), real(0));
        return;
    }
    // Pair-local frame (precision policy §11.2): differences are formed against the pair's own
    // centroid so a millimetre gap is resolved against the primitive size, not the scene size.
    const real3 p0 = x[id.x], p1 = x[id.y], p2 = x[id.z], p3 = x[id.w];
    if (drop_parallel && type[i] == kEE && edges_parallel(p0, p1, p2, p3, par_eps)) {
        // Inert: a zero gradient contributes nothing to energy, gradient or Hessian; a gap of
        // +delta drives the multiplier to zero so the decay removes the pair (MS §8).
        gap[i] = delta;
        grad4[4 * i + 0] = grad4[4 * i + 1] = grad4[4 * i + 2] = grad4[4 * i + 3] =
            make_real3(real(0), real(0), real(0));
        return;
    }
    const real3 c = (p0 + p1 + p2 + p3) * real(0.25);
    real d;
    real3 g[4], nrm;
    bool degenerate = false;
    if (type[i] == kPT) pt_distance_gradient(p0 - c, p1 - c, p2 - c, p3 - c, d, g, nrm, degenerate);
    else                ee_distance_gradient(p0 - c, p1 - c, p2 - c, p3 - c, d, g, nrm, degenerate);
    if (degenerate) atomicAdd(n_degenerate, 1);
    gap[i] = d - delta;
    grad4[4 * i + 0] = g[0];
    grad4[4 * i + 1] = g[1];
    grad4[4 * i + 2] = g[2];
    grad4[4 * i + 3] = g[3];
}

// c_i(x̂) = base_i + Σ_s ∇d_s · (x̂_s − x_s)  (displacement form, MS §5.1)
CUDA_INLINE_CALLABLE real constraint_value(int i, const int4* idx, const real* base, const real3* grad4,
                                           const real3* x_hat, const real3* x_anchor) {
    const int4 id = idx[i];
    real c = base[i];
    c += dot(grad4[4 * i + 0], x_hat[id.x] - x_anchor[id.x]);
    if (id.z >= 0) {
        c += dot(grad4[4 * i + 1], x_hat[id.y] - x_anchor[id.y]);
        c += dot(grad4[4 * i + 2], x_hat[id.z] - x_anchor[id.z]);
        c += dot(grad4[4 * i + 3], x_hat[id.w] - x_anchor[id.w]);
    }
    return c;
}

__global__ void k_slack(int n, const int4* __restrict__ idx, const real* __restrict__ gap,
                        const real3* __restrict__ grad4, const real* __restrict__ lambda, real mu,
                        const real3* __restrict__ x_hat, const real3* __restrict__ x_anchor,
                        real* __restrict__ slack, real* __restrict__ gap_folded) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real c = constraint_value(i, idx, gap, grad4, x_hat, x_anchor);
    const real w = c - lambda[i] / mu;
    const real s = w > real(0) ? w : real(0);
    slack[i] = s;
    gap_folded[i] = gap[i] - s - lambda[i] / mu;
}

__global__ void k_pair_scalar_grad(int n, const int4* __restrict__ idx, const real* __restrict__ gap_folded,
                                   const real3* __restrict__ grad4, const real* __restrict__ coeff,
                                   const real3* __restrict__ x_hat, const real3* __restrict__ x_anchor,
                                   real* __restrict__ q) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = coeff[i] * constraint_value(i, idx, gap_folded, grad4, x_hat, x_anchor);
}

__global__ void k_pair_energy(int n, const int4* __restrict__ idx, const real* __restrict__ gap_folded,
                              const real3* __restrict__ grad4, const real* __restrict__ coeff,
                              const real3* __restrict__ x_hat, const real3* __restrict__ x_anchor,
                              real* __restrict__ e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real v = constraint_value(i, idx, gap_folded, grad4, x_hat, x_anchor);
    e[i] = real(0.5) * coeff[i] * v * v;
}

// σ_i = c_i Σ_{free slots} ∇d_s · x_in[s]   (pass 1 of the rank-one product, §6.5)
__global__ void k_pair_sigma(const int* __restrict__ skip, int n, int n_free, const int4* __restrict__ idx, const real3* __restrict__ grad4,
                             const real* __restrict__ coeff, const real3* __restrict__ x_in, real* __restrict__ sigma) {
    if (skip != nullptr && *skip) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    real s = real(0);
    if (id.x < n_free) s += dot(grad4[4 * i + 0], x_in[id.x]);
    if (id.z >= 0) {
        if (id.y < n_free) s += dot(grad4[4 * i + 1], x_in[id.y]);
        if (id.z < n_free) s += dot(grad4[4 * i + 2], x_in[id.z]);
        if (id.w < n_free) s += dot(grad4[4 * i + 3], x_in[id.w]);
    }
    sigma[i] = coeff[i] * s;
}

// y_v += Σ_{k ∈ row v} scalar[pair(k)] · vp_grad[k]   (pass 2; gradient and SpMV share it)
__global__ void k_vertex_gather(const int* __restrict__ skip, int n_free, const int* __restrict__ vp_ptr, const int* __restrict__ vp_pair,
                                const real3* __restrict__ vp_grad, const real* __restrict__ scalar,
                                real3* __restrict__ y) {
    if (skip != nullptr && *skip) return;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    const int b = vp_ptr[v], e = vp_ptr[v + 1];
    if (b == e) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = b; k < e; ++k) {
        const double s = double(scalar[vp_pair[k]]);
        const real3 g = vp_grad[k];
        ax += s * double(g.x);
        ay += s * double(g.y);
        az += s * double(g.z);
    }
    y[v] = y[v] + make_real3(real(ax), real(ay), real(az));
}

__global__ void k_contact_diag(int n_free, const int* __restrict__ vp_ptr, const int* __restrict__ vp_pair,
                               const real3* __restrict__ vp_grad, const real* __restrict__ coeff,
                               real* __restrict__ diag) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    double m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
    for (int k = vp_ptr[v]; k < vp_ptr[v + 1]; ++k) {
        const double c = double(coeff[vp_pair[k]]);
        const real3 g = vp_grad[k];
        const double gx = double(g.x), gy = double(g.y), gz = double(g.z);
        m00 += c * gx * gx; m01 += c * gx * gy; m02 += c * gx * gz;
        m11 += c * gy * gy; m12 += c * gy * gz; m22 += c * gz * gz;
    }
    real* d = diag + 9 * v;
    d[0] = real(m00); d[1] = real(m01); d[2] = real(m02);
    d[3] = real(m01); d[4] = real(m11); d[5] = real(m12);
    d[6] = real(m02); d[7] = real(m12); d[8] = real(m22);
}

__global__ void k_zero_diag(int n_free, real* __restrict__ diag) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    for (int j = 0; j < 9; ++j) diag[9 * v + j] = real(0);
}

__global__ void k_coeff(int n, const int* __restrict__ ninact, real gamma_factor, real mu, real* __restrict__ coeff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    coeff[i] = pow(gamma_factor, real(ninact[i])) * mu;
}

// MS §8: λ ← max(0, λ − μ c), n ← 0 when active else n + 1.
__global__ void k_update_lambda(int n, const int4* __restrict__ idx, const real* __restrict__ gap,
                                const real3* __restrict__ grad4, real mu, const real3* __restrict__ x_hat,
                                const real3* __restrict__ x_anchor, real* __restrict__ lambda,
                                int* __restrict__ ninact) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real c = constraint_value(i, idx, gap, grad4, x_hat, x_anchor);
    const real t = lambda[i] - mu * c;
    if (t > real(0)) { lambda[i] = t; ninact[i] = 0; }
    else             { lambda[i] = real(0); ninact[i] += 1; }
}

// ---- incidence CSR over free vertices --------------------------------------------------
__global__ void k_count_incidence(int n, int n_free, const int4* __restrict__ idx, int* __restrict__ count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    if (id.x < n_free) atomicAdd(&count[id.x], 1);
    if (id.z >= 0) {
        if (id.y < n_free) atomicAdd(&count[id.y], 1);
        if (id.z < n_free) atomicAdd(&count[id.z], 1);
        if (id.w < n_free) atomicAdd(&count[id.w], 1);
    }
}

__global__ void k_fill_incidence(int n, int n_free, const int4* __restrict__ idx, const int* __restrict__ vp_ptr,
                                 int* __restrict__ fill, int* __restrict__ vp_pair,
                                 unsigned char* __restrict__ vp_slot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    const int vs[4] = {id.x, id.y, id.z, id.w};
    const int ns = id.z >= 0 ? 4 : 1;
    for (int s = 0; s < ns; ++s) {
        const int v = vs[s];
        if (v < n_free) {
            const int pos = vp_ptr[v] + atomicAdd(&fill[v], 1);
            vp_pair[pos] = i;
            vp_slot[pos] = (unsigned char)s;
        }
    }
}

// Per-row insertion sort by (pair, slot): the atomic fill above is order-nondeterministic, this
// makes the gather order canonical (rows are short).
__global__ void k_sort_rows(int n_free, const int* __restrict__ vp_ptr, int* __restrict__ vp_pair,
                            unsigned char* __restrict__ vp_slot) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    const int b = vp_ptr[v], e = vp_ptr[v + 1];
    for (int i = b + 1; i < e; ++i) {
        const int p = vp_pair[i];
        const unsigned char s = vp_slot[i];
        int j = i - 1;
        while (j >= b && (vp_pair[j] > p || (vp_pair[j] == p && vp_slot[j] > s))) {
            vp_pair[j + 1] = vp_pair[j];
            vp_slot[j + 1] = vp_slot[j];
            --j;
        }
        vp_pair[j + 1] = p;
        vp_slot[j + 1] = s;
    }
}

__global__ void k_pack_status(const unsigned long long* __restrict__ alpha_bits,
                              const unsigned long long* __restrict__ count,
                              const unsigned long long* __restrict__ pres_count, unsigned long long* __restrict__ out) {
    out[0] = *alpha_bits;
    out[1] = *count;
    out[2] = *pres_count;
}

__global__ void k_vp_grad(const int* __restrict__ d_n_inc, int bound, const int* __restrict__ vp_pair,
                          const unsigned char* __restrict__ vp_slot, const real3* __restrict__ grad4,
                          real3* __restrict__ vp_grad) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_inc = min(*d_n_inc, bound);   // the device-resident incidence total
    if (k >= n_inc) return;
    vp_grad[k] = grad4[4 * vp_pair[k] + vp_slot[k]];
}

// =======================================================================================
// Friction (MS §13): semi-implicit, lagged normal force, frozen basis and weights
// =======================================================================================

// f0, f1/‖u‖ and the f2 term of IPC's C1-clamped friction (MS §13).
CUDA_INLINE_CALLABLE real fr_f0(real y, real eps) {
    if (y >= eps) return y;
    return -(y * y * y) / (real(3) * eps * eps) + (y * y) / eps + eps / real(3);
}
CUDA_INLINE_CALLABLE real fr_f1_over_y(real y, real eps) {
    if (y >= eps) return real(1) / y;
    return (real(2) * eps - y) / (eps * eps);
}

// Friction snapshot: F̂_i = max(0, λ_i − μ (d_i(x^t) − δ_i)); the pair keeps its frozen
// closest-point weights and tangent basis. `keep` marks the pairs that carry a positive force.
__global__ void k_friction_snapshot(int n, const unsigned char* __restrict__ type, const int4* __restrict__ idx,
                                    const real* __restrict__ xi, real d_hat, real mu,
                                    const real* __restrict__ lambda, const real3* __restrict__ x,
                                    const real3* __restrict__ plane_o, const real3* __restrict__ plane_n,
                                    const real3* __restrict__ sphere_c, const real* __restrict__ sphere_r1,
                                    const unsigned char* __restrict__ sphere_inv, const int* __restrict__ sphere_cid,
                                    const int* __restrict__ contact_id, const int* __restrict__ plane_cid,
                                    const real* __restrict__ table_friction, int n_cid, bool use_lambda,
                                    int* __restrict__ keep, real* __restrict__ force, real4* __restrict__ weight,
                                    real3* __restrict__ basis) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    const real delta = d_hat + xi[i];
    real d = real(0);
    real3 nrm = make_real3(real(0), real(1), real(0));
    real4 w = make_real4(real(1), real(0), real(0), real(0));
    int cid_a = 0, cid_b = 0;
    if (type[i] == kPH) {
        (void)plane_distance_gradient(x[id.x], plane_o[id.y], plane_n[id.y], d);
        nrm = plane_n[id.y];
        w = make_real4(real(1), real(0), real(0), real(0));
        cid_a = contact_id[id.x];
        cid_b = plane_cid[id.y];
    } else if (type[i] == kPS) {
        // MS §12.3: the radius change is normal to the wall, so no collider-velocity term.
        nrm = sphere_distance_gradient(x[id.x], sphere_c[id.y], sphere_r1[id.y], sphere_inv[id.y] != 0, d);
        w = make_real4(real(1), real(0), real(0), real(0));
        cid_a = contact_id[id.x];
        cid_b = sphere_cid[id.y];
    } else {
        const real3 p0 = x[id.x], p1 = x[id.y], p2 = x[id.z], p3 = x[id.w];
        const real3 c = (p0 + p1 + p2 + p3) * real(0.25);
        real3 g[4];
        bool degenerate = false;
        if (type[i] == kPT) pt_distance_gradient(p0 - c, p1 - c, p2 - c, p3 - c, d, g, nrm, degenerate);
        else                ee_distance_gradient(p0 - c, p1 - c, p2 - c, p3 - c, d, g, nrm, degenerate);
        // The distance gradient of both types is w_s n̂ with Σ w_s = 0 and w_0 = 1: recover the
        // relative-motion weights directly from it (MS §13 uses the same closest-point weights).
        w = make_real4(dot(g[0], nrm), dot(g[1], nrm), dot(g[2], nrm), dot(g[3], nrm));
        cid_a = contact_id[id.x];
        cid_b = contact_id[id.z >= 0 ? id.z : id.x];
    }
    const real c_lin = d - delta;
    const real F = use_lambda ? lambda[i] : (lambda[i] - mu * c_lin);
    const real mu_f = table_friction[cid_a * n_cid + cid_b];
    const real fval = (F > real(0) && mu_f > real(0)) ? mu_f * F : real(0);
    keep[i] = fval > real(0) ? 1 : 0;
    force[i] = fval;
    weight[i] = w;
    // Orthonormal tangent basis of the contact plane.
    real3 e1 = cross(nrm, make_real3(real(0), real(0), real(1)));
    if (dot(e1, e1) < real(1e-12)) e1 = cross(nrm, make_real3(real(0), real(1), real(0)));
    e1 = normalize(e1);
    const real3 e2 = normalize(cross(nrm, e1));
    basis[2 * i + 0] = e1;
    basis[2 * i + 1] = e2;
}

// u_i = T^T Σ_s w_s (x̂_s − x^t_s) ∈ R²
CUDA_INLINE_CALLABLE void friction_u(int i, const int4* idx, const real4* weight, const real3* basis,
                                     const real3* x_hat, const real3* x_prev, real& u0, real& u1) {
    const int4 id = idx[i];
    const real4 w = weight[i];
    real3 rel = w.x * (x_hat[id.x] - x_prev[id.x]);
    if (id.z >= 0) {
        rel = rel + w.y * (x_hat[id.y] - x_prev[id.y]);
        rel = rel + w.z * (x_hat[id.z] - x_prev[id.z]);
        rel = rel + w.w * (x_hat[id.w] - x_prev[id.w]);
    }
    u0 = dot(basis[2 * i + 0], rel);
    u1 = dot(basis[2 * i + 1], rel);
}

__global__ void k_friction_energy(int n, const int4* __restrict__ idx, const real4* __restrict__ weight,
                                  const real3* __restrict__ basis, const real* __restrict__ force, real eps,
                                  const real3* __restrict__ x_hat, const real3* __restrict__ x_prev,
                                  real* __restrict__ e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real u0, u1;
    friction_u(i, idx, weight, basis, x_hat, x_prev, u0, u1);
    e[i] = force[i] * fr_f0(sqrt(u0 * u0 + u1 * u1), eps);
}

// pass 1: g2_i = F̂ f1(‖u‖)/‖u‖ u   (gradient) — stored as a real2 per pair
__global__ void k_friction_grad_scalar(int n, const int4* __restrict__ idx, const real4* __restrict__ weight,
                                       const real3* __restrict__ basis, const real* __restrict__ force, real eps,
                                       const real3* __restrict__ x_hat, const real3* __restrict__ x_prev,
                                       real2* __restrict__ g2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real u0, u1;
    friction_u(i, idx, weight, basis, x_hat, x_prev, u0, u1);
    const real y = sqrt(u0 * u0 + u1 * u1);
    const real f = force[i] * fr_f1_over_y(y, eps);
    g2[i] = make_real2(f * u0, f * u1);
}

// pass 1 of the SpMV: p_i = H2_i (T^T Σ w_s x_s), with H2 the PSD-projected 2x2 friction Hessian.
__global__ void k_friction_spmv_scalar(const int* __restrict__ skip, int n, int n_free, const int4* __restrict__ idx,
                                       const real4* __restrict__ weight, const real3* __restrict__ basis,
                                       const real* __restrict__ h2a, const real3* __restrict__ x_in,
                                       real2* __restrict__ p2) {
    if (skip != nullptr && *skip) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    const real4 w = weight[i];
    real3 rel = make_real3(real(0), real(0), real(0));
    if (id.x < n_free) rel = rel + w.x * x_in[id.x];
    if (id.z >= 0) {
        if (id.y < n_free) rel = rel + w.y * x_in[id.y];
        if (id.z < n_free) rel = rel + w.z * x_in[id.z];
        if (id.w < n_free) rel = rel + w.w * x_in[id.w];
    }
    const real q0 = dot(basis[2 * i + 0], rel);
    const real q1 = dot(basis[2 * i + 1], rel);
    const real a = h2a[3 * i + 0], b = h2a[3 * i + 1], c = h2a[3 * i + 2];   // [[a,b],[b,c]]
    p2[i] = make_real2(a * q0 + b * q1, b * q0 + c * q1);
}

// H2 = PSD projection of the IPC 2x2 friction Hessian, cached once per Newton step.
__global__ void k_friction_hessian(int n, const int4* __restrict__ idx, const real4* __restrict__ weight,
                                   const real3* __restrict__ basis, const real* __restrict__ force, real eps,
                                   const real3* __restrict__ x_hat, const real3* __restrict__ x_prev,
                                   real* __restrict__ h2a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    real u0, u1;
    friction_u(i, idx, weight, basis, x_hat, x_prev, u0, u1);
    const real F = force[i];
    const real y2 = u0 * u0 + u1 * u1;
    const real y = sqrt(y2);
    real a, b, c;
    if (y >= eps) {
        // exact tangential part: F/‖u‖ (I − uu^T/‖u‖²), already PSD
        const real s = F / y;
        a = s * (real(1) - u0 * u0 / y2);
        b = s * (-u0 * u1 / y2);
        c = s * (real(1) - u1 * u1 / y2);
    } else if (y2 == real(0)) {
        const real s = F * fr_f1_over_y(y, eps);
        a = s; b = real(0); c = s;
    } else {
        const real f1 = fr_f1_over_y(y, eps);
        const real f2 = -real(1) / (eps * eps);           // f2 term of the C1 clamp
        a = F * (f2 / y * u0 * u0 + f1);
        b = F * (f2 / y * u0 * u1);
        c = F * (f2 / y * u1 * u1 + f1);
        // PSD projection of the symmetric 2x2
        const real tr = a + c, det = a * c - b * b;
        const real disc = sqrt(fmax(real(0), tr * tr / real(4) - det));
        real l0 = tr / real(2) - disc, l1 = tr / real(2) + disc;
        if (l0 < real(0) || l1 < real(0)) {
            l0 = l0 < real(0) ? real(0) : l0;
            l1 = l1 < real(0) ? real(0) : l1;
            // eigenvector of l1
            real vx = b, vy = l1 - a;
            if (fabs(vx) + fabs(vy) < real(1e-20)) { vx = real(1); vy = real(0); }
            const real inv = real(1) / sqrt(vx * vx + vy * vy);
            vx *= inv; vy *= inv;
            const real wx = -vy, wy = vx;
            a = l1 * vx * vx + l0 * wx * wx;
            b = l1 * vx * vy + l0 * wx * wy;
            c = l1 * vy * vy + l0 * wy * wy;
        }
    }
    h2a[3 * i + 0] = a; h2a[3 * i + 1] = b; h2a[3 * i + 2] = c;
}

// pass 2 for friction: y_v += w_v (T p2_i) over the incidence CSR.
__global__ void k_friction_gather(const int* __restrict__ skip, int n_free, const int* __restrict__ ptr, const int* __restrict__ pair,
                                  const unsigned char* __restrict__ slot, const real4* __restrict__ weight,
                                  const real3* __restrict__ basis, const real2* __restrict__ p2,
                                  real3* __restrict__ y) {
    if (skip != nullptr && *skip) return;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = ptr[v]; k < ptr[v + 1]; ++k) {
        const int i = pair[k];
        const real4 w4 = weight[i];
        const real w = slot[k] == 0 ? w4.x : (slot[k] == 1 ? w4.y : (slot[k] == 2 ? w4.z : w4.w));
        const real2 p = p2[i];
        const real3 t = w * (p.x * basis[2 * i + 0] + p.y * basis[2 * i + 1]);
        ax += double(t.x); ay += double(t.y); az += double(t.z);
    }
    y[v] = y[v] + make_real3(real(ax), real(ay), real(az));
}

__global__ void k_friction_diag(int n_free, const int* __restrict__ ptr, const int* __restrict__ pair,
                                const unsigned char* __restrict__ slot, const real4* __restrict__ weight,
                                const real3* __restrict__ basis, const real* __restrict__ h2a,
                                real* __restrict__ diag) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    double m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = ptr[v]; k < ptr[v + 1]; ++k) {
        const int i = pair[k];
        const real4 w4 = weight[i];
        const real w = slot[k] == 0 ? w4.x : (slot[k] == 1 ? w4.y : (slot[k] == 2 ? w4.z : w4.w));
        const real ww = w * w;
        const real3 e1 = basis[2 * i + 0], e2 = basis[2 * i + 1];
        const real a = h2a[3 * i + 0], b = h2a[3 * i + 1], c = h2a[3 * i + 2];
        // w² T H2 T^T  with T = [e1 e2]
        const real3 c1 = a * e1 + b * e2;   // T H2 column 1
        const real3 c2 = b * e1 + c * e2;   // T H2 column 2
        const real3 r0 = make_real3(c1.x, c2.x, real(0));
        (void)r0;
        m[0] += double(ww) * (double(c1.x) * double(e1.x) + double(c2.x) * double(e2.x));
        m[1] += double(ww) * (double(c1.x) * double(e1.y) + double(c2.x) * double(e2.y));
        m[2] += double(ww) * (double(c1.x) * double(e1.z) + double(c2.x) * double(e2.z));
        m[4] += double(ww) * (double(c1.y) * double(e1.y) + double(c2.y) * double(e2.y));
        m[5] += double(ww) * (double(c1.y) * double(e1.z) + double(c2.y) * double(e2.z));
        m[8] += double(ww) * (double(c1.z) * double(e1.z) + double(c2.z) * double(e2.z));
    }
    m[3] = m[1]; m[6] = m[2]; m[7] = m[5];
    for (int j = 0; j < 9; ++j) diag[9 * v + j] += real(m[j]);
}

// =======================================================================================
// CCD stage
// =======================================================================================

struct HitSink {
    unsigned long long* alpha_bits;
    unsigned long long* T_v;
    const key_t* keys;
    int n_keys;
    bool filter_all;
    key_t* cand_key;
    int4* cand_idx;
    real* cand_toi;
    real* cand_xi;
    unsigned char* cand_type;
    unsigned long long* cand_count;   // 64-bit: a release-phase pass can report more than 2^31 hits
    int cand_capacity;
    bool filtered;      // store only the hits that pass the tie filter against the complete T_v
    double tie_tol;
    unsigned long long* pres_count;   // CCD hits between prescribed-only primitives: counted, not admitted (IS §5.x)
};

DEVICE_INLINE_CALLABLE void report_hit(const HitSink& sink, key_t key, int4 id, real toi, real xi, int nverts,
                                     unsigned char type) {
    atomicMin(reinterpret_cast<unsigned long long*>(sink.alpha_bits), d2bits(double(toi)));
    const bool member = key_member(sink.keys, sink.n_keys, key);
    if (!member || sink.filter_all) {
        const int vs[4] = {id.x, id.y, id.z, id.w};
        for (int s = 0; s < nverts; ++s)
            atomicMin(reinterpret_cast<unsigned long long*>(&sink.T_v[vs[s]]), d2bits(double(toi)));
    }
    if (member) return;
    if (sink.filtered) {
        // IS §12.3 item 15 (bounded hit storage): a pass repeated past the storage cap. T_v is
        // complete from the pass before, so the earliest-impact filter of k_keep_flag can be
        // applied here and only the hits the merge would keep are stored.
        const int vs[4] = {id.x, id.y, id.z, id.w};
        double tmax = bits2d(sink.T_v[vs[0]]);
        for (int s = 1; s < nverts; ++s) tmax = fmax(tmax, bits2d(sink.T_v[vs[s]]));
        if (double(toi) > tmax + sink.tie_tol) return;
    }
    const unsigned long long pos = atomicAdd(sink.cand_count, 1ull);
    if (pos < static_cast<unsigned long long>(sink.cand_capacity)) {
        sink.cand_key[pos] = key;
        sink.cand_idx[pos] = id;
        sink.cand_toi[pos] = toi;
        sink.cand_xi[pos] = xi;
        sink.cand_type[pos] = type;
    }
}

__global__ void k_plane_ccd(int n_surf_v, int n_planes, const int* __restrict__ surf_vert,
                            const PrimInfo* __restrict__ vinfo, const real* __restrict__ vthick,
                            const real3* __restrict__ plane_o, const real3* __restrict__ plane_n,
                            const int* __restrict__ plane_cid, const unsigned char* __restrict__ table, int n_cid,
                            const real3* __restrict__ xa, const real3* __restrict__ xh,
                            const unsigned char* __restrict__ is_prescribed, real s, HitSink sink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_surf_v * n_planes) return;
    const int sv = i / n_planes, h = i % n_planes;
    const int v = surf_vert[sv];
    if (!table[vinfo[sv].contact_id() * n_cid + plane_cid[h]]) return;
    const real xi = vthick[sv];
    real toi;
    if (!plane_toi(xa[v], xh[v] - xa[v], plane_o[h], plane_n[h], xi, s, toi)) return;
    if (is_prescribed[v]) {   // IS §5.x: a prescribed vertex driven into the plane is counted, not admitted (MS §4.1)
        if (!sink.filtered) atomicAdd(sink.pres_count, 1ull);
        return;
    }
    report_hit(sink, make_key(kPH, sv, h), make_int4(v, h, -1, -1), toi, xi, 1, (unsigned char)kPH);
}

// Vertex vs analytic sphere (MS §10.1): closed-form TOI with the radius interpolated from r0 to
// r1 over the step, like a prescribed vertex's motion (MS §12.3). Enumerated like planes.
__global__ void k_sphere_ccd(int n_surf_v, int n_spheres, const int* __restrict__ surf_vert,
                             const PrimInfo* __restrict__ vinfo, const real* __restrict__ vthick,
                             const real3* __restrict__ sphere_c, const real* __restrict__ sphere_r0,
                             const real* __restrict__ sphere_r1, const unsigned char* __restrict__ sphere_inv,
                             const int* __restrict__ sphere_cid, const unsigned char* __restrict__ table, int n_cid,
                             const real3* __restrict__ xa, const real3* __restrict__ xh,
                             const unsigned char* __restrict__ is_prescribed, real s, HitSink sink) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_surf_v * n_spheres) return;
    const int sv = i / n_spheres, q = i % n_spheres;
    const int v = surf_vert[sv];
    if (!table[vinfo[sv].contact_id() * n_cid + sphere_cid[q]]) return;
    const real xi = vthick[sv];
    real toi;
    if (!sphere_toi(xa[v], xh[v] - xa[v], sphere_c[q], sphere_r0[q], sphere_r1[q], sphere_inv[q] != 0, xi, s, toi))
        return;
    if (is_prescribed[v]) {   // IS §5.x: counted, not admitted
        if (!sink.filtered) atomicAdd(sink.pres_count, 1ull);
        return;
    }
    report_hit(sink, make_key(kPS, sv, q), make_int4(v, q, -1, -1), toi, xi, 1, (unsigned char)kPS);
}

__global__ void k_box_verts(int n, const int* __restrict__ sv, const real* __restrict__ thick,
                            const real3* __restrict__ xa, const real3* __restrict__ xh,
                            culbvh::Bound<float>* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = sv[i];
    out[i] = swept_box(xa[v], xh[v], thick[i]);
}
__global__ void k_box_edges(int n, const int2* __restrict__ e, const real* __restrict__ thick,
                            const real3* __restrict__ xa, const real3* __restrict__ xh,
                            culbvh::Bound<float>* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    culbvh::Bound<float> b = swept_box(xa[e[i].x], xh[e[i].x], thick[i]);
    b.absorb(swept_box(xa[e[i].y], xh[e[i].y], thick[i]));
    out[i] = b;
}
__global__ void k_box_tris(int n, const int3* __restrict__ t, const real* __restrict__ thick,
                           const real3* __restrict__ xa, const real3* __restrict__ xh,
                           culbvh::Bound<float>* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    culbvh::Bound<float> b = swept_box(xa[t[i].x], xh[t[i].x], thick[i]);
    b.absorb(swept_box(xa[t[i].y], xh[t[i].y], thick[i]));
    b.absorb(swept_box(xa[t[i].z], xh[t[i].z], thick[i]));
    out[i] = b;
}

__device__ __forceinline__ void narrow_pt_pair(int i, const int2* __restrict__ raw, const int* __restrict__ surf_vert,
                            const int3* __restrict__ surf_tri, const PrimInfo* __restrict__ vinfo,
                            const PrimInfo* __restrict__ tinfo, const real* __restrict__ vthick,
                            const real* __restrict__ tthick, const unsigned char* __restrict__ table, int n_cid,
                            const real3* __restrict__ xa, const real3* __restrict__ xh, real s, int max_iter,
                            HitSink sink, int n_tris, int n_svert, int n_verts, int* __restrict__ dbg) {
    const int ti = raw[i].x, vi = raw[i].y;
    if (ti < 0 || ti >= n_tris || vi < 0 || vi >= n_svert) {
        if (atomicAdd(dbg, 1) < 8) printf("narrow_pt: candidate %d = (%d, %d) outside [0, %d) x [0, %d)\n", i, ti, vi, n_tris, n_svert);
        return;
    }
    const int3 t = surf_tri[ti];
    const int v = surf_vert[vi];
    if (v < 0 || v >= n_verts || t.x < 0 || t.x >= n_verts || t.y < 0 || t.y >= n_verts || t.z < 0 || t.z >= n_verts) {
        if (atomicAdd(dbg, 1) < 8) printf("narrow_pt: candidate %d = (%d, %d) has vertices (%d | %d %d %d) outside [0, %d)\n", i, ti, vi, v, t.x, t.y, t.z, n_verts);
        return;
    }
    if (v == t.x || v == t.y || v == t.z) return;
    if (!admissible_by_scene(vinfo[vi], tinfo[ti], table, n_cid)) return;
    const bool pres_only = vinfo[vi].all_prescribed() && tinfo[ti].all_prescribed();
    const real xi = vthick[vi] + tthick[ti];
    const real3 c = (xa[v] + xa[t.x] + xa[t.y] + xa[t.z]) * real(0.25);
    real toi;
    if (!ACCD::vertex_triangle_ccd(xa[v] - c, xa[t.x] - c, xa[t.y] - c, xa[t.z] - c,
                                   xh[v] - c, xh[t.x] - c, xh[t.y] - c, xh[t.z] - c,
                                   toi, unsigned(max_iter), xi, s, real(1)))
        return;
    if (pres_only) {   // IS §5.x: a prescribed motion driven into a boundary is counted, not admitted (MS §4.1)
        if (!sink.filtered) atomicAdd(sink.pres_count, 1ull);
        return;
    }
    report_hit(sink, make_key(kPT, vi, ti), make_int4(v, t.x, t.y, t.z), toi, xi, 4, (unsigned char)kPT);
}
// Grid-stride over the chunk's candidates, whose count lives on the device (IS §12.3 item 15):
// a fixed launch whatever the count, no readback before it.
__global__ void k_narrow_pt(const int* __restrict__ d_n_raw, int cap, const int2* __restrict__ raw, const int* __restrict__ surf_vert,
                            const int3* __restrict__ surf_tri, const PrimInfo* __restrict__ vinfo,
                            const PrimInfo* __restrict__ tinfo, const real* __restrict__ vthick,
                            const real* __restrict__ tthick, const unsigned char* __restrict__ table, int n_cid,
                            const real3* __restrict__ xa, const real3* __restrict__ xh, real s, int max_iter,
                            HitSink sink, int n_tris, int n_svert, int n_verts, int* __restrict__ dbg) {
    const int n_raw = min(*d_n_raw, cap);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_raw; i += gridDim.x * blockDim.x)
        narrow_pt_pair(i, raw, surf_vert, surf_tri, vinfo, tinfo, vthick, tthick, table, n_cid, xa, xh, s, max_iter, sink, n_tris, n_svert, n_verts, dbg);
}

__device__ __forceinline__ void narrow_ee_pair(int i, const int2* __restrict__ raw, const int2* __restrict__ surf_edge,
                            const PrimInfo* __restrict__ einfo, const real* __restrict__ ethick,
                            const unsigned char* __restrict__ table, int n_cid, const real3* __restrict__ xa,
                            const real3* __restrict__ xh, real s, int max_iter, bool drop_parallel, real par_eps,
                            HitSink sink, int n_edges, int n_verts, int* __restrict__ dbg) {
    int ea = raw[i].x, eb = raw[i].y;
    // Release-phase fault hunt (2026-09-07): bounds guards that name the first bad candidates.
    if (ea < 0 || ea >= n_edges || eb < 0 || eb >= n_edges) {
        if (atomicAdd(dbg, 1) < 8) printf("narrow_ee: candidate %d = (%d, %d) outside [0, %d)\n", i, ea, eb, n_edges);
        return;
    }
    if (ea == eb) return;
    if (ea > eb) { const int tmp = ea; ea = eb; eb = tmp; }
    const int2 a = surf_edge[ea], b = surf_edge[eb];
    if (a.x < 0 || a.x >= n_verts || a.y < 0 || a.y >= n_verts || b.x < 0 || b.x >= n_verts || b.y < 0 || b.y >= n_verts) {
        if (atomicAdd(dbg, 1) < 8)
            printf("narrow_ee: candidate %d = (%d, %d) has vertices (%d %d | %d %d) outside [0, %d)\n", i, ea, eb, a.x, a.y, b.x, b.y, n_verts);
        return;
    }
    if (a.x == b.x || a.x == b.y || a.y == b.x || a.y == b.y) return;
    if (!admissible_by_scene(einfo[ea], einfo[eb], table, n_cid)) return;
    const bool pres_only = einfo[ea].all_prescribed() && einfo[eb].all_prescribed();
    if (drop_parallel && edges_parallel(xa[a.x], xa[a.y], xa[b.x], xa[b.y], par_eps)) return;
    const real xi = ethick[ea] + ethick[eb];
    const real3 c = (xa[a.x] + xa[a.y] + xa[b.x] + xa[b.y]) * real(0.25);
    real toi;
    if (!ACCD::edge_edge_ccd(xa[a.x] - c, xa[a.y] - c, xa[b.x] - c, xa[b.y] - c,
                             xh[a.x] - c, xh[a.y] - c, xh[b.x] - c, xh[b.y] - c,
                             toi, unsigned(max_iter), xi, s, real(1)))
        return;
    if (pres_only) {   // IS §5.x: counted, not admitted
        if (!sink.filtered) atomicAdd(sink.pres_count, 1ull);
        return;
    }
    report_hit(sink, make_key(kEE, ea, eb), make_int4(a.x, a.y, b.x, b.y), toi, xi, 4, (unsigned char)kEE);
}
// Grid-stride over the chunk's candidates, whose count lives on the device (IS §12.3 item 15):
// a fixed launch whatever the count, no readback before it.
__global__ void k_narrow_ee(const int* __restrict__ d_n_raw, int cap, const int2* __restrict__ raw, const int2* __restrict__ surf_edge,
                            const PrimInfo* __restrict__ einfo, const real* __restrict__ ethick,
                            const unsigned char* __restrict__ table, int n_cid, const real3* __restrict__ xa,
                            const real3* __restrict__ xh, real s, int max_iter, bool drop_parallel, real par_eps,
                            HitSink sink, int n_edges, int n_verts, int* __restrict__ dbg) {
    const int n_raw = min(*d_n_raw, cap);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_raw; i += gridDim.x * blockDim.x)
        narrow_ee_pair(i, raw, surf_edge, einfo, ethick, table, n_cid, xa, xh, s, max_iter, drop_parallel, par_eps, sink, n_edges, n_verts, dbg);
}

__global__ void k_keep_flag(int n, const int4* __restrict__ idx, const real* __restrict__ toi,
                            const unsigned long long* __restrict__ T_v, double tol, int* __restrict__ keep) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int4 id = idx[i];
    double tmax = bits2d(T_v[id.x]);
    if (id.z >= 0) {
        tmax = fmax(tmax, bits2d(T_v[id.y]));
        tmax = fmax(tmax, bits2d(T_v[id.z]));
        tmax = fmax(tmax, bits2d(T_v[id.w]));
    }
    keep[i] = (double(toi[i]) <= tmax + tol) ? 1 : 0;
}

__global__ void k_remove_flag(int n, const int* __restrict__ ninact, int remove_after, int* __restrict__ keep) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keep[i] = ninact[i] > remove_after ? 0 : 1;
}

__global__ void k_fill_bits(int n, unsigned long long v, unsigned long long* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = v;
}

// Debug DCD sweep (§6.12): minimum unsigned distance over admissible pairs with a free vertex.
__device__ __forceinline__ void dcd_pt_pair(int i, const int2* __restrict__ raw, const int* __restrict__ surf_vert,
                         const int3* __restrict__ surf_tri, const PrimInfo* __restrict__ vinfo,
                         const PrimInfo* __restrict__ tinfo, const real* __restrict__ vthick,
                         const real* __restrict__ tthick, const unsigned char* __restrict__ table, int n_cid,
                         const real3* __restrict__ x, unsigned long long* __restrict__ min_bits,
                         int* __restrict__ violations) {
    const int ti = raw[i].x, vi = raw[i].y;
    const int3 t = surf_tri[ti];
    const int v = surf_vert[vi];
    if (v == t.x || v == t.y || v == t.z) return;
    if (!admissible(vinfo[vi], tinfo[ti], table, n_cid)) return;
    const real3 c = (x[v] + x[t.x] + x[t.y] + x[t.z]) * real(0.25);
    real u0, u1;
    real3 nrm;
    const real d2 = vertex_triangle_distance_square(x[v] - c, x[t.x] - c, x[t.y] - c, x[t.z] - c, u0, u1, nrm);
    const double d = sqrt(double(d2)) - double(vthick[vi] + tthick[ti]);
    atomicMin(min_bits, d2bits(fmax(0.0, d + 1.0)));   // shifted so the bit trick stays valid
    if (d < 0.0) atomicAdd(violations, 1);
}
// Grid-stride over the chunk's candidates, whose count lives on the device (IS §12.3 item 15):
// a fixed launch whatever the count, no readback before it.
__global__ void k_dcd_pt(const int* __restrict__ d_n_raw, int cap, const int2* __restrict__ raw, const int* __restrict__ surf_vert,
                         const int3* __restrict__ surf_tri, const PrimInfo* __restrict__ vinfo,
                         const PrimInfo* __restrict__ tinfo, const real* __restrict__ vthick,
                         const real* __restrict__ tthick, const unsigned char* __restrict__ table, int n_cid,
                         const real3* __restrict__ x, unsigned long long* __restrict__ min_bits,
                         int* __restrict__ violations) {
    const int n_raw = min(*d_n_raw, cap);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_raw; i += gridDim.x * blockDim.x)
        dcd_pt_pair(i, raw, surf_vert, surf_tri, vinfo, tinfo, vthick, tthick, table, n_cid, x, min_bits, violations);
}

}  // namespace

// =======================================================================================
// Impl
// =======================================================================================
struct ContactSystem::Impl {
    const SceneBuffers& sb;
    ContactParams params;
    int n_free = 0;

    // active set, always sorted by key
    int n_pairs = 0;
    int n_pt = 0, n_ee = 0, n_ph = 0;
    DeviceArray<key_t> key;
    DeviceArray<unsigned char> type;
    DeviceArray<int4> idx;
    DeviceArray<real> xi, lambda, gap, gap_folded, slack, coeff, pair_scalar;
    DeviceArray<int> ninact;
    DeviceArray<real3> grad4;

    // incidence over free vertices
    int n_inc = 0;
    DeviceArray<int> vp_ptr, vp_pair, vp_count, vp_fill;
    DeviceArray<unsigned char> vp_slot;
    DeviceArray<real3> vp_grad;
    DeviceArray<real> diag;

    // friction snapshot (frozen for the step)
    int n_fr = 0, n_fr_inc = 0;
    ActiveSetUnion aset_union;     // device union of C and the kept hits (IS §6.10 step 7)
    DevicePool fr_pool;
    DeviceArray<int> fr_sel;
    DeviceArray<int4> fr_idx;
    DeviceArray<real4> fr_weight;
    DeviceArray<real3> fr_basis;
    DeviceArray<real> fr_force, fr_h2;
    DeviceArray<real2> fr_scalar;
    DeviceArray<int> fr_ptr, fr_pair, fr_count, fr_fill;
    DeviceArray<unsigned char> fr_slot;

    // candidates
    DeviceArray<key_t> cand_key;
    DeviceArray<int4> cand_idx;
    DeviceArray<real> cand_toi, cand_xi;
    DeviceArray<unsigned char> cand_type;
    DeviceArray<int> cand_keep;
    DeviceVar<unsigned long long> cand_count;   // hits reported in the current pass (64-bit)
    DeviceVar<unsigned long long> pres_count;   // hits between prescribed-only primitives (IS §5.x diagnostic)
    DeviceArray<unsigned long long> status{3};   // {alpha bits, hit count, prescribed-only hits} packed for one readback
    DeviceArray<unsigned long long> T_v;
    DeviceVar<unsigned long long> alpha_bits;
    DeviceVar<int> n_degenerate;
    DeviceVar<int> dbg_count;        // release-phase fault hunt: out-of-range candidates seen
    DeviceVar<unsigned long long> dcd_min;
    DeviceVar<int> dcd_violations;

    DeviceArray<int> keep_flag;

    // broad phase
    DeviceArray<culbvh::Bound<float>> box_v, box_e, box_t;
    culbvh::LBVHStacklessLite tri_tree, edge_tree;
    bool trees_built = false;
    DeviceArray<int2> raw_pt, raw_ee;

    DeviceArray<real> scratch_r;
    DeviceArray<double> scratch_d;

    Impl(const SceneBuffers& s, const Scene& scene, const ContactDesc& cd) : sb(s) {
        n_free = sb.n_free;
        params.d_hat = real(cd.d_hat);
        params.gamma_factor = real(cd.decay_factor);
        params.remove_after = std::max(1, int(std::ceil(std::log(cd.decay_remove_threshold) /
                                                        std::log(std::max(1e-9, cd.decay_factor)))));
        params.accd_s = real(cd.ccd.s);
        params.accd_max_iter = cd.ccd.max_iter;
        params.toi_tie_tol = real(cd.toi_tie_tolerance);
        params.toi_filter_all = (cd.toi_filter_domain == "all");
        params.exclude_one_ring = cd.exclude_one_ring_pairs;
        params.drop_parallel_ee = cd.drop_parallel_edge_pairs;
        params.rebuild_quality_ratio = real(cd.ccd.rebuild_quality_ratio);
        params.friction_enable = cd.friction.enable;
        params.eps_v = real(cd.friction.eps_v);
        params.friction_normal_force_lambda = (cd.friction.normal_force == "lambda");

        vp_ptr.resize(size_t(n_free) + 1);
        vp_count.resize(size_t(n_free) + 1);
        vp_fill.resize(size_t(n_free) + 1);
        fr_ptr.resize(size_t(n_free) + 1);
        fr_count.resize(size_t(n_free) + 1);
        fr_fill.resize(size_t(n_free) + 1);
        diag.resize(size_t(9) * std::max(1, n_free));
        if (n_free) { k_zero_diag<<<grid_for(n_free), kBlock>>>(n_free, diag.data()); CS_CUDA_KERNEL_CHECK(); }
        T_v.resize(std::max(1, sb.n_vertices));
        scratch_r.resize(1024);
        cand_count.set(0);
        alpha_bits.set(d2bits(1.0));
        n_degenerate.set(0);
        dbg_count.set(0);
        dcd_min.set(d2bits(1e30));
        dcd_violations.set(0);
        ensure_candidate_capacity(std::max(4096, sb.n_surf_v));
        box_v.resize(std::max(1, sb.n_surf_v));
        box_e.resize(std::max(1, sb.n_surf_e));
        box_t.resize(std::max(1, sb.n_surf_t));
        raw_pt.resize(size_t(kRawCapacity));
        raw_ee.resize(size_t(kRawCapacity));
        (void)scene;
    }

    void ensure_candidate_capacity(int n) {
        if (int(cand_key.size()) >= n) return;
        cand_key.resize(n); cand_idx.resize(n); cand_toi.resize(n); cand_xi.resize(n);
        cand_type.resize(n); cand_keep.resize(n);
    }

    void resize_pair_scratch(int n) {
        gap.loose_resize(n); gap_folded.loose_resize(n); slack.loose_resize(n);
        coeff.loose_resize(n); pair_scalar.loose_resize(n); grad4.loose_resize(size_t(4) * n);
    }

    // ---- broad phase ------------------------------------------------------------------
    void update_boxes(const real3* xa, const real3* xh) {
        if (sb.n_surf_v) {
            k_box_verts<<<grid_for(sb.n_surf_v), kBlock>>>(sb.n_surf_v, sb.surf_vert.data(),
                                                           sb.surf_vert_thickness.data(), xa, xh, box_v.data());
        }
        if (sb.n_surf_e) {
            k_box_edges<<<grid_for(sb.n_surf_e), kBlock>>>(sb.n_surf_e, sb.surf_edge.data(),
                                                           sb.surf_edge_thickness.data(), xa, xh, box_e.data());
        }
        if (sb.n_surf_t) {
            k_box_tris<<<grid_for(sb.n_surf_t), kBlock>>>(sb.n_surf_t, sb.surf_tri.data(),
                                                          sb.surf_tri_thickness.data(), xa, xh, box_t.data());
        }
        CS_CUDA_KERNEL_CHECK();
    }

    void build_trees(bool structural) {
        if (!trees_built) {
            if (sb.n_surf_t) tri_tree.compute(box_t.data(), size_t(sb.n_surf_t));
            if (sb.n_surf_e) edge_tree.compute(box_e.data(), size_t(sb.n_surf_e));
            CS_SYNC_POINT("lbvh compute");
            trees_built = true;
        } else if (structural) {
            if (sb.n_surf_t) tri_tree.refit_structure();
            if (sb.n_surf_e) edge_tree.refit_structure();
            CS_SYNC_POINT("lbvh refit_structure");
        } else {
            if (sb.n_surf_t) tri_tree.refit();      // §12.3 item 1: bounds-only per outer iteration
            if (sb.n_surf_e) edge_tree.refit();
            CS_SYNC_POINT("lbvh refit");
        }
    }

    // ---- broad + narrow phase in chunks (IS §12.3 item 15) ----------------------------
    // The candidate buffers used to hold every broad-phase pair of the iteration at once (up
    // to 3e8 entries on the compressor, doubling on overflow: several GB). Now the query
    // objects are visited in chunks of their Morton order, each chunk's candidates go through
    // the narrow phase at once, and only the hits (which are few) accumulate. The chunk length
    // adapts: halved when a chunk overflows the fixed buffer, doubled after a pass in which
    // every chunk stayed below a quarter of it. The hit set is independent of the chunking.
    static constexpr int kRawCapacity = 16 << 20;   // int2 entries per buffer: 128 MB
    static constexpr int kNarrowGrid = 4096;         // blocks of the grid-stride narrow-phase launches
    int chunk_pt = 0, chunk_ee = 0;                  // current chunk lengths (query objects)
    // Bounded hit storage: the candidate arrays grow only up to hit_capacity entries (the
    // default is ~190 MB). A pass that overflows it is repeated with the tie filter applied at
    // report time (`HitSink::filtered`), so only the survivors are stored; the compressor's
    // release phase reports >1e8 hits per pass of which the merge keeps a few hundred thousand.
    static constexpr int kHitCapacity = 4 << 20;
    int hit_capacity = kHitCapacity;
    bool sink_filtered = false;                      // the current pass stores only tie-filter survivors
    // Entries the current pass may store: an unfiltered pass never uses more than hit_capacity
    // even when the arrays grew larger for the survivors of an earlier filtered pass.
    int storage_cap() const { return sink_filtered ? int(cand_key.size()) : std::min(int(cand_key.size()), hit_capacity); }

    // One chunked pass of a query type. `query(q0, q1, d_count)` enqueues the broad-phase query
    // of the range with its pair count accumulated into *d_count; `narrow(d_count)` enqueues
    // the narrow phase over the buffer bounded by that count on the device. Nothing in the
    // pass synchronises until the counts of all its ranges are read back together; ranges that
    // overflowed the buffer are redone in halves (their hits were partly reported already, the
    // duplicates fall out in the merge's unique pass).
    DeviceArray<int> chunk_counts;
    template <class Query, class Narrow>
    long long chunked_pass(int n_query, int& chunk, int cap, Query query, Narrow narrow) {
        long long candidates = 0;
        if (n_query <= 0) return 0;
        if (chunk <= 0 || chunk > n_query) chunk = n_query;
        std::vector<std::pair<int, int>> ranges;
        for (int q0 = 0; q0 < n_query; q0 += chunk) ranges.emplace_back(q0, std::min(n_query, q0 + chunk));
        std::vector<int> counts;
        int peak = 0;
        while (!ranges.empty()) {
            const int m = int(ranges.size());
            chunk_counts.loose_resize(size_t(m));
            CS_CUDA_CHECK(cudaMemsetAsync(chunk_counts.data(), 0, sizeof(int) * size_t(m), 0));
            for (int k = 0; k < m; ++k) {
                query(size_t(ranges[k].first), size_t(ranges[k].second), chunk_counts.data() + k);
                narrow(chunk_counts.data() + k);
            }
            counts.resize(size_t(m));
            chunk_counts.download(counts.data(), size_t(m));   // the pass's one synchronisation
            std::vector<std::pair<int, int>> redo;
            for (int k = 0; k < m; ++k) {
                const int q0 = ranges[k].first, q1 = ranges[k].second;
                if (counts[k] >= cap || counts[k] < 0) {   // < 0: a wrapped counter, defensive
                    if (q1 - q0 == 1)
                        throw std::runtime_error("contact: a single query object produced more than " +
                                                 std::to_string(cap) + " broad-phase candidates");
                    const int mid = q0 + (q1 - q0) / 2;
                    redo.emplace_back(q0, mid);
                    redo.emplace_back(mid, q1);
                    chunk = std::max(1, std::min(chunk, mid - q0));
                } else {
                    candidates += counts[k];
                    peak = std::max(peak, counts[k]);
                }
            }
            ranges.swap(redo);
        }
        if (peak < cap / 4 && chunk < n_query) chunk = std::min(n_query, 2 * chunk);
        return candidates;
    }

    // ---- incidence -------------------------------------------------------------------
    void build_incidence(const int4* pair_idx, int n, DeviceArray<int>& ptr, DeviceArray<int>& count,
                         DeviceArray<int>& fill, DeviceArray<int>& pair, DeviceArray<unsigned char>& slot, int& n_out) {
        count.zero();
        if (n) {
            k_count_incidence<<<grid_for(n), kBlock>>>(n, n_free, pair_idx, count.data());
            CS_CUDA_KERNEL_CHECK();
        }
        thrust::exclusive_scan(thrust::device, thrust::device_ptr<int>(count.data()),
                               thrust::device_ptr<int>(count.data() + n_free + 1),
                               thrust::device_ptr<int>(ptr.data()));
        // IS §12.3 item 13: the incidence total stays on the device (ptr[n_free]); every pair has
        // at most four free vertices, so 4 n bounds it and is what the consumers launch with.
        const int total = 4 * n;
        n_out = total;
        pair.loose_resize(std::max(1, total));
        slot.loose_resize(std::max(1, total));
        fill.zero();
        if (n) {
            k_fill_incidence<<<grid_for(n), kBlock>>>(n, n_free, pair_idx, ptr.data(), fill.data(), pair.data(), slot.data());
            CS_CUDA_KERNEL_CHECK();
            k_sort_rows<<<grid_for(n_free), kBlock>>>(n_free, ptr.data(), pair.data(), slot.data());
            CS_CUDA_KERNEL_CHECK();
        }
    }

    // ---- per outer iteration -----------------------------------------------------------
    void linearize(const real3* xa) {
        ScopedTimer t("contact");
        const int n = n_pairs;
        if (n) {
            n_degenerate.set(0);
            k_linearize<<<grid_for(n), kBlock>>>(n, type.data(), idx.data(), xi.data(), params.d_hat, xa,
                                                 sb.plane_o.data(), sb.plane_n.data(), sb.sphere_c.data(),
                                                 sb.sphere_r1.data(), sb.sphere_inverted.data(),
                                                 params.drop_parallel_ee, params.parallel_ee_eps, gap.data(),
                                                 grad4.data(), n_degenerate.data());
            CS_CUDA_KERNEL_CHECK();
            // IS §12.3 item 13: the degenerate-pair count is a diagnostic; it is read back only
            // when the debug log level asks for it, not on every outer iteration.
            if (log().should_log(spdlog::level::debug)) {
                const int nd = n_degenerate.get();
                if (nd) log().warn("contact: {} degenerate pair distances at the anchor", nd);
            }
            k_coeff<<<grid_for(n), kBlock>>>(n, ninact.data(), params.gamma_factor, params.mu, coeff.data());
            CS_CUDA_KERNEL_CHECK();
        }
        build_incidence(idx.data(), n, vp_ptr, vp_count, vp_fill, vp_pair, vp_slot, n_inc);
        vp_grad.loose_resize(std::max(1, n_inc));
        if (n_inc) {
            k_vp_grad<<<grid_for(n_inc), kBlock>>>(vp_ptr.data() + n_free, n_inc, vp_pair.data(), vp_slot.data(),
                                                   grad4.data(), vp_grad.data());
            CS_CUDA_KERNEL_CHECK();
        }
        update_hessian(xa);   // valid diagonal even before the first Newton step
    }

    // ---- per Newton step ----------------------------------------------------------------
    // MS §5.5: the slack is frozen for the whole Newton step (including its line search), so
    // this runs once per step, never per line-search trial.
    void update_slack(const real3* xh, const real3* xa) {
        if (!n_pairs) return;
        k_slack<<<grid_for(n_pairs), kBlock>>>(n_pairs, idx.data(), gap.data(), grad4.data(), lambda.data(),
                                               params.mu, xh, xa, slack.data(), gap_folded.data());
        CS_CUDA_KERNEL_CHECK();
    }

    // Rebuilds the preconditioner diagonal: the contact part (write) plus, when friction is
    // active, its PSD-projected 2x2 Hessian re-linearized at the current iterate (add). Also
    // once per Newton step — k_friction_diag accumulates, so it must not run per trial.
    void update_hessian(const real3* xh) {
        if (!n_free) return;
        k_contact_diag<<<grid_for(n_free), kBlock>>>(n_free, vp_ptr.data(), vp_pair.data(), vp_grad.data(),
                                                     coeff.data(), diag.data());
        CS_CUDA_KERNEL_CHECK();
        if (!n_fr) return;
        k_friction_hessian<<<grid_for(n_fr), kBlock>>>(n_fr, fr_idx.data(), fr_weight.data(), fr_basis.data(),
                                                       fr_force.data(), params.eps_v * sb.dt, xh, sb.x_prev.data(),
                                                       fr_h2.data());
        CS_CUDA_KERNEL_CHECK();
        k_friction_diag<<<grid_for(n_free), kBlock>>>(n_free, fr_ptr.data(), fr_pair.data(), fr_slot.data(),
                                                      fr_weight.data(), fr_basis.data(), fr_h2.data(), diag.data());
        CS_CUDA_KERNEL_CHECK();
    }

    void add_gradient(const real3* xh, const real3* xa, real3* G) {
        if (!n_free) return;
        if (n_pairs) {
            k_pair_scalar_grad<<<grid_for(n_pairs), kBlock>>>(n_pairs, idx.data(), gap_folded.data(), grad4.data(),
                                                              coeff.data(), xh, xa, pair_scalar.data());
            CS_CUDA_KERNEL_CHECK();
            k_vertex_gather<<<grid_for(n_free), kBlock>>>(nullptr, n_free, vp_ptr.data(), vp_pair.data(), vp_grad.data(),
                                                          pair_scalar.data(), G);
            CS_CUDA_KERNEL_CHECK();
        }
        if (n_fr) {
            k_friction_grad_scalar<<<grid_for(n_fr), kBlock>>>(n_fr, fr_idx.data(), fr_weight.data(), fr_basis.data(),
                                                               fr_force.data(), params.eps_v * sb.dt, xh,
                                                               sb.x_prev.data(), fr_scalar.data());
            CS_CUDA_KERNEL_CHECK();
            k_friction_gather<<<grid_for(n_free), kBlock>>>(nullptr, n_free, fr_ptr.data(), fr_pair.data(), fr_slot.data(),
                                                            fr_weight.data(), fr_basis.data(), fr_scalar.data(), G);
            CS_CUDA_KERNEL_CHECK();
        }
    }

    void spmv_add(const real3* x_in, real3* y, cudaStream_t stream, const int* skip) {
        if (!n_free) return;
        if (n_pairs) {
            k_pair_sigma<<<grid_for(n_pairs), kBlock, 0, stream>>>(skip, n_pairs, n_free, idx.data(), grad4.data(), coeff.data(), x_in,
                                                        pair_scalar.data());
            CS_CUDA_KERNEL_CHECK();
            k_vertex_gather<<<grid_for(n_free), kBlock, 0, stream>>>(skip, n_free, vp_ptr.data(), vp_pair.data(), vp_grad.data(),
                                                          pair_scalar.data(), y);
            CS_CUDA_KERNEL_CHECK();
        }
        if (n_fr) {
            k_friction_spmv_scalar<<<grid_for(n_fr), kBlock, 0, stream>>>(skip, n_fr, n_free, fr_idx.data(), fr_weight.data(),
                                                               fr_basis.data(), fr_h2.data(), x_in, fr_scalar.data());
            CS_CUDA_KERNEL_CHECK();
            k_friction_gather<<<grid_for(n_free), kBlock, 0, stream>>>(skip, n_free, fr_ptr.data(), fr_pair.data(), fr_slot.data(),
                                                            fr_weight.data(), fr_basis.data(), fr_scalar.data(), y);
            CS_CUDA_KERNEL_CHECK();
        }
    }

    void energy_to(const real3* xh, const real3* xa, double* d_pairs, double* d_friction) {
        if (n_pairs) {
            if (scratch_r.size() < size_t(n_pairs)) scratch_r.resize(n_pairs);
            k_pair_energy<<<grid_for(n_pairs), kBlock>>>(n_pairs, idx.data(), gap_folded.data(), grad4.data(),
                                                         coeff.data(), xh, xa, scratch_r.data());
            CS_CUDA_KERNEL_CHECK();
        }
        reduce_sum_to(scratch_r.data(), n_pairs, scratch_d, d_pairs);
        if (n_fr) {
            if (scratch_r.size() < size_t(n_fr)) scratch_r.resize(n_fr);
            k_friction_energy<<<grid_for(n_fr), kBlock>>>(n_fr, fr_idx.data(), fr_weight.data(), fr_basis.data(),
                                                          fr_force.data(), params.eps_v * sb.dt, xh, sb.x_prev.data(),
                                                          scratch_r.data());
            CS_CUDA_KERNEL_CHECK();
        }
        reduce_sum_to(scratch_r.data(), n_fr, scratch_d, d_friction);
    }

    double energy(const real3* xh, const real3* xa) {
        double e = 0.0;
        if (n_pairs) {
            if (scratch_r.size() < size_t(n_pairs)) scratch_r.resize(n_pairs);
            k_pair_energy<<<grid_for(n_pairs), kBlock>>>(n_pairs, idx.data(), gap_folded.data(), grad4.data(),
                                                         coeff.data(), xh, xa, scratch_r.data());
            CS_CUDA_KERNEL_CHECK();
            e += reduce_sum(scratch_r.data(), n_pairs, scratch_d);
        }
        if (n_fr) {
            if (scratch_r.size() < size_t(n_fr)) scratch_r.resize(n_fr);
            k_friction_energy<<<grid_for(n_fr), kBlock>>>(n_fr, fr_idx.data(), fr_weight.data(), fr_basis.data(),
                                                          fr_force.data(), params.eps_v * sb.dt, xh, sb.x_prev.data(),
                                                          scratch_r.data());
            CS_CUDA_KERNEL_CHECK();
            e += reduce_sum(scratch_r.data(), n_fr, scratch_d);
        }
        return e;
    }

    void update_multipliers(const real3* xh, const real3* xa) {
        if (!n_pairs) return;
        k_update_lambda<<<grid_for(n_pairs), kBlock>>>(n_pairs, idx.data(), gap.data(), grad4.data(), params.mu, xh, xa,
                                                       lambda.data(), ninact.data());
        CS_CUDA_KERNEL_CHECK();
    }

    // ---- CCD + active set ---------------------------------------------------------------
    HitSink make_sink() {
        HitSink s;
        s.alpha_bits = alpha_bits.data();
        s.T_v = T_v.data();
        s.keys = key.data();
        s.n_keys = n_pairs;
        s.filter_all = params.toi_filter_all;
        s.cand_key = cand_key.data();
        s.cand_idx = cand_idx.data();
        s.cand_toi = cand_toi.data();
        s.cand_xi = cand_xi.data();
        s.cand_type = cand_type.data();
        s.cand_count = cand_count.data();
        s.cand_capacity = storage_cap();
        s.filtered = sink_filtered;
        s.tie_tol = double(params.toi_tie_tol);
        s.pres_count = pres_count.data();
        return s;
    }

    bool candidates_and_hits(const real3* xa, const real3* xh, double alpha_cap, CcdResult& r) {
        cand_count.set(0);
        alpha_bits.set(d2bits(alpha_cap));
        if (!sink_filtered) {   // a filtered pass reuses the complete T_v (and the prescribed-only count) of the pass before it
            k_fill_bits<<<grid_for(sb.n_vertices), kBlock>>>(sb.n_vertices, d2bits(2.0), T_v.data());
            CS_CUDA_KERNEL_CHECK();
            pres_count.set(0);
        }
        HitSink sink = make_sink();
        const int nvh = sb.n_surf_v * sb.n_planes;
        if (nvh) {
            k_plane_ccd<<<grid_for(nvh), kBlock>>>(sb.n_surf_v, sb.n_planes, sb.surf_vert.data(),
                                                   sb.surf_vert_info.data(), sb.surf_vert_thickness.data(),
                                                   sb.plane_o.data(), sb.plane_n.data(), sb.plane_contact_id.data(),
                                                   sb.table_enabled.data(), sb.n_contact_ids, xa, xh,
                                                   sb.is_prescribed.data(), params.accd_s, sink);
            CS_CUDA_KERNEL_CHECK();
        }
        const int nvs = sb.n_surf_v * sb.n_spheres;
        if (nvs) {
            k_sphere_ccd<<<grid_for(nvs), kBlock>>>(sb.n_surf_v, sb.n_spheres, sb.surf_vert.data(),
                                                    sb.surf_vert_info.data(), sb.surf_vert_thickness.data(),
                                                    sb.sphere_c.data(), sb.sphere_r0.data(), sb.sphere_r1.data(),
                                                    sb.sphere_inverted.data(), sb.sphere_contact_id.data(),
                                                    sb.table_enabled.data(), sb.n_contact_ids, xa, xh,
                                                    sb.is_prescribed.data(), params.accd_s, sink);
            CS_CUDA_KERNEL_CHECK();
        }
        long long candidates = 0;
        if (sb.n_surf_t && sb.n_surf_v) {
            tri_tree.prepare_other(box_v.data(), size_t(sb.n_surf_v));
            CS_SYNC_POINT("lbvh prepare_other (ccd)");
            candidates += chunked_pass(sb.n_surf_v, chunk_pt, kRawCapacity,
                [&](size_t q0, size_t q1, int* d_count) { tri_tree.query_other_range_async(raw_pt.data(), size_t(kRawCapacity), box_v.data(), q0, q1, d_count); CS_SYNC_POINT("lbvh query pt (ccd)"); },
                [&](const int* d_n_raw) {
            k_narrow_pt<<<kNarrowGrid, kBlock>>>(d_n_raw, kRawCapacity, raw_pt.data(), sb.surf_vert.data(),
                                                           sb.surf_tri.data(), sb.surf_vert_info.data(),
                                                           sb.surf_tri_info.data(), sb.surf_vert_thickness.data(),
                                                           sb.surf_tri_thickness.data(), sb.table_enabled.data(),
                                                           sb.n_contact_ids, xa, xh, params.accd_s,
                                                           params.accd_max_iter, sink, sb.n_surf_t, sb.n_surf_v,
                                                           sb.n_vertices, dbg_count.data());
            CS_CUDA_KERNEL_CHECK();
                });
        }
        if (sb.n_surf_e) {
            candidates += chunked_pass(sb.n_surf_e, chunk_ee, kRawCapacity,
                [&](size_t q0, size_t q1, int* d_count) { edge_tree.query_range_async(raw_ee.data(), size_t(kRawCapacity), q0, q1, d_count); CS_SYNC_POINT("lbvh query ee (ccd)"); },
                [&](const int* d_n_raw) {
            k_narrow_ee<<<kNarrowGrid, kBlock>>>(d_n_raw, kRawCapacity, raw_ee.data(), sb.surf_edge.data(),
                                                           sb.surf_edge_info.data(), sb.surf_edge_thickness.data(),
                                                           sb.table_enabled.data(), sb.n_contact_ids, xa, xh,
                                                           params.accd_s, params.accd_max_iter,
                                                           params.drop_parallel_ee, params.parallel_ee_eps, sink,
                                                           sb.n_surf_e, sb.n_vertices, dbg_count.data());
            CS_CUDA_KERNEL_CHECK();
                });
        }
        r.candidates = candidates + static_cast<long long>(sb.n_surf_v) * (sb.n_planes + sb.n_spheres);
        // IS §12.3 item 13: the step bound and the hit count come back in one copy.
        k_pack_status<<<1, 1>>>(alpha_bits.data(), cand_count.data(), pres_count.data(), status.data());
        CS_CUDA_KERNEL_CHECK();
        unsigned long long st[3];
        status.download(st, 3);
        r.prescribed_hits = static_cast<long long>(st[2]);
        const unsigned long long count64 = st[1];
        if (count64 > static_cast<unsigned long long>(storage_cap())) {
            // Bounded hit storage (IS §12.3 item 15b): grow only up to hit_capacity. Past it the
            // pass is repeated filtered; the survivors are exactly what k_keep_flag keeps, so
            // the merge sees the same set. A filtered pass that still overflows must grow.
            // The count is 64-bit: the compressor's release phase reports more than 2^31 hits
            // in one pass, and a wrapped int count once produced negative store positions.
            if (!sink_filtered && count64 > static_cast<unsigned long long>(hit_capacity)) {
                sink_filtered = true;
                ++r.filtered_passes;
            } else {
                unsigned long long want = count64 * 3ull / 2ull + 4096ull;
                if (!sink_filtered) want = std::min(want, static_cast<unsigned long long>(hit_capacity));
                if (want > 0x3fffffffull)
                    throw std::runtime_error("contact: " + std::to_string(count64) +
                                             " tie-filter survivors in one CCD pass exceed the hit storage limit");
                ensure_candidate_capacity(static_cast<int>(want));
            }
            return false;
        }
        r.hits = static_cast<int>(count64);
        r.alpha = bits2d(st[0]);
        return true;
    }

    CcdResult ccd_and_update(const real3* xa, const real3* xh, double alpha_cap) {
        CcdResult r;
        r.alpha = alpha_cap;
        {
            ScopedTimer t("ccd");
            if (sb.n_surf_t || sb.n_surf_e) {
                update_boxes(xa, xh);
                build_trees(/*structural=*/false);
            }
            sink_filtered = false;
            while (!candidates_and_hits(xa, xh, alpha_cap, r)) {}
            sink_filtered = false;
            if (r.hits) {
                k_keep_flag<<<grid_for(r.hits), kBlock>>>(r.hits, cand_idx.data(), cand_toi.data(), T_v.data(),
                                                          double(params.toi_tie_tol), cand_keep.data());
                CS_CUDA_KERNEL_CHECK();
            }
            keep_flag.loose_resize(std::max(1, n_pairs));
            if (n_pairs) {
                k_remove_flag<<<grid_for(n_pairs), kBlock>>>(n_pairs, ninact.data(), params.remove_after, keep_flag.data());
                CS_CUDA_KERNEL_CHECK();
            }
        }
        merge(r);
        return r;
    }

    // Device union of C and the kept hits (implementation spec §6.10 step 7): stable select of
    // the survivors, radix sort + unique of the kept hits, merge-path set union, one gather.
    // The per-type counts come back with the size (one short readback); everything else stays
    // on the device. The host merge this replaced cost 29% of the compressor's step at 1.7 M
    // pairs (doc/squishy-press-reproduction.md §3).
    void merge(CcdResult& r) {
        ScopedTimer t("active_set");
        const UnionResult u = aset_union.unite(n_pairs, key, type, idx, xi, lambda, ninact, keep_flag.data(), r.hits,
                                               cand_key.data(), cand_type.data(), cand_idx.data(), cand_xi.data(),
                                               cand_keep.data());
        r.removed = u.removed;
        r.kept = u.kept;
        n_pt = u.counts[kPT];
        n_ee = u.counts[kEE];
        n_ph = u.n_new - n_pt - n_ee;   // PH and PS pairs, as the host merge counted them
        resize_pair_scratch(std::max(1, u.n_new));
        n_pairs = u.n_new;
        r.n_pt = n_pt; r.n_ee = n_ee; r.n_ph = n_ph;
    }

    // ---- step boundary ------------------------------------------------------------------
    void begin_step(const real3* xa, const real3* xh, real mu) {
        params.mu = mu;
        if (params.friction_enable) friction_snapshot(xa);
        else n_fr = 0;
        if (sb.n_surf_t || sb.n_surf_e) {
            update_boxes(xa, xh);
            build_trees(/*structural=*/true);   // §12.3 item 1: rebuild once per step
        }
    }

    // MS §13: snapshot from the carried-over active set, linearized at x^t (= the anchor here).
    void friction_snapshot(const real3* x_t) {
        n_fr = 0;
        if (!n_pairs) return;
        DeviceArray<int> keep(n_pairs);
        DeviceArray<real> force(n_pairs);
        DeviceArray<real4> weight(n_pairs);
        DeviceArray<real3> basis(size_t(2) * n_pairs);
        k_friction_snapshot<<<grid_for(n_pairs), kBlock>>>(n_pairs, type.data(), idx.data(), xi.data(), params.d_hat,
                                                           params.mu, lambda.data(), x_t, sb.plane_o.data(),
                                                           sb.plane_n.data(), sb.sphere_c.data(), sb.sphere_r1.data(),
                                                           sb.sphere_inverted.data(), sb.sphere_contact_id.data(),
                                                           sb.contact_id.data(),
                                                           sb.plane_contact_id.data(), sb.table_friction.data(),
                                                           sb.n_contact_ids, params.friction_normal_force_lambda,
                                                           keep.data(), force.data(), weight.data(), basis.data());
        CS_CUDA_KERNEL_CHECK();
        // compact on the device (once per step)
        n_fr = compact_friction(n_pairs, keep.data(), idx.data(), weight.data(), basis.data(), force.data(), fr_idx,
                                fr_weight, fr_basis, fr_force, fr_sel, fr_pool);
        if (!n_fr) return;
        fr_h2.resize(size_t(3) * n_fr);
        fr_scalar.resize(n_fr);
        build_incidence(fr_idx.data(), n_fr, fr_ptr, fr_count, fr_fill, fr_pair, fr_slot, n_fr_inc);
        log().debug("friction: {} pairs from {} active constraints", n_fr, n_pairs);
    }

    double debug_min_distance(const real3* x, int& violations) {
        dcd_min.set(d2bits(1e30));
        dcd_violations.set(0);
        // The last CCD's swept boxes and tree are a superset of the pairs close at x, so the
        // vertex-triangle candidates are re-enumerated from them in chunks and each chunk is
        // checked at x.
        if (sb.n_surf_t && sb.n_surf_v) {
            tri_tree.prepare_other(box_v.data(), size_t(sb.n_surf_v));
            CS_SYNC_POINT("lbvh prepare_other (dcd)");
            chunked_pass(sb.n_surf_v, chunk_pt, kRawCapacity,
                [&](size_t q0, size_t q1, int* d_count) { tri_tree.query_other_range_async(raw_pt.data(), size_t(kRawCapacity), box_v.data(), q0, q1, d_count); CS_SYNC_POINT("lbvh query pt (dcd)"); },
                [&](const int* d_n_raw) {
                    k_dcd_pt<<<kNarrowGrid, kBlock>>>(d_n_raw, kRawCapacity, raw_pt.data(), sb.surf_vert.data(),
                                                          sb.surf_tri.data(), sb.surf_vert_info.data(),
                                                          sb.surf_tri_info.data(), sb.surf_vert_thickness.data(),
                                                          sb.surf_tri_thickness.data(), sb.table_enabled.data(),
                                                          sb.n_contact_ids, x, dcd_min.data(), dcd_violations.data());
                    CS_CUDA_KERNEL_CHECK();
                });
        }
        violations = dcd_violations.get();
        const double shifted = bits2d(dcd_min.get());
        return shifted >= 1e29 ? 1e30 : shifted - 1.0;
    }
};

// =======================================================================================
// Public API
// =======================================================================================
ContactSystem::ContactSystem(const SceneBuffers& sb, const Scene& scene, const ContactDesc& desc)
    : impl_(new Impl(sb, scene, desc)) {}
ContactSystem::~ContactSystem() = default;

ContactParams& ContactSystem::params() { return impl_->params; }
const ContactParams& ContactSystem::params() const { return impl_->params; }

void ContactSystem::begin_step(const real3* x_anchor, const real3* x_hat, real mu) { impl_->begin_step(x_anchor, x_hat, mu); }
void ContactSystem::linearize(const real3* x_anchor) { impl_->linearize(x_anchor); }
void ContactSystem::update_slack(const real3* x_hat, const real3* x_anchor) { impl_->update_slack(x_hat, x_anchor); }
void ContactSystem::update_hessian(const real3* x_hat) { impl_->update_hessian(x_hat); }
void ContactSystem::add_gradient(const real3* x_hat, const real3* x_anchor, real3* G) { impl_->add_gradient(x_hat, x_anchor, G); }
const real* ContactSystem::diagonal_blocks() const { return impl_->diag.data(); }
void ContactSystem::spmv_add(const real3* x_in, real3* y, cudaStream_t stream, const int* skip) const {
    impl_->spmv_add(x_in, y, stream, skip);
}
ContactSpmvView ContactSystem::spmv_view() const {
    const Impl& I = *impl_;
    ContactSpmvView v;
    v.n_pairs = I.n_pairs; v.n_free = I.n_free;
    v.idx = I.idx.data(); v.grad4 = I.grad4.data(); v.coeff = I.coeff.data();
    v.pair_scalar = const_cast<real*>(I.pair_scalar.data());
    v.vp_ptr = I.vp_ptr.data(); v.vp_pair = I.vp_pair.data(); v.vp_grad = I.vp_grad.data();
    v.n_fr = I.n_fr;
    v.fr_idx = I.fr_idx.data(); v.fr_weight = I.fr_weight.data(); v.fr_basis = I.fr_basis.data();
    v.fr_h2 = I.fr_h2.data(); v.fr_scalar = const_cast<real2*>(I.fr_scalar.data());
    v.fr_ptr = I.fr_ptr.data(); v.fr_pair = I.fr_pair.data(); v.fr_slot = I.fr_slot.data();
    return v;
}

double ContactSystem::energy(const real3* x_hat, const real3* x_anchor) { return impl_->energy(x_hat, x_anchor); }
void ContactSystem::energy_to(const real3* x_hat, const real3* x_anchor, double* d_pairs, double* d_friction) {
    impl_->energy_to(x_hat, x_anchor, d_pairs, d_friction);
}
void ContactSystem::update_multipliers(const real3* x_hat, const real3* x_anchor) { impl_->update_multipliers(x_hat, x_anchor); }
CcdResult ContactSystem::ccd_and_update(const real3* x_anchor, const real3* x_hat, double alpha_cap) {
    return impl_->ccd_and_update(x_anchor, x_hat, alpha_cap);
}
void ContactSystem::set_d_hat(real d_hat) { impl_->params.d_hat = d_hat; }
void ContactSystem::set_mu(real mu) { impl_->params.mu = mu; }
int ContactSystem::n_pairs() const { return impl_->n_pairs; }
int ContactSystem::n_free() const { return impl_->n_free; }
bool ContactSystem::has_friction_pairs() const { return impl_->n_fr > 0; }
void ContactSystem::debug_set_hit_capacity(int entries) { impl_->hit_capacity = std::max(1, entries); }

double ContactSystem::debug_min_distance(const real3* x, int& n_violations) { return impl_->debug_min_distance(x, n_violations); }

}  // namespace cs
