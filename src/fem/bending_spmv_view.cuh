#pragma once
// POD view of the bending system's SpMV data for the fused CG iteration (IS §12.3 item 14);
// the device functions are the bodies of k_sigma / k_gather in bending.cu, expression for
// expression. Keep them in step.
#ifndef CS_BENDING_SPMV_VIEW_CUH
#define CS_BENDING_SPMV_VIEW_CUH

#include "core/typedef.cuh"

namespace cs {

namespace bending_detail {
__device__ __forceinline__ real dot3(const real3& u, const real3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
__device__ __forceinline__ real3 add3(const real3& u, const real3& v) { return make_real3(u.x + v.x, u.y + v.y, u.z + v.z); }
}  // namespace bending_detail

struct BendingSpmvView {
    int n = 0, n_free = 0;
    real h2 = real(0);
    const int4* idx = nullptr;
    const real3* grad = nullptr;   // 4 per hinge
    const real* coeff = nullptr;
    real* sigma = nullptr;
    const int* vh_ptr = nullptr;
    const int* vh_hinge = nullptr;
    const unsigned char* vh_slot = nullptr;
};

__device__ __forceinline__ void bending_hinge_sigma(const BendingSpmvView& V, int h, const real3* __restrict__ p) {
    const int4 id = V.idx[h];
    const int v[4] = {id.x, id.y, id.z, id.w};
    real s = real(0);
    for (int k = 0; k < 4; ++k)
        if (v[k] < V.n_free) s += bending_detail::dot3(V.grad[4 * h + k], p[v[k]]);
    V.sigma[h] = V.coeff[h] * s;
}

__device__ __forceinline__ void bending_vertex_gather(const BendingSpmvView& V, int v, real3* __restrict__ y) {
    const int b = V.vh_ptr[v], e = V.vh_ptr[v + 1];
    if (b == e) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = b; k < e; ++k) {
        const int h = V.vh_hinge[k];
        const double s = double(V.sigma[h]) * double(V.h2);
        const real3 g = V.grad[4 * h + V.vh_slot[k]];
        ax += s * double(g.x);
        ay += s * double(g.y);
        az += s * double(g.z);
    }
    y[v] = bending_detail::add3(y[v], make_real3(real(ax), real(ay), real(az)));
}

}  // namespace cs

#endif  // CS_BENDING_SPMV_VIEW_CUH
