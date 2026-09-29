#pragma once
// POD views of the contact system's SpMV data for kernels that fuse the contact term with the
// rest of the CG iteration (implementation spec §12.3 item 14). The device functions here are
// the bodies of k_pair_sigma / k_vertex_gather / k_friction_spmv_scalar / k_friction_gather in
// contact_system.cu, expression for expression, so a fused iteration forms bitwise the same
// products as the kernel path; keep them in step.
#ifndef CS_CONTACT_SPMV_VIEW_CUH
#define CS_CONTACT_SPMV_VIEW_CUH

#include "core/typedef.cuh"

namespace cs {

struct ContactSpmvView {
    int n_pairs = 0, n_free = 0;
    const int4* idx = nullptr;
    const real3* grad4 = nullptr;
    const real* coeff = nullptr;
    real* pair_scalar = nullptr;
    const int* vp_ptr = nullptr;
    const int* vp_pair = nullptr;
    const real3* vp_grad = nullptr;
    int n_fr = 0;
    const int4* fr_idx = nullptr;
    const real4* fr_weight = nullptr;
    const real3* fr_basis = nullptr;
    const real* fr_h2 = nullptr;
    real2* fr_scalar = nullptr;
    const int* fr_ptr = nullptr;
    const int* fr_pair = nullptr;
    const unsigned char* fr_slot = nullptr;
};

// pass 1 of the contact term for pair i (k_pair_sigma)
__device__ __forceinline__ void contact_pair_sigma(const ContactSpmvView& V, int i, const real3* __restrict__ x_in) {
    const int4 id = V.idx[i];
    real s = real(0);
    if (id.x < V.n_free) s += dot(V.grad4[4 * i + 0], x_in[id.x]);
    if (id.z >= 0) {
        if (id.y < V.n_free) s += dot(V.grad4[4 * i + 1], x_in[id.y]);
        if (id.z < V.n_free) s += dot(V.grad4[4 * i + 2], x_in[id.z]);
        if (id.w < V.n_free) s += dot(V.grad4[4 * i + 3], x_in[id.w]);
    }
    V.pair_scalar[i] = V.coeff[i] * s;
}

// pass 2 of the contact term for vertex v (k_vertex_gather): y_v += sum over its pairs
__device__ __forceinline__ void contact_vertex_gather(const ContactSpmvView& V, int v, real3* __restrict__ y) {
    const int b = V.vp_ptr[v], e = V.vp_ptr[v + 1];
    if (b == e) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = b; k < e; ++k) {
        const double s = double(V.pair_scalar[V.vp_pair[k]]);
        const real3 g = V.vp_grad[k];
        ax += s * double(g.x);
        ay += s * double(g.y);
        az += s * double(g.z);
    }
    y[v] = y[v] + make_real3(real(ax), real(ay), real(az));
}

// friction pass 1 for pair i (k_friction_spmv_scalar)
__device__ __forceinline__ void friction_pair_scalar(const ContactSpmvView& V, int i, const real3* __restrict__ x_in) {
    const int4 id = V.fr_idx[i];
    const real4 w = V.fr_weight[i];
    real3 rel = make_real3(real(0), real(0), real(0));
    if (id.x < V.n_free) rel = rel + w.x * x_in[id.x];
    if (id.z >= 0) {
        if (id.y < V.n_free) rel = rel + w.y * x_in[id.y];
        if (id.z < V.n_free) rel = rel + w.z * x_in[id.z];
        if (id.w < V.n_free) rel = rel + w.w * x_in[id.w];
    }
    const real q0 = dot(V.fr_basis[2 * i + 0], rel);
    const real q1 = dot(V.fr_basis[2 * i + 1], rel);
    const real a = V.fr_h2[3 * i + 0], b = V.fr_h2[3 * i + 1], c = V.fr_h2[3 * i + 2];   // [[a,b],[b,c]]
    V.fr_scalar[i] = make_real2(a * q0 + b * q1, b * q0 + c * q1);
}

// friction pass 2 for vertex v (k_friction_gather)
__device__ __forceinline__ void friction_vertex_gather(const ContactSpmvView& V, int v, real3* __restrict__ y) {
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = V.fr_ptr[v]; k < V.fr_ptr[v + 1]; ++k) {
        const int i = V.fr_pair[k];
        const real4 w4 = V.fr_weight[i];
        const real w = V.fr_slot[k] == 0 ? w4.x : (V.fr_slot[k] == 1 ? w4.y : (V.fr_slot[k] == 2 ? w4.z : w4.w));
        const real2 p = V.fr_scalar[i];
        const real3 t = w * (p.x * V.fr_basis[2 * i + 0] + p.y * V.fr_basis[2 * i + 1]);
        ax += double(t.x); ay += double(t.y); az += double(t.z);
    }
    y[v] = y[v] + make_real3(real(ax), real(ay), real(az));
}

}  // namespace cs

#endif  // CS_CONTACT_SPMV_VIEW_CUH
