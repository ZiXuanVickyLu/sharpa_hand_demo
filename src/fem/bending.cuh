#pragma once
// Discrete-shell bending of cloth bodies (math spec §3.7, implementation spec §6.4b), in the
// Gauss-Newton form: E_h = k_b (|e|/h) (theta - theta_rest)^2 per interior edge, gradient
// c_h (theta - theta_rest) grad theta and Hessian c_h grad theta grad theta^T with
// c_h = 2 k_b |e| / h, which is PSD and rank one. It therefore never touches the BSR matrix:
// like a contact pair (§6.5) it is applied matrix-free through a per-hinge scalar and a
// per-vertex gather over a static vertex->hinge incidence, and only its 3x3 diagonal blocks
// enter the block-Jacobi preconditioner. Everything the solver sees is scaled by h^2 like the
// rest of the elastic energy. Deterministic: fixed gather order, no atomics.
#include "core/typedef.cuh"
#include "fem/bending_spmv_view.cuh"
#include "core/device_buffer.cuh"
#include "scene/scene.h"

namespace cs {

class BendingSystem {
public:
    // Extracts every interior edge of every cloth body whose material has k_bend > 0.
    explicit BendingSystem(const Scene& scene);
    ~BendingSystem();

    int n_hinges() const;
    bool empty() const { return n_hinges() == 0; }

    // Per-hinge theta and grad theta at x (once per Newton step, MS §3.7: the Gauss-Newton
    // Hessian is frozen at the linearization point like the contact slack).
    void cache(const real3* x);
    // sum_h h2 k_b (|e|/h) (theta(x) - theta_rest)^2, recomputing theta at x (line search).
    double energy(const real3* x, real h2);
    // The raw hinge sum (not scaled by h2) written to the device double at d_out, no host sync.
    void energy_to(const real3* x, double* d_out);
    // G_v += h2 c_h (theta_h - theta_rest) grad theta_{h,v} over free vertices, from the cache.
    void add_gradient(real h2, real3* G);
    // y_v += h2 sum_h c_h grad theta_{h,v} (sum_w grad theta_{h,w} . p_w), free vertices only.
    void spmv_add(real h2, const real3* p, real3* y, cudaStream_t stream = 0, const int* skip = nullptr) const;
    // POD view of the SpMV data for the fused CG kernel (IS §12.3 item 14).
    struct BendingSpmvView spmv_view(real h2) const;
    // diag_v (9 per free vertex, row-major) += h2 sum_h c_h grad theta_{h,v} grad theta_{h,v}^T.
    void add_diagonal_blocks(real h2, real* diag9) const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace cs
