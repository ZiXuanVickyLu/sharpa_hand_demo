#pragma once
// Two-level additive preconditioner: per-body rigid-mode coarse correction on top of block-Jacobi
// (doc/al-ipc-implementation-spec.md §6.6, "Rigid-mode coarse correction").
//   M^-1 = D^-1 + sum_b P_b G_b^+ P_b^T,   P_b|_i = [ I3 | -[rho_i]x ],  rho_i = x_i - c_b
//   G_b  = P_b^T (A_nc P_b) + sum_{i in b} P_b|_i^T C_i P_b|_i
// A_nc is the operator WITHOUT its inter-body (contact) part, C_i the per-row diagonal block of
// that part. Per-body loops are serial in one device thread, so every sum has a fixed order.
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace cs {

struct LinearOperator;

class RigidCoarse {
public:
    // Free-row range [begin, begin + count) of every body. Bodies with more than max_rows rows
    // (or none) get no correction.
    void setup(const std::vector<int>& begin, const std::vector<int>& count, int n_rows, int max_rows);
    // Build for the current Newton step. `A_no_contact` must not couple rows of different bodies;
    // `contact_diag9` (9 per row, row-major, may be null) is the per-row diagonal block of the
    // inter-body part; `x` the current positions of the rows; `held` (may be null) marks rows
    // that are pinned this step. Runs on the default stream and synchronises once.
    void build(const LinearOperator& A_no_contact, const real* contact_diag9, const real3* x, const int* held);
    // z += P G^+ P^T r on `stream`; returns at once when `done` is non-null and *done != 0.
    void apply(const real3* r, real3* z, cudaStream_t stream, const int* done) const;
    bool active() const { return n_active_ > 0; }
    int bodies() const { return n_active_; }

private:
    int n_rows_ = 0, n_active_ = 0;
    const int* held_ = nullptr;
    DeviceArray<int> body_begin_, body_count_;   // active bodies only
    DeviceArray<int> row_body_;                  // per row: index into the active list, or -1
    DeviceArray<real3> rho_;                     // per row: x_i - c_b
    DeviceArray<real3> v_, w_;                   // mode vector and its image
    DeviceArray<double> centroid_;               // 3 per active body
    DeviceArray<double> G_, Ginv_;               // 36 per active body, row-major
    mutable DeviceArray<double> y_;              // 6 per active body
    std::vector<double> h_G_, h_Ginv_;
};

}  // namespace cs
