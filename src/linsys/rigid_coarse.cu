// rigid_coarse.cu - per-body rigid-mode coarse correction (IS §6.6).
#include "linsys/rigid_coarse.cuh"
#include "linsys/pcg.cuh"
#include "linsys/linsys_detail.cuh"
#include "core/cuda_check.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace cs {

using linsys_detail::grid_for;
using linsys_detail::kBlock;

namespace {

// Column k (0..5) of P|_i for rho: translations e_k, then rotations e_{k-3} x rho.
__device__ __forceinline__ void mode_column(int k, const real3& rho, double& cx, double& cy, double& cz)
{
    const double rx = static_cast<double>(rho.x), ry = static_cast<double>(rho.y), rz = static_cast<double>(rho.z);
    switch (k) {
        case 0: cx = 1.0; cy = 0.0; cz = 0.0; break;
        case 1: cx = 0.0; cy = 1.0; cz = 0.0; break;
        case 2: cx = 0.0; cy = 0.0; cz = 1.0; break;
        case 3: cx = 0.0; cy = -rz; cz = ry; break;    // e_x x rho
        case 4: cx = rz; cy = 0.0; cz = -rx; break;    // e_y x rho
        default: cx = -ry; cy = rx; cz = 0.0; break;   // e_z x rho
    }
}

__device__ __forceinline__ bool row_held(const int* __restrict__ held, int i) { return held != nullptr && held[i] != 0; }

// One thread per active body: centroid of its free, non-held rows.
__global__ void k_centroid(int nb, const int* __restrict__ begin, const int* __restrict__ count,
                           const int* __restrict__ held, const real3* __restrict__ x, double* __restrict__ c)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double sx = 0.0, sy = 0.0, sz = 0.0;
    int m = 0;
    for (int i = begin[b]; i < begin[b] + count[b]; ++i) {
        if (row_held(held, i)) continue;
        sx += static_cast<double>(x[i].x); sy += static_cast<double>(x[i].y); sz += static_cast<double>(x[i].z);
        ++m;
    }
    const double inv = m > 0 ? 1.0 / static_cast<double>(m) : 0.0;
    c[3 * b] = sx * inv; c[3 * b + 1] = sy * inv; c[3 * b + 2] = sz * inv;
}

__global__ void k_rho(int n, const int* __restrict__ row_body, const double* __restrict__ c, const real3* __restrict__ x,
                      real3* __restrict__ rho)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = row_body[i];
    if (b < 0) { rho[i] = make_real3(real(0), real(0), real(0)); return; }
    rho[i] = make_real3(static_cast<real>(static_cast<double>(x[i].x) - c[3 * b]),
                        static_cast<real>(static_cast<double>(x[i].y) - c[3 * b + 1]),
                        static_cast<real>(static_cast<double>(x[i].z) - c[3 * b + 2]));
}

// v = column k of P over all active bodies (zero on held rows and rows without a body).
__global__ void k_mode_vector(int n, int k, const int* __restrict__ row_body, const int* __restrict__ held,
                              const real3* __restrict__ rho, real3* __restrict__ v)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (row_body[i] < 0 || row_held(held, i)) { v[i] = make_real3(real(0), real(0), real(0)); return; }
    double cx, cy, cz;
    mode_column(k, rho[i], cx, cy, cz);
    v[i] = make_real3(static_cast<real>(cx), static_cast<real>(cy), static_cast<real>(cz));
}

// One thread per active body: column l of G_b, G[k, l] = sum_i P|_i[:, k] . (w_i + C_i P|_i[:, l]).
__global__ void k_coarse_column(int nb, int l, const int* __restrict__ begin, const int* __restrict__ count,
                                const int* __restrict__ held, const real3* __restrict__ rho, const real3* __restrict__ w,
                                const real* __restrict__ C9, double* __restrict__ G)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double col[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = begin[b]; i < begin[b] + count[b]; ++i) {
        if (row_held(held, i)) continue;
        double ax = static_cast<double>(w[i].x), ay = static_cast<double>(w[i].y), az = static_cast<double>(w[i].z);
        if (C9 != nullptr) {
            double px, py, pz;
            mode_column(l, rho[i], px, py, pz);
            const real* c = C9 + 9 * static_cast<std::size_t>(i);
            ax += static_cast<double>(c[0]) * px + static_cast<double>(c[1]) * py + static_cast<double>(c[2]) * pz;
            ay += static_cast<double>(c[3]) * px + static_cast<double>(c[4]) * py + static_cast<double>(c[5]) * pz;
            az += static_cast<double>(c[6]) * px + static_cast<double>(c[7]) * py + static_cast<double>(c[8]) * pz;
        }
        for (int k = 0; k < 6; ++k) {
            double qx, qy, qz;
            mode_column(k, rho[i], qx, qy, qz);
            col[k] += qx * ax + qy * ay + qz * az;
        }
    }
    for (int k = 0; k < 6; ++k) G[36 * static_cast<std::size_t>(b) + 6 * k + l] = col[k];
}

// One thread per active body: y_b = Ginv_b (P_b^T r).
__global__ void k_coarse_restrict(const int* __restrict__ done, int nb, const int* __restrict__ begin,
                                  const int* __restrict__ count, const int* __restrict__ held,
                                  const real3* __restrict__ rho, const real3* __restrict__ r,
                                  const double* __restrict__ Ginv, double* __restrict__ y)
{
    if (done != nullptr && *done) return;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double c[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = begin[b]; i < begin[b] + count[b]; ++i) {
        if (row_held(held, i)) continue;
        const double rx = static_cast<double>(r[i].x), ry = static_cast<double>(r[i].y), rz = static_cast<double>(r[i].z);
        for (int k = 0; k < 6; ++k) {
            double qx, qy, qz;
            mode_column(k, rho[i], qx, qy, qz);
            c[k] += qx * rx + qy * ry + qz * rz;
        }
    }
    const double* g = Ginv + 36 * static_cast<std::size_t>(b);
    for (int k = 0; k < 6; ++k) {
        double s = 0.0;
        for (int l = 0; l < 6; ++l) s += g[6 * k + l] * c[l];
        y[6 * static_cast<std::size_t>(b) + k] = s;
    }
}

// One thread per row: z_i += P|_i y_b.
__global__ void k_coarse_prolong(const int* __restrict__ done, int n, const int* __restrict__ row_body,
                                 const int* __restrict__ held, const real3* __restrict__ rho,
                                 const double* __restrict__ y, real3* __restrict__ z)
{
    if (done != nullptr && *done) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = row_body[i];
    if (b < 0 || row_held(held, i)) return;
    const double* yb = y + 6 * static_cast<std::size_t>(b);
    double dx = 0.0, dy = 0.0, dz = 0.0;
    for (int k = 0; k < 6; ++k) {
        double qx, qy, qz;
        mode_column(k, rho[i], qx, qy, qz);
        dx += qx * yb[k]; dy += qy * yb[k]; dz += qz * yb[k];
    }
    z[i] = make_real3(static_cast<real>(static_cast<double>(z[i].x) + dx), static_cast<real>(static_cast<double>(z[i].y) + dy),
                      static_cast<real>(static_cast<double>(z[i].z) + dz));
}

}  // namespace

void RigidCoarse::setup(const std::vector<int>& begin, const std::vector<int>& count, int n_rows, int max_rows)
{
    n_rows_ = n_rows;
    std::vector<int> b, c, row(static_cast<std::size_t>(std::max(n_rows, 0)), -1);
    for (std::size_t k = 0; k < begin.size(); ++k) {
        if (count[k] <= 0 || count[k] > max_rows) continue;
        const int id = static_cast<int>(b.size());
        b.push_back(begin[k]);
        c.push_back(count[k]);
        for (int i = begin[k]; i < begin[k] + count[k] && i < n_rows; ++i) row[static_cast<std::size_t>(i)] = id;
    }
    n_active_ = static_cast<int>(b.size());
    if (n_active_ == 0) return;
    body_begin_.upload(b);
    body_count_.upload(c);
    row_body_.upload(row);
    const std::size_t n = static_cast<std::size_t>(n_rows);
    rho_.resize(n); v_.resize(n); w_.resize(n);
    centroid_.resize(3 * static_cast<std::size_t>(n_active_));
    G_.resize(36 * static_cast<std::size_t>(n_active_));
    Ginv_.resize(36 * static_cast<std::size_t>(n_active_));
    y_.resize(6 * static_cast<std::size_t>(n_active_));
    h_G_.resize(36 * static_cast<std::size_t>(n_active_));
    h_Ginv_.resize(36 * static_cast<std::size_t>(n_active_));
}

void RigidCoarse::build(const LinearOperator& A, const real* contact_diag9, const real3* x, const int* held)
{
    if (n_active_ == 0) return;
    held_ = held;
    const int n = n_rows_, nb = n_active_;
    const unsigned gb = grid_for(nb), gn = grid_for(n);
    k_centroid<<<gb, kBlock>>>(nb, body_begin_.data(), body_count_.data(), held, x, centroid_.data());
    CS_CUDA_KERNEL_CHECK();
    k_rho<<<gn, kBlock>>>(n, row_body_.data(), centroid_.data(), x, rho_.data());
    CS_CUDA_KERNEL_CHECK();
    for (int l = 0; l < 6; ++l) {
        k_mode_vector<<<gn, kBlock>>>(n, l, row_body_.data(), held, rho_.data(), v_.data());
        CS_CUDA_KERNEL_CHECK();
        A.apply(v_.data(), w_.data(), 0, nullptr);
        k_coarse_column<<<gb, kBlock>>>(nb, l, body_begin_.data(), body_count_.data(), held, rho_.data(), w_.data(),
                                        contact_diag9, G_.data());
        CS_CUDA_KERNEL_CHECK();
    }
    G_.download(h_G_.data(), h_G_.size());
    for (int b = 0; b < nb; ++b) {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> Gm(h_G_.data() + 36 * static_cast<std::size_t>(b));
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> Gi(h_Ginv_.data() + 36 * static_cast<std::size_t>(b));
        const Eigen::Matrix<double, 6, 6> S = 0.5 * (Gm + Gm.transpose());
        Gi.setZero();
        if (!S.allFinite()) continue;   // leave this body uncorrected
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(S);
        const double lmax = es.eigenvalues().cwiseAbs().maxCoeff();
        if (!(lmax > 0.0)) continue;
        Eigen::Matrix<double, 6, 6> inv = Eigen::Matrix<double, 6, 6>::Zero();
        for (int k = 0; k < 6; ++k) {
            const double lam = es.eigenvalues()[k];
            if (lam > 1e-12 * lmax) inv += (1.0 / lam) * es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose();
        }
        Gi = inv;
    }
    Ginv_.upload(h_Ginv_);
}

void RigidCoarse::apply(const real3* r, real3* z, cudaStream_t stream, const int* done) const
{
    if (n_active_ == 0) return;
    const int n = n_rows_, nb = n_active_;
    k_coarse_restrict<<<grid_for(nb), kBlock, 0, stream>>>(done, nb, body_begin_.data(), body_count_.data(), held_,
                                                           rho_.data(), r, Ginv_.data(), y_.data());
    CS_CUDA_KERNEL_CHECK();
    k_coarse_prolong<<<grid_for(n), kBlock, 0, stream>>>(done, n, row_body_.data(), held_, rho_.data(), y_.data(), z);
    CS_CUDA_KERNEL_CHECK();
}

}  // namespace cs
