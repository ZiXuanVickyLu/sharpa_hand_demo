// bending.cu - discrete-shell bending, Gauss-Newton form (math spec §3.7). The dihedral angle
// and its gradient follow Codim-IPC's DIHEDRAL_ANGLE.h as distilled in libuipc's
// dihedral_angle.h: hinge (a, b, c, d) with mid-edge b-c and opposite vertices a and d.
#include "fem/bending.cuh"
#include <Eigen/Geometry>
#include "core/cuda_check.h"
#include "core/log.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cs {

namespace {

constexpr int kBlock = 256;
inline unsigned grid_for(int n) { return unsigned(std::max(1, (n + kBlock - 1) / kBlock)); }

__device__ __forceinline__ real3 cross3(const real3& u, const real3& v) {
    return make_real3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
__device__ __forceinline__ real dot3(const real3& u, const real3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
__device__ __forceinline__ real3 scale3(const real3& u, real s) { return make_real3(u.x * s, u.y * s, u.z * s); }
__device__ __forceinline__ real3 sub3(const real3& u, const real3& v) { return make_real3(u.x - v.x, u.y - v.y, u.z - v.z); }
__device__ __forceinline__ real3 add3(const real3& u, const real3& v) { return make_real3(u.x + v.x, u.y + v.y, u.z + v.z); }

// Signed dihedral angle of the hinge (a, b, c, d): 0 when flat, sign by the side d bends to.
__device__ __forceinline__ real dihedral(const real3& a, const real3& b, const real3& c, const real3& d) {
    const real3 n1 = cross3(sub3(b, a), sub3(c, a));
    const real3 n2 = cross3(sub3(c, d), sub3(b, d));
    const real den = sqrt(dot3(n1, n1) * dot3(n2, n2));
    real cs = den > real(0) ? dot3(n1, n2) / den : real(1);
    cs = fmin(real(1), fmax(real(-1), cs));
    real th = acos(cs);
    if (dot3(cross3(n2, n1), sub3(b, c)) < real(0)) th = -th;
    return th;
}

// grad theta with respect to a, b, c, d.
__device__ __forceinline__ void dihedral_gradient(const real3& a, const real3& b, const real3& c, const real3& d,
                                                  real3* g) {
    const real3 e0 = sub3(c, b), e1 = sub3(a, b), e2 = sub3(d, b), e3 = sub3(a, c), e4 = sub3(d, c);
    const real3 n1 = cross3(e0, e1), n2 = cross3(e2, e0);
    const real n1s = fmax(dot3(n1, n1), real(1e-300)), n2s = fmax(dot3(n2, n2), real(1e-300));
    const real e0n = fmax(sqrt(dot3(e0, e0)), real(1e-150));
    g[0] = scale3(n1, -e0n / n1s);
    g[1] = add3(scale3(n1, -dot3(e0, e3) / (e0n * n1s)), scale3(n2, -dot3(e0, e4) / (e0n * n2s)));
    g[2] = add3(scale3(n1, dot3(e0, e1) / (e0n * n1s)), scale3(n2, dot3(e0, e2) / (e0n * n2s)));
    g[3] = scale3(n2, -e0n / n2s);
}

__global__ void k_cache(int n, const int4* __restrict__ idx, const real3* __restrict__ x, real* __restrict__ theta,
                        real3* __restrict__ grad) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    const int4 id = idx[h];
    const real3 a = x[id.x], b = x[id.y], c = x[id.z], d = x[id.w];
    theta[h] = dihedral(a, b, c, d);
    dihedral_gradient(a, b, c, d, grad + 4 * h);
}

__global__ void k_energy(int n, const int4* __restrict__ idx, const real* __restrict__ coeff,
                         const real* __restrict__ rest, const real3* __restrict__ x, double* __restrict__ out) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    const int4 id = idx[h];
    const double dth = double(dihedral(x[id.x], x[id.y], x[id.z], x[id.w])) - double(rest[h]);
    out[h] = 0.5 * double(coeff[h]) * dth * dth;   // c_h/2 (theta - rest)^2 = k_b |e|/h (..)^2
}

// pass 1 of the SpMV: s_h = c_h sum_w grad_{h,w} . p_w over free w
__global__ void k_sigma(const int* __restrict__ skip, int n, int n_free, const int4* __restrict__ idx, const real3* __restrict__ grad,
                        const real* __restrict__ coeff, const real3* __restrict__ p, real* __restrict__ sigma) {
    if (skip != nullptr && *skip) return;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    const int4 id = idx[h];
    const int v[4] = {id.x, id.y, id.z, id.w};
    real s = real(0);
    for (int k = 0; k < 4; ++k)
        if (v[k] < n_free) s += dot3(grad[4 * h + k], p[v[k]]);
    sigma[h] = coeff[h] * s;
}

// gradient scalar: s_h = c_h (theta_h - rest_h)
__global__ void k_grad_scalar(int n, const real* __restrict__ theta, const real* __restrict__ rest,
                              const real* __restrict__ coeff, real* __restrict__ sigma) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    sigma[h] = coeff[h] * (theta[h] - rest[h]);
}

// pass 2: y_v += h2 sum_{k in row v} sigma[hinge(k)] grad[hinge(k), slot(k)]
__global__ void k_gather(const int* __restrict__ skip, int n_free, real h2, const int* __restrict__ ptr, const int* __restrict__ hinge,
                         const unsigned char* __restrict__ slot, const real3* __restrict__ grad,
                         const real* __restrict__ sigma, real3* __restrict__ y) {
    if (skip != nullptr && *skip) return;
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    const int b = ptr[v], e = ptr[v + 1];
    if (b == e) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = b; k < e; ++k) {
        const int h = hinge[k];
        const double s = double(sigma[h]) * double(h2);
        const real3 g = grad[4 * h + slot[k]];
        ax += s * double(g.x);
        ay += s * double(g.y);
        az += s * double(g.z);
    }
    y[v] = add3(y[v], make_real3(real(ax), real(ay), real(az)));
}

__global__ void k_diag(int n_free, real h2, const int* __restrict__ ptr, const int* __restrict__ hinge,
                       const unsigned char* __restrict__ slot, const real3* __restrict__ grad,
                       const real* __restrict__ coeff, real* __restrict__ diag) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_free) return;
    const int b = ptr[v], e = ptr[v + 1];
    if (b == e) return;
    double m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = b; k < e; ++k) {
        const int h = hinge[k];
        const double c = double(coeff[h]) * double(h2);
        const real3 g = grad[4 * h + slot[k]];
        const double gg[3] = {double(g.x), double(g.y), double(g.z)};
        for (int r = 0; r < 3; ++r)
            for (int cc = 0; cc < 3; ++cc) m[3 * r + cc] += c * gg[r] * gg[cc];
    }
    for (int i = 0; i < 9; ++i) diag[9 * v + i] += real(m[i]);
}

__global__ void k_reduce_sum(int n, const double* __restrict__ in, double* __restrict__ out) {
    // single block, fixed order: n is the hinge count of a cloth mesh (thousands to millions),
    // one thread walks a fixed stride and a shared tree folds the 256 partials
    __shared__ double sh[kBlock];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += kBlock) acc += in[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int w = kBlock / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = sh[0];
}

}  // namespace

struct BendingSystem::Impl {
    int n = 0, n_free = 0;
    DeviceArray<int4> idx;
    DeviceArray<real> coeff, rest, theta, sigma;
    DeviceArray<real3> grad;                     // 4 per hinge
    DeviceArray<int> vh_ptr, vh_hinge;
    DeviceArray<unsigned char> vh_slot;
    DeviceArray<double> scratch;

    explicit Impl(const Scene& scene) : n_free(scene.n_free) {
        // Hinges from each cloth body's triangles (its surf_tris are all of them, MS §4.1).
        std::vector<int4> hidx;
        std::vector<real> hcoeff, hrest;
        for (size_t b = 0; b < scene.bodies.size(); ++b) {
            const BodyInfo& bi = scene.bodies[b];
            if (bi.kind != BodyKind::Deformable || bi.material_id < 0) continue;
            const MaterialDesc& mat = scene.materials[size_t(bi.material_id)];
            if (!is_cloth_model(mat.model) || mat.k_bend <= 0.0) continue;
            // edge (lo, hi) -> up to two (triangle index, opposite vertex)
            std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> edge_faces;
            for (size_t f = 0; f < scene.surf_tris.size(); ++f) {
                if (scene.surf_tri_body[f] != int(b)) continue;
                const auto& t = scene.surf_tris[f];
                for (int k = 0; k < 3; ++k) {
                    int u = t[size_t(k)], w = t[size_t((k + 1) % 3)], o = t[size_t((k + 2) % 3)];
                    if (u > w) std::swap(u, w);
                    edge_faces[{u, w}].push_back({int(f), o});
                }
            }
            for (const auto& [edge, faces] : edge_faces) {
                if (faces.size() != 2) continue;   // boundary edge (or non-manifold): no hinge
                const int a = faces[0].second, bb = edge.first, c = edge.second, d = faces[1].second;
                if (scene.is_prescribed[size_t(a)] && scene.is_prescribed[size_t(bb)] && scene.is_prescribed[size_t(c)] &&
                    scene.is_prescribed[size_t(d)])
                    continue;  // no free DOF: constant energy
                const Vec3 A = scene.x0.col(a), B = scene.x0.col(bb), C = scene.x0.col(c), D = scene.x0.col(d);
                const double len = (C - B).norm();
                const double area1 = 0.5 * ((B - A).cross(C - A)).norm();
                const double area2 = 0.5 * ((C - D).cross(B - D)).norm();
                const double hbar = (area1 + area2) / (3.0 * len);
                if (!(len > 0.0) || !(hbar > 0.0)) continue;
                // rest angle with the same formula the kernel uses
                const Vec3 n1 = (B - A).cross(C - A), n2 = (C - D).cross(B - D);
                double cs = n1.dot(n2) / std::sqrt(n1.squaredNorm() * n2.squaredNorm());
                cs = std::min(1.0, std::max(-1.0, cs));
                double th = std::acos(cs);
                if (n2.cross(n1).dot(B - C) < 0.0) th = -th;
                hidx.push_back(make_int4(a, bb, c, d));
                hcoeff.push_back(real(2.0 * mat.k_bend * len / hbar));
                hrest.push_back(real(th));
            }
        }
        n = int(hidx.size());
        if (n == 0) return;
        idx.upload(hidx);
        coeff.upload(hcoeff);
        rest.upload(hrest);
        theta.resize(size_t(n));
        sigma.resize(size_t(n));
        grad.resize(size_t(4) * n);
        // vertex -> hinge incidence over free vertices, counting sort in hinge order
        std::vector<int> ptr(size_t(n_free) + 1, 0);
        for (const int4& h : hidx) {
            const int v[4] = {h.x, h.y, h.z, h.w};
            for (int k = 0; k < 4; ++k)
                if (v[k] < n_free) ++ptr[size_t(v[k]) + 1];
        }
        for (int i = 0; i < n_free; ++i) ptr[size_t(i) + 1] += ptr[i];
        std::vector<int> fill(ptr.begin(), ptr.end() - 1);
        std::vector<int> hh;
        std::vector<unsigned char> hs;
        hh.resize(static_cast<size_t>(ptr[n_free]));
        hs.resize(static_cast<size_t>(ptr[n_free]));
        for (int h = 0; h < n; ++h) {
            const int v[4] = {hidx[h].x, hidx[h].y, hidx[h].z, hidx[h].w};
            for (int k = 0; k < 4; ++k)
                if (v[k] < n_free) {
                    const int pos = fill[v[k]]++;
                    hh[size_t(pos)] = h;
                    hs[size_t(pos)] = (unsigned char)k;
                }
        }
        vh_ptr.upload(ptr);
        vh_hinge.upload(hh);
        vh_slot.upload(hs);
        scratch.resize(size_t(n) + 1);
        log().info("BendingSystem: {} hinges over {} free vertices", n, n_free);
    }
};

BendingSystem::BendingSystem(const Scene& scene) : impl_(new Impl(scene)) {}
BendingSystem::~BendingSystem() { delete impl_; }
int BendingSystem::n_hinges() const { return impl_->n; }

void BendingSystem::cache(const real3* x) {
    Impl& I = *impl_;
    if (!I.n) return;
    k_cache<<<grid_for(I.n), kBlock>>>(I.n, I.idx.data(), x, I.theta.data(), I.grad.data());
    CS_CUDA_KERNEL_CHECK();
}

void BendingSystem::energy_to(const real3* x, double* d_out) {
    Impl& I = *impl_;
    if (!I.n) {
        k_reduce_sum<<<1, kBlock>>>(0, nullptr, d_out);
        CS_CUDA_KERNEL_CHECK();
        return;
    }
    k_energy<<<grid_for(I.n), kBlock>>>(I.n, I.idx.data(), I.coeff.data(), I.rest.data(), x, I.scratch.data());
    CS_CUDA_KERNEL_CHECK();
    k_reduce_sum<<<1, kBlock>>>(I.n, I.scratch.data(), d_out);
    CS_CUDA_KERNEL_CHECK();
}

double BendingSystem::energy(const real3* x, real h2) {
    Impl& I = *impl_;
    if (!I.n) return 0.0;
    k_energy<<<grid_for(I.n), kBlock>>>(I.n, I.idx.data(), I.coeff.data(), I.rest.data(), x, I.scratch.data());
    CS_CUDA_KERNEL_CHECK();
    k_reduce_sum<<<1, kBlock>>>(I.n, I.scratch.data(), I.scratch.data() + I.n);
    CS_CUDA_KERNEL_CHECK();
    double e = 0.0;
    CS_CUDA_CHECK(cudaMemcpy(&e, I.scratch.data() + I.n, sizeof(double), cudaMemcpyDeviceToHost));
    return double(h2) * e;
}

void BendingSystem::add_gradient(real h2, real3* G) {
    Impl& I = *impl_;
    if (!I.n || !I.n_free) return;
    k_grad_scalar<<<grid_for(I.n), kBlock>>>(I.n, I.theta.data(), I.rest.data(), I.coeff.data(), I.sigma.data());
    CS_CUDA_KERNEL_CHECK();
    k_gather<<<grid_for(I.n_free), kBlock>>>(nullptr, I.n_free, h2, I.vh_ptr.data(), I.vh_hinge.data(), I.vh_slot.data(),
                                             I.grad.data(), I.sigma.data(), G);
    CS_CUDA_KERNEL_CHECK();
}

BendingSpmvView BendingSystem::spmv_view(real h2) const {
    const Impl& I = *impl_;
    BendingSpmvView v;
    v.n = I.n; v.n_free = I.n_free; v.h2 = h2;
    v.idx = I.idx.data(); v.grad = I.grad.data(); v.coeff = I.coeff.data();
    v.sigma = const_cast<real*>(I.sigma.data());
    v.vh_ptr = I.vh_ptr.data(); v.vh_hinge = I.vh_hinge.data(); v.vh_slot = I.vh_slot.data();
    return v;
}

void BendingSystem::spmv_add(real h2, const real3* p, real3* y, cudaStream_t stream, const int* skip) const {
    Impl& I = *impl_;
    if (!I.n || !I.n_free) return;
    k_sigma<<<grid_for(I.n), kBlock, 0, stream>>>(skip, I.n, I.n_free, I.idx.data(), I.grad.data(), I.coeff.data(), p, I.sigma.data());
    CS_CUDA_KERNEL_CHECK();
    k_gather<<<grid_for(I.n_free), kBlock, 0, stream>>>(skip, I.n_free, h2, I.vh_ptr.data(), I.vh_hinge.data(), I.vh_slot.data(),
                                             I.grad.data(), I.sigma.data(), y);
    CS_CUDA_KERNEL_CHECK();
}

void BendingSystem::add_diagonal_blocks(real h2, real* diag9) const {
    Impl& I = *impl_;
    if (!I.n || !I.n_free) return;
    k_diag<<<grid_for(I.n_free), kBlock>>>(I.n_free, h2, I.vh_ptr.data(), I.vh_hinge.data(), I.vh_slot.data(),
                                           I.grad.data(), I.coeff.data(), diag9);
    CS_CUDA_KERNEL_CHECK();
}

}  // namespace cs
