#include "cuda_smoke.h"
#include "typedef.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace cs {

namespace {

__global__ void k_smoke(const real3* __restrict__ a, const real3* __restrict__ b, real* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = dot(a[i], b[i]) + length(cross(a[i], b[i]));
}

}  // namespace

int cuda_smoke() {
    const int n = 4096;
    std::vector<real3> ha(n), hb(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = make_real3(real(1), real(0), real(0));
        hb[i] = make_real3(real(0), real(1), real(0));
    }
    real3 *da = nullptr, *db = nullptr;
    real* dout = nullptr;
    if (cudaMalloc(&da, n * sizeof(real3)) != cudaSuccess) return 10;
    if (cudaMalloc(&db, n * sizeof(real3)) != cudaSuccess) return 11;
    if (cudaMalloc(&dout, n * sizeof(real)) != cudaSuccess) return 12;
    cudaMemcpy(da, ha.data(), n * sizeof(real3), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), n * sizeof(real3), cudaMemcpyHostToDevice);
    k_smoke<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "cuda_smoke: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 13;
    }
    std::vector<real> hout(n);
    cudaMemcpy(hout.data(), dout, n * sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    for (int i = 0; i < n; ++i) {
        // dot = 0, |cross| = 1
        if (hout[i] < real(0.999999) || hout[i] > real(1.000001)) return 14;
    }
    return 0;
}

}  // namespace cs
