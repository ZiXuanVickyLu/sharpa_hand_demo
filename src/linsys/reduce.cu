// reduce.cu - deterministic two-level reductions (doc/al-ipc-implementation-spec.md §11.1).
//
// Level 1: the input is cut into `blocks` contiguous chunks, a function of n alone; every
// thread of a block folds its strided elements sequentially in double, then the block folds
// its kBlock partials with a fixed shared-memory tree. Level 2: a single block folds the
// per-block partials the same way and writes the result behind them in `scratch`. No
// atomics and no dependence on the launch geometry beyond n, so a given input reduces to a
// bitwise identical result on every run.
#include "linsys/reduce.cuh"

#include <cmath>
#include <cstddef>
#include <limits>

namespace cs {
namespace {

constexpr int kBlock = 256;       // threads per block; power of two for the tree
constexpr int kMaxBlocks = 1024;  // level-2 input bound (4 partials per thread)
constexpr int kMinChunk = kBlock * 8;

struct SumOp {
    __host__ __device__ static double identity() { return 0.0; }
    __device__ static double apply(double a, double b) { return a + b; }
};
struct MaxOp {
    __host__ __device__ static double identity() { return -std::numeric_limits<double>::infinity(); }
    __device__ static double apply(double a, double b) { return fmax(a, b); }
};
struct MinOp {
    __host__ __device__ static double identity() { return std::numeric_limits<double>::infinity(); }
    __device__ static double apply(double a, double b) { return fmin(a, b); }
};

struct LoadReal {
    const real* a;
    __device__ double operator()(int i) const { return static_cast<double>(a[i]); }
};
struct LoadDot {
    const real3* a;
    const real3* b;
    __device__ double operator()(int i) const
    {
        const real3 x = a[i];
        const real3 y = b[i];
        return static_cast<double>(x.x) * static_cast<double>(y.x) + static_cast<double>(x.y) * static_cast<double>(y.y) +
               static_cast<double>(x.z) * static_cast<double>(y.z);
    }
};
struct LoadSumSq {
    const real3* a;
    __device__ double operator()(int i) const
    {
        const real3 x = a[i];
        const double x0 = x.x, x1 = x.y, x2 = x.z;
        return x0 * x0 + x1 * x1 + x2 * x2;
    }
};
struct LoadMaxAbs {
    const real3* a;
    __device__ double operator()(int i) const
    {
        const real3 x = a[i];
        return fmax(fmax(fabs(static_cast<double>(x.x)), fabs(static_cast<double>(x.y))), fabs(static_cast<double>(x.z)));
    }
};
struct LoadMaxAbsDiff {
    const real3* x;
    const real3* y;
    __device__ double operator()(int i) const
    {
        const real3 a = x[i];
        const real3 b = y[i];
        const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
        const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
        const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
        return fmax(fmax(fabs(dx), fabs(dy)), fabs(dz));
    }
};

template <class Op>
__device__ __forceinline__ double block_fold(double acc, double* sh)
{
    sh[threadIdx.x] = acc;
    __syncthreads();
#pragma unroll
    for (int w = kBlock / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) sh[threadIdx.x] = Op::apply(sh[threadIdx.x], sh[threadIdx.x + w]);
        __syncthreads();
    }
    return sh[0];
}

template <class Op, class Load>
__global__ void __launch_bounds__(kBlock) k_reduce_level1(int n, int chunk, Load load, double* __restrict__ partials)
{
    __shared__ double sh[kBlock];
    const long long begin = static_cast<long long>(blockIdx.x) * chunk;
    long long end = begin + chunk;
    if (end > n) end = n;
    double acc = Op::identity();
    for (long long i = begin + threadIdx.x; i < end; i += kBlock) acc = Op::apply(acc, load(static_cast<int>(i)));
    const double v = block_fold<Op>(acc, sh);
    if (threadIdx.x == 0) partials[blockIdx.x] = v;
}

template <class Op>
__global__ void __launch_bounds__(kBlock) k_reduce_level2(int m, const double* __restrict__ partials, double* __restrict__ out)
{
    __shared__ double sh[kBlock];
    double acc = Op::identity();
    for (int i = threadIdx.x; i < m; i += kBlock) acc = Op::apply(acc, partials[i]);
    const double v = block_fold<Op>(acc, sh);
    if (threadIdx.x == 0) *out = v;
}

// Grid layout as a pure function of n: at most kMaxBlocks chunks, each a multiple of kBlock.
void reduce_layout(int n, int& blocks, int& chunk)
{
    long long b = (static_cast<long long>(n) + kMinChunk - 1) / kMinChunk;
    if (b < 1) b = 1;
    if (b > kMaxBlocks) b = kMaxBlocks;
    long long c = (static_cast<long long>(n) + b - 1) / b;
    c = (c + kBlock - 1) / kBlock * kBlock;
    b = (static_cast<long long>(n) + c - 1) / c;
    blocks = static_cast<int>(b);
    chunk = static_cast<int>(c);
}

template <class Op, class Load>
double reduce_impl(int n, const Load& load, DeviceArray<double>& scratch)
{
    if (n <= 0) return Op::identity();  // 0.0 for sums, -/+inf for max/min
    int blocks = 0, chunk = 0;
    reduce_layout(n, blocks, chunk);
    const std::size_t need = static_cast<std::size_t>(blocks) + 1;
    if (scratch.size() < need) scratch.resize(need);
    double* partials = scratch.data();
    k_reduce_level1<Op, Load><<<static_cast<unsigned>(blocks), kBlock>>>(n, chunk, load, partials);
    CS_CUDA_KERNEL_CHECK();
    k_reduce_level2<Op><<<1, kBlock>>>(blocks, partials, partials + blocks);
    CS_CUDA_KERNEL_CHECK();
    double out = 0.0;
    CS_CUDA_CHECK(cudaMemcpy(&out, partials + blocks, sizeof(double), cudaMemcpyDeviceToHost));
    return out;
}

template <class Op, class Load>
void reduce_impl_to(int n, const Load& load, DeviceArray<double>& scratch, double* d_out)
{
    if (n <= 0) {
        if (scratch.size() < 1) scratch.resize(1);
        k_reduce_level2<Op><<<1, kBlock>>>(0, scratch.data(), d_out);   // folds nothing: the identity
        CS_CUDA_KERNEL_CHECK();
        return;
    }
    int blocks = 0, chunk = 0;
    reduce_layout(n, blocks, chunk);
    const std::size_t need = static_cast<std::size_t>(blocks) + 1;
    if (scratch.size() < need) scratch.resize(need);
    double* partials = scratch.data();
    k_reduce_level1<Op, Load><<<static_cast<unsigned>(blocks), kBlock>>>(n, chunk, load, partials);
    CS_CUDA_KERNEL_CHECK();
    k_reduce_level2<Op><<<1, kBlock>>>(blocks, partials, d_out);
    CS_CUDA_KERNEL_CHECK();
}

}  // namespace

double reduce_sum(const real* a, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<SumOp>(n, LoadReal{a}, scratch);
}

void reduce_sum_to(const real* a, int n, DeviceArray<double>& scratch, double* d_out)
{
    reduce_impl_to<SumOp>(n, LoadReal{a}, scratch, d_out);
}

double reduce_dot(const real3* a, const real3* b, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<SumOp>(n, LoadDot{a, b}, scratch);
}

double reduce_sum_sq(const real3* a, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<SumOp>(n, LoadSumSq{a}, scratch);
}

double reduce_max(const real* a, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<MaxOp>(n, LoadReal{a}, scratch);
}

double reduce_min(const real* a, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<MinOp>(n, LoadReal{a}, scratch);
}

double reduce_max_abs(const real3* a, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<MaxOp>(n, LoadMaxAbs{a}, scratch);
}

double reduce_max_abs_diff(const real3* x, const real3* y, int n, DeviceArray<double>& scratch)
{
    return reduce_impl<MaxOp>(n, LoadMaxAbsDiff{x, y}, scratch);
}

}  // namespace cs
