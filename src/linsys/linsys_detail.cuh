#pragma once
// Internal device helpers shared by the linsys translation units (not a contract header).
// Block layout everywhere: 9 reals row-major (b[0..2] = first row); arithmetic in double.
#include "core/typedef.cuh"

namespace cs {
namespace linsys_detail {

constexpr int kBlock = 256;

inline unsigned grid_for(int n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }

// (sx, sy, sz) += B x for the row-major 3x3 block at b and x given in double. B may be the
// double matrix or its float mirror (§12.4 Tier A); the products are always double.
template <typename B>
__device__ __forceinline__ void block_mul_acc(const B* __restrict__ b, double x0, double x1, double x2,
                                              double& sx, double& sy, double& sz)
{
    sx += static_cast<double>(b[0]) * x0 + static_cast<double>(b[1]) * x1 + static_cast<double>(b[2]) * x2;
    sy += static_cast<double>(b[3]) * x0 + static_cast<double>(b[4]) * x1 + static_cast<double>(b[5]) * x2;
    sz += static_cast<double>(b[6]) * x0 + static_cast<double>(b[7]) * x1 + static_cast<double>(b[8]) * x2;
}

// D r for the row-major 3x3 block at d (used by the block-Jacobi preconditioner).
__device__ __forceinline__ real3 block_apply(const real* __restrict__ d, const real3& r)
{
    double sx = 0.0, sy = 0.0, sz = 0.0;
    block_mul_acc(d, static_cast<double>(r.x), static_cast<double>(r.y), static_cast<double>(r.z), sx, sy, sz);
    return make_real3(static_cast<real>(sx), static_cast<real>(sy), static_cast<real>(sz));
}

}  // namespace linsys_detail
}  // namespace cs
