#pragma once
// contact_solver M0: closed-form vertex - plane time of impact (math spec 10.1).
#ifndef CS_PLANE_CCD_CUH
#define CS_PLANE_CCD_CUH

#include "../core/typedef.cuh"

namespace cs {

/**
 * Vertex p moving by dp against the half-space {x : n^T (x - o) >= 0} with unit normal n,
 * minimum separation xi and early-stop ratio s (math spec 10.1):
 *   gap g = n^T (p - o) - xi;
 *   no hit (toi = 1, return false) when n^T dp >= 0 (moving away or tangentially);
 *   else toi = min(1, (1 - s) g / (-n^T dp)), a hit iff toi < 1.
 * At toi the gap equals s * g (exact, since the gap is affine in t). A vertex that already
 * violates the separation (g <= 0) and approaches gets toi = 0 (clamped; the AL loop never
 * queries from such a state, math spec 10.2).
 */
CUDA_INLINE_CALLABLE bool plane_toi(const real3& p, const real3& dp,
                                    const real3& o, const real3& n_unit,
                                    real xi, real s, real& toi) {
    const real g = dot(n_unit, p - o) - xi;
    const real v = dot(n_unit, dp);
    if (v >= static_cast<real>(0)) {
        toi = static_cast<real>(1);
        return false;
    }
    const real t = (static_cast<real>(1) - s) * g / (-v);
    toi = clamp(t, static_cast<real>(0), static_cast<real>(1));
    return toi < static_cast<real>(1);
}

} // namespace cs

#endif // CS_PLANE_CCD_CUH
