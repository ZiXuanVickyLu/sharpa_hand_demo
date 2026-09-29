#pragma once
// Analytic sphere colliders (math spec §4.1 PS, §10.1, §12.3): unsigned distance, gradient, and
// the closed-form time of impact of a vertex against a sphere whose radius moves linearly across
// the step. Added for the trapped squishy balls of [Z25] Fig. 21, whose container is an
// inverted sphere of prescribed, shrinking radius.
#ifndef CS_SPHERE_CCD_CUH
#define CS_SPHERE_CCD_CUH

#include "../core/typedef.cuh"

namespace cs {

/**
 * Distance of p to the sphere (c, r). Solid sphere: d = |p - c| - r, gradient (p - c)/|p - c|.
 * Inverted sphere (the body lives inside): d = r - |p - c|, gradient -(p - c)/|p - c|. The
 * gradient is a unit vector either way, so PS pairs use every PH formula with n := grad d.
 * A vertex exactly at the centre has no direction; +y is returned, as the degenerate PT case
 * does, so nothing downstream divides by zero.
 */
CUDA_INLINE_CALLABLE real3 sphere_distance_gradient(const real3& p, const real3& c, real r, bool inverted,
                                                    real& d) {
    const real3 u = p - c;
    const real len = sqrt(dot(u, u));
    real3 n = (len > real(0)) ? u * (real(1) / len) : make_real3(real(0), real(1), real(0));
    if (inverted) {
        d = r - len;
        n = -n;
    } else {
        d = len - r;
    }
    return n;
}

/**
 * Vertex p moving by dp against a sphere centred at c whose radius moves from r0 to r1 over
 * the same theta in [0, 1] (math spec §10.1, §12.3), with minimum separation xi and early-stop
 * ratio s. Returns true and toi < 1 on a hit; the gap at toi is exactly s times the initial gap.
 *
 * Let u = p - c, r(t) = r0 + t r', g0 = d(0) - xi. The gap reaches s g0 when |u + t dp| = q(t),
 * with q(t) = r(t) - s g0 - xi (inverted) or r(t) + s g0 + xi (solid): squaring gives the
 * quadratic (r'^2 - |dp|^2) t^2 + 2 (r' q0 - u.dp) t + (q0^2 - |u|^2) = 0, whose smallest root in
 * (0, 1] with q(t) >= 0 is the impact. Squaring can only add roots with q(t) < 0, which the sign
 * test discards. An already-violating vertex (g0 <= 0) that keeps approaching gets toi = 0, as a
 * plane does; one that recedes is not a hit.
 */
CUDA_INLINE_CALLABLE bool sphere_toi(const real3& p, const real3& dp, const real3& c, real r0, real r1,
                                     bool inverted, real xi, real s, real& toi) {
    toi = real(1);
    const real3 u = p - c;
    const real len0 = sqrt(dot(u, u));
    const real g0 = (inverted ? (r0 - len0) : (len0 - r0)) - xi;
    // Rate of change of the gap at t = 0: negative means approaching.
    const real3 n = (len0 > real(0)) ? u * (real(1) / len0) : make_real3(real(0), real(1), real(0));
    const real rp = r1 - r0;
    const real gdot = inverted ? (rp - dot(n, dp)) : (dot(n, dp) - rp);
    if (g0 <= real(0)) {
        if (gdot < real(0)) { toi = real(0); return true; }
        return false;
    }
    const real q0 = inverted ? (r0 - s * g0 - xi) : (r0 + s * g0 + xi);
    const real A = rp * rp - dot(dp, dp);
    const real B = real(2) * (rp * q0 - dot(u, dp));
    const real C = q0 * q0 - len0 * len0;
    // Candidates: the roots of A t^2 + B t + C = 0 (linear when A vanishes).
    real cand[2];
    int nc = 0;
    if (fabs(A) <= real(1e-14) * (fabs(B) + fabs(C) + real(1e-300))) {
        if (B != real(0)) cand[nc++] = -C / B;
    } else {
        const real disc = B * B - real(4) * A * C;
        if (disc < real(0)) return false;
        const real sq = sqrt(disc);
        // Numerically stable pair of roots.
        const real qq = real(-0.5) * (B + (B >= real(0) ? sq : -sq));
        if (qq != real(0)) { cand[nc++] = qq / A; cand[nc++] = C / qq; }
        else cand[nc++] = real(0);
    }
    real best = real(1);
    for (int k = 0; k < nc; ++k) {
        const real t = cand[k];
        if (!(t > real(0)) || t > real(1)) continue;
        const real qt = q0 + rp * t;
        if (qt < real(0)) continue;
        if (t < best) best = t;
    }
    toi = best;
    return toi < real(1);
}

}  // namespace cs

#endif  // CS_SPHERE_CCD_CUH
