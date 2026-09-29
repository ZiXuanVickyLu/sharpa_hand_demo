#pragma once
// contact_solver M0: unsigned distance and its gradient for the contact pair types
// (math spec 4.2), assembled on top of the vendored GTE closest-point queries of
// distance.cuh. Header-only so the linearization kernel and the tests share one
// implementation. Always evaluated in `real` (double in the default build, precision
// policy 11.1); callers translate a pair to its own frame before calling (11.2).
#ifndef CS_DISTANCE_GRADIENT_CUH
#define CS_DISTANCE_GRADIENT_CUH

#include "../core/typedef.cuh"
#include "distance.cuh"

namespace cs {

namespace detail {

    // Distances at or below this are treated as zero: the closest-point direction is then
    // unusable and the gradient falls back to a geometric normal (math spec 4.2, d_tiny).
    CUDA_INLINE_CALLABLE real distance_tiny() { return static_cast<real>(1e-12); }

    // Some unit vector orthogonal to e (e non-zero, not necessarily unit).
    CUDA_INLINE_CALLABLE real3 any_unit_perpendicular(const real3& e) {
        const real ax = fabs(e.x), ay = fabs(e.y), az = fabs(e.z);
        const real3 h = (ax <= ay && ax <= az) ? make_real3(1, 0, 0)
                      : ((ay <= az) ? make_real3(0, 1, 0) : make_real3(0, 0, 1));
        const real3 c = cross(e, h);
        const real l = length(c);
        if (l > static_cast<real>(0)) return c / l;
        return make_real3(0, 0, 1);
    }

    // true when v is a usable direction (finite, not ~zero)
    CUDA_INLINE_CALLABLE bool usable_direction(const real3& v) {
        const real nn = squaredLength(v);
        return nn > static_cast<real>(0.25) && nn < static_cast<real>(4);
    }

} // namespace detail

/**
 * Point - triangle distance and gradient (math spec 4.2, PT).
 *
 * Closest point r = t0*u0 + t1*u1 + t2*(1-u0-u1) from cs::vertex_triangle_distance_square,
 * d = |p - r|, n = (p - r) / d, and with the barycentric weights frozen
 *   g[0] = n,  g[1] = -u0 n,  g[2] = -u1 n,  g[3] = -(1-u0-u1) n     (sum = 0).
 * Degenerate (d <= 1e-12 or no usable direction): `degenerate` is set, n is the unit triangle
 * normal oriented toward p (+normal when p lies on the plane) and g is built from it with the
 * same weights. The distance query's own zero-normal cutoff (d^2 < 1e-10) is not used here.
 */
CUDA_INLINE_CALLABLE void pt_distance_gradient(const real3& p,
                                               const real3& t0, const real3& t1, const real3& t2,
                                               real& d, real3 g[4], real3& n, bool& degenerate) {
    real u0 = 0, u1 = 0;
    real3 n_query;   // the query's normal is recomputed below from the closest point
    const real d2 = vertex_triangle_distance_square(p, t0, t1, t2, u0, u1, n_query);
    const real u2 = static_cast<real>(1) - u0 - u1;
    const real3 r = t0 * u0 + t1 * u1 + t2 * u2;
    d = sqrt(d2);

    degenerate = !(d > detail::distance_tiny());
    if (!degenerate) {
        n = (p - r) / d;
        degenerate = !detail::usable_direction(n);
    }
    if (degenerate) {
        real3 nt = cross(t1 - t0, t2 - t0);
        const real l = length(nt);
        if (l > static_cast<real>(0)) {
            nt = nt / l;
        } else {
            // degenerate triangle: any direction orthogonal to its longest edge
            const real3 e = (squaredLength(t1 - t0) >= squaredLength(t2 - t0)) ? (t1 - t0) : (t2 - t0);
            nt = (squaredLength(e) > static_cast<real>(0)) ? detail::any_unit_perpendicular(e) : make_real3(0, 0, 1);
        }
        if (dot(nt, p - t0) < static_cast<real>(0)) nt = -nt;   // toward p; + when on the plane
        n = nt;
    }
    g[0] = n;
    g[1] = -u0 * n;
    g[2] = -u1 * n;
    g[3] = -u2 * n;
}

/**
 * Edge - edge distance and gradient (math spec 4.2, EE).
 *
 * Convention of cs::edge_edge_distance_square(a0,a1,b0,b1,u0,v0,n), verified in distance.cu:
 * GTE returns parameter[0] = s with closest_a = a0 + s (a1 - a0) and the wrapper stores
 * u0 = 1 - s, hence closest_a = a0*u0 + a1*(1-u0); likewise closest_b = b0*v0 + b1*(1-v0).
 * d = |closest_a - closest_b|, n = (closest_a - closest_b) / d (from b toward a), and with
 * the weights frozen
 *   g[0] = u0 n,  g[1] = (1-u0) n,  g[2] = -v0 n,  g[3] = -(1-v0) n     (sum = 0).
 * Degenerate (d <= 1e-12 or no usable direction): `degenerate` is set and
 * n = normalize((a1-a0) x (b1-b0)) oriented from b toward a (by the edge midpoints); for
 * parallel edges (|cross| ~ 0) the perpendicular from closest_b to closest_a (the point-edge
 * sub-case: the component of closest_a - closest_b orthogonal to the edge direction) is used,
 * or, when that vanishes too, an arbitrary unit vector orthogonal to the edges.
 */
CUDA_INLINE_CALLABLE void ee_distance_gradient(const real3& a0, const real3& a1,
                                               const real3& b0, const real3& b1,
                                               real& d, real3 g[4], real3& n, bool& degenerate) {
    real u0 = 0, v0 = 0;
    real3 n_query;
    const real d2 = edge_edge_distance_square(a0, a1, b0, b1, u0, v0, n_query);
    const real u1 = static_cast<real>(1) - u0;
    const real v1 = static_cast<real>(1) - v0;
    const real3 ca = a0 * u0 + a1 * u1;
    const real3 cb = b0 * v0 + b1 * v1;
    d = sqrt(d2);

    degenerate = !(d > detail::distance_tiny());
    if (!degenerate) {
        n = (ca - cb) / d;
        degenerate = !detail::usable_direction(n);
    }
    if (degenerate) {
        const real3 ea = a1 - a0;
        const real3 eb = b1 - b0;
        const real3 c = cross(ea, eb);
        const real cl = length(c);
        const real scale = length(ea) * length(eb);
        if (cl > static_cast<real>(1e-8) * scale && cl > static_cast<real>(0)) {
            real3 nn = c / cl;
            const real3 mid_a = (a0 + a1) * static_cast<real>(0.5);
            const real3 mid_b = (b0 + b1) * static_cast<real>(0.5);
            if (dot(nn, mid_a - mid_b) < static_cast<real>(0)) nn = -nn;   // from b toward a
            n = nn;
        } else {
            // parallel (or a vanishing edge): perpendicular from closest_b to closest_a
            const real3 e = (squaredLength(ea) >= squaredLength(eb)) ? ea : eb;
            const real ee = squaredLength(e);
            real3 r = ca - cb;
            if (ee > static_cast<real>(0)) r = r - e * (dot(r, e) / ee);
            const real rl = length(r);
            if (rl > detail::distance_tiny()) {
                n = r / rl;
            } else {
                n = (ee > static_cast<real>(0)) ? detail::any_unit_perpendicular(e) : make_real3(0, 0, 1);
            }
        }
    }
    g[0] = u0 * n;
    g[1] = u1 * n;
    g[2] = -v0 * n;
    g[3] = -v1 * n;
}

/**
 * Point - plane signed distance and gradient (math spec 4.2, PH): d = n^T (p - o) for the
 * unit normal n; the gradient with respect to p is n itself (returned).
 */
CUDA_INLINE_CALLABLE real3 plane_distance_gradient(const real3& p, const real3& o, const real3& n_unit, real& d) {
    d = dot(n_unit, p - o);
    return n_unit;
}

} // namespace cs

#endif // CS_DISTANCE_GRADIENT_CUH
