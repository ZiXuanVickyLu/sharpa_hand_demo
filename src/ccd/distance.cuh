//
// Created by birdpeople on 8/24/2023.
// Vendored into contact_solver (M0). Header-only: these closest-point queries are device
// helpers shared by the solver kernels, the gradient assembly (distance_gradient.cuh) and
// the tests, and cs_device resolves its own device symbols, so consumers cannot device-link
// against out-of-line definitions.
//

#pragma once
#ifndef DISTANCE_H
#define DISTANCE_H

#include "../core/typedef.cuh"
#include "libcuGTE/DistTriangle3Triangle3.h"
#include "libcuGTE/DistSegmentSegment.h"
#include "libcuGTE/DistPointSegment.h"

namespace cs{

        using pt = real3;

        // Below this squared distance the closest-point direction is not returned (n = 0);
        // the gradient assembly of distance_gradient.cuh uses its own fallback instead.
        CUDA_INLINE_CALLABLE real distance_query_zero_normal_dsqr() { return static_cast<real>(1e-10); }

        namespace detail {
            CUDA_INLINE_CALLABLE gte::Vector3<real> to_gte(const pt& v) { return gte::Vector3<real>{v.x, v.y, v.z}; }

            CUDA_INLINE_CALLABLE pt direction_or_zero(const pt& diff, real dsqr) {
                if (dsqr < distance_query_zero_normal_dsqr()) return make_real3(0, 0, 0);
                return diff / sqrt(dsqr);
            }
        }

        CUDA_INLINE_CALLABLE real vertex_vertex_distance_square(const pt &p, const pt &q) {
            return squaredLength(p - q);
        }

        CUDA_INLINE_CALLABLE real edge_edge_distance_square(const pt &p0, const pt &p1, const pt &q0, const pt &q1) {
            gte::Segment3<real> edge0(detail::to_gte(p0), detail::to_gte(p1));
            gte::Segment3<real> edge1(detail::to_gte(q0), detail::to_gte(q1));
            auto dcp = gte::DCPSegment3Segment3<real>{};
            // ComputeRobust: GTE's operator() divides by det = a c - b^2 and is inaccurate for
            // (near-)parallel segments in floating point (distance off by 1e-4 and host/device
            // disagreement in test_distance_gradient); the robust boundary search is exact there.
            auto result = dcp.ComputeRobust(edge0, edge1);
            return result.sqrDistance;
        }

        // Convention: GTE returns parameter[0] = s with closest_p = p0 + s (p1 - p0) and
        // parameter[1] = t likewise on q. The wrapper stores u0 = 1 - s, v0 = 1 - t, i.e.
        // closest_p = p0 * u0 + p1 * (1 - u0), closest_q = q0 * v0 + q1 * (1 - v0), and
        // n = (closest_p - closest_q) / d (zero when d^2 < 1e-10).
        CUDA_INLINE_CALLABLE real edge_edge_distance_square(const pt &p0, const pt &p1, const pt &q0, const pt &q1,
                                                            real &u0, real &v0, pt &n) {
            gte::Segment3<real> edge0(detail::to_gte(p0), detail::to_gte(p1));
            gte::Segment3<real> edge1(detail::to_gte(q0), detail::to_gte(q1));
            auto dcp = gte::DCPSegment3Segment3<real>{};
            auto result = dcp.ComputeRobust(edge0, edge1);   // see the 4-argument overload
            u0 = static_cast<real>(1) - result.parameter[0];
            v0 = static_cast<real>(1) - result.parameter[1];
            n = detail::direction_or_zero(p0 * u0 + p1 * result.parameter[0] - (q0 * v0 + q1 * result.parameter[1]),
                                          result.sqrDistance);
            return result.sqrDistance;
        }

        CUDA_INLINE_CALLABLE real vertex_triangle_distance_square(const pt &p0, const pt &q0, const pt &q1, const pt &q2) {
            gte::Triangle3<real> tri(detail::to_gte(q0), detail::to_gte(q1), detail::to_gte(q2));
            auto dcp = gte::DCPPoint3Triangle3<real>{};
            auto res = dcp(detail::to_gte(p0), tri);
            return res.sqrDistance;
        }

        // Convention: u0, u1 are the barycentric weights of q0 and q1 (q2 has 1 - u0 - u1),
        // closest = q0 u0 + q1 u1 + q2 (1 - u0 - u1), n = (p0 - closest) / d (zero when d^2 < 1e-10).
        CUDA_INLINE_CALLABLE real vertex_triangle_distance_square(const pt &p0, const pt &q0, const pt &q1, const pt &q2,
                                                                  real &u0, real &u1, pt &n) {
            gte::Triangle3<real> tri(detail::to_gte(q0), detail::to_gte(q1), detail::to_gte(q2));
            auto dcp = gte::DCPPoint3Triangle3<real>{};
            auto res = dcp(detail::to_gte(p0), tri);
            u0 = res.barycentric[0];
            u1 = res.barycentric[1];
            n = detail::direction_or_zero(p0 - (q0 * u0 + q1 * u1 + q2 * (static_cast<real>(1) - u0 - u1)),
                                          res.sqrDistance);
            return res.sqrDistance;
        }

        CUDA_INLINE_CALLABLE real vertex_edge_distance_square(const pt &p, const pt &q0, const pt &q1) {
            gte::Segment3<real> edge(detail::to_gte(q0), detail::to_gte(q1));
            auto dcp = gte::DCPPoint3Segment3<real>{};
            auto res = dcp(detail::to_gte(p), edge);
            return res.sqrDistance;
        }

        // Convention: u0 is the weight of q0 and v0 = 1 - u0 the weight of q1,
        // n = (p - u0 q0 - v0 q1) / d (zero when d^2 < 1e-10).
        CUDA_INLINE_CALLABLE real vertex_edge_distance_square(const pt &p, const pt &q0, const pt &q1,
                                                              real &u0, real &v0, pt &n) {
            gte::Segment3<real> edge(detail::to_gte(q0), detail::to_gte(q1));
            auto dcp = gte::DCPPoint3Segment3<real>{};
            auto res = dcp(detail::to_gte(p), edge);
            u0 = static_cast<real>(1) - res.parameter;
            v0 = res.parameter;
            n = detail::direction_or_zero(p - u0 * q0 - v0 * q1, res.sqrDistance);
            return res.sqrDistance;
        }

        CUDA_INLINE_CALLABLE real triangle_triangle_distance_square(const pt &p0, const pt &p1, const pt &p2,
                                                                    const pt &q0, const pt &q1, const pt &q2) {
            gte::Triangle3<real> t1(detail::to_gte(p0), detail::to_gte(p1), detail::to_gte(p2));
            gte::Triangle3<real> t2(detail::to_gte(q0), detail::to_gte(q1), detail::to_gte(q2));
            auto dcp = gte::DCPTriangle3Triangle3<real>{};
            auto result = dcp(t1, t2);
            return result.sqrDistance;
        }

}

#endif //DISTANCE_H
