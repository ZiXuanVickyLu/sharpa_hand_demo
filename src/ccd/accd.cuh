//
// Created by birdpeople on 8/24/2023.
// Vendored into contact_solver (M0) with the fixes of implementation spec 3.2. Header-only:
// ACCD is a device helper shared by the solver kernels and the tests, and cs_device resolves
// its own device symbols, so consumers cannot device-link against out-of-line definitions.
//

#pragma once
#ifndef ACCD_H
#define ACCD_H
#include "../core/vector_type_t.h"
#include "../core/typedef.cuh"

namespace cs{

    /**
     * Additive CCD (Li, Kaufman, Jiang 2021, "Codimensional IPC") with minimum separation,
     * math spec 10.1.
     *
     * Contract shared by every routine. Inputs are the start positions x and the end
     * positions x' of the pair's vertices (linear motion x(t) = x + t (x' - x), t in [0,1]),
     * the minimum separation xi >= 0 and the early-stop ratio s in (0,1). Let d(t) be the
     * unsigned distance of the pair and g(t) = d(t) - xi its gap.
     *
     *   Returns true (hit) with toi in [0,1) when the gap would shrink to s * g(0) before the
     *   end of the motion, else false with toi = 1.
     *
     *   toi is a conservative lower bound of that time: g(toi) >= s * g(0) at the returned
     *   sample and g(t) > 0 on [0, toi] (the pair never comes within xi along the way). The
     *   advancement steps by (1 - s) * g / l_p where l_p = |p_0| + max_m |p_m| (PT) or
     *   max(|a_0|,|a_1|) + max(|b_0|,|b_1|) (EE) is the bound on the relative motion of the
     *   *mean-removed* displacements; every step is clamped to tau - t, so the last sample lies
     *   exactly at the end of the motion and the miss loop stops at t = tau (= 1). (The vendored
     *   loop evaluated the extrapolated path up to one step past tau and could report a hit for
     *   an event that happens only after the motion ends.)
     *
     *   Already inside the separation (g(0) <= 0, which the AL loop never produces, math spec
     *   10.2): the gap is measured by the smallest vertex-vertex distance of the pair instead
     *   (edge_edge_ccd's original fallback, now shared by every routine), so that a positive
     *   step exists and the vertices never coincide; if even that is <= xi the routine reports
     *   a hit at toi = 0. The conservative bound above then holds for that substituted gap.
     *
     *   max_iter caps the advancement loop; on exhaustion the accumulated time is returned as a
     *   hit (conservative).
     *
     * The *_on_smem variants are bitwise identical to the register variants and keep the
     * working set in caller-provided shared memory (accd_smem_size_per_query bytes per query).
     */
    struct ACCD{
        using pt = real3;

        static CUDA_INLINE_CALLABLE bool edge_edge_ccd(
                const pt& p0, const pt& p1, const pt& q0, const pt& q1,
                const pt& p0t, const pt& p1t, const pt& q0t, const pt& q1t,
                real& toi,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

        static CUDA_INLINE_CALLABLE bool vertex_triangle_ccd(
                const pt& p0, const pt& q0, const pt& q1, const pt& q2,
                const pt& p0t, const pt& q0t, const pt& q1t, const pt& q2t,
                real& toi,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

        static CUDA_INLINE_CALLABLE bool vertex_edge_ccd(
                const pt& p0, const pt& q0, const pt& q1,
                const pt& p0t, const pt& q0t, const pt& q1t,
                real& toi,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

        // on smem method: 16 points per query
        // [0..3] vertex, [4..7] displacement, [8..11] mean-removed displacement, [12..15] scalars.
        static constexpr size_t accd_smem_points_per_query = 16;
        static constexpr size_t accd_smem_size_per_query = sizeof(pt) * accd_smem_points_per_query;

        static CUDA_INLINE_CALLABLE bool edge_edge_ccd_on_smem(
                const pt& p0, const pt& p1, const pt& q0, const pt& q1,
                const pt& p0t, const pt& p1t, const pt& q0t, const pt& q1t,
                real& toi,
                pt* smem,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

        static CUDA_INLINE_CALLABLE bool vertex_triangle_ccd_on_smem(
                const pt& p0, const pt& q0, const pt& q1, const pt& q2,
                const pt& p0t, const pt& q0t, const pt& q1t, const pt& q2t,
                real& toi,
                pt* smem,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

        static CUDA_INLINE_CALLABLE bool vertex_edge_ccd_on_smem(
                const pt& p0, const pt& q0, const pt& q1,
                const pt& p0t, const pt& q0t, const pt& q1t,
                real& toi,
                pt* smem,
                unsigned int max_iter = 100,
                real xi = static_cast<real>(1e-3),
                real s = static_cast<real>(1e-1),
                real tau = static_cast<real>(1));

    };

}

// ------------------------------------------------------------------------------------
// implementation
// ------------------------------------------------------------------------------------
#include "distance.cuh"

// Additive CCD, see the contract above. Changes against the coupled_solver version
// (implementation spec 3.2): l_p of vertex_triangle_ccd uses the mean-removed displacement of
// every vertex; the in-loop step factor is (1 - s) everywhere; the "already inside the minimum
// separation" fallback of edge_edge_ccd is shared by every routine; the miss loop of
// vertex_edge_ccd stops at toi >= tau like the others; each step is clamped to tau - toi so
// the path is never evaluated beyond the end of the motion; triangle_triangle_ccd was removed.
//
// The register and the shared-memory variants are written with textually identical
// arithmetic so that they produce bitwise identical results (test_accd checks this).

namespace cs {

    namespace accd_detail {
        // Smallest squared vertex-vertex distance of a pair: the fallback gap when the pair
        // is already inside the minimum separation.
        CUDA_INLINE_CALLABLE real min_vv_dsqr_ee(const real3& p0, const real3& p1,
                                                 const real3& q0, const real3& q1) {
            real m = squaredLength(p0 - q0);
            m = min(m, squaredLength(p1 - q0));
            m = min(m, squaredLength(p0 - q1));
            m = min(m, squaredLength(p1 - q1));
            return m;
        }

        CUDA_INLINE_CALLABLE real min_vv_dsqr_vt(const real3& p, const real3& q0,
                                                 const real3& q1, const real3& q2) {
            real m = squaredLength(p - q0);
            m = min(m, squaredLength(p - q1));
            m = min(m, squaredLength(p - q2));
            return m;
        }

        CUDA_INLINE_CALLABLE real min_vv_dsqr_ve(const real3& p, const real3& q0, const real3& q1) {
            real m = squaredLength(p - q0);
            m = min(m, squaredLength(p - q1));
            return m;
        }
    } // namespace accd_detail

    // ------------------------------------------------------------------------------------
    // edge - edge
    // ------------------------------------------------------------------------------------
    CUDA_INLINE_CALLABLE bool ACCD::edge_edge_ccd(const pt &p0, const pt &p1,
                                           const pt &q0, const pt &q1,
                                           const pt &p0t, const pt &p1t,
                                           const pt &q0t, const pt &q1t,
                                           real &toi,
                                           unsigned int max_iter,
                                           real xi,
                                           real s,
                                           real tau) {
        pt vertex[4] = {p0, p1, q0, q1};
        pt displacement[4] = {p0t - p0, p1t - p1, q0t - q0, q1t - q1};
        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        pt dis_eval[4] = {displacement[0] - eval, displacement[1] - eval,
                          displacement[2] - eval, displacement[3] - eval};
        const real lp = max(length(dis_eval[0]), length(dis_eval[1])) +
                        max(length(dis_eval[2]), length(dis_eval[3]));
        if (lp <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        real dsqr = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        real dFunc = dsqr - xi * xi;
        if (dFunc <= static_cast<real>(0)) {
            // already inside the minimum separation: measure the gap between vertices instead
            dsqr = accd_detail::min_vv_dsqr_ee(vertex[0], vertex[1], vertex[2], vertex[3]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        real dis_current = sqrt(dsqr);
        const real g = s * dFunc / (dis_current + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            // step clamped to the end of the motion: never evaluate the extrapolated path beyond tau
            const real tl = min((static_cast<real>(1) - s) * dFunc / (lp * (dis_current + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                dsqr = accd_detail::min_vv_dsqr_ee(vertex[0], vertex[1], vertex[2], vertex[3]);
                dFunc = dsqr - xi * xi;
                if (dFunc <= static_cast<real>(0)) break;   // inside even by the fallback: keep the last safe time
            }
            dis_current = sqrt(dsqr);
            const real g_current = dFunc / (dis_current + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    // ------------------------------------------------------------------------------------
    // vertex - triangle
    // ------------------------------------------------------------------------------------
    CUDA_INLINE_CALLABLE bool ACCD::vertex_triangle_ccd(const pt &p0,
                                                 const pt &q0, const pt &q1, const pt &q2,
                                                 const pt &p0t,
                                                 const pt &q0t, const pt &q1t, const pt &q2t,
                                                 real &toi,
                                                 unsigned int max_iter,
                                                 real xi,
                                                 real s,
                                                 real tau) {
        pt vertex[4] = {p0, q0, q1, q2};
        pt displacement[4] = {p0t - p0, q0t - q0, q1t - q1, q2t - q2};
        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        pt dis_eval[4] = {displacement[0] - eval, displacement[1] - eval,
                          displacement[2] - eval, displacement[3] - eval};
        // l_p = |p_0| + max_m |p_m| on the mean-removed displacements (the vendored code used
        // displacement[3] for the last vertex, which is neither a bound nor mean-removed).
        const real lp = length(dis_eval[0]) +
                        max(max(length(dis_eval[1]), length(dis_eval[2])), length(dis_eval[3]));
        if (lp <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        real dsqr = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        real dFunc = dsqr - xi * xi;
        if (dFunc <= static_cast<real>(0)) {
            dsqr = accd_detail::min_vv_dsqr_vt(vertex[0], vertex[1], vertex[2], vertex[3]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        real dis_current = sqrt(dsqr);
        const real g = s * dFunc / (dis_current + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            // step clamped to the end of the motion: never evaluate the extrapolated path beyond tau
            const real tl = min((static_cast<real>(1) - s) * dFunc / (lp * (dis_current + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                dsqr = accd_detail::min_vv_dsqr_vt(vertex[0], vertex[1], vertex[2], vertex[3]);
                dFunc = dsqr - xi * xi;
                if (dFunc <= static_cast<real>(0)) break;
            }
            dis_current = sqrt(dsqr);
            const real g_current = dFunc / (dis_current + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    // ------------------------------------------------------------------------------------
    // vertex - edge
    // ------------------------------------------------------------------------------------
    CUDA_INLINE_CALLABLE bool ACCD::vertex_edge_ccd(const pt &p0,
                                             const pt &q0, const pt &q1,
                                             const pt &p0t,
                                             const pt &q0t, const pt &q1t,
                                             real &toi,
                                             unsigned int max_iter,
                                             real xi,
                                             real s,
                                             real tau) {
        pt vertex[3] = {p0, q0, q1};
        pt displacement[3] = {p0t - p0, q0t - q0, q1t - q1};
        pt eval = (displacement[0] + displacement[1] + displacement[2]) / static_cast<real>(3);
        pt dis_eval[3] = {displacement[0] - eval, displacement[1] - eval, displacement[2] - eval};
        const real lp = length(dis_eval[0]) + max(length(dis_eval[1]), length(dis_eval[2]));
        if (lp <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        real dsqr = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
        real dFunc = dsqr - xi * xi;
        if (dFunc <= static_cast<real>(0)) {
            dsqr = accd_detail::min_vv_dsqr_ve(vertex[0], vertex[1], vertex[2]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        real dis_current = sqrt(dsqr);
        const real g = s * dFunc / (dis_current + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            // step clamped to the end of the motion: never evaluate the extrapolated path beyond tau
            const real tl = min((static_cast<real>(1) - s) * dFunc / (lp * (dis_current + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 3; ++i) vertex[i] += tl * dis_eval[i];
            dsqr = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
            dFunc = dsqr - xi * xi;
            if (dFunc <= static_cast<real>(0)) {
                dsqr = accd_detail::min_vv_dsqr_ve(vertex[0], vertex[1], vertex[2]);
                dFunc = dsqr - xi * xi;
                if (dFunc <= static_cast<real>(0)) break;
            }
            dis_current = sqrt(dsqr);
            const real g_current = dFunc / (dis_current + xi);
            if (toi > static_cast<real>(0) && g_current < g) break;
            toi += tl;
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    // ------------------------------------------------------------------------------------
    // shared-memory variants (same arithmetic, working set in caller-provided smem)
    // ------------------------------------------------------------------------------------
    CUDA_INLINE_CALLABLE bool ACCD::edge_edge_ccd_on_smem(const pt &p0, const pt &p1,
                                                   const pt &q0, const pt &q1,
                                                   const pt &p0t, const pt &p1t,
                                                   const pt &q0t, const pt &q1t,
                                                   real &toi,
                                                   pt *smem,
                                                   unsigned int max_iter,
                                                   real xi,
                                                   real s,
                                                   real tau) {
        // shared layout (per query):
        // [0..3]   vertex
        // [4..7]   displacement
        // [8..11]  dis_eval
        // [12..15] scratch (reals): [0] lp, [1] dsqr, [2] dFunc, [3] dis_current, [4] g, [5] tl, [6] g_current
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);

        vertex[0] = p0; vertex[1] = p1; vertex[2] = q0; vertex[3] = q1;
        displacement[0] = p0t - p0;
        displacement[1] = p1t - p1;
        displacement[2] = q0t - q0;
        displacement[3] = q1t - q1;

        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;
        dis_eval[3] = displacement[3] - eval;

        scratch[0] = max(length(dis_eval[0]), length(dis_eval[1])) +
                     max(length(dis_eval[2]), length(dis_eval[3]));
        if (scratch[0] <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        scratch[1] = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        scratch[2] = scratch[1] - xi * xi;
        if (scratch[2] <= static_cast<real>(0)) {
            scratch[1] = accd_detail::min_vv_dsqr_ee(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        scratch[3] = sqrt(scratch[1]);
        scratch[4] = s * scratch[2] / (scratch[3] + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            scratch[5] = min((static_cast<real>(1) - s) * scratch[2] / (scratch[0] * (scratch[3] + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += scratch[5] * dis_eval[i];
            scratch[1] = edge_edge_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                scratch[1] = accd_detail::min_vv_dsqr_ee(vertex[0], vertex[1], vertex[2], vertex[3]);
                scratch[2] = scratch[1] - xi * xi;
                if (scratch[2] <= static_cast<real>(0)) break;
            }
            scratch[3] = sqrt(scratch[1]);
            scratch[6] = scratch[2] / (scratch[3] + xi);
            if (toi > static_cast<real>(0) && scratch[6] < scratch[4]) break;
            toi += scratch[5];
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_INLINE_CALLABLE bool ACCD::vertex_triangle_ccd_on_smem(const pt &p0,
                                                         const pt &q0, const pt &q1, const pt &q2,
                                                         const pt &p0t,
                                                         const pt &q0t, const pt &q1t, const pt &q2t,
                                                         real &toi,
                                                         pt *smem,
                                                         unsigned int max_iter,
                                                         real xi,
                                                         real s,
                                                         real tau) {
        // shared layout as in edge_edge_ccd_on_smem
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);

        vertex[0] = p0; vertex[1] = q0; vertex[2] = q1; vertex[3] = q2;
        displacement[0] = p0t - p0;
        displacement[1] = q0t - q0;
        displacement[2] = q1t - q1;
        displacement[3] = q2t - q2;

        pt eval = (displacement[0] + displacement[1] + displacement[2] + displacement[3]) / static_cast<real>(4);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;
        dis_eval[3] = displacement[3] - eval;

        scratch[0] = length(dis_eval[0]) +
                     max(max(length(dis_eval[1]), length(dis_eval[2])), length(dis_eval[3]));
        if (scratch[0] <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        scratch[1] = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
        scratch[2] = scratch[1] - xi * xi;
        if (scratch[2] <= static_cast<real>(0)) {
            scratch[1] = accd_detail::min_vv_dsqr_vt(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        scratch[3] = sqrt(scratch[1]);
        scratch[4] = s * scratch[2] / (scratch[3] + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            scratch[5] = min((static_cast<real>(1) - s) * scratch[2] / (scratch[0] * (scratch[3] + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 4; ++i) vertex[i] += scratch[5] * dis_eval[i];
            scratch[1] = vertex_triangle_distance_square(vertex[0], vertex[1], vertex[2], vertex[3]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                scratch[1] = accd_detail::min_vv_dsqr_vt(vertex[0], vertex[1], vertex[2], vertex[3]);
                scratch[2] = scratch[1] - xi * xi;
                if (scratch[2] <= static_cast<real>(0)) break;
            }
            scratch[3] = sqrt(scratch[1]);
            scratch[6] = scratch[2] / (scratch[3] + xi);
            if (toi > static_cast<real>(0) && scratch[6] < scratch[4]) break;
            toi += scratch[5];
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

    CUDA_INLINE_CALLABLE bool ACCD::vertex_edge_ccd_on_smem(const pt &p0,
                                                     const pt &q0, const pt &q1,
                                                     const pt &p0t,
                                                     const pt &q0t, const pt &q1t,
                                                     real &toi,
                                                     pt *smem,
                                                     unsigned int max_iter,
                                                     real xi,
                                                     real s,
                                                     real tau) {
        // shared layout as in edge_edge_ccd_on_smem (slots 3, 7, 11 unused)
        pt *vertex = smem;
        pt *displacement = smem + 4;
        pt *dis_eval = smem + 8;
        real *scratch = reinterpret_cast<real *>(smem + 12);

        vertex[0] = p0; vertex[1] = q0; vertex[2] = q1;
        displacement[0] = p0t - p0;
        displacement[1] = q0t - q0;
        displacement[2] = q1t - q1;

        pt eval = (displacement[0] + displacement[1] + displacement[2]) / static_cast<real>(3);
        dis_eval[0] = displacement[0] - eval;
        dis_eval[1] = displacement[1] - eval;
        dis_eval[2] = displacement[2] - eval;

        scratch[0] = length(dis_eval[0]) + max(length(dis_eval[1]), length(dis_eval[2]));
        if (scratch[0] <= static_cast<real>(0)) {
            toi = static_cast<real>(1);
            return false;
        }

        scratch[1] = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
        scratch[2] = scratch[1] - xi * xi;
        if (scratch[2] <= static_cast<real>(0)) {
            scratch[1] = accd_detail::min_vv_dsqr_ve(vertex[0], vertex[1], vertex[2]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                toi = static_cast<real>(0);
                return true;
            }
        }

        scratch[3] = sqrt(scratch[1]);
        scratch[4] = s * scratch[2] / (scratch[3] + xi);
        toi = static_cast<real>(0);

        unsigned int ite_current = 0;
        while (ite_current < max_iter) {
            scratch[5] = min((static_cast<real>(1) - s) * scratch[2] / (scratch[0] * (scratch[3] + xi)), tau - toi);
            ++ite_current;
            for (int i = 0; i < 3; ++i) vertex[i] += scratch[5] * dis_eval[i];
            scratch[1] = vertex_edge_distance_square(vertex[0], vertex[1], vertex[2]);
            scratch[2] = scratch[1] - xi * xi;
            if (scratch[2] <= static_cast<real>(0)) {
                scratch[1] = accd_detail::min_vv_dsqr_ve(vertex[0], vertex[1], vertex[2]);
                scratch[2] = scratch[1] - xi * xi;
                if (scratch[2] <= static_cast<real>(0)) break;
            }
            scratch[3] = sqrt(scratch[1]);
            scratch[6] = scratch[2] / (scratch[3] + xi);
            if (toi > static_cast<real>(0) && scratch[6] < scratch[4]) break;
            toi += scratch[5];
            if (toi >= tau) {
                toi = static_cast<real>(1);
                return false;
            }
        }
        toi = clamp(toi, static_cast<real>(0), static_cast<real>(1));
        return true;
    }

} // namespace cs

#endif //ACCD_H
