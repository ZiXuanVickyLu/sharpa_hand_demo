#pragma once
// elastic_models.cuh - isotropic hyperelastic models in singular values, their analytic
// eigen-systems and the per-vertex-block closed form (math spec §3.2-3.4 and Appendix B).
//
// Everything here is templated on the scalar T and is CUDA_INLINE_CALLABLE, so the tests run
// the very same code on the host in double while the kernels instantiate T = real.
// Matrices are column-major arrays of 9 (M[3*c + r] is row r, column c), the convention of
// svd3.cuh and of the DmInv / cache arrays of SceneBuffers and ElementCache.
//
// Models (MaterialParams::model): 0 = SNH, 1 = NH, 2 = COR.
//   SNH  Ψ = mu/2 (I_C - 3) + lambda_hat/2 (J - alpha_hat)^2
//   NH   Ψ = mu/2 (I_C - 3) - mu ln J + lambda/2 ln^2 J           (J <= 0: +huge)
//   COR  Ψ = mu Σ (σ_i - 1)^2 + lambda/2 (J - 1)^2
// The twist/flip eigenvalues use the algebraically simplified forms of the math-spec table
// (MS §3.3), which never divide by σ_i - σ_j and therefore contain the σ_i -> σ_j limit
// Ψ_ii - Ψ_ij exactly.
#ifndef CS_FEM_ELASTIC_MODELS_CUH
#define CS_FEM_ELASTIC_MODELS_CUH

#include "core/typedef.cuh"
#include "core/svd3.cuh"
#include "gpu/scene_buffers.cuh"
#include <cmath>
#include <math.h>

namespace cs {
namespace fem {

// Cloth membrane models (MS §3.5) live in the same enum: their elements are triangles padded
// to the tet layout (F with a zero third column, a 3x2 PK1 with a zero third column, the nine
// eigenvalue slots filled as MS §3.6 prescribes) so every downstream kernel is shared.
enum Model : int { kSNH = 0, kNH = 1, kCOR = 2, kClothStVK = 3, kClothCOR = 4, kClothSNH = 5 };
CUDA_INLINE_CALLABLE constexpr bool is_cloth(int model) { return model >= kClothStVK; }

// ---------------------------------------------------------------------------------------------
// Scalar shims (explicit per type so the host never picks an integer/double overload silently).
// ---------------------------------------------------------------------------------------------
template <typename T> struct sc;
template <> struct sc<double> {
    static CUDA_INLINE_CALLABLE double sqrt(double x) { return ::sqrt(x); }
    static CUDA_INLINE_CALLABLE double abs(double x) { return ::fabs(x); }
    static CUDA_INLINE_CALLABLE double log(double x) { return ::log(x); }
    static CUDA_INLINE_CALLABLE double max(double a, double b) { return ::fmax(a, b); }
    static CUDA_INLINE_CALLABLE double min(double a, double b) { return ::fmin(a, b); }
    static CUDA_INLINE_CALLABLE double huge() { return 1e300; }
    static CUDA_INLINE_CALLABLE double tiny() { return 1e-300; }
    static CUDA_INLINE_CALLABLE double eps() { return 2.220446049250313e-16; }
};
template <> struct sc<float> {
    static CUDA_INLINE_CALLABLE float sqrt(float x) { return ::sqrtf(x); }
    static CUDA_INLINE_CALLABLE float abs(float x) { return ::fabsf(x); }
    static CUDA_INLINE_CALLABLE float log(float x) { return ::logf(x); }
    static CUDA_INLINE_CALLABLE float max(float a, float b) { return ::fmaxf(a, b); }
    static CUDA_INLINE_CALLABLE float min(float a, float b) { return ::fminf(a, b); }
    static CUDA_INLINE_CALLABLE float huge() { return 1e30f; }
    static CUDA_INLINE_CALLABLE float tiny() { return 1e-30f; }
    static CUDA_INLINE_CALLABLE float eps() { return 1.1920929e-7f; }
};

// Energy returned for an NH element with J <= 0 (the line search rejects such a step).
template <typename T> CUDA_INLINE_CALLABLE T huge_energy() { return sc<T>::huge(); }

// ---------------------------------------------------------------------------------------------
// Material parameters in the scalar of the caller.
// ---------------------------------------------------------------------------------------------
template <typename T> struct ModelParams {
    int model;
    T mu;
    T lambda;
    T lambda_hat;
    T alpha_hat;
};

template <typename T> CUDA_INLINE_CALLABLE ModelParams<T> model_params(const MaterialParams& m)
{
    ModelParams<T> p;
    p.model = m.model;
    p.mu = T(m.mu);
    p.lambda = T(m.lambda);
    p.lambda_hat = T(m.lambda_hat);
    p.alpha_hat = T(m.alpha_hat);
    return p;
}

// ---------------------------------------------------------------------------------------------
// 3x3 helpers (column-major arrays of 9, no aliasing between inputs and outputs)
// ---------------------------------------------------------------------------------------------
template <typename T> CUDA_INLINE_CALLABLE void m3_mul(const T* A, const T* B, T* C)  // C = A B
{
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            C[3 * c + r] = A[r] * B[3 * c] + A[3 + r] * B[3 * c + 1] + A[6 + r] * B[3 * c + 2];
}

template <typename T> CUDA_INLINE_CALLABLE void m3_mul_ABt(const T* A, const T* B, T* C)  // C = A B^T
{
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) C[3 * c + r] = A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c];
}

template <typename T> CUDA_INLINE_CALLABLE void m3_mul_AtB(const T* A, const T* B, T* C)  // C = A^T B
{
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            C[3 * c + r] = A[3 * r] * B[3 * c] + A[3 * r + 1] * B[3 * c + 1] + A[3 * r + 2] * B[3 * c + 2];
}

template <typename T> CUDA_INLINE_CALLABLE void m3_mulvec(const T* A, const T* v, T* out)  // out = A v
{
    for (int r = 0; r < 3; ++r) out[r] = A[r] * v[0] + A[3 + r] * v[1] + A[6 + r] * v[2];
}

template <typename T> CUDA_INLINE_CALLABLE void m3_mulvec_t(const T* A, const T* v, T* out)  // out = A^T v
{
    for (int r = 0; r < 3; ++r) out[r] = A[3 * r] * v[0] + A[3 * r + 1] * v[1] + A[3 * r + 2] * v[2];
}

template <typename T> CUDA_INLINE_CALLABLE T m3_det(const T* M)
{
    return M[0] * (M[4] * M[8] - M[7] * M[5]) - M[3] * (M[1] * M[8] - M[7] * M[2]) + M[6] * (M[1] * M[5] - M[4] * M[2]);
}

template <typename T> CUDA_INLINE_CALLABLE T m3_frob2(const T* M)
{
    T s = T(0);
    for (int i = 0; i < 9; ++i) s += M[i] * M[i];
    return s;
}

template <typename T> CUDA_INLINE_CALLABLE void v3_cross(const T* a, const T* b, T* out)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// ∂J/∂F = cofactor matrix = [f1 x f2, f2 x f0, f0 x f1] (columns f_c of F).
template <typename T> CUDA_INLINE_CALLABLE void m3_dJdF(const T* F, T* out)
{
    v3_cross(F + 3, F + 6, out);
    v3_cross(F + 6, F + 0, out + 3);
    v3_cross(F + 0, F + 3, out + 6);
}

// ---------------------------------------------------------------------------------------------
// Element kinematics (MS §3.1)
// ---------------------------------------------------------------------------------------------
// F = D_s DmInv with D_s = [x1 - x0, x2 - x0, x3 - x0].
template <typename T>
CUDA_INLINE_CALLABLE void deformation_gradient(const T* x0, const T* x1, const T* x2, const T* x3, const T* DmInv, T* F)
{
    T Ds[9];
    for (int r = 0; r < 3; ++r) {
        Ds[r] = x1[r] - x0[r];
        Ds[3 + r] = x2[r] - x0[r];
        Ds[6 + r] = x3[r] - x0[r];
    }
    m3_mul(Ds, DmInv, F);
}

// Shape gradient a_a of local vertex a: a_m = row (m-1) of DmInv for m = 1..3, a_0 = -Σ a_m.
template <typename T> CUDA_INLINE_CALLABLE void shape_gradient(const T* DmInv, int a, T* out)
{
    if (a == 0) {
        for (int k = 0; k < 3; ++k) out[k] = -(DmInv[3 * k] + DmInv[3 * k + 1] + DmInv[3 * k + 2]);
    } else {
        const int m = a - 1;
        for (int k = 0; k < 3; ++k) out[k] = DmInv[3 * k + m];
    }
}

// ---------------------------------------------------------------------------------------------
// Energy densities (MS §3.2)
// ---------------------------------------------------------------------------------------------
template <typename T> CUDA_INLINE_CALLABLE T psi_sigma(const ModelParams<T>& p, const T* s)
{
    const T J = s[0] * s[1] * s[2];
    switch (p.model) {
    case kSNH: {
        const T IC = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        const T d = J - p.alpha_hat;
        return T(0.5) * p.mu * (IC - T(3)) + T(0.5) * p.lambda_hat * d * d;
    }
    case kNH: {
        if (!(J > T(0))) return huge_energy<T>();
        const T IC = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        const T lnJ = sc<T>::log(J);
        return T(0.5) * p.mu * (IC - T(3)) - p.mu * lnJ + T(0.5) * p.lambda * lnJ * lnJ;
    }
    default: {
        const T d0 = s[0] - T(1), d1 = s[1] - T(1), d2 = s[2] - T(1), dJ = J - T(1);
        return p.mu * (d0 * d0 + d1 * d1 + d2 * d2) + T(0.5) * p.lambda * dJ * dJ;
    }
    }
}

// SNH / NH straight from F (no SVD). COR is not handled here (needs σ): see psi_F.
template <typename T> CUDA_INLINE_CALLABLE T psi_F_nosvd(const ModelParams<T>& p, const T* F)
{
    const T IC = m3_frob2(F);
    const T J = m3_det(F);
    if (p.model == kSNH) {
        const T d = J - p.alpha_hat;
        return T(0.5) * p.mu * (IC - T(3)) + T(0.5) * p.lambda_hat * d * d;
    }
    if (!(J > T(0))) return huge_energy<T>();
    const T lnJ = sc<T>::log(J);
    return T(0.5) * p.mu * (IC - T(3)) - p.mu * lnJ + T(0.5) * p.lambda * lnJ * lnJ;
}

// Any model from F (COR through the signed SVD).
template <typename T> CUDA_INLINE_CALLABLE T psi_F(const ModelParams<T>& p, const T* F)
{
    if (p.model == kCOR) {
        T U[9], S[3], V[9];
        svd3(F, U, S, V);
        return psi_sigma(p, S);
    }
    return psi_F_nosvd(p, F);
}

// ---------------------------------------------------------------------------------------------
// First Piola-Kirchhoff stress (MS §3.4). U, V are only read for COR (R = U V^T).
// ---------------------------------------------------------------------------------------------
template <typename T>
CUDA_INLINE_CALLABLE void pk1(const ModelParams<T>& p, const T* F, const T* U, const T* V, T* P)
{
    T cof[9];
    m3_dJdF(F, cof);
    const T J = m3_det(F);
    switch (p.model) {
    case kSNH: {
        const T c = p.lambda_hat * (J - p.alpha_hat);
        for (int i = 0; i < 9; ++i) P[i] = p.mu * F[i] + c * cof[i];
        break;
    }
    case kNH: {
        // mu (F - F^-T) + lambda ln J F^-T with F^-T = cof / J.
        // J <= 0 is outside the model's domain (the inversion guard of MS 10.4 keeps every
        // accepted iterate and every line-search sample at J > 0). Dividing by a clamped J
        // there would overflow to +-Inf and poison the gradient of the *rejected* state with
        // NaNs, so return the finite (meaningless) elastic part instead.
        if (!(J > T(0))) {
            for (int i = 0; i < 9; ++i) P[i] = p.mu * F[i];
            break;
        }
        const T c = (p.lambda * sc<T>::log(J) - p.mu) / J;
        for (int i = 0; i < 9; ++i) P[i] = p.mu * F[i] + c * cof[i];
        break;
    }
    default: {
        T R[9];
        m3_mul_ABt(U, V, R);
        const T c = p.lambda * (J - T(1));
        for (int i = 0; i < 9; ++i) P[i] = T(2) * p.mu * (F[i] - R[i]) + c * cof[i];
        break;
    }
    }
}

// ---------------------------------------------------------------------------------------------
// Derivatives in singular values (MS §3.3 table). Pair index p = 0, 1, 2 <-> (i, j) = (0,1),
// (0,2), (1,2); k is the third index. tau / phi are the UNprojected twist / flip eigenvalues.
// ---------------------------------------------------------------------------------------------
template <typename T> struct SigmaDerivs {
    T psi;      // Ψ
    T dpsi[3];  // Ψ_i
    T A[9];     // Ψ_ij (symmetric, column-major)
    T tau[3];   // (Ψ_i + Ψ_j) / (σ_i + σ_j)
    T phi[3];   // (Ψ_i - Ψ_j) / (σ_i - σ_j)
};

CUDA_INLINE_CALLABLE constexpr int pair_i(int p) { return p == 2 ? 1 : 0; }
CUDA_INLINE_CALLABLE constexpr int pair_j(int p) { return p == 0 ? 1 : 2; }
CUDA_INLINE_CALLABLE constexpr int pair_k(int p) { return 2 - p; }

template <typename T> CUDA_INLINE_CALLABLE void sigma_derivs(const ModelParams<T>& p, const T* s, SigmaDerivs<T>& d)
{
    const T J = s[0] * s[1] * s[2];
    d.psi = psi_sigma(p, s);
    switch (p.model) {
    case kSNH: {
        const T Ja = p.lambda_hat * (J - p.alpha_hat);
        const T off = p.lambda_hat * (T(2) * J - p.alpha_hat);
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            d.dpsi[i] = p.mu * s[i] + Ja * s[j] * s[k];
            d.A[3 * i + i] = p.mu + p.lambda_hat * s[j] * s[j] * s[k] * s[k];
        }
        for (int q = 0; q < 3; ++q) {
            const int i = pair_i(q), j = pair_j(q), k = pair_k(q);
            d.A[3 * j + i] = d.A[3 * i + j] = off * s[k];
            d.tau[q] = p.mu + Ja * s[k];
            d.phi[q] = p.mu - Ja * s[k];
        }
        break;
    }
    case kNH: {
        const T Js = sc<T>::max(J, sc<T>::tiny());
        const T lnJ = sc<T>::log(Js);
        const T c = p.lambda * lnJ - p.mu;
        for (int i = 0; i < 3; ++i) {
            const T si = s[i];
            d.dpsi[i] = p.mu * si + c / si;
            d.A[3 * i + i] = p.mu + (p.lambda - c) / (si * si);
        }
        for (int q = 0; q < 3; ++q) {
            const int i = pair_i(q), j = pair_j(q);
            const T inv = T(1) / (s[i] * s[j]);
            d.A[3 * j + i] = d.A[3 * i + j] = p.lambda * inv;
            d.tau[q] = p.mu + c * inv;
            d.phi[q] = p.mu - c * inv;
        }
        break;
    }
    default: {
        const T Jm = p.lambda * (J - T(1));
        const T off = p.lambda * (T(2) * J - T(1));
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            d.dpsi[i] = T(2) * p.mu * (s[i] - T(1)) + Jm * s[j] * s[k];
            d.A[3 * i + i] = T(2) * p.mu + p.lambda * s[j] * s[j] * s[k] * s[k];
        }
        for (int q = 0; q < 3; ++q) {
            const int i = pair_i(q), j = pair_j(q), k = pair_k(q);
            d.A[3 * j + i] = d.A[3 * i + j] = off * s[k];
            // 4 mu / (σ_i + σ_j) is genuinely singular at σ_i = -σ_j (non-unique polar rotation);
            // keep the denominator away from zero so no NaN enters the matrix.
            T sum = s[i] + s[j];
            const T floor_ = T(1e-6) * sc<T>::max(T(1), sc<T>::abs(s[i]));
            if (sc<T>::abs(sum) < floor_) sum = (sum < T(0)) ? -floor_ : floor_;
            d.tau[q] = T(2) * p.mu - T(4) * p.mu / sum + Jm * s[k];
            d.phi[q] = T(2) * p.mu - Jm * s[k];
        }
        break;
    }
    }
}

// ---------------------------------------------------------------------------------------------
// Shells (MS §3.5, §3.6): everything in the padded tet layout.
// ---------------------------------------------------------------------------------------------
// Membrane energy density in the two singular values.
template <typename T> CUDA_INLINE_CALLABLE T psi2_sigma(const ModelParams<T>& p, T s0, T s1)
{
    const T J = s0 * s1;
    const T IC = s0 * s0 + s1 * s1;
    switch (p.model) {
    case kClothStVK: {
        // mu ||E||_F^2 + lambda/2 tr^2 E with E = (F^T F - I)/2: in singular values
        // E = diag((s_i^2 - 1)/2).
        const T e0 = T(0.5) * (s0 * s0 - T(1)), e1 = T(0.5) * (s1 * s1 - T(1));
        return p.mu * (e0 * e0 + e1 * e1) + T(0.5) * p.lambda * (e0 + e1) * (e0 + e1);
    }
    case kClothSNH: {
        const T d = J - p.alpha_hat;
        return T(0.5) * p.mu * (IC - T(2)) + T(0.5) * p.lambda_hat * d * d;
    }
    default: {  // kClothCOR
        const T d0 = s0 - T(1), d1 = s1 - T(1), dJ = J - T(1);
        return p.mu * (d0 * d0 + d1 * d1) + T(0.5) * p.lambda * dJ * dJ;
    }
    }
}

// Psi_i, the 2x2 A_ij, the in-plane twist/flip pair and the two out-of-plane eigenvalues
// nu_i = Psi_i / sigma_i, written straight into the padded nine-slot layout of eigen_system:
//   eig = { a0, a1, 0 (scaling, the third pinned to 0), tau01, nu0, nu1 (twist slots), phi01, nu0, nu1 (flip slots) }
// with the pair convention of elem_block: pair 1 = (0,2) carries nu_0, pair 2 = (1,2) carries nu_1.
// Both the twist and the flip slot of an out-of-plane pair carry nu_i: elem_block's
// (tau+phi)/2 then equals nu_i and its (phi-tau)/2 vanishes, which is exactly the
// n n^T v_i v_i^T block of MS §3.6 once the zero third component of b_a kills the rest.
template <typename T>
CUDA_INLINE_CALLABLE void shell_eigen_system(const ModelParams<T>& p, T s0, T s1, T* eig, T* q, T* dpsi)
{
    const T J = s0 * s1;
    T A00, A01, A11, tau, phi;
    switch (p.model) {
    case kClothStVK: {
        // Psi_i = mu s_i (s_i^2 - 1) + lambda/2 s_i (I_C - 2)
        const T IC2 = s0 * s0 + s1 * s1 - T(2);
        dpsi[0] = p.mu * s0 * (s0 * s0 - T(1)) + T(0.5) * p.lambda * s0 * IC2;
        dpsi[1] = p.mu * s1 * (s1 * s1 - T(1)) + T(0.5) * p.lambda * s1 * IC2;
        A00 = p.mu * (T(3) * s0 * s0 - T(1)) + T(0.5) * p.lambda * (IC2 + T(2) * s0 * s0);
        A11 = p.mu * (T(3) * s1 * s1 - T(1)) + T(0.5) * p.lambda * (IC2 + T(2) * s1 * s1);
        A01 = p.lambda * s0 * s1;
        // (Psi_0 + Psi_1)/(s0 + s1) and (Psi_0 - Psi_1)/(s0 - s1), simplified so the flip
        // limit s0 -> s1 is exact: Psi_i = s_i [ mu (s_i^2 - 1) + lambda/2 IC2 ].
        tau = p.mu * (s0 * s0 - s0 * s1 + s1 * s1 - T(1)) + T(0.5) * p.lambda * IC2;
        phi = p.mu * (s0 * s0 + s0 * s1 + s1 * s1 - T(1)) + T(0.5) * p.lambda * IC2;
        break;
    }
    case kClothSNH: {
        const T Ja = p.lambda_hat * (J - p.alpha_hat);
        dpsi[0] = p.mu * s0 + Ja * s1;
        dpsi[1] = p.mu * s1 + Ja * s0;
        A00 = p.mu + p.lambda_hat * s1 * s1;
        A11 = p.mu + p.lambda_hat * s0 * s0;
        A01 = p.lambda_hat * (T(2) * J - p.alpha_hat);
        tau = p.mu + Ja;
        phi = p.mu - Ja;
        break;
    }
    default: {  // kClothCOR
        const T Jm = p.lambda * (J - T(1));
        dpsi[0] = T(2) * p.mu * (s0 - T(1)) + Jm * s1;
        dpsi[1] = T(2) * p.mu * (s1 - T(1)) + Jm * s0;
        A00 = T(2) * p.mu + p.lambda * s1 * s1;
        A11 = T(2) * p.mu + p.lambda * s0 * s0;
        A01 = p.lambda * (T(2) * J - T(1));
        T sum = s0 + s1;
        const T floor_ = T(1e-6) * sc<T>::max(T(1), sc<T>::abs(s0));
        if (sc<T>::abs(sum) < floor_) sum = floor_;
        tau = T(2) * p.mu - T(4) * p.mu / sum + Jm;
        phi = T(2) * p.mu - Jm;
        break;
    }
    }
    // scaling block: 2x2 eigen-decomposition, padded with a zero third eigenpair on e3
    const T half = T(0.5) * (A00 + A11);
    const T diff = T(0.5) * (A00 - A11);
    const T rad = sc<T>::sqrt(diff * diff + A01 * A01);
    eig[0] = half + rad;
    eig[1] = half - rad;
    eig[2] = T(0);
    T c, sn;
    if (rad > T(0)) {
        // eigenvector of eig[0]: (A01, eig0 - A00) or (eig0 - A11, A01), whichever is larger
        const T v0x = A01, v0y = eig[0] - A00;
        const T w0x = eig[0] - A11, w0y = A01;
        if (v0x * v0x + v0y * v0y >= w0x * w0x + w0y * w0y) { c = v0x; sn = v0y; }
        else { c = w0x; sn = w0y; }
        const T nrm = sc<T>::sqrt(c * c + sn * sn);
        if (nrm > T(0)) { c /= nrm; sn /= nrm; } else { c = T(1); sn = T(0); }
    } else { c = T(1); sn = T(0); }
    for (int i = 0; i < 9; ++i) q[i] = T(0);
    q[0] = c;  q[1] = sn;          // column 0 = q_0
    q[3] = -sn; q[4] = c;          // column 1 = q_1
    q[8] = T(1);                   // column 2 = e3 (dropped by the zero third component of b_a)
    // out-of-plane eigenvalues nu_i = Psi_i / sigma_i (limit at sigma -> 0: keep it finite)
    const T tiny = T(1e-12);
    const T nu0 = dpsi[0] / sc<T>::max(s0, tiny);
    const T nu1 = dpsi[1] / sc<T>::max(s1, tiny);
    eig[3] = tau; eig[4] = nu0; eig[5] = nu1;
    eig[6] = phi; eig[7] = nu0; eig[8] = nu1;
}

// Thin SVD of the padded F (column-major 3x3 with a zero third column): F = U diag(s0, s1) V^T.
// V is a 2x2 rotation from the eigen-decomposition of F^T F, u_i = F v_i / s_i, n = u_0 x u_1.
// Writes the padded rotations Ubar = [u0 u1 n] and Vbar = diag(V, 1) in column-major 3x3.
template <typename T> CUDA_INLINE_CALLABLE void shell_svd(const T* F, T* Ub, T* S, T* Vb)
{
    // F^T F, upper-left 2x2 (columns f0, f1 of F)
    const T* f0 = F; const T* f1 = F + 3;
    const T a = f0[0] * f0[0] + f0[1] * f0[1] + f0[2] * f0[2];
    const T b = f0[0] * f1[0] + f0[1] * f1[1] + f0[2] * f1[2];
    const T c = f1[0] * f1[0] + f1[1] * f1[1] + f1[2] * f1[2];
    const T half = T(0.5) * (a + c), diff = T(0.5) * (a - c);
    const T rad = sc<T>::sqrt(diff * diff + b * b);
    const T l0 = sc<T>::max(half + rad, T(0)), l1 = sc<T>::max(half - rad, T(0));
    T cs_, sn;
    if (rad > T(0)) {
        const T vx = b, vy = l0 - a, wx = l0 - c, wy = b;
        if (vx * vx + vy * vy >= wx * wx + wy * wy) { cs_ = vx; sn = vy; } else { cs_ = wx; sn = wy; }
        const T nrm = sc<T>::sqrt(cs_ * cs_ + sn * sn);
        if (nrm > T(0)) { cs_ /= nrm; sn /= nrm; } else { cs_ = T(1); sn = T(0); }
    } else { cs_ = T(1); sn = T(0); }
    S[0] = sc<T>::sqrt(l0);
    S[1] = sc<T>::sqrt(l1);
    S[2] = T(0);
    // V = [v0 v1] with v0 = (cs, sn), v1 = (-sn, cs): a rotation
    T u0[3], u1[3];
    for (int r = 0; r < 3; ++r) {
        u0[r] = cs_ * f0[r] + sn * f1[r];
        u1[r] = -sn * f0[r] + cs_ * f1[r];
    }
    const T tiny = T(1e-30);
    if (S[0] > tiny) for (int r = 0; r < 3; ++r) u0[r] /= S[0];
    else { u0[0] = T(1); u0[1] = T(0); u0[2] = T(0); }
    if (S[1] > tiny * sc<T>::max(T(1), S[0])) {
        for (int r = 0; r < 3; ++r) u1[r] /= S[1];
        // re-orthogonalise against u0 (cheap insurance for nearly degenerate sigma_1)
        const T d = u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2];
        for (int r = 0; r < 3; ++r) u1[r] -= d * u0[r];
        const T n1 = sc<T>::sqrt(u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
        if (n1 > tiny) for (int r = 0; r < 3; ++r) u1[r] /= n1;
    } else {
        // any unit vector orthogonal to u0
        T t[3] = {T(0), T(1), T(0)};
        if (sc<T>::abs(u0[1]) > T(0.9)) { t[0] = T(1); t[1] = T(0); }
        const T d = u0[0] * t[0] + u0[1] * t[1] + u0[2] * t[2];
        for (int r = 0; r < 3; ++r) u1[r] = t[r] - d * u0[r];
        const T n1 = sc<T>::sqrt(u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
        for (int r = 0; r < 3; ++r) u1[r] /= n1;
    }
    T n[3];
    v3_cross(u0, u1, n);
    for (int r = 0; r < 3; ++r) { Ub[r] = u0[r]; Ub[3 + r] = u1[r]; Ub[6 + r] = n[r]; }
    for (int i = 0; i < 9; ++i) Vb[i] = T(0);
    Vb[0] = cs_; Vb[1] = sn; Vb[3] = -sn; Vb[4] = cs_; Vb[8] = T(1);
}

// PK1 of a membrane, padded: P = Ubar diag(Psi_0, Psi_1, 0) Vbar^T (zero third column).
template <typename T> CUDA_INLINE_CALLABLE void shell_pk1(const T* Ub, const T* dpsi, const T* Vb, T* P)
{
    T D[9];
    for (int i = 0; i < 9; ++i) D[i] = T(0);
    D[0] = dpsi[0]; D[4] = dpsi[1];
    T UD[9];
    m3_mul(Ub, D, UD);
    m3_mul_ABt(UD, Vb, P);
}

// ---------------------------------------------------------------------------------------------
// Symmetric 3x3 eigen-solver: cyclic Jacobi with exact rotations, 8 sweeps (converges
// quadratically; 3x3 needs 3-5). A = Q diag(evals) Q^T, Q orthonormal (column-major, column s
// is eigenvector s). Unsorted.
// ---------------------------------------------------------------------------------------------
template <typename T> CUDA_INLINE_CALLABLE void sym_eig3(const T* A, T* evals, T* Q)
{
    T a[9];
    for (int i = 0; i < 9; ++i) a[i] = A[i];
    for (int i = 0; i < 9; ++i) Q[i] = T(0);
    Q[0] = Q[4] = Q[8] = T(1);
    constexpr int kSweeps = 8;
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        for (int q = 0; q < 3; ++q) {
            const int p = pair_i(q), r = pair_j(q);
            const T apq = a[3 * r + p];
            if (apq == T(0)) continue;
            const T app = a[3 * p + p], arr = a[3 * r + r];
            const T theta = (arr - app) / (T(2) * apq);
            T t;
            if (sc<T>::abs(theta) > T(1e15)) {
                t = T(1) / (T(2) * theta);
            } else {
                t = T(1) / (sc<T>::abs(theta) + sc<T>::sqrt(theta * theta + T(1)));
                if (theta < T(0)) t = -t;
            }
            const T c = T(1) / sc<T>::sqrt(t * t + T(1));
            const T s = t * c;
            // A' = P^T A P with P_pp = P_rr = c, P_pr = s, P_rp = -s
            a[3 * p + p] = app - t * apq;
            a[3 * r + r] = arr + t * apq;
            a[3 * r + p] = T(0);
            a[3 * p + r] = T(0);
            const int o = 3 - p - r;  // the remaining index
            const T aop = a[3 * p + o], aor = a[3 * r + o];
            const T nop = c * aop - s * aor;
            const T nor = s * aop + c * aor;
            a[3 * p + o] = nop;
            a[3 * o + p] = nop;
            a[3 * r + o] = nor;
            a[3 * o + r] = nor;
            // Q' = Q P
            for (int m = 0; m < 3; ++m) {
                const T qp = Q[3 * p + m], qr = Q[3 * r + m];
                Q[3 * p + m] = c * qp - s * qr;
                Q[3 * r + m] = s * qp + c * qr;
            }
        }
        // Stop once the off-diagonal is at the roundoff level of the diagonal (3 sweeps is the
        // typical count for 3x3; the sweep cap stays as the guarantee).
        const T off = sc<T>::abs(a[3]) + sc<T>::abs(a[6]) + sc<T>::abs(a[7]);
        const T dia = sc<T>::abs(a[0]) + sc<T>::abs(a[4]) + sc<T>::abs(a[8]);
        if (off <= sc<T>::eps() * dia) break;
    }
    evals[0] = a[0];
    evals[1] = a[4];
    evals[2] = a[8];
}

// ---------------------------------------------------------------------------------------------
// The nine analytic eigenvalues of ∂²Ψ/∂F² (unprojected) and the scaling eigenvectors:
//   eig = { a0, a1, a2, tau01, tau02, tau12, phi01, phi02, phi12 }, q = [q0 q1 q2] column-major.
// Eigenvectors of the 9x9 Hessian: Q_s = U diag(q_s) V^T, twist Q_ij = U (e_i e_j^T - e_j e_i^T) V^T / sqrt2,
// flip Q_ij = U (e_i e_j^T + e_j e_i^T) V^T / sqrt2 (MS §3.3).
// ---------------------------------------------------------------------------------------------
template <typename T> CUDA_INLINE_CALLABLE void eigen_system(const ModelParams<T>& p, const T* s, T* eig, T* q)
{
    SigmaDerivs<T> d;
    sigma_derivs(p, s, d);
    sym_eig3(d.A, eig, q);
    for (int i = 0; i < 3; ++i) {
        eig[3 + i] = d.tau[i];
        eig[6 + i] = d.phi[i];
    }
}

template <typename T> CUDA_INLINE_CALLABLE void project_eig(T* eig)
{
    for (int i = 0; i < 9; ++i) eig[i] = sc<T>::max(eig[i], T(0));
}

// ---------------------------------------------------------------------------------------------
// Appendix B: out (ROW-major 3x3) = scale * U S_ab U^T with B_ab = b_a b_b^T,
//   S_ab = Σ_s a_s (q_s q_s^T) ∘ B + Σ_(i,j) [ (tau+phi)/2 (B_jj e_i e_i^T + B_ii e_j e_j^T)
//                                            + (phi-tau)/2 (B_ji e_i e_j^T + B_ij e_j e_i^T) ].
// eig must already be projected. scale carries h^2 V_e.
// ---------------------------------------------------------------------------------------------
template <typename T>
CUDA_INLINE_CALLABLE void elem_block(const T* U, const T* eig, const T* q, const T* ba, const T* bb, T scale, T* out)
{
    T B[9];
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) B[3 * c + r] = ba[r] * bb[c];
    T S[9];
    for (int i = 0; i < 9; ++i) S[i] = T(0);
    for (int s = 0; s < 3; ++s) {
        const T as = eig[s];
        const T* qs = q + 3 * s;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) S[3 * c + r] += as * qs[r] * qs[c] * B[3 * c + r];
    }
    for (int p = 0; p < 3; ++p) {
        const int i = pair_i(p), j = pair_j(p);
        const T tau = eig[3 + p], phi = eig[6 + p];
        const T h = T(0.5) * (tau + phi), g = T(0.5) * (phi - tau);
        S[3 * i + i] += h * B[3 * j + j];
        S[3 * j + j] += h * B[3 * i + i];
        S[3 * j + i] += g * B[3 * i + j];  // S(i,j) += g B(j,i)
        S[3 * i + j] += g * B[3 * j + i];  // S(j,i) += g B(i,j)
    }
    T US[9], M[9];
    m3_mul(U, S, US);
    m3_mul_ABt(US, U, M);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[3 * r + c] = scale * M[3 * c + r];
}

}  // namespace fem
}  // namespace cs

#endif  // CS_FEM_ELASTIC_MODELS_CUH
