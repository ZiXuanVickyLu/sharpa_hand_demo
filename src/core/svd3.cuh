// svd3.cuh - signed 3x3 singular value decomposition for host and device code.
//
// Adapted from NVIDIA Warp, warp/native/svd.h (Apache-2.0; the vendored copy with its license is
// ext/warp_svd/). Warp's file is itself a templated port of Eric V. Jang's svd3_cuda (MIT), an
// implementation of McAdams, Selle, Tamstorf, Teran, Sifakis, "Computing the Singular Value
// Decomposition of 3x3 matrices with minimal branching and elementary floating point
// operations", University of Wisconsin-Madison TR 1690, 2011.
//
// What was changed relative to warp/native/svd.h:
//   * only the scalar-level routines are kept (config, recipSqrt, Jacobi conjugation, approximate
//     Givens quaternion, QR Givens, singular-value sort, the core SVD); Warp's mat_t/vec_t/half
//     code, the 2x2 SVD, QR/eig wrappers and the adjoints are not vendored;
//   * CUDA_CALLABLE -> CUDA_INLINE_CALLABLE, namespace wp -> cs::svd_detail, the math shims of
//     builtin.h are replaced by the small `scalar<T>` table below;
//   * the accumulated Jacobi quaternion is renormalised before it is turned into V (see svd_core);
//   * the public entry points (cs::svd3 / cs::polar3) add an exact power-of-two prescale and a
//     canonicalisation pass that enforces the signed-SVD convention documented below.
//
// The original attribution block of warp/native/svd.h follows verbatim.
//
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The MIT License (MIT)
//
// Copyright (c) 2014 Eric V. Jang
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Source: https://github.com/ericjang/svd3/blob/master/svd3_cuda/svd3_cuda.h
//
// ---------------------------------------------------------------------------------------------
// Convention (math spec MS 3.2, implementation spec 3 and 11):
//
//     A = U * diag(S) * V^T
//
//   * U and V are proper rotations: det U = det V = +1 (up to rounding);
//   * S[0] >= S[1] >= |S[2]| >= 0, i.e. magnitudes sorted descending, with the sign of det A
//     carried by S[2] alone:  S[2] < 0  <=>  det A < 0;
//   * matrices are column-major arrays of 9 (A[c*3 + r] is row r, column c), matching
//     double3x3/float3x3 (columns c0, c1, c2);
//   * the routine is scale invariant: the input is prescaled by an exact power of two so that the
//     absolute epsilons of the Jacobi/QR stages act relative to max|a_ij|;
//   * precision: svd_config<double> runs 8 Jacobi sweeps with 1e-12 epsilons, svd_config<float>
//     6 sweeps (Warp: 4, see svd_config) with 1e-6 epsilons. Measured (test/test_svd3.cu, u =
//     unit roundoff): reconstruction and orthogonality ~20u for well-conditioned input (3e-15
//     double, 2e-6 float). For nearly rank-1 input with sigma3 != sigma2 the reconstruction error
//     grows like u * sigma1/sigma2 (the QR residual r23, a rounding residual of A^T A of size
//     u*sigma1^2, is divided by ||A v2|| = sigma2 and then dropped); exact rank-1 input and
//     degenerate pairs (sigma2 = sigma3) stay at ~u. Deformation gradients never approach that
//     regime.
//
// polar3 returns the rotation-variant polar decomposition A = R * S_sym with R = U V^T in SO(3)
// and S_sym = V diag(S) V^T symmetric; S_sym is indefinite (one negative eigenvalue) when A is
// inverted, which is the convention of the elasticity literature (Irving et al. 2004).
// ---------------------------------------------------------------------------------------------
#pragma once
#ifndef CS_SVD3_CUH
#define CS_SVD3_CUH

#include "typedef.h"
#include "double3x3.h"
#include "float3x3.h"
#include <cmath>
#include <math.h>

namespace cs {
namespace svd_detail {

// ---------------------------------------------------------------------------------------------
// Precision configuration: Jacobi sweep count and the epsilons of the Givens gates.
// ---------------------------------------------------------------------------------------------
// Float runs 6 sweeps instead of Warp's 4: measured on 100k random matrices (test_svd3 and the
// sweep study behind it), 4 sweeps leave ~2% of inputs with a reconstruction error above 1e-5
// (worst 7e-3, identical in double with 4 sweeps, i.e. non-convergence rather than rounding);
// 5 sweeps reach 8e-6, 6 sweeps 4e-6 (the float floor), and more sweeps slowly degrade again
// because the accumulated quaternion drifts on degenerate spectra. Double converges by 7 sweeps
// (9e-15); 8 keeps a one-sweep margin.
template <typename Type> struct svd_config {
    static constexpr float SVD_EPSILON = 1.e-6f;
    static constexpr float QR_GIVENS_EPSILON = 1.e-6f;
    static constexpr int JACOBI_ITERATIONS = 6;
};

template <> struct svd_config<double> {
    static constexpr double SVD_EPSILON = 1.e-12;
    static constexpr double QR_GIVENS_EPSILON = 1.e-12;
    static constexpr int JACOBI_ITERATIONS = 8;
};

// ---------------------------------------------------------------------------------------------
// Scalar math shims: resolve to the CUDA device intrinsics in device code and to the C library on
// the host. Unqualified `sqrt`/`abs`/`max` would silently pick integer or double overloads on
// the host, so every call site goes through this table.
// ---------------------------------------------------------------------------------------------
template <typename Type> struct scalar;

template <> struct scalar<float> {
    static CUDA_INLINE_CALLABLE float sqrt(float x) { return ::sqrtf(x); }
    static CUDA_INLINE_CALLABLE float abs(float x) { return ::fabsf(x); }
    static CUDA_INLINE_CALLABLE float max(float a, float b) { return ::fmaxf(a, b); }
    static CUDA_INLINE_CALLABLE float frexp(float x, int* e) { return ::frexpf(x, e); }
    static CUDA_INLINE_CALLABLE float ldexp(float x, int e) { return ::ldexpf(x, e); }
    static CUDA_INLINE_CALLABLE bool finite(float x) { return ::isfinite(x); }
    static CUDA_INLINE_CALLABLE float rsqrt(float x) {
#if defined(__CUDA_ARCH__)
        return ::rsqrtf(x);
#else
        return 1.0f / ::sqrtf(x);
#endif
    }
};

template <> struct scalar<double> {
    static CUDA_INLINE_CALLABLE double sqrt(double x) { return ::sqrt(x); }
    static CUDA_INLINE_CALLABLE double abs(double x) { return ::fabs(x); }
    static CUDA_INLINE_CALLABLE double max(double a, double b) { return ::fmax(a, b); }
    static CUDA_INLINE_CALLABLE double frexp(double x, int* e) { return ::frexp(x, e); }
    static CUDA_INLINE_CALLABLE double ldexp(double x, int e) { return ::ldexp(x, e); }
    static CUDA_INLINE_CALLABLE bool finite(double x) { return ::isfinite(x); }
    static CUDA_INLINE_CALLABLE double rsqrt(double x) {
#if defined(__CUDA_ARCH__)
        return ::rsqrt(x);
#else
        return 1.0 / ::sqrt(x);
#endif
    }
};

template <typename Type> CUDA_INLINE_CALLABLE Type recipSqrt(Type x) { return scalar<Type>::rsqrt(x); }

// ---------------------------------------------------------------------------------------------
// Warp's scalar routines (names kept so the file diffs against warp/native/svd.h).
// ---------------------------------------------------------------------------------------------
template <typename Type> CUDA_INLINE_CALLABLE void condSwap(bool c, Type& X, Type& Y)
{
    // used in step 2
    Type Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

template <typename Type> CUDA_INLINE_CALLABLE void condNegSwap(bool c, Type& X, Type& Y)
{
    // used in step 2 and 3
    Type Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}

// matrix multiplication M = A * B
template <typename Type>
CUDA_INLINE_CALLABLE void multAB(
    Type a11, Type a12, Type a13,
    Type a21, Type a22, Type a23,
    Type a31, Type a32, Type a33,
    //
    Type b11, Type b12, Type b13,
    Type b21, Type b22, Type b23,
    Type b31, Type b32, Type b33,
    //
    Type& m11, Type& m12, Type& m13,
    Type& m21, Type& m22, Type& m23,
    Type& m31, Type& m32, Type& m33)
{
    m11 = a11 * b11 + a12 * b21 + a13 * b31;
    m12 = a11 * b12 + a12 * b22 + a13 * b32;
    m13 = a11 * b13 + a12 * b23 + a13 * b33;
    m21 = a21 * b11 + a22 * b21 + a23 * b31;
    m22 = a21 * b12 + a22 * b22 + a23 * b32;
    m23 = a21 * b13 + a22 * b23 + a23 * b33;
    m31 = a31 * b11 + a32 * b21 + a33 * b31;
    m32 = a31 * b12 + a32 * b22 + a33 * b32;
    m33 = a31 * b13 + a32 * b23 + a33 * b33;
}

// matrix multiplication M = Transpose[A] * B
template <typename Type>
CUDA_INLINE_CALLABLE void multAtB(
    Type a11, Type a12, Type a13,
    Type a21, Type a22, Type a23,
    Type a31, Type a32, Type a33,
    //
    Type b11, Type b12, Type b13,
    Type b21, Type b22, Type b23,
    Type b31, Type b32, Type b33,
    //
    Type& m11, Type& m12, Type& m13,
    Type& m21, Type& m22, Type& m23,
    Type& m31, Type& m32, Type& m33)
{
    m11 = a11 * b11 + a21 * b21 + a31 * b31;
    m12 = a11 * b12 + a21 * b22 + a31 * b32;
    m13 = a11 * b13 + a21 * b23 + a31 * b33;
    m21 = a12 * b11 + a22 * b21 + a32 * b31;
    m22 = a12 * b12 + a22 * b22 + a32 * b32;
    m23 = a12 * b13 + a22 * b23 + a32 * b33;
    m31 = a13 * b11 + a23 * b21 + a33 * b31;
    m32 = a13 * b12 + a23 * b22 + a33 * b32;
    m33 = a13 * b13 + a23 * b23 + a33 * b33;
}

template <typename Type>
CUDA_INLINE_CALLABLE void quatToMat3(
    const Type* qV,
    Type& m11, Type& m12, Type& m13,
    Type& m21, Type& m22, Type& m23,
    Type& m31, Type& m32, Type& m33)
{
    Type w = qV[3];
    Type x = qV[0];
    Type y = qV[1];
    Type z = qV[2];

    Type qxx = x * x;
    Type qyy = y * y;
    Type qzz = z * z;
    Type qxz = x * z;
    Type qxy = x * y;
    Type qyz = y * z;
    Type qwx = w * x;
    Type qwy = w * y;
    Type qwz = w * z;

    m11 = Type(1) - Type(2) * (qyy + qzz);
    m12 = Type(2) * (qxy - qwz);
    m13 = Type(2) * (qxz + qwy);
    m21 = Type(2) * (qxy + qwz);
    m22 = Type(1) - Type(2) * (qxx + qzz);
    m23 = Type(2) * (qyz - qwx);
    m31 = Type(2) * (qxz - qwy);
    m32 = Type(2) * (qyz + qwx);
    m33 = Type(1) - Type(2) * (qxx + qyy);
}

template <typename Type>
CUDA_INLINE_CALLABLE void approximateGivensQuaternion(Type a11, Type a12, Type a22, Type& ch, Type& sh)
{
    /*
     * Given givens angle computed by approximateGivensAngles,
     * compute the corresponding rotation quaternion.
     */
    constexpr double _gamma = 5.82842712474619;   // FOUR_GAMMA_SQUARED = sqrt(8)+3;
    constexpr double _cstar = 0.9238795325112867; // cos(pi/8)
    constexpr double _sstar = 0.3826834323650898; // sin(p/8)

    ch = Type(2) * (a11 - a22);
    sh = a12;
    bool b = Type(_gamma) * sh * sh < ch * ch;
    Type w = recipSqrt(ch * ch + sh * sh);
    ch = b ? w * ch : Type(_cstar);
    sh = b ? w * sh : Type(_sstar);
}

template <typename Type>
CUDA_INLINE_CALLABLE void jacobiConjugation(
    const int x, const int y, const int z,
    Type& s11, Type& s21, Type& s22, Type& s31, Type& s32, Type& s33,
    Type* qV)
{
    Type ch, sh;
    approximateGivensQuaternion(s11, s21, s22, ch, sh);

    Type scale = ch * ch + sh * sh;
    Type a = (ch * ch - sh * sh) / scale;
    Type b = (Type(2) * sh * ch) / scale;

    // make temp copy of S
    Type _s11 = s11;
    Type _s21 = s21;
    Type _s22 = s22;
    Type _s31 = s31;
    Type _s32 = s32;
    Type _s33 = s33;

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    s31 = a * _s31 + b * _s32;
    s32 = -b * _s31 + a * _s32;
    s33 = _s33;

    // update cumulative rotation qV
    Type tmp[3];
    tmp[0] = qV[0] * sh;
    tmp[1] = qV[1] * sh;
    tmp[2] = qV[2] * sh;
    sh *= qV[3];

    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qV[z] += sh;
    qV[3] -= tmp[z]; // w
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    _s11 = s22;
    _s21 = s32;
    _s22 = s33;
    _s31 = s21;
    _s32 = s31;
    _s33 = s11;
    s11 = _s11;
    s21 = _s21;
    s22 = _s22;
    s31 = _s31;
    s32 = _s32;
    s33 = _s33;
}

template <typename Type> CUDA_INLINE_CALLABLE Type dist2(Type x, Type y, Type z) { return x * x + y * y + z * z; }

// finds transformation that diagonalizes a symmetric matrix
template <typename Type>
CUDA_INLINE_CALLABLE void jacobiEigenanlysis( // symmetric matrix
    Type& s11,
    Type& s21, Type& s22,
    Type& s31, Type& s32, Type& s33,
    // quaternion representation of V
    Type* qV)
{
    qV[3] = 1;
    qV[0] = 0;
    qV[1] = 0;
    qV[2] = 0; // follow same indexing convention as GLM
    constexpr int ITERS = svd_config<Type>::JACOBI_ITERATIONS;
    for (int i = 0; i < ITERS; i++) {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        jacobiConjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,1
        jacobiConjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV); // p,q = 1,2
        jacobiConjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV); // p,q = 0,2
    }
}

// Sorts the columns of B = A V by descending norm, applying the same (determinant-preserving,
// swap-and-negate) permutation to V. After the QR step below the diagonal of R therefore comes
// out as (||b1||, ||b2||, +-||b3||): sorted by magnitude with the sign in the last entry.
template <typename Type>
CUDA_INLINE_CALLABLE void sortSingularValues( // matrix that we want to decompose
    Type& b11, Type& b12, Type& b13,
    Type& b21, Type& b22, Type& b23,
    Type& b31, Type& b32, Type& b33,
    // sort V simultaneously
    Type& v11, Type& v12, Type& v13,
    Type& v21, Type& v22, Type& v23,
    Type& v31, Type& v32, Type& v33)
{
    Type rho1 = dist2(b11, b21, b31);
    Type rho2 = dist2(b12, b22, b32);
    Type rho3 = dist2(b13, b23, b33);
    bool c;
    c = rho1 < rho2;
    condNegSwap(c, b11, b12);
    condNegSwap(c, v11, v12);
    condNegSwap(c, b21, b22);
    condNegSwap(c, v21, v22);
    condNegSwap(c, b31, b32);
    condNegSwap(c, v31, v32);
    condSwap(c, rho1, rho2);
    c = rho1 < rho3;
    condNegSwap(c, b11, b13);
    condNegSwap(c, v11, v13);
    condNegSwap(c, b21, b23);
    condNegSwap(c, v21, v23);
    condNegSwap(c, b31, b33);
    condNegSwap(c, v31, v33);
    condSwap(c, rho1, rho3);
    c = rho2 < rho3;
    condNegSwap(c, b12, b13);
    condNegSwap(c, v12, v13);
    condNegSwap(c, b22, b23);
    condNegSwap(c, v22, v23);
    condNegSwap(c, b32, b33);
    condNegSwap(c, v32, v33);
}

template <typename Type> CUDA_INLINE_CALLABLE void QRGivensQuaternion(Type a1, Type a2, Type& ch, Type& sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    const Type epsilon = svd_config<Type>::QR_GIVENS_EPSILON;
    Type rho = scalar<Type>::sqrt(a1 * a1 + a2 * a2);

    sh = rho > epsilon ? a2 : Type(0);
    ch = scalar<Type>::abs(a1) + scalar<Type>::max(rho, epsilon);
    bool b = a1 < Type(0);
    condSwap(b, sh, ch);
    Type w = recipSqrt(ch * ch + sh * sh);
    ch *= w;
    sh *= w;
}

template <typename Type>
CUDA_INLINE_CALLABLE void QRDecomposition( // matrix that we want to decompose
    Type b11, Type b12, Type b13,
    Type b21, Type b22, Type b23,
    Type b31, Type b32, Type b33,
    // output Q
    Type& q11, Type& q12, Type& q13,
    Type& q21, Type& q22, Type& q23,
    Type& q31, Type& q32, Type& q33,
    // output R
    Type& r11, Type& r12, Type& r13,
    Type& r21, Type& r22, Type& r23,
    Type& r31, Type& r32, Type& r33)
{
    Type ch1, sh1, ch2, sh2, ch3, sh3;
    Type a, b;

    // first givens rotation (ch,0,0,sh)
    QRGivensQuaternion(b11, b21, ch1, sh1);
    a = Type(1) - Type(2) * sh1 * sh1;
    b = Type(2) * ch1 * sh1;
    // apply B = Q' * B
    r11 = a * b11 + b * b21;
    r12 = a * b12 + b * b22;
    r13 = a * b13 + b * b23;
    r21 = -b * b11 + a * b21;
    r22 = -b * b12 + a * b22;
    r23 = -b * b13 + a * b23;
    r31 = b31;
    r32 = b32;
    r33 = b33;

    // second givens rotation (ch,0,-sh,0)
    QRGivensQuaternion(r11, r31, ch2, sh2);
    a = Type(1) - Type(2) * sh2 * sh2;
    b = Type(2) * ch2 * sh2;
    // apply B = Q' * B;
    b11 = a * r11 + b * r31;
    b12 = a * r12 + b * r32;
    b13 = a * r13 + b * r33;
    b21 = r21;
    b22 = r22;
    b23 = r23;
    b31 = -b * r11 + a * r31;
    b32 = -b * r12 + a * r32;
    b33 = -b * r13 + a * r33;

    // third givens rotation (ch,sh,0,0)
    QRGivensQuaternion(b22, b32, ch3, sh3);
    a = Type(1) - Type(2) * sh3 * sh3;
    b = Type(2) * ch3 * sh3;
    // R is now set to desired value
    r11 = b11;
    r12 = b12;
    r13 = b13;
    r21 = a * b21 + b * b31;
    r22 = a * b22 + b * b32;
    r23 = a * b23 + b * b33;
    r31 = -b * b21 + a * b31;
    r32 = -b * b22 + a * b32;
    r33 = -b * b23 + a * b33;

    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    // the number of floating point operations for three quaternion multiplications
    // is more or less comparable to the explicit form of the joined matrix.
    // certainly more memory-efficient!
    Type sh12 = sh1 * sh1;
    Type sh22 = sh2 * sh2;
    Type sh32 = sh3 * sh3;

    q11 = (Type(-1) + Type(2) * sh12) * (Type(-1) + Type(2) * sh22);
    q12 = Type(4) * ch2 * ch3 * (Type(-1) + Type(2) * sh12) * sh2 * sh3
        + Type(2) * ch1 * sh1 * (Type(-1) + Type(2) * sh32);
    q13 = Type(4) * ch1 * ch3 * sh1 * sh3
        - Type(2) * ch2 * (Type(-1) + Type(2) * sh12) * sh2 * (Type(-1) + Type(2) * sh32);

    q21 = Type(2) * ch1 * sh1 * (Type(1) - Type(2) * sh22);
    q22 = Type(-8) * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (Type(-1) + Type(2) * sh12) * (Type(-1) + Type(2) * sh32);
    q23 = Type(-2) * ch3 * sh3 + Type(4) * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (Type(-1) + Type(2) * sh32));

    q31 = Type(2) * ch2 * sh2;
    q32 = Type(2) * ch3 * (Type(1) - Type(2) * sh22) * sh3;
    q33 = (Type(-1) + Type(2) * sh22) * (Type(-1) + Type(2) * sh32);
}

// Warp's `_svd`: A = U S V^T with S returned as the full (nearly diagonal) R factor of the QR of
// A V; the off-diagonal entries of S are the residual of the Jacobi convergence and are dropped
// by the caller.
template <typename Type>
CUDA_INLINE_CALLABLE void svd_core( // input A
    Type a11, Type a12, Type a13,
    Type a21, Type a22, Type a23,
    Type a31, Type a32, Type a33,
    // output U
    Type& u11, Type& u12, Type& u13,
    Type& u21, Type& u22, Type& u23,
    Type& u31, Type& u32, Type& u33,
    // output S
    Type& s11, Type& s12, Type& s13,
    Type& s21, Type& s22, Type& s23,
    Type& s31, Type& s32, Type& s33,
    // output V
    Type& v11, Type& v12, Type& v13,
    Type& v21, Type& v22, Type& v23,
    Type& v31, Type& v32, Type& v33)
{
    // normal equations matrix
    Type ATA11, ATA12, ATA13;
    Type ATA21, ATA22, ATA23;
    Type ATA31, ATA32, ATA33;

    multAtB(
        a11, a12, a13, a21, a22, a23, a31, a32, a33, a11, a12, a13, a21, a22, a23, a31, a32, a33, ATA11, ATA12, ATA13,
        ATA21, ATA22, ATA23, ATA31, ATA32, ATA33);

    // symmetric eigenalysis
    Type qV[4];
    jacobiEigenanlysis(ATA11, ATA21, ATA22, ATA31, ATA32, ATA33, qV);
    // Added to Warp's routine: renormalise the accumulated quaternion. Every conjugation
    // multiplies its norm by sqrt(ch^2 + sh^2), which is 1 only up to rounding, and quatToMat3
    // assumes a unit quaternion; without this V drifts from orthogonality by about
    // (3 x sweeps) ulp, i.e. ~1e-5 in float on degenerate spectra where every step rotates by pi/4.
    {
        const Type qn = recipSqrt(qV[0] * qV[0] + qV[1] * qV[1] + qV[2] * qV[2] + qV[3] * qV[3]);
        qV[0] *= qn;
        qV[1] *= qn;
        qV[2] *= qn;
        qV[3] *= qn;
    }
    quatToMat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);

    Type b11, b12, b13;
    Type b21, b22, b23;
    Type b31, b32, b33;
    multAB(
        a11, a12, a13, a21, a22, a23, a31, a32, a33, v11, v12, v13, v21, v22, v23, v31, v32, v33, b11, b12, b13, b21,
        b22, b23, b31, b32, b33);

    // sort singular values and find V
    sortSingularValues(b11, b12, b13, b21, b22, b23, b31, b32, b33, v11, v12, v13, v21, v22, v23, v31, v32, v33);

    // QR decomposition
    QRDecomposition(
        b11, b12, b13, b21, b22, b23, b31, b32, b33, u11, u12, u13, u21, u22, u23, u31, u32, u33, s11, s12, s13, s21,
        s22, s23, s31, s32, s33);
}

// --- helpers of the canonicalisation pass (column-major 3x3 arrays) ---------------------------

template <typename Type> CUDA_INLINE_CALLABLE Type det3(const Type M[9])
{
    return M[0] * (M[4] * M[8] - M[7] * M[5]) - M[3] * (M[1] * M[8] - M[7] * M[2]) + M[6] * (M[1] * M[5] - M[4] * M[2]);
}

template <typename Type> CUDA_INLINE_CALLABLE void swap_columns(Type M[9], int i, int j)
{
    for (int r = 0; r < 3; ++r) {
        Type t = M[3 * i + r];
        M[3 * i + r] = M[3 * j + r];
        M[3 * j + r] = t;
    }
}

template <typename Type> CUDA_INLINE_CALLABLE void negate_column(Type M[9], int i)
{
    M[3 * i] = -M[3 * i];
    M[3 * i + 1] = -M[3 * i + 1];
    M[3 * i + 2] = -M[3 * i + 2];
}

// Swapping the same pair of columns in U and V leaves U diag(S) V^T unchanged.
template <typename Type> CUDA_INLINE_CALLABLE void swap_pair(Type U[9], Type S[3], Type V[9], int i, int j)
{
    Type t = S[i];
    S[i] = S[j];
    S[j] = t;
    swap_columns(U, i, j);
    swap_columns(V, i, j);
}

}  // namespace svd_detail

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------

/// Signed SVD of a column-major 3x3 matrix: A = U diag(S) V^T with det U = det V = +1 and
/// S[0] >= S[1] >= |S[2]|, sign(S[2]) = sign(det A). See the header comment for the convention.
template <typename T>
CUDA_INLINE_CALLABLE void svd3(const T A[9], T U[9], T S[3], T V[9])
{
    using M = svd_detail::scalar<T>;

    // 1. Exact power-of-two prescale: the Jacobi/QR epsilons are absolute, so the core is run on
    //    a matrix with max|a_ij| in [0.5, 1). Skipped for zero, non-finite and denormal input
    //    (the core handles the zero matrix; nothing sensible can be done for the others).
    T amax = T(0);
    for (int i = 0; i < 9; ++i) amax = M::max(amax, M::abs(A[i]));
    T sc = T(1), inv = T(1);
    if (amax > T(0) && M::finite(amax)) {
        int e = 0;
        (void)M::frexp(amax, &e);
        const T s_try = M::ldexp(T(1), e);
        const T i_try = M::ldexp(T(1), -e);
        if (s_try > T(0) && M::finite(i_try)) {
            sc = s_try;
            inv = i_try;
        }
    }
    const T a11 = A[0] * inv, a21 = A[1] * inv, a31 = A[2] * inv;
    const T a12 = A[3] * inv, a22 = A[4] * inv, a32 = A[5] * inv;
    const T a13 = A[6] * inv, a23 = A[7] * inv, a33 = A[8] * inv;

    // 2. Warp's McAdams SVD (row-major scalar interface, column-major arrays on our side).
    T s12, s13, s21, s23, s31, s32;
    svd_detail::svd_core(
        a11, a12, a13, a21, a22, a23, a31, a32, a33,
        U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8],
        S[0], s12, s13, s21, S[1], s23, s31, s32, S[2],
        V[0], V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

    // 3. Canonicalise.
    //    (a) magnitudes descending: Warp sorts the columns of A V by norm before the QR, so this
    //        only re-orders ties broken the other way by rounding (e.g. pure rotations);
    if (M::abs(S[1]) > M::abs(S[0])) svd_detail::swap_pair(U, S, V, 0, 1);
    if (M::abs(S[2]) > M::abs(S[1])) svd_detail::swap_pair(U, S, V, 1, 2);
    if (M::abs(S[1]) > M::abs(S[0])) svd_detail::swap_pair(U, S, V, 0, 1);
    //    (b) S[0], S[1] >= 0: a negative entry moves its sign into the matching column of U;
    if (S[0] < T(0)) { S[0] = -S[0]; svd_detail::negate_column(U, 0); }
    if (S[1] < T(0)) { S[1] = -S[1]; svd_detail::negate_column(U, 1); }
    //    (c) proper rotations: a reflection in U or V is moved into the sign of S[2].
    if (svd_detail::det3(U) < T(0)) { svd_detail::negate_column(U, 2); S[2] = -S[2]; }
    if (svd_detail::det3(V) < T(0)) { svd_detail::negate_column(V, 2); S[2] = -S[2]; }

    // 4. Undo the prescale (exact).
    S[0] *= sc;
    S[1] *= sc;
    S[2] *= sc;
}

/// Rotation-variant polar decomposition A = R S_sym, R = U V^T in SO(3), S_sym = V diag(S) V^T
/// (symmetric, indefinite when det A < 0). Column-major arrays.
template <typename T>
CUDA_INLINE_CALLABLE void polar3(const T A[9], T R[9], T S_sym[9])
{
    T U[9], S[3], V[9];
    svd3(A, U, S, V);
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            // R(r,c) = sum_k U(r,k) V(c,k)
            R[3 * c + r] = U[r] * V[c] + U[3 + r] * V[3 + c] + U[6 + r] * V[6 + c];
            // S_sym(r,c) = sum_k V(r,k) S[k] V(c,k)
            S_sym[3 * c + r] = V[r] * S[0] * V[c] + V[3 + r] * S[1] * V[3 + c] + V[6 + r] * S[2] * V[6 + c];
        }
    }
}

// --- double3x3 / float3x3 overloads (real3x3 is an alias of one of them) ---------------------

CUDA_INLINE_CALLABLE void svd3(const double3x3& A, double3x3& U, double3& S, double3x3& V)
{
    const double a[9] = {A.m00, A.m10, A.m20, A.m01, A.m11, A.m21, A.m02, A.m12, A.m22};
    double u[9], s[3], v[9];
    svd3(a, u, s, v);
    U = double3x3(u);
    V = double3x3(v);
    S = make_double3(s[0], s[1], s[2]);
}

CUDA_INLINE_CALLABLE void svd3(const float3x3& A, float3x3& U, float3& S, float3x3& V)
{
    const float a[9] = {A.m00, A.m10, A.m20, A.m01, A.m11, A.m21, A.m02, A.m12, A.m22};
    float u[9], s[3], v[9];
    svd3(a, u, s, v);
    U = float3x3(u);
    V = float3x3(v);
    S = make_float3(s[0], s[1], s[2]);
}

CUDA_INLINE_CALLABLE void polar3(const double3x3& A, double3x3& R, double3x3& S_sym)
{
    const double a[9] = {A.m00, A.m10, A.m20, A.m01, A.m11, A.m21, A.m02, A.m12, A.m22};
    double r[9], s[9];
    polar3(a, r, s);
    R = double3x3(r);
    S_sym = double3x3(s);
}

CUDA_INLINE_CALLABLE void polar3(const float3x3& A, float3x3& R, float3x3& S_sym)
{
    const float a[9] = {A.m00, A.m10, A.m20, A.m01, A.m11, A.m21, A.m02, A.m12, A.m22};
    float r[9], s[9];
    polar3(a, r, s);
    R = float3x3(r);
    S_sym = float3x3(s);
}

}  // namespace cs

#endif  // CS_SVD3_CUH
