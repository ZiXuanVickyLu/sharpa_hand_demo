#pragma once
#ifndef TYPE_DEF_CUH
#define TYPE_DEF_CUH

#include "typedef.h"
#include "float3x3.h"
#include "double3x3.h"
#include "vector_type_t.h"
// Only include CUDA device code when compiling with nvcc
#ifdef __CUDACC__
#endif
#include <cuda/api.hpp>
#include <pch.h>

namespace cs {

    #ifdef CS_USE_DOUBLE
        using real = double;
        using real2 = double2;
        using real3 = double3;
        #ifdef CCCL_VERSION_GREATER_EQUAL_13_0
            using real4 = double4_16a;
        #else
            using real4 = double4;
        #endif
        using real3x3 = double3x3;
    #else
        using real = float;
        using real2 = float2;
        using real3 = float3;
        using real4 = float4;
        using real3x3 = float3x3;
    #endif
   

    inline __host__ __device__ real2 make_real2(real x, real y) {
    #ifdef CS_USE_DOUBLE
        return make_double2(x, y);
    #else
        return make_float2(x, y);
    #endif
    }

    inline __host__ __device__ real2 make_real2(real v) {
    #ifdef CS_USE_DOUBLE
        return make_double2(v, v);
    #else
        return make_float2(v, v);
    #endif
    }
    
    inline __host__ __device__ real2 make_real2(const int2 &v) {
    #ifdef CS_USE_DOUBLE
        return make_double2(static_cast<double>(v.x), static_cast<double>(v.y));
    #else
        return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
    #endif
    }
    
    inline __host__ __device__ real2 make_real2(const uint2 &v) {
    #ifdef CS_USE_DOUBLE
        return make_double2(static_cast<double>(v.x), static_cast<double>(v.y));
    #else
        return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
    #endif
    }
    
    inline __host__ __device__ real3 make_real3(real x, real y, real z) {
    #ifdef CS_USE_DOUBLE
        return make_double3(x, y, z);
    #else
        return make_float3(x, y, z);
    #endif
    }
    
    inline __host__ __device__ real3 make_real3(real v) {
    #ifdef CS_USE_DOUBLE
        return make_double3(v, v, v);
    #else
        return make_float3(v, v, v);
    #endif
    }
    
    inline __host__ __device__ real3 make_real3(const real4 &v) {
    #ifdef CS_USE_DOUBLE
        return make_double3(v.x, v.y, v.z);
    #else
        return make_float3(v.x, v.y, v.z);
    #endif
    }

    // CCCL version > 13.0, the double4 is replaced by double4_16a.
    #ifdef CCCL_VERSION_GREATER_EQUAL_13_0
    inline __host__ __device__ real4 make_real4(real x, real y, real z, real w) {
    #ifdef CS_USE_DOUBLE
        return make_double4_16a(x, y, z, w);
    #else
        return make_float4(x, y, z, w);
    #endif
    }
    
    inline __host__ __device__ real4 make_real4(real v) {
    #ifdef CS_USE_DOUBLE
        return make_double4_16a(v, v, v, v);
    #else
        return make_float4(v, v, v, v);
    #endif
    }
    
    inline __host__ __device__ real4 make_real4(const real3 &v) {
    #ifdef CS_USE_DOUBLE
        return make_double4_16a(v.x, v.y, v.z, 0.0);
    #else
        return make_float4(v.x, v.y, v.z, 0.0f);
    #endif
    }
    
    inline __host__ __device__ real4 make_real4(const real3 &v, real w) {
    #ifdef CS_USE_DOUBLE
        return make_double4_16a(v.x, v.y, v.z, w);
    #else
        return make_float4(v.x, v.y, v.z, w);
    #endif
    }
    // CCCL version < 13.0, we go back to double4.
    #else
    inline __host__ __device__ real4 make_real4(real x, real y, real z, real w) {
    #ifdef CS_USE_DOUBLE
        return make_double4(x, y, z, w);
    #else
        return make_float4(x, y, z, w);
    #endif
    }
    inline __host__ __device__ real4 make_real4(real v) {
    #ifdef CS_USE_DOUBLE
        return make_double4(v, v, v, v);
    #else
        return make_float4(v, v, v, v);
    #endif
    }
    inline __host__ __device__ real4 make_real4(const real3 &v) {
    #ifdef CS_USE_DOUBLE
        return make_double4(v.x, v.y, v.z, 0.0);
    #else
        return make_float4(v.x, v.y, v.z, 0.0f);
    #endif
    }
    inline __host__ __device__ real4 make_real4(const real3 &v, real w) {
    #ifdef CS_USE_DOUBLE
        return make_double4(v.x, v.y, v.z, w);
    #else
        return make_float4(v.x, v.y, v.z, w);
    #endif
    }
    #endif


    #ifdef __CUDACC__
    #ifdef CS_USE_DOUBLE
    __device__ __forceinline__ double atomicAdd_double(double* address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        
        return __longlong_as_double(old);
    }
    

    __device__ __forceinline__ real atomicAdd_real(real* address, real val) {
        return atomicAdd_double(address, val);
    }
    #else
    // For float, use native atomicAdd
    __device__ __forceinline__ real atomicAdd_real(real* address, real val) {
        return atomicAdd(address, val);
    }
    #endif

    // Atomic min for NON-NEGATIVE reals only: for values >= 0 the IEEE bit
    // pattern is monotone as a signed integer, so an integer atomicMin works.
    #ifdef CS_USE_DOUBLE
    __device__ __forceinline__ void atomicMin_real_nonneg(real* address, real val) {
        atomicMin(reinterpret_cast<long long*>(address), __double_as_longlong(val));
    }
    #else
    __device__ __forceinline__ void atomicMin_real_nonneg(real* address, real val) {
        atomicMin(reinterpret_cast<int*>(address), __float_as_int(val));
    }
    #endif

    // Scatter a 3x3 contact Hessian block onto a per-vertex accumulator.
    __device__ __forceinline__ void atomicAdd_real3x3(real3x3* address, const real3x3& val) {
        atomicAdd_real(&address->c0.x, val.c0.x);
        atomicAdd_real(&address->c0.y, val.c0.y);
        atomicAdd_real(&address->c0.z, val.c0.z);
        atomicAdd_real(&address->c1.x, val.c1.x);
        atomicAdd_real(&address->c1.y, val.c1.y);
        atomicAdd_real(&address->c1.z, val.c1.z);
        atomicAdd_real(&address->c2.x, val.c2.x);
        atomicAdd_real(&address->c2.y, val.c2.y);
        atomicAdd_real(&address->c2.z, val.c2.z);
    }
    #endif // __CUDACC__

    /**
     * @brief Anisotropic contact stiffness block for one contact pair.
     *
     * Resists motion along the contact normal with @p k_normal and motion in the
     * tangent plane with @p k_tangent:  k_n*n*n^T + k_t*(I - n*n^T).
     * @p n must be unit length. Passing k_tangent == k_normal reproduces the
     * isotropic block k*I.
     */
    CUDA_INLINE_CALLABLE real3x3 contact_stiffness_block(const real3& n, real k_normal, real k_tangent) {
        real3x3 block = out_dot(n, n) * (k_normal - k_tangent);
        block.m00 += k_tangent;
        block.m11 += k_tangent;
        block.m22 += k_tangent;
        return block;
    }

    /**
     * @brief Solve M*x = b for a symmetric positive-definite 3x3 M.
     *
     * Returns zero for a singular M, mirroring the scalar path's
     * `inv_diag = denom != 0 ? 1/denom : 0` fallback for massless/unconstrained rows.
     */
    CUDA_INLINE_CALLABLE real3 solve_spd3(const real3x3& M, const real3& b) {
        if (det(M) == static_cast<real>(0)) {
            return make_real3(0, 0, 0);
        }
        return inverse(M) * b;
    }

    template<typename T>
    using DBuffer = std::optional<cuda::unique_span<T>>;
    
    template<typename T>
    using HBuffer = std::optional<std::vector<T>>;

}

#endif // TYPE_DEF_H
