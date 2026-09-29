//
// Created by birdpeople on 8/9/2023.
//

#ifndef VECTOR_TYPE_T_H
#define VECTOR_TYPE_T_H

#include "typedef.h"
#include <cuda_runtime.h>
#ifdef CCCL_VERSION_GREATER_EQUAL_13_0
#include <cccl/thrust/extrema.h>
#else
#include <thrust/extrema.h>
#endif // CCCL_VERSION_GREATER_EQUAL_13_0
#include <vector_types.h>
#include <vector_functions.hpp>
#include <vector_functions.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

    template <typename T>
    CUDA_INLINE_CALLABLE constexpr T min(const T &a, const T &b) {
        return b < a ? b : a;
    }

    template <typename T>
    CUDA_INLINE_CALLABLE constexpr T max(const T &a, const T &b) {
        return b > a ? b : a;
    }

    template <typename T>
    CUDA_INLINE_CALLABLE constexpr T clamp(const T &v, const T &minv, const T &maxv) {
        return min(max(v, minv), maxv);
    }

    template<typename T>
    CUDA_INLINE_CALLABLE void swap(T &a, T &b) {
        T tmp = a;
        a = b;
        b = tmp;
    }

    template<typename T>
    CUDA_INLINE_CALLABLE T pow2(const T &x) {
        return x * x;
    }

    template<typename T>
    CUDA_INLINE_CALLABLE T pow3(const T &x) {
        return x * x * x;
    }

    using unit = uint32_t;

    CUDA_INLINE_CALLABLE float3 getXYZ(const float4 &v) {
        return make_float3(v.x, v.y, v.z);
    }

    CUDA_INLINE_CALLABLE double3 getXYZ(const double4 &v) {
        return make_double3(v.x, v.y, v.z);
    }

    CUDA_INLINE_CALLABLE int2 make_int2(const float2 &v) {
        return make_int2(static_cast<int32_t>(v.x), static_cast<int32_t>(v.y));
    }
    CUDA_INLINE_CALLABLE int2 make_int2(const int3 &v) {
        return make_int2(v.x, v.y);
    }
    CUDA_INLINE_CALLABLE int2 make_int2(const uint3 &v) {
        return make_int2(static_cast<int32_t>(v.x), static_cast<int32_t>(v.y));
    }
    CUDA_INLINE_CALLABLE bool operator==(const int2 &v0, const int2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const int2 &v0, const int2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator==(const int2 &v0, const uint2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const int2 &v0, const uint2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE uint2 operator+(const int2 &v0, const uint2 &v1) {
        return make_uint2(v0.x + v1.x, v0.y + v1.y);
    }
    CUDA_INLINE_CALLABLE int2 operator+(const int2 &v0, const int2 &v1) {
        return make_int2(v0.x + v1.x, v0.y + v1.y);
    }
    CUDA_INLINE_CALLABLE int2 operator*(const int2 &v0, const int2 &v1) {
        return make_int2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE int2 operator*(uint32_t s, const int2 &v) {
        return make_int2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE int2 operator*(const int2 &v, uint32_t s) {
        return make_int2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE int2 &operator*=(int2 &v0, const int2 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        return v0;
    }
    CUDA_INLINE_CALLABLE int2 &operator*=(int2 &v, uint32_t s) {
        v.x *= s;
        v.y *= s;
        return v;
    }
    CUDA_INLINE_CALLABLE int2 operator/(const int2 &v0, const int2 &v1) {
        return make_int2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE int2 operator/(const int2 &v, uint32_t s) {
        return make_int2(v.x / s, v.y / s);
    }
    CUDA_INLINE_CALLABLE uint2 operator/(const int2 &v0, const uint2 &v1) {
        return make_uint2(v0.x / v1.x, v0.y / v1.y);
    }

    CUDA_INLINE_CALLABLE uint2 make_uint2(const float2 &v) {
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    CUDA_INLINE_CALLABLE uint2 make_uint2(const int3 &v) {
        return make_uint2(static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y));
    }
    CUDA_INLINE_CALLABLE uint2 make_uint2(const uint3 &v) {
        return make_uint2(v.x, v.y);
    }
    CUDA_INLINE_CALLABLE bool operator==(const uint2 &v0, const uint2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const uint2 &v0, const uint2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator==(const uint2 &v0, const int2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const uint2 &v0, const int2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE uint2 operator+(const uint2 &v0, const uint2 &v1) {
        return make_uint2(v0.x + v1.x, v0.y + v1.y);
    }
    CUDA_INLINE_CALLABLE uint2 &operator+=(uint2 &v, uint32_t s) {
        v.x += s;
        v.y += s;
        return v;
    }
    CUDA_INLINE_CALLABLE uint2 operator-(const uint2 &v, uint32_t s) {
        return make_uint2(v.x - s, v.y - s);
    }
    CUDA_INLINE_CALLABLE uint2 &operator-=(uint2 &v, uint32_t s) {
        v.x -= s;
        v.y -= s;
        return v;
    }
    CUDA_INLINE_CALLABLE uint2 operator*(const uint2 &v0, const uint2 &v1) {
        return make_uint2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE uint2 operator*(float s, const uint2 &v) {
        return make_uint2(static_cast<uint32_t>(s * v.x), static_cast<uint32_t>(s * v.y));
    }
    CUDA_INLINE_CALLABLE uint2 operator*(const uint2 &v, float s) {
        return make_uint2(static_cast<uint32_t>(s * v.x), static_cast<uint32_t>(s * v.y));
    }
    CUDA_INLINE_CALLABLE uint2 &operator*=(uint2 &v0, const uint2 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        return v0;
    }
    CUDA_INLINE_CALLABLE uint2 &operator*=(uint2 &v, uint32_t s) {
        v.x *= s;
        v.y *= s;
        return v;
    }
    CUDA_INLINE_CALLABLE uint2 operator/(const uint2 &v0, const uint2 &v1) {
        return make_uint2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE uint2 operator/(const uint2 &v0, const int2 &v1) {
        return make_uint2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE uint2 operator/(const uint2 &v, uint32_t s) {
        return make_uint2(v.x / s, v.y / s);
    }
    CUDA_INLINE_CALLABLE uint2 &operator/=(uint2 &v, uint32_t s) {
        v.x /= s;
        v.y /= s;
        return v;
    }
    CUDA_INLINE_CALLABLE uint2 operator%(const uint2 &v0, const uint2 &v1) {
        return make_uint2(v0.x % v1.x, v0.y % v1.y);
    }
    CUDA_INLINE_CALLABLE uint2 operator<<(const uint2 &v, uint32_t s) {
        return make_uint2(v.x << s, v.y << s);
    }
    CUDA_INLINE_CALLABLE uint2 &operator<<=(uint2 &v, uint32_t s) {
        v = v << s;
        return v;
    }
    CUDA_INLINE_CALLABLE uint2 operator>>(const uint2 &v, uint32_t s) {
        return make_uint2(v.x >> s, v.y >> s);
    }
    CUDA_INLINE_CALLABLE uint2 &operator>>=(uint2 &v, uint32_t s) {
        v = v >> s;
        return v;
    }

    CUDA_INLINE_CALLABLE float2 make_float2(float v) {
        return make_float2(v, v);
    }
    CUDA_INLINE_CALLABLE float2 make_float2(const int2 &v) {
        return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
    }
    CUDA_INLINE_CALLABLE float2 make_float2(const uint2 &v) {
        return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
    }
    CUDA_INLINE_CALLABLE bool operator==(const float2 &v0, const float2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const float2 &v0, const float2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE float2 operator-(const float2 &v) {
        return make_float2(-v.x, -v.y);
    }
    CUDA_INLINE_CALLABLE float2 operator+(const float2 &v0, const float2 &v1) {
        return make_float2(v0.x + v1.x, v0.y + v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator-(const float2 &v0, const float2 &v1) {
        return make_float2(v0.x - v1.x, v0.y - v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator*(const float2 &v0, const float2 &v1) {
        return make_float2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator*(float s, const float2 &v) {
        return make_float2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE float2 operator*(const float2 &v, float s) {
        return make_float2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE float2 &operator*=(float2 &v, float s) {
        v = v * s;
        return v;
    }
    CUDA_INLINE_CALLABLE float2 operator*(const int2 &v0, const float2 &v1) {
        return make_float2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator*(const float2 &v0, const int2 &v1) {
        return make_float2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator/(const float2 &v0, const float2 &v1) {
        return make_float2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator/(const float2 &v0, const int2 &v1) {
        return make_float2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE float2 operator/(const float2 &v, float s) {
        float r = 1 / s;
        return r * v;
    }
    CUDA_INLINE_CALLABLE float2 &operator/=(float2 &v, float s) {
        v = v / s;
        return v;
    }

    CUDA_INLINE_CALLABLE float3 make_float3(float v) {
        return make_float3(v, v, v);
    }
    CUDA_INLINE_CALLABLE float3 make_float3(const float4 &v) {
        return make_float3(v.x, v.y, v.z);
    }
    CUDA_INLINE_CALLABLE bool operator==(const float3 &v0, const float3 &v1) {
        return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const float3 &v0, const float3 &v1) {
        return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z;
    }
    CUDA_INLINE_CALLABLE float3 operator-(const float3 &v) {
        return make_float3(-v.x, -v.y, -v.z);
    }
    CUDA_INLINE_CALLABLE float3 operator+(const float3 &v0, const float3 &v1) {
        return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
    }
    CUDA_INLINE_CALLABLE float3 &operator+=(float3 &v0, const float3 &v1) {
        v0.x += v1.x;
        v0.y += v1.y;
        v0.z += v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE float3 operator-(const float3 &v0, const float3 &v1) {
        return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
    }
    CUDA_INLINE_CALLABLE float3 &operator-=(float3 &v0, const float3 &v1) {
        v0.x -= v1.x;
        v0.y -= v1.y;
        v0.z -= v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE float3 operator*(const float3 &v0, const float3 &v1) {
        return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
    }
    CUDA_INLINE_CALLABLE float3 operator*(float s, const float3 &v) {
        return make_float3(s * v.x, s * v.y, s * v.z);
    }
    CUDA_INLINE_CALLABLE float3 operator*(const float3 &v, float s) {
        return make_float3(s * v.x, s * v.y, s * v.z);
    }
    CUDA_INLINE_CALLABLE float3 &operator*=(float3 &v0, const float3 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        v0.z *= v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE float3 &operator*=(float3 &v, float s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        return v;
    }
    CUDA_INLINE_CALLABLE float3 operator/(const float3 &v0, const float3 &v1) {
        return make_float3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
    }
    CUDA_INLINE_CALLABLE float3 operator/(const float3 &v, float s) {
        float r = 1 / s;
        return r * v;
    }
    CUDA_INLINE_CALLABLE float3 safeDivide(const float3 &v0, const float3 &v1) {
        return make_float3(
                v1.x != 0.0f ? v0.x / v1.x : 0.0f,
                v1.y != 0.0f ? v0.y / v1.y : 0.0f,
                v1.z != 0.0f ? v0.z / v1.z : 0.0f);
    }
    CUDA_INLINE_CALLABLE float3 safeDivide(const float3 &v, float d) {
        return d != 0.0f ? (v / d) : make_float3(0.0f);
    }
    CUDA_INLINE_CALLABLE float3 &operator/=(float3 &v, float s) {
        float r = 1 / s;
        return v *= r;
    }

    CUDA_INLINE_CALLABLE float4 make_float4(float v) {
        return make_float4(v, v, v, v);
    }
    CUDA_INLINE_CALLABLE float4 make_float4(const float3 &v) {
        return make_float4(v.x, v.y, v.z, 0.0f);
    }
    CUDA_INLINE_CALLABLE float4 make_float4(const float3 &v, float w) {
        return make_float4(v.x, v.y, v.z, w);
    }
    CUDA_INLINE_CALLABLE bool operator==(const float4 &v0, const float4 &v1) {
        return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const float4 &v0, const float4 &v1) {
        return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z || v0.w != v1.w;
    }
    CUDA_INLINE_CALLABLE float4 operator-(const float4 &v) {
        return make_float4(-v.x, -v.y, -v.z, -v.w);
    }
    CUDA_INLINE_CALLABLE float4 operator+(const float4 &v0, const float4 &v1) {
        return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
    }
    CUDA_INLINE_CALLABLE float4 &operator+=(float4 &v0, const float4 &v1) {
        v0.x += v1.x;
        v0.y += v1.y;
        v0.z += v1.z;
        v0.w += v1.w;
        return v0;
    }
    CUDA_INLINE_CALLABLE float4 operator-(const float4 &v0, const float4 &v1) {
        return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
    }
    CUDA_INLINE_CALLABLE float4 &operator-=(float4 &v0, const float4 &v1) {
        v0.x -= v1.x;
        v0.y -= v1.y;
        v0.z -= v1.z;
        v0.w -= v1.w;
        return v0;
    }
    CUDA_INLINE_CALLABLE float4 operator*(const float4 &v0, const float4 &v1) {
        return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
    }
    CUDA_INLINE_CALLABLE float4 operator*(float s, const float4 &v) {
        return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
    }
    CUDA_INLINE_CALLABLE float4 operator*(const float4 &v, float s) {
        return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
    }
    CUDA_INLINE_CALLABLE float4 &operator*=(float4 &v0, const float4 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        v0.z *= v1.z;
        v0.w *= v1.w;
        return v0;
    }
    CUDA_INLINE_CALLABLE float4 &operator*=(float4 &v, float s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        return v;
    }
    CUDA_INLINE_CALLABLE float4 operator/(const float4 &v0, const float4 &v1) {
        return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
    }
    CUDA_INLINE_CALLABLE float4 operator/(const float4 &v, float s) {
        float r = 1 / s;
        return r * v;
    }
    CUDA_INLINE_CALLABLE float4 &operator/=(float4 &v, float s) {
        float r = 1 / s;
        return v *= r;
    }

    CUDA_INLINE_CALLABLE double2 make_double2(double v) {
        return make_double2(v, v);
    }
    CUDA_INLINE_CALLABLE double2 make_double2(const int2 &v) {
        return make_double2(static_cast<double>(v.x), static_cast<double>(v.y));
    }
    CUDA_INLINE_CALLABLE double2 make_double2(const uint2 &v) {
        return make_double2(static_cast<double>(v.x), static_cast<double>(v.y));
    }
    CUDA_INLINE_CALLABLE bool operator==(const double2 &v0, const double2 &v1) {
        return v0.x == v1.x && v0.y == v1.y;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const double2 &v0, const double2 &v1) {
        return v0.x != v1.x || v0.y != v1.y;
    }
    CUDA_INLINE_CALLABLE double2 operator-(const double2 &v) {
        return make_double2(-v.x, -v.y);
    }
    CUDA_INLINE_CALLABLE double2 operator+(const double2 &v0, const double2 &v1) {
        return make_double2(v0.x + v1.x, v0.y + v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator-(const double2 &v0, const double2 &v1) {
        return make_double2(v0.x - v1.x, v0.y - v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator*(const double2 &v0, const double2 &v1) {
        return make_double2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator*(double s, const double2 &v) {
        return make_double2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE double2 operator*(const double2 &v, double s) {
        return make_double2(s * v.x, s * v.y);
    }
    CUDA_INLINE_CALLABLE double2 &operator*=(double2 &v, double s) {
        v = v * s;
        return v;
    }
    CUDA_INLINE_CALLABLE double2 operator*(const int2 &v0, const double2 &v1) {
        return make_double2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator*(const double2 &v0, const int2 &v1) {
        return make_double2(v0.x * v1.x, v0.y * v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator/(const double2 &v0, const double2 &v1) {
        return make_double2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator/(const double2 &v0, const int2 &v1) {
        return make_double2(v0.x / v1.x, v0.y / v1.y);
    }
    CUDA_INLINE_CALLABLE double2 operator/(const double2 &v, double s) {
        double r = 1 / s;
        return r * v;
    }
    CUDA_INLINE_CALLABLE double2 &operator/=(double2 &v, double s) {
        v = v / s;
        return v;
    }

    CUDA_INLINE_CALLABLE double3 make_double3(double v) {
        return make_double3(v, v, v);
    }
    CUDA_INLINE_CALLABLE double3 make_double3(const double4 &v) {
        return make_double3(v.x, v.y, v.z);
    }
    CUDA_INLINE_CALLABLE bool operator==(const double3 &v0, const double3 &v1) {
        return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const double3 &v0, const double3 &v1) {
        return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z;
    }
    CUDA_INLINE_CALLABLE double3 operator-(const double3 &v) {
        return make_double3(-v.x, -v.y, -v.z);
    }
    CUDA_INLINE_CALLABLE double3 operator+(const double3 &v0, const double3 &v1) {
        return make_double3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
    }
    CUDA_INLINE_CALLABLE double3 &operator+=(double3 &v0, const double3 &v1) {
        v0.x += v1.x;
        v0.y += v1.y;
        v0.z += v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE double3 operator-(const double3 &v0, const double3 &v1) {
        return make_double3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
    }
    CUDA_INLINE_CALLABLE double3 &operator-=(double3 &v0, const double3 &v1) {
        v0.x -= v1.x;
        v0.y -= v1.y;
        v0.z -= v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE double3 operator*(const double3 &v0, const double3 &v1) {
        return make_double3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
    }
    CUDA_INLINE_CALLABLE double3 operator*(double s, const double3 &v) {
        return make_double3(s * v.x, s * v.y, s * v.z);
    }
    CUDA_INLINE_CALLABLE double3 operator*(const double3 &v, double s) {
        return make_double3(s * v.x, s * v.y, s * v.z);
    }
    CUDA_INLINE_CALLABLE double3 &operator*=(double3 &v0, const double3 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        v0.z *= v1.z;
        return v0;
    }
    CUDA_INLINE_CALLABLE double3 &operator*=(double3 &v, double s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        return v;
    }
    CUDA_INLINE_CALLABLE double3 operator/(const double3 &v0, const double3 &v1) {
        return make_double3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
    }
    CUDA_INLINE_CALLABLE double3 operator/(const double3 &v, double s) {
        double r = 1 / s;
        return r * v;
    }
    CUDA_INLINE_CALLABLE double3 safeDivide(const double3 &v0, const double3 &v1) {
        return make_double3(
                v1.x != 0.0f ? v0.x / v1.x : 0.0f,
                v1.y != 0.0f ? v0.y / v1.y : 0.0f,
                v1.z != 0.0f ? v0.z / v1.z : 0.0f);
    }
    CUDA_INLINE_CALLABLE double3 safeDivide(const double3 &v, double d) {
        return d != 0.0f ? (v / d) : make_double3(0.0f);
    }
    CUDA_INLINE_CALLABLE double3 &operator/=(double3 &v, double s) {
        double r = 1 / s;
        return v *= r;
    }


#ifndef CCCL_VERSION_GREATER_EQUAL_13_0
    CUDA_INLINE_CALLABLE double4 make_double4(double v) {
        return make_double4(v, v, v, v);
    }
    CUDA_INLINE_CALLABLE double4 make_double4(const double3 &v) {
        return make_double4(v.x, v.y, v.z, 0.0f);
    }
    CUDA_INLINE_CALLABLE double4 make_double4(const double3 &v, double w) {
        return make_double4(v.x, v.y, v.z, w);
    }
    CUDA_INLINE_CALLABLE bool operator==(const double4 &v0, const double4 &v1) {
        return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const double4 &v0, const double4 &v1) {
        return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z || v0.w != v1.w;
    }

    CUDA_INLINE_CALLABLE double4 operator-(const double4 &v) {
        return make_double4(-v.x, -v.y, -v.z, -v.w);
    }

    CUDA_INLINE_CALLABLE double4 operator+(const double4 &v0, const double4 &v1) {
        return make_double4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
    }

    CUDA_INLINE_CALLABLE double4 &operator+=(double4 &v0, const double4 &v1) {
        v0.x += v1.x;
        v0.y += v1.y;
        v0.z += v1.z;
        v0.w += v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4 operator-(const double4 &v0, const double4 &v1) {
        return make_double4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
    }

    CUDA_INLINE_CALLABLE double4 &operator-=(double4 &v0, const double4 &v1) {
        v0.x -= v1.x;
        v0.y -= v1.y;
        v0.z -= v1.z;
        v0.w -= v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4 operator*(const double4 &v0, const double4 &v1) {
        return make_double4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
    }

    CUDA_INLINE_CALLABLE double4 operator*(double s, const double4 &v) {
        return make_double4(s * v.x, s * v.y, s * v.z, s * v.w);
    }

    CUDA_INLINE_CALLABLE double4 operator*(const double4 &v, double s) {
        return make_double4(s * v.x, s * v.y, s * v.z, s * v.w);
    }

    CUDA_INLINE_CALLABLE double4 &operator*=(double4 &v0, const double4 &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        v0.z *= v1.z;
        v0.w *= v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4 &operator*=(double4 &v, double s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        return v;
    }

    CUDA_INLINE_CALLABLE double4 operator/(const double4 &v0, const double4 &v1) {
        return make_double4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
    }

    CUDA_INLINE_CALLABLE double4 operator/(const double4 &v, double s) {
        double r = 1 / s;
        return r * v;
    }

    CUDA_INLINE_CALLABLE double4 &operator/=(double4 &v, double s) {
        double r = 1 / s;
        return v *= r;
    }

    CUDA_INLINE_CALLABLE double4 min(const double4 &v0, const double4 &v1) {
        return make_double4(fmin(v0.x, v1.x),
                            fmin(v0.y, v1.y),
                            fmin(v0.z, v1.z),
                            fmin(v0.w, v1.w));
    }

    CUDA_INLINE_CALLABLE double4 max(const double4 &v0, const double4 &v1) {
        return make_double4(fmax(v0.x, v1.x),
                            fmax(v0.y, v1.y),
                            fmax(v0.z, v1.z),
                            fmax(v0.w, v1.w));
    }

    CUDA_INLINE_CALLABLE double dot(const double4 &v0, const double4 &v1) {
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
    }
#else // cuda version >13.0

    CUDA_INLINE_CALLABLE double4_16a make_double4(double v) {
        return make_double4_16a(v, v, v, v);
    }
    CUDA_INLINE_CALLABLE double4_16a make_double4(const double3 &v) {
        return make_double4_16a(v.x, v.y, v.z, 0.0f);
    }
    CUDA_INLINE_CALLABLE double4_16a make_double4(const double3 &v, double w) {
        return make_double4_16a(v.x, v.y, v.z, w);
    }
    CUDA_INLINE_CALLABLE bool operator==(const double4_16a &v0, const double4_16a &v1) {
        return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
    }
    CUDA_INLINE_CALLABLE bool operator!=(const double4_16a &v0, const double4_16a &v1) {
        return v0.x != v1.x || v0.y != v1.y || v0.z != v1.z || v0.w != v1.w;
    }

    CUDA_INLINE_CALLABLE double4_16a operator-(const double4_16a &v) {
        return make_double4_16a(-v.x, -v.y, -v.z, -v.w);
    }

    CUDA_INLINE_CALLABLE double4_16a operator+(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
    }

    CUDA_INLINE_CALLABLE double4_16a &operator+=(double4_16a &v0, const double4_16a &v1) {
        v0.x += v1.x;
        v0.y += v1.y;
        v0.z += v1.z;
        v0.w += v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4_16a operator-(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
    }

    CUDA_INLINE_CALLABLE double4_16a &operator-=(double4_16a &v0, const double4_16a &v1) {
        v0.x -= v1.x;
        v0.y -= v1.y;
        v0.z -= v1.z;
        v0.w -= v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4_16a operator*(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
    }

    CUDA_INLINE_CALLABLE double4_16a operator*(double s, const double4_16a &v) {
        return make_double4_16a(s * v.x, s * v.y, s * v.z, s * v.w);
    }

    CUDA_INLINE_CALLABLE double4_16a operator*(const double4_16a &v, double s) {
        return make_double4_16a(s * v.x, s * v.y, s * v.z, s * v.w);
    }

    CUDA_INLINE_CALLABLE double4_16a &operator*=(double4_16a &v0, const double4_16a &v1) {
        v0.x *= v1.x;
        v0.y *= v1.y;
        v0.z *= v1.z;
        v0.w *= v1.w;
        return v0;
    }

    CUDA_INLINE_CALLABLE double4_16a &operator*=(double4_16a &v, double s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
        v.w *= s;
        return v;
    }

    CUDA_INLINE_CALLABLE double4_16a operator/(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
    }

    CUDA_INLINE_CALLABLE double4_16a operator/(const double4_16a &v, double s) {
        double r = 1 / s;
        return r * v;
    }

    CUDA_INLINE_CALLABLE double4_16a &operator/=(double4_16a &v, double s) {
        double r = 1 / s;
        return v *= r;
    }

    CUDA_INLINE_CALLABLE double4_16a min(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(fmin(v0.x, v1.x),
                            fmin(v0.y, v1.y),
                            fmin(v0.z, v1.z),
                            fmin(v0.w, v1.w));
    }

    CUDA_INLINE_CALLABLE double4_16a max(const double4_16a &v0, const double4_16a &v1) {
        return make_double4_16a(fmax(v0.x, v1.x),
                            fmax(v0.y, v1.y),
                            fmax(v0.z, v1.z),
                            fmax(v0.w, v1.w));
    }

    CUDA_INLINE_CALLABLE double dot(const double4_16a &v0, const double4_16a &v1) {
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
    }

#endif 
    // Linear interpolation operator for points (float3/double3)
    CUDA_INLINE_CALLABLE float3 lerp(const float3 &p0, const float3 &p1, float t) {
        return make_float3(
            p0.x + t * (p1.x - p0.x),
            p0.y + t * (p1.y - p0.y),
            p0.z + t * (p1.z - p0.z)
        );
    }

    CUDA_INLINE_CALLABLE double3 lerp(const double3 &p0, const double3 &p1, double t) {
        return make_double3(
            p0.x + t * (p1.x - p0.x),
            p0.y + t * (p1.y - p0.y),
            p0.z + t * (p1.z - p0.z)
        );
    }

    CUDA_INLINE_CALLABLE int2 min(const int2 &v0, const int2 &v1) {
        return make_int2(min(v0.x, v1.x),
                         min(v0.y, v1.y));
    }

    CUDA_INLINE_CALLABLE int2 max(const int2 &v0, const int2 &v1) {
        return make_int2(max(v0.x, v1.x),
                         max(v0.y, v1.y));
    }

    CUDA_INLINE_CALLABLE uint2 min(const uint2 &v0, const uint2 &v1) {
        return make_uint2(min(v0.x, v1.x),
                          min(v0.y, v1.y));
    }
    CUDA_INLINE_CALLABLE uint2 max(const uint2 &v0, const uint2 &v1) {
        return make_uint2(max(v0.x, v1.x),
                          max(v0.y, v1.y));
    }

    CUDA_INLINE_CALLABLE float2 min(const float2 &v0, const float2 &v1) {
        return make_float2(min(v0.x, v1.x),
                           min(v0.y, v1.y));
    }
    CUDA_INLINE_CALLABLE float2 max(const float2 &v0, const float2 &v1) {
        return make_float2(max(v0.x, v1.x),
                           max(v0.y, v1.y));
    }
    CUDA_INLINE_CALLABLE float cross(const float2 &v0, const float2 &v1) {
        return v0.x * v1.y - v0.y * v1.x;
    }

    CUDA_INLINE_CALLABLE float3 min(const float3 &v0, const float3 &v1) {
        return make_float3(fmin(v0.x, v1.x),
                           fmin(v0.y, v1.y),
                           fmin(v0.z, v1.z));
    }
    CUDA_INLINE_CALLABLE float3 max(const float3 &v0, const float3 &v1) {
        return make_float3(fmax(v0.x, v1.x),
                           fmax(v0.y, v1.y),
                           fmax(v0.z, v1.z));
    }
    CUDA_INLINE_CALLABLE float dot(const float3 &v0, const float3 &v1) {
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    }
    CUDA_INLINE_CALLABLE float3 cross(const float3 &v0, const float3 &v1) {
        return make_float3(v0.y * v1.z - v0.z * v1.y,
                           v0.z * v1.x - v0.x * v1.z,
                           v0.x * v1.y - v0.y * v1.x);
    }
    CUDA_INLINE_CALLABLE float squaredDistance(const float3 &p0, const float3 &p1) {
        float3 d = p1 - p0;
        return dot(d, d);
    }
    CUDA_INLINE_CALLABLE float length(const float3 &v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    CUDA_INLINE_CALLABLE float squaredLength(const float3 &v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
    CUDA_INLINE_CALLABLE float3 normalize(const float3 &v) {
        return v / length(v);
    }

    CUDA_INLINE_CALLABLE float4 min(const float4 &v0, const float4 &v1) {
        return make_float4(fmin(v0.x, v1.x),
                           fmin(v0.y, v1.y),
                           fmin(v0.z, v1.z),
                           fmin(v0.w, v1.w));
    }
    CUDA_INLINE_CALLABLE float4 max(const float4 &v0, const float4 &v1) {
        return make_float4(fmax(v0.x, v1.x),
                           fmax(v0.y, v1.y),
                           fmax(v0.z, v1.z),
                           fmax(v0.w, v1.w));
    }
    CUDA_INLINE_CALLABLE float dot(const float4 &v0, const float4 &v1) {
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
    }

        CUDA_INLINE_CALLABLE int32_t floatToOrderedInt(float fVal) {
    #if defined(__CUDA_ARCH__)
            int32_t iVal = __float_as_int(fVal);
    #else
            int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
    #endif
            return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
        }

        CUDA_INLINE_CALLABLE float orderedIntToFloat(int32_t iVal) {
            int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
    #if defined(__CUDA_ARCH__)
            return __int_as_float(orgVal);
    #else
            return *reinterpret_cast<float*>(&orgVal);
    #endif
        }



    struct float3AsOrderedInt {
        int32_t x, y, z;

        CUDA_CALLABLE float3AsOrderedInt() : x(0), y(0), z(0) {
        }
        CUDA_CALLABLE float3AsOrderedInt(const float3 &v) :
                x(floatToOrderedInt(v.x)), y(floatToOrderedInt(v.y)), z(floatToOrderedInt(v.z)) {
        }

        CUDA_CALLABLE explicit operator float3() const {
            return make_float3(orderedIntToFloat(x), orderedIntToFloat(y), orderedIntToFloat(z));
        }
    };

    CUDA_INLINE_CALLABLE double2 min(const double2 &v0, const double2 &v1) {
        return make_double2(min(v0.x, v1.x),
                            min(v0.y, v1.y));
    }

    CUDA_INLINE_CALLABLE double2 max(const double2 &v0, const double2 &v1) {
        return make_double2(max(v0.x, v1.x),
                            max(v0.y, v1.y));
    }

    CUDA_INLINE_CALLABLE double cross(const double2 &v0, const double2 &v1) {
        return v0.x * v1.y - v0.y * v1.x;
    }

    CUDA_INLINE_CALLABLE double3 min(const double3 &v0, const double3 &v1) {
        return make_double3(fmin(v0.x, v1.x),
                            fmin(v0.y, v1.y),
                            fmin(v0.z, v1.z));
    }

    CUDA_INLINE_CALLABLE double3 max(const double3 &v0, const double3 &v1) {
        return make_double3(fmax(v0.x, v1.x),
                            fmax(v0.y, v1.y),
                            fmax(v0.z, v1.z));
    }

    CUDA_INLINE_CALLABLE double dot(const double3 &v0, const double3 &v1) {
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    }

    CUDA_INLINE_CALLABLE double3 cross(const double3 &v0, const double3 &v1) {
        return make_double3(v0.y * v1.z - v0.z * v1.y,
                            v0.z * v1.x - v0.x * v1.z,
                            v0.x * v1.y - v0.y * v1.x);
    }

    CUDA_INLINE_CALLABLE double squaredDistance(const double3 &p0, const double3 &p1) {
        double3 d = p1 - p0;
        return dot(d, d);
    }

    CUDA_INLINE_CALLABLE double length(const double3 &v) {
        // was sqrtf (float rounding of a double length): contact_solver precision policy 11.1
        return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    CUDA_INLINE_CALLABLE double squaredLength(const double3 &v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    CUDA_INLINE_CALLABLE double3 normalize(const double3 &v) {
        return v / length(v);
    }

   

    CUDA_INLINE_CALLABLE int64_t doubleToOrderedLongLong(double fVal) {
    #if defined(__CUDA_ARCH__)
        int64_t iVal = __double_as_longlong(fVal);
    #else
        int64_t iVal = *reinterpret_cast<int64_t*>(&fVal);
    #endif
        return (iVal >= 0) ? iVal : iVal ^ 0xFFFFFFFF;
    }

    CUDA_INLINE_CALLABLE double orderedLongLongToDouble(int64_t iVal) {
        int64_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0xFFFFFFFF;
    #if defined(__CUDA_ARCH__)
        return __longlong_as_double(orgVal);
    #else
        return *reinterpret_cast<double*>(&orgVal);
    #endif
    }

    struct double3AsOrderedLongLong {
        int64_t x, y, z;

        CUDA_CALLABLE double3AsOrderedLongLong() : x(0), y(0), z(0) {
        }
        CUDA_CALLABLE double3AsOrderedLongLong(const double3 &v) :
                x(doubleToOrderedLongLong(v.x)), y(doubleToOrderedLongLong(v.y)), z(doubleToOrderedLongLong(v.z)) {
        }

        CUDA_CALLABLE explicit operator double3() const {
            return make_double3(orderedLongLongToDouble(x), orderedLongLongToDouble(y), orderedLongLongToDouble(z));
        }
    };

#endif
