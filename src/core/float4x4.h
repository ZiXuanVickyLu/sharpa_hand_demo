//
// Created by birdpeople on 12/12/2025.
//

#ifndef FLOAT4X4_H
#define FLOAT4X4_H
#include <typedef.h>
#include "vector_type_t.h"
#include "double4x4.h"

struct __align__(16) float4x4 {
    union {
        struct { float m00, m10, m20, m30; };
        float4 c0;
    };
    union {
        struct { float m01, m11, m21, m31; };
        float4 c1;
    };
    union {
        struct { float m02, m12, m22, m32; };
        float4 c2;
    };
    union {
        struct { float m03, m13, m23, m33; };
        float4 c3;
    };

    CUDA_CALLABLE float4x4() :
            c0(make_float4(1, 0, 0, 0)),
            c1(make_float4(0, 1, 0, 0)),
            c2(make_float4(0, 0, 1, 0)),
            c3(make_float4(0, 0, 0, 1)) { }

    CUDA_CALLABLE float4x4(const float array[16]) :
            m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
            m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
            m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
            m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }

    CUDA_CALLABLE float4x4(const float4 &col0, const float4 &col1, const float4 &col2, const float4 &col3) :
            c0(col0), c1(col1), c2(col2), c3(col3)
    { }

    CUDA_CALLABLE float4x4 operator+() const { return *this; }
    CUDA_CALLABLE float4x4 operator-() const { return {-c0, -c1, -c2, -c3}; }

    CUDA_CALLABLE float4x4 &operator+=(const float4x4 &mat) {
        c0 += mat.c0;
        c1 += mat.c1;
        c2 += mat.c2;
        c3 += mat.c3;
        return *this;
    }
    CUDA_CALLABLE float4x4 &operator-=(const float4x4 &mat) {
        c0 -= mat.c0;
        c1 -= mat.c1;
        c2 -= mat.c2;
        c3 -= mat.c3;
        return *this;
    }
    CUDA_CALLABLE float4x4 &operator*=(float s) {
        c0 *= s;
        c1 *= s;
        c2 *= s;
        c3 *= s;
        return *this;
    }
    CUDA_CALLABLE float4x4 &operator*=(const float4x4 &mat) {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        c0 = make_float4(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = make_float4(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = make_float4(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = make_float4(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }

    CUDA_CALLABLE float4 operator*(const float4 &v) const {
        const float4 r[] = { row(0), row(1), row(2), row(3) };
        return make_float4(dot(r[0], v),
                           dot(r[1], v),
                           dot(r[2], v),
                           dot(r[3], v));
    }

    CUDA_INLINE_CALLABLE float4 row(unsigned int r) const {
        switch (r) {
            case 0:
                return make_float4(m00, m01, m02, m03);
            case 1:
                return make_float4(m10, m11, m12, m13);
            case 2:
                return make_float4(m20, m21, m22, m23);
            case 3:
                return make_float4(m30, m31, m32, m33);
            default:
                return make_float4(0, 0, 0, 0);
        }
    }

    CUDA_INLINE_CALLABLE float4x4 &transpose() {
        float temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    static CUDA_INLINE_CALLABLE float4x4 identity4x4() {
        return {make_float4(1, 0, 0, 0),
                make_float4(0, 1, 0, 0),
                make_float4(0, 0, 1, 0),
                make_float4(0, 0, 0, 1)};
    }

    static CUDA_INLINE_CALLABLE float4x4 zeros4x4() {
        return {make_float4(0, 0, 0, 0),
                make_float4(0, 0, 0, 0),
                make_float4(0, 0, 0, 0),
                make_float4(0, 0, 0, 0)};
    }

    static CUDA_INLINE_CALLABLE float4x4 ones4x4() {
        return {make_float4(1, 1, 1, 1),
                make_float4(1, 1, 1, 1),
                make_float4(1, 1, 1, 1),
                make_float4(1, 1, 1, 1)};
    }
};

//return a * b^T
CUDA_INLINE_CALLABLE float4x4 out_dot(const float4 &a, const float4 &b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

CUDA_INLINE_CALLABLE float4x4 operator+(const float4x4 &a, const float4x4 &b){
    float4x4 ret = a;
    ret += b;
    return ret;
}

CUDA_INLINE_CALLABLE float4x4 operator-(const float4x4 &a, const float4x4 &b) {
    float4x4 ret = a;
    ret -= b;
    return ret;
}

CUDA_INLINE_CALLABLE float4x4 operator*(const float4x4 &a, float b) {
    float4x4 ret = a;
    ret *= b;
    return ret;
}

CUDA_INLINE_CALLABLE float4x4 operator*(float a, const float4x4 &b) {
    float4x4 ret = b;
    ret *= a;
    return ret;
}

CUDA_INLINE_CALLABLE float4x4 operator*(const float4x4 &a, const float4x4 &b) {
    float4x4 ret = a;
    ret *= b;
    return ret;
}

//this will not change the input matrix
CUDA_INLINE_CALLABLE float4x4 transpose(const float4x4 &mat) {
    float4x4 ret = mat;
    return ret.transpose();
}

//this will not change the input matrix
CUDA_INLINE_CALLABLE float4x4 scale4x4(const float4 &s) {
    return {s.x * make_float4(1, 0, 0, 0),
            s.y * make_float4(0, 1, 0, 0),
            s.z * make_float4(0, 0, 1, 0),
            s.w * make_float4(0, 0, 0, 1)};
}

//this will not change the input matrix
CUDA_INLINE_CALLABLE float4x4 scale4x4(float sx, float sy, float sz, float sw) {
    return scale4x4(make_float4(sx, sy, sz, sw));
}

//this will not change the input matrix
CUDA_INLINE_CALLABLE float4x4 scale4x4(float s) {
    return scale4x4(make_float4(s, s, s, s));
}

CUDA_INLINE_CALLABLE float4x4 from_double4x4 (const double4x4 &mat){
    return         {make_float4(static_cast<float>(mat.c0.x), static_cast<float>(mat.c0.y), static_cast<float>(mat.c0.z), static_cast<float>(mat.c0.w)),
                    make_float4(static_cast<float>(mat.c1.x), static_cast<float>(mat.c1.y), static_cast<float>(mat.c1.z), static_cast<float>(mat.c1.w)),
                    make_float4(static_cast<float>(mat.c2.x), static_cast<float>(mat.c2.y), static_cast<float>(mat.c2.z), static_cast<float>(mat.c2.w)),
                    make_float4(static_cast<float>(mat.c3.x), static_cast<float>(mat.c3.y), static_cast<float>(mat.c3.z), static_cast<float>(mat.c3.w))};
}

CUDA_INLINE_CALLABLE double4x4 to_double4x4(const float4x4 &mat){
    return          {make_double4(static_cast<double>(mat.c0.x), static_cast<double>(mat.c0.y), static_cast<double>(mat.c0.z), static_cast<double>(mat.c0.w)),
                     make_double4(static_cast<double>(mat.c1.x), static_cast<double>(mat.c1.y), static_cast<double>(mat.c1.z), static_cast<double>(mat.c1.w)),
                     make_double4(static_cast<double>(mat.c2.x), static_cast<double>(mat.c2.y), static_cast<double>(mat.c2.z), static_cast<double>(mat.c2.w)),
                     make_double4(static_cast<double>(mat.c3.x), static_cast<double>(mat.c3.y), static_cast<double>(mat.c3.z), static_cast<double>(mat.c3.w))};
}

CUDA_INLINE_CALLABLE double4x4 from_float4x4 (const float4x4 &mat){
    return          {make_double4(static_cast<double>(mat.c0.x), static_cast<double>(mat.c0.y), static_cast<double>(mat.c0.z), static_cast<double>(mat.c0.w)),
                     make_double4(static_cast<double>(mat.c1.x), static_cast<double>(mat.c1.y), static_cast<double>(mat.c1.z), static_cast<double>(mat.c1.w)),
                     make_double4(static_cast<double>(mat.c2.x), static_cast<double>(mat.c2.y), static_cast<double>(mat.c2.z), static_cast<double>(mat.c2.w)),
                     make_double4(static_cast<double>(mat.c3.x), static_cast<double>(mat.c3.y), static_cast<double>(mat.c3.z), static_cast<double>(mat.c3.w))};
}

CUDA_INLINE_CALLABLE float4x4 to_float4x4(const double4x4 & mat){
    return         {make_float4(static_cast<float>(mat.c0.x), static_cast<float>(mat.c0.y), static_cast<float>(mat.c0.z), static_cast<float>(mat.c0.w)),
                    make_float4(static_cast<float>(mat.c1.x), static_cast<float>(mat.c1.y), static_cast<float>(mat.c1.z), static_cast<float>(mat.c1.w)),
                    make_float4(static_cast<float>(mat.c2.x), static_cast<float>(mat.c2.y), static_cast<float>(mat.c2.z), static_cast<float>(mat.c2.w)),
                    make_float4(static_cast<float>(mat.c3.x), static_cast<float>(mat.c3.y), static_cast<float>(mat.c3.z), static_cast<float>(mat.c3.w))};
}

#endif //FLOAT4X4_H

