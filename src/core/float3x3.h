//
// Created by birdpeople on 8/9/2023.
//

#ifndef FLOAT3X3_H
#define FLOAT3X3_H
#include <typedef.h>
#include "vector_type_t.h"
#include "double3x3.h"

    struct float3x3 {
        union {
            struct { float m00, m10, m20; };
            float3 c0;
        };
        union {
            struct { float m01, m11, m21; };
            float3 c1;
        };
        union {
            struct { float m02, m12, m22; };
            float3 c2;
        };

        CUDA_CALLABLE  float3x3() :
                c0(make_float3(1, 0, 0)),
                c1(make_float3(0, 1, 0)),
                c2(make_float3(0, 0, 1)) { }

        CUDA_CALLABLE float3x3(const float array[9]) :
                m00(array[0]), m10(array[1]), m20(array[2]),
                m01(array[3]), m11(array[4]), m21(array[5]),
                m02(array[6]), m12(array[7]), m22(array[8]) { }

        CUDA_CALLABLE  float3x3(const float3 &col0, const float3 &col1, const float3 &col2) :
                c0(col0), c1(col1), c2(col2)
        { }

        CUDA_CALLABLE float3x3 operator+() const { return *this; }
        CUDA_CALLABLE float3x3 operator-() const { return {-c0, -c1, -c2}; }

        CUDA_CALLABLE float3x3 &operator+=(const float3x3 &mat) {
            c0 += mat.c0;
            c1 += mat.c1;
            c2 += mat.c2;
            return *this;
        }
        CUDA_CALLABLE float3x3 &operator-=(const float3x3 &mat) {
            c0 -= mat.c0;
            c1 -= mat.c1;
            c2 -= mat.c2;
            return *this;
        }
        CUDA_CALLABLE float3x3 &operator*=(float s) {
            c0 *= s;
            c1 *= s;
            c2 *= s;
            return *this;
        }
        CUDA_CALLABLE float3x3 &operator*=(const float3x3 &mat) {
            const float3 r[] = { row(0), row(1), row(2) };
            c0 = make_float3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
            c1 = make_float3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
            c2 = make_float3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
            return *this;
        }

        CUDA_CALLABLE float3 operator*(const float3 &v) const {
            const float3 r[] = { row(0), row(1), row(2) };
            return make_float3(dot(r[0], v),
                               dot(r[1], v),
                               dot(r[2], v));
        }

        CUDA_INLINE_CALLABLE float3 row(unsigned int r) const {

            switch (r) {
                case 0:
                    return make_float3(m00, m01, m02);
                case 1:
                    return make_float3(m10, m11, m12);
                case 2:
                    return make_float3(m20, m21, m22);
                default:
                    return make_float3(0, 0, 0);
            }
        }

        CUDA_INLINE_CALLABLE float3x3 &inverse() {
            float det = 1.0f / (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21 -
                                m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
            float3x3 m;
            m.m00 = det * (m11 * m22 - m12 * m21); m.m01 = -det * (m01 * m22 - m02 * m21); m.m02 = det * (m01 * m12 - m02 * m11);
            m.m10 = -det * (m10 * m22 - m12 * m20); m.m11 = det * (m00 * m22 - m02 * m20); m.m12 = -det * (m00 * m12 - m02 * m10);
            m.m20 = det * (m10 * m21 - m11 * m20); m.m21 = -det * (m00 * m21 - m01 * m20); m.m22 = det * (m00 * m11 - m01 * m10);
            *this = m;

            return *this;
        }

        CUDA_INLINE_CALLABLE float3x3 &transpose() {
            float temp;
            temp = m10; m10 = m01; m01 = temp;
            temp = m20; m20 = m02; m02 = temp;
            temp = m21; m21 = m12; m12 = temp;
            return *this;
        }
        static CUDA_INLINE_CALLABLE float3x3 identity3x3() {
            return {make_float3(1, 0, 0),
                    make_float3(0, 1, 0),
                    make_float3(0, 0, 1)};
        }

        static CUDA_INLINE_CALLABLE float3x3 zeros3x3() {
            return {make_float3(0, 0, 0),
                    make_float3(0, 0, 0),
                    make_float3(0, 0, 0)};
        }

        static CUDA_INLINE_CALLABLE float3x3 ones3x3() {
            return {make_float3(1, 1, 1),
                    make_float3(1, 1, 1),
                    make_float3(1, 1, 1)};
        }
    };

//return a * b^T
    CUDA_INLINE_CALLABLE float3x3 out_dot(const float3 &a, const float3 &b) {

        return {a * b.x, a * b.y, a * b.z};
    }

    CUDA_INLINE_CALLABLE float3x3 operator+(const float3x3 &a, const float3x3 &b){
        float3x3 ret = a;
        ret += b;
        return ret;
    }

    CUDA_INLINE_CALLABLE float3x3 operator-(const float3x3 &a, const float3x3 &b) {
        float3x3 ret = a;
        ret -= b;
        return ret;
    }

    CUDA_INLINE_CALLABLE float3x3 operator*(const float3x3 &a, float b) {
        float3x3 ret = a;
        ret *= b;
        return ret;
    }

    CUDA_INLINE_CALLABLE float3x3 operator*(float a, const float3x3 &b) {
        float3x3 ret = b;
        ret *= a;
        return ret;
    }

    CUDA_INLINE_CALLABLE float3x3 operator*(const float3x3 &a, const float3x3 &b) {
        float3x3 ret = a;
        ret *= b;
        return ret;
    }
    CUDA_INLINE_CALLABLE float det(const float3x3 &mat){
        float res;
        res = ( mat.m00 * mat.m11 * mat.m22 +
                mat.m01 * mat.m12 * mat.m20 +
                mat.m02 * mat.m10 * mat.m21 -
                mat.m02 * mat.m11 * mat.m20 -
                mat.m01 * mat.m10 * mat.m22 -
                mat.m00 * mat.m12 * mat.m21);
        return res;
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 transpose(const float3x3 &mat) {
        float3x3 ret = mat;
        return ret.transpose();
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 inverse(const float3x3 &mat) {
        float3x3 ret = mat;
        return ret.inverse();
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 scale3x3(const float3 &s) {
        return {s.x * make_float3(1, 0, 0),
                s.y * make_float3(0, 1, 0),
                s.z * make_float3(0, 0, 1)};
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 scale3x3(float sx, float sy, float sz) {
        return scale3x3(make_float3(sx, sy, sz));
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 scale3x3(float s) {
        return scale3x3(make_float3(s, s, s));
    }

//this will not change the input matrix
    CUDA_INLINE_CALLABLE float3x3 rotate3x3(float angle, const float3 &axis) {
        float3x3 matrix;
        float3 nAxis = normalize(axis);
        float s = std::sin(angle);
        float c = std::cos(angle);
        float oneMinusC = 1 - c;

        matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
        matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
        matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
        matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
        matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
        matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
        matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
        matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
        matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

        return matrix;
    }

    CUDA_INLINE_CALLABLE float3x3 rotate3x3(float angle, float ax, float ay, float az) {
        return rotate3x3(angle, make_float3(ax, ay, az));
    }
    CUDA_INLINE_CALLABLE float3x3 rotateX3x3(float angle) { return rotate3x3(angle, make_float3(1, 0, 0)); }
    CUDA_INLINE_CALLABLE float3x3 rotateY3x3(float angle) { return rotate3x3(angle, make_float3(0, 1, 0)); }
    CUDA_INLINE_CALLABLE float3x3 rotateZ3x3(float angle) { return rotate3x3(angle, make_float3(0, 0, 1)); }

    CUDA_INLINE_CALLABLE float3x3 from_double3x3 (const double3x3 &mat){
        return         {make_float3(static_cast<float>(mat.c0.x), static_cast<float>(mat.c0.y), static_cast<float>(mat.c0.z)),
                        make_float3(static_cast<float>(mat.c1.x), static_cast<float>(mat.c1.y), static_cast<float>(mat.c1.z)),
                        make_float3(static_cast<float>(mat.c2.x), static_cast<float>(mat.c2.y), static_cast<float>(mat.c2.z))};
    }
    CUDA_INLINE_CALLABLE double3x3 to_double3x3(const float3x3 &mat){
        return          {make_double3(static_cast<double>(mat.c0.x), static_cast<double>(mat.c0.y), static_cast<double>(mat.c0.z)),
                         make_double3(static_cast<double>(mat.c1.x), static_cast<double>(mat.c1.y), static_cast<double>(mat.c1.z)),
                         make_double3(static_cast<double>(mat.c2.x), static_cast<double>(mat.c2.y), static_cast<double>(mat.c2.z))};
    }
    CUDA_INLINE_CALLABLE double3x3 from_float3x3 (const float3x3 &mat){
        return          {make_double3(static_cast<double>(mat.c0.x), static_cast<double>(mat.c0.y), static_cast<double>(mat.c0.z)),
                         make_double3(static_cast<double>(mat.c1.x), static_cast<double>(mat.c1.y), static_cast<double>(mat.c1.z)),
                         make_double3(static_cast<double>(mat.c2.x), static_cast<double>(mat.c2.y), static_cast<double>(mat.c2.z))};
    }
    CUDA_INLINE_CALLABLE float3x3 to_float3x3(const double3x3 & mat){
        return         {make_float3(static_cast<float>(mat.c0.x), static_cast<float>(mat.c0.y), static_cast<float>(mat.c0.z)),
                        make_float3(static_cast<float>(mat.c1.x), static_cast<float>(mat.c1.y), static_cast<float>(mat.c1.z)),
                        make_float3(static_cast<float>(mat.c2.x), static_cast<float>(mat.c2.y), static_cast<float>(mat.c2.z))};
    }

#endif //FLOAT3X3_H
