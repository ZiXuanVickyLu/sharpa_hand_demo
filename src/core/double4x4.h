//
// Created by birdpeople on 12/12/2025.
//

#ifndef DOUBLE4X4_H
#define DOUBLE4X4_H
#include <typedef.h>
#include "vector_type_t.h"

// Define a type alias to handle compatibility between double4 and double4_16a
#ifdef CCCL_VERSION_GREATER_EQUAL_13_0
    // CUDA >= 13.0 uses double4_16a for better alignment
    using double4_t = double4_16a;
    #define make_double4_t make_double4_16a
#else
    // CUDA < 13.0 uses standard double4
    using double4_t = double4;
    #define make_double4_t make_double4
#endif

struct __align__(16) double4x4 {
    union {
        struct { double m00, m10, m20, m30; };
        double4_t c0;
    };
    union {
        struct { double m01, m11, m21, m31; };
        double4_t c1;
    };
    union {
        struct { double m02, m12, m22, m32; };
        double4_t c2;
    };
    union {
        struct { double m03, m13, m23, m33; };
        double4_t c3;
    };

    CUDA_CALLABLE double4x4() :
            c0(make_double4_t(1, 0, 0, 0)),
            c1(make_double4_t(0, 1, 0, 0)),
            c2(make_double4_t(0, 0, 1, 0)),
            c3(make_double4_t(0, 0, 0, 1)) { }

    CUDA_CALLABLE double4x4(const double array[16]) :
            m00(array[0]), m10(array[1]), m20(array[2]), m30(array[3]),
            m01(array[4]), m11(array[5]), m21(array[6]), m31(array[7]),
            m02(array[8]), m12(array[9]), m22(array[10]), m32(array[11]),
            m03(array[12]), m13(array[13]), m23(array[14]), m33(array[15]) { }

    CUDA_CALLABLE double4x4(const double4_t &col0, const double4_t &col1, const double4_t &col2, const double4_t &col3) :
            c0(col0), c1(col1), c2(col2), c3(col3)
    { }

    CUDA_CALLABLE double4x4 operator+() const { return *this; }
    CUDA_CALLABLE double4x4 operator-() const { return {-c0, -c1, -c2, -c3}; }

    CUDA_CALLABLE double4x4 &operator+=(const double4x4 &mat) {
        c0 += mat.c0;
        c1 += mat.c1;
        c2 += mat.c2;
        c3 += mat.c3;
        return *this;
    }
    CUDA_CALLABLE double4x4 &operator-=(const double4x4 &mat) {
        c0 -= mat.c0;
        c1 -= mat.c1;
        c2 -= mat.c2;
        c3 -= mat.c3;
        return *this;
    }
    CUDA_CALLABLE double4x4 &operator*=(double s) {
        c0 *= s;
        c1 *= s;
        c2 *= s;
        c3 *= s;
        return *this;
    }
    CUDA_CALLABLE double4x4 &operator*=(const double4x4 &mat) {
        const double4_t r[] = { row(0), row(1), row(2), row(3) };
        c0 = make_double4_t(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0), dot(r[3], mat.c0));
        c1 = make_double4_t(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1), dot(r[3], mat.c1));
        c2 = make_double4_t(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2), dot(r[3], mat.c2));
        c3 = make_double4_t(dot(r[0], mat.c3), dot(r[1], mat.c3), dot(r[2], mat.c3), dot(r[3], mat.c3));
        return *this;
    }

    CUDA_CALLABLE double4_t operator*(const double4_t &v) const {
        const double4_t r[] = { row(0), row(1), row(2), row(3) };
        return make_double4_t(dot(r[0], v),
                              dot(r[1], v),
                              dot(r[2], v),
                              dot(r[3], v));
    }

    CUDA_INLINE_CALLABLE double4_t row(unsigned int r) const {
        switch (r) {
            case 0:
                return make_double4_t(m00, m01, m02, m03);
            case 1:
                return make_double4_t(m10, m11, m12, m13);
            case 2:
                return make_double4_t(m20, m21, m22, m23);
            case 3:
                return make_double4_t(m30, m31, m32, m33);
            default:
                return make_double4_t(0, 0, 0, 0);
        }
    }

    CUDA_INLINE_CALLABLE double4x4 &transpose() {
        double temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m30; m30 = m03; m03 = temp;
        temp = m21; m21 = m12; m12 = temp;
        temp = m31; m31 = m13; m13 = temp;
        temp = m32; m32 = m23; m23 = temp;
        return *this;
    }

    static CUDA_INLINE_CALLABLE double4x4 identity4x4() {
        return         {make_double4_t(1, 0, 0, 0),
                        make_double4_t(0, 1, 0, 0),
                        make_double4_t(0, 0, 1, 0),
                        make_double4_t(0, 0, 0, 1)};
    }

    static CUDA_INLINE_CALLABLE double4x4 zeros4x4() {
        return          {make_double4_t(0, 0, 0, 0),
                         make_double4_t(0, 0, 0, 0),
                         make_double4_t(0, 0, 0, 0),
                         make_double4_t(0, 0, 0, 0)};
    }

    static CUDA_INLINE_CALLABLE double4x4 ones4x4() {
        return          {make_double4_t(1, 1, 1, 1),
                         make_double4_t(1, 1, 1, 1),
                         make_double4_t(1, 1, 1, 1),
                         make_double4_t(1, 1, 1, 1)};
    }
};

//return a * b^T
CUDA_INLINE_CALLABLE double4x4 out_dot(const double4_t &a, const double4_t &b) {
    return double4x4{a * b.x, a * b.y, a * b.z, a * b.w};
}

CUDA_INLINE_CALLABLE double4x4 operator+(const double4x4 &a, const double4x4 &b){
    double4x4 ret = a;
    ret += b;
    return ret;
}

CUDA_INLINE_CALLABLE double4x4 operator-(const double4x4 &a, const double4x4 &b) {
    double4x4 ret = a;
    ret -= b;
    return ret;
}

CUDA_INLINE_CALLABLE double4x4 operator*(const double4x4 &a, double b) {
    double4x4 ret = a;
    ret *= b;
    return ret;
}

CUDA_INLINE_CALLABLE double4x4 operator*(double a, const double4x4 &b) {
    double4x4 ret = b;
    ret *= a;
    return ret;
}

CUDA_INLINE_CALLABLE double4x4 operator*(const double4x4 &a, const double4x4 &b) {
    double4x4 ret = a;
    ret *= b;
    return ret;
}

CUDA_INLINE_CALLABLE double4x4 transpose(const double4x4 &mat) {
    double4x4 ret = mat;
    return ret.transpose();
}

CUDA_INLINE_CALLABLE double4x4 scale4x4(const double4_t &s) {
    return         {s.x * make_double4_t(1, 0, 0, 0),
                    s.y * make_double4_t(0, 1, 0, 0),
                    s.z * make_double4_t(0, 0, 1, 0),
                    s.w * make_double4_t(0, 0, 0, 1)};
}

CUDA_INLINE_CALLABLE double4x4 scale4x4(double sx, double sy, double sz, double sw) {
    return scale4x4(make_double4_t(sx, sy, sz, sw));
}

CUDA_INLINE_CALLABLE double4x4 scale4x4(double s) {
    return scale4x4(make_double4_t(s, s, s, s));
}

#endif //DOUBLE4X4_H

