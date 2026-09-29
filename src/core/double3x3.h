//
// Created by birdpeople on 9/12/2023.
//

#ifndef DOUBLE3X3_H
#define DOUBLE3X3_H
#include <typedef.h>
#include "vector_type_t.h"
struct double3x3 {
    union {
        struct { double m00, m10, m20; };
        double3 c0;
    };
    union {
        struct { double m01, m11, m21; };
        double3 c1;
    };
    union {
        struct { double m02, m12, m22; };
        double3 c2;
    };

    CUDA_CALLABLE  double3x3() :
            c0(make_double3(1, 0, 0)),
            c1(make_double3(0, 1, 0)),
            c2(make_double3(0, 0, 1)) { }

    CUDA_CALLABLE double3x3(const double array[9]) :
            m00(array[0]), m10(array[1]), m20(array[2]),
            m01(array[3]), m11(array[4]), m21(array[5]),
            m02(array[6]), m12(array[7]), m22(array[8]) { }

    CUDA_CALLABLE  double3x3(const double3 &col0, const double3 &col1, const double3 &col2) :
            c0(col0), c1(col1), c2(col2)
    { }

    CUDA_CALLABLE double3x3 operator+() const { return *this; }
    CUDA_CALLABLE double3x3 operator-() const { return {-c0, -c1, -c2}; }

    CUDA_CALLABLE double3x3 &operator+=(const double3x3 &mat) {
        c0 += mat.c0;
        c1 += mat.c1;
        c2 += mat.c2;
        return *this;
    }
    CUDA_CALLABLE double3x3 &operator-=(const double3x3 &mat) {
        c0 -= mat.c0;
        c1 -= mat.c1;
        c2 -= mat.c2;
        return *this;
    }
    CUDA_CALLABLE double3x3 &operator*=(double s) {
        c0 *= s;
        c1 *= s;
        c2 *= s;
        return *this;
    }
    CUDA_CALLABLE double3x3 &operator*=(const double3x3 &mat) {
        const double3 r[] = { row(0), row(1), row(2) };
        c0 = make_double3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
        c1 = make_double3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
        c2 = make_double3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
        return *this;
    }

    CUDA_CALLABLE double3 operator*(const double3 &v) const {
        const double3 r[] = { row(0), row(1), row(2) };
        return make_double3(dot(r[0], v),
                           dot(r[1], v),
                           dot(r[2], v));
    }

    CUDA_INLINE_CALLABLE double3 row(unsigned int r) const {

        switch (r) {
            case 0:
                return make_double3(m00, m01, m02);
            case 1:
                return make_double3(m10, m11, m12);
            case 2:
                return make_double3(m20, m21, m22);
            default:
                return make_double3(0, 0, 0);
        }
    }

    CUDA_INLINE_CALLABLE double3x3 &inverse() {
        double det = 1.0f / (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21 -
                            m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
        double3x3 m;
        m.m00 = det * (m11 * m22 - m12 * m21); m.m01 = -det * (m01 * m22 - m02 * m21); m.m02 = det * (m01 * m12 - m02 * m11);
        m.m10 = -det * (m10 * m22 - m12 * m20); m.m11 = det * (m00 * m22 - m02 * m20); m.m12 = -det * (m00 * m12 - m02 * m10);
        m.m20 = det * (m10 * m21 - m11 * m20); m.m21 = -det * (m00 * m21 - m01 * m20); m.m22 = det * (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }

    CUDA_INLINE_CALLABLE double3x3 &transpose() {
        double temp;
        temp = m10; m10 = m01; m01 = temp;
        temp = m20; m20 = m02; m02 = temp;
        temp = m21; m21 = m12; m12 = temp;
        return *this;
    }

    static CUDA_INLINE_CALLABLE double3x3 identity3x3() {
        return         {make_double3(1, 0, 0),
                        make_double3(0, 1, 0),
                        make_double3(0, 0, 1)};
    }

    static CUDA_INLINE_CALLABLE double3x3 zeros3x3() {
        return          {make_double3(0, 0, 0),
                         make_double3(0, 0, 0),
                         make_double3(0, 0, 0)};
    }

    static CUDA_INLINE_CALLABLE double3x3 ones3x3() {
        return          {make_double3(1, 1, 1),
                         make_double3(1, 1, 1),
                         make_double3(1, 1, 1)};
    }
};

//return a * b^T
CUDA_INLINE_CALLABLE double3x3 out_dot(const double3 &a, const double3 &b) {

    return double3x3{a * b.x, a * b.y, a * b.z};
}

CUDA_INLINE_CALLABLE double3x3 operator+(const double3x3 &a, const double3x3 &b){
    double3x3 ret = a;
    ret += b;
    return ret;
}

CUDA_INLINE_CALLABLE double3x3 operator-(const double3x3 &a, const double3x3 &b) {
    double3x3 ret = a;
    ret -= b;
    return ret;
}

CUDA_INLINE_CALLABLE double3x3 operator*(const double3x3 &a, double b) {
    double3x3 ret = a;
    ret *= b;
    return ret;
}

CUDA_INLINE_CALLABLE double3x3 operator*(double a, const double3x3 &b) {
    double3x3 ret = b;
    ret *= a;
    return ret;
}

CUDA_INLINE_CALLABLE double3x3 operator*(const double3x3 &a, const double3x3 &b) {
    double3x3 ret = a;
    ret *= b;
    return ret;
}
CUDA_INLINE_CALLABLE double det(const double3x3 &mat){
    double res;
    res = ( mat.m00 * mat.m11 * mat.m22 +
            mat.m01 * mat.m12 * mat.m20 +
            mat.m02 * mat.m10 * mat.m21 -
            mat.m02 * mat.m11 * mat.m20 -
            mat.m01 * mat.m10 * mat.m22 -
            mat.m00 * mat.m12 * mat.m21);
    return res;
}

CUDA_INLINE_CALLABLE double3x3 transpose(const double3x3 &mat) {
    double3x3 ret = mat;
    return ret.transpose();
}
CUDA_INLINE_CALLABLE double3x3 inverse(const double3x3 &mat) {
    double3x3 ret = mat;
    return ret.inverse();
}

CUDA_INLINE_CALLABLE double3x3 scale3x3(const double3 &s) {
    return         {s.x * make_double3(1, 0, 0),
                    s.y * make_double3(0, 1, 0),
                    s.z * make_double3(0, 0, 1)};
}
CUDA_INLINE_CALLABLE double3x3 scale3x3(double sx, double sy, double sz) {
    return scale3x3(make_double3(sx, sy, sz));
}
CUDA_INLINE_CALLABLE double3x3 scale3x3(double s) {
    return scale3x3(make_double3(s, s, s));
}

CUDA_INLINE_CALLABLE double3x3 rotate3x3(double angle, const double3 &axis) {
    double3x3 matrix;
    double3 nAxis = normalize(axis);
    double s = std::sin(angle);
    double c = std::cos(angle);
    double oneMinusC = 1 - c;

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
CUDA_INLINE_CALLABLE double3x3 rotate3x3(double angle, double ax, double ay, double az) {
    return rotate3x3(angle, make_double3(ax, ay, az));
}
CUDA_INLINE_CALLABLE double3x3 rotateX3x3(double angle) { return rotate3x3(angle, make_double3(1, 0, 0)); }
CUDA_INLINE_CALLABLE double3x3 rotateY3x3(double angle) { return rotate3x3(angle, make_double3(0, 1, 0)); }
CUDA_INLINE_CALLABLE double3x3 rotateZ3x3(double angle) { return rotate3x3(angle, make_double3(0, 0, 1)); }

#endif //DOUBLE3X3_H
