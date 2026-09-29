// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2023
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2023.08.08

#pragma once

// The triangle is represented as an array of three vertices.  The dimension
// N must be 2 or larger.

#include "Vector.h"
#include "array.h"

namespace gte
{
    template <size_t N, typename Real>
    class Triangle
    {
    public:
        // Construction and destruction.  The default constructor sets
        // the/ vertices to (0,..,0), (1,0,...,0) and (0,1,0,...,0).
        CUDA_CALLABLE Triangle()
            :
            v{ Vector<N, Real>::Zero(), Vector<N, Real>::Unit(0), Vector<N, Real>::Unit(1) }
        {
        }

        CUDA_CALLABLE Triangle(Vector<N, Real> const& v0, Vector<N, Real> const& v1, Vector<N, Real> const& v2)
            :
            v{ v0, v1, v2 }
        {
        }


        CUDA_CALLABLE explicit Triangle(gte::array<Vector<N, Real>, 3> const& inV)
            :
            v(inV)
        {
        }

        // Public member access.
        gte::array<Vector<N, Real>, 3> v;

    public:
        // Comparisons to support sorted containers.
        CUDA_CALLABLE bool operator==(Triangle const& triangle) const
        {
            return v == triangle.v;
        }

        CUDA_CALLABLE bool operator!=(Triangle const& triangle) const
        {
            return v != triangle.v;
        }

        CUDA_CALLABLE bool operator< (Triangle const& triangle) const
        {
            return v < triangle.v;
        }

        CUDA_CALLABLE bool operator<=(Triangle const& triangle) const
        {
            return v <= triangle.v;
        }

        CUDA_CALLABLE bool operator> (Triangle const& triangle) const
        {
            return v > triangle.v;
        }

        CUDA_CALLABLE bool operator>=(Triangle const& triangle) const
        {
            return v >= triangle.v;
        }
    };

    // Template aliases for convenience.
    template <typename Real>
    using Triangle2 = Triangle<2, Real>;

    template <typename Real>
    using Triangle3 = Triangle<3, Real>;
}
