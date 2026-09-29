//
// Created by birdpeople on 8/25/2023.
//

#ifndef ARRAY_CUH
#define ARRAY_CUH
#include <cuda_runtime.h>
#include <array>
#include "common.h"
 // a light weight thread-only array used the same as std::array, only if typename T is constructible in CUDA kernel
 // the reversed and forward iterator is not implemented yet, all traited as random access iterator

namespace gte {
    template<typename T, size_t N>
    struct array {
        using value_type = T;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using reference = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using self_type = array<T, N>;

        CUDA_CALLABLE constexpr reference operator[](size_type i) noexcept { return data_[i]; }

        CUDA_CALLABLE constexpr const_reference operator[](size_type i) const noexcept{ return data_[i]; }

        CUDA_CALLABLE constexpr reference at(size_type i) { return data_[i]; }

        CUDA_CALLABLE constexpr const_reference at(size_type i) const { return data_[i]; }

        CUDA_CALLABLE constexpr reference front() noexcept{ return data_[0]; }

        CUDA_CALLABLE constexpr const_reference front() const noexcept{ return data_[0]; }

        CUDA_CALLABLE constexpr reference back() noexcept{ return data_[N - 1]; }

        CUDA_CALLABLE constexpr const_reference back() const noexcept{ return data_[N - 1]; }

        CUDA_CALLABLE constexpr pointer data() noexcept{ return data_; }

        CUDA_CALLABLE constexpr const_pointer data() const noexcept{ return data_; }

        CUDA_CALLABLE constexpr iterator begin() noexcept{ return data_; }

        CUDA_CALLABLE constexpr const_iterator begin() const noexcept{ return data_; }

        CUDA_CALLABLE constexpr const_iterator cbegin() const noexcept{ return data_; }

        CUDA_CALLABLE constexpr iterator end() noexcept{ return data_ + N; }

        CUDA_CALLABLE constexpr const_iterator end() const noexcept{ return data_ + N; }

        CUDA_CALLABLE constexpr const_iterator cend() const noexcept{ return data_ + N; }

        CUDA_CALLABLE constexpr bool empty() const noexcept{ return N == 0; }

        CUDA_CALLABLE constexpr size_type size() const noexcept{ return size_type(N);};

        CUDA_CALLABLE constexpr void fill(const value_type& value) {
            for (size_type i = 0; i < N; ++i) data_[i] = value;
        }

        CUDA_CALLABLE constexpr void swap(self_type& rhs) noexcept{
            for (size_type i = 0; i < N; ++i) {
                auto tmp = data_[i];
                data_[i] = rhs.data_[i];
                rhs.data_[i] = tmp;
            }
        }

        value_type data_[N];
    };

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr  bool operator == (const array<T,N>& rhs, const array<T,N>& lhs) {
        for (typename array<T,N>::size_type i = 0; i < N; ++i) {
            if (rhs[i] != lhs[i]) return false;
        }
        return true;
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr bool operator != (const array<T,N>& rhs, const array<T,N>& lhs) {
        return !(rhs == lhs);
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr bool operator < (const array<T,N>& rhs, const array<T,N>& lhs) {
        for (typename array<T,N>::size_type i = 0; i < N; ++i) {
            if (rhs[i] < lhs[i]) return true;
            if (rhs[i] > lhs[i]) return false;
        }
        return false;
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr  bool operator > (const array<T,N>& rhs, const array<T,N>& lhs) {
        return lhs < rhs;
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr  bool operator <= (const array<T,N>& rhs, const array<T,N>& lhs) {
        return !(rhs > lhs);
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr  bool operator >= (const array<T,N>& rhs, const array<T,N>& lhs) {
        return !(rhs < lhs);
    }

    template<typename T, size_t N>
    CUDA_CALLABLE constexpr void swap(array<T,N>& rhs, array<T,N>& lhs) noexcept{
        rhs.swap(lhs);
    }


    template< size_t I, class T, size_t N >
    CUDA_CALLABLE constexpr T& get( array<T,N>& a ) noexcept{
        static_assert( I < N, "array index out of bounds" );
        return a[I];
    }

    template< size_t I, class T, size_t N >
    CUDA_CALLABLE constexpr const T& get( const array<T,N>& a ) noexcept{
        static_assert( I < N, "array index out of bounds" );
        return a[I];
    }

    template< std::size_t I, class T, std::size_t N >
    CUDA_CALLABLE constexpr T&& get( std::array<T,N>&& a ) noexcept{
        static_assert( I < N, "array index out of bounds" );
        return std::move(a[I]);
    }

    template< size_t I, class T, size_t N >
    CUDA_CALLABLE constexpr const T&& get( const array<T,N>&& a ) noexcept{
        static_assert( I < N, "array index out of bounds" );
        return std::move(a[I]);
    }

}
#endif //ARRAY_CUH
