// device_buffer.cuh - RAII device arrays.
//
//   DeviceArray<T>  owning, move-only cudaMalloc'd array with size/capacity semantics:
//                     resize(n, keep)   exact growth to n (capacity never shrinks)
//                     loose_resize(n)   capacity grows by 1.5x, contents kept (pair/candidate
//                                       buffers of implementation spec 4)
//                     reserve, fill (kernel, nvcc translation units only), zero, upload,
//                     download, copy_from, swap, release
//   DeviceVar<T>    one T on the device with set()/get().
//
// T must be trivially copyable (buffers are moved with cudaMemcpy). Every CUDA call goes through
// CS_CUDA_CHECK; the destructor swallows errors, as destructors must.
#pragma once
#ifndef CS_DEVICE_BUFFER_CUH
#define CS_DEVICE_BUFFER_CUH

#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cs {

#ifdef __CUDACC__
namespace detail {

template <typename T>
__global__ void k_fill(T* __restrict__ p, std::size_t n, T value)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        p[i] = value;
    }
}

}  // namespace detail
#endif

namespace detail {
// CS_TRACE_ALLOC=1 in the environment logs every device reallocation above 64 MB to stderr
// with the free memory at that moment: the way to find what grows before an out-of-memory.
inline void trace_alloc(std::size_t bytes, std::size_t elem, std::size_t old_bytes)
{
    static const bool on = std::getenv("CS_TRACE_ALLOC") != nullptr;
    if (!on || bytes < (std::size_t(64) << 20)) return;
    std::size_t f = 0, t = 0;
    (void)cudaMemGetInfo(&f, &t);
    std::fprintf(stderr, "[alloc] %.1f MB (elem %zu B, was %.1f MB), free %.1f MB\n", bytes / 1048576.0, elem,
                 old_bytes / 1048576.0, f / 1048576.0);
}
}  // namespace detail

template <typename T>
class DeviceArray {
    static_assert(std::is_trivially_copyable<T>::value, "DeviceArray<T> requires a trivially copyable T");

public:
    using value_type = T;

    DeviceArray() = default;
    explicit DeviceArray(std::size_t n) { resize(n); }
    DeviceArray(const T* src, std::size_t n) { upload(src, n); }
    explicit DeviceArray(const std::vector<T>& v) { upload(v); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& o) noexcept : ptr_(o.ptr_), size_(o.size_), cap_(o.cap_)
    {
        o.ptr_ = nullptr;
        o.size_ = 0;
        o.cap_ = 0;
    }
    DeviceArray& operator=(DeviceArray&& o) noexcept
    {
        if (this != &o) {
            release_nothrow();
            ptr_ = o.ptr_;
            size_ = o.size_;
            cap_ = o.cap_;
            o.ptr_ = nullptr;
            o.size_ = 0;
            o.cap_ = 0;
        }
        return *this;
    }
    ~DeviceArray() { release_nothrow(); }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return cap_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    bool empty() const noexcept { return size_ == 0; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    /// Sets the size to n. Grows the allocation to exactly n when n > capacity(); the capacity
    /// never shrinks. With keep_contents the first min(old size, n) elements survive a
    /// reallocation, otherwise the contents are unspecified after growth.
    void resize(std::size_t n, bool keep_contents = false)
    {
        if (n > cap_) reallocate(n, keep_contents);
        size_ = n;
    }

    /// Sets the size to n, growing the capacity geometrically (x1.5, at least n) and keeping
    /// the contents. Never shrinks.
    void loose_resize(std::size_t n)
    {
        if (n > cap_) {
            std::size_t grown = cap_ + cap_ / 2 + 1;
            reallocate(grown > n ? grown : n, true);
        }
        size_ = n;
    }

    /// Ensures capacity() >= n, keeping size and contents.
    void reserve(std::size_t n)
    {
        if (n > cap_) reallocate(n, true);
    }

#ifdef __CUDACC__
    /// Writes value to every element (kernel; only available in nvcc translation units).
    void fill(const T& value)
    {
        if (size_ == 0) return;
        const unsigned block = 256;
        const std::size_t want = (size_ + block - 1) / block;
        const unsigned grid = static_cast<unsigned>(want < 65535 ? want : 65535);
        detail::k_fill<T><<<grid, block>>>(ptr_, size_, value);
        CS_CUDA_KERNEL_CHECK();
    }
#endif

    /// Sets every byte of the first size() elements to zero.
    void zero()
    {
        if (size_ == 0) return;
        CS_CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
    }

    /// resize(n) followed by a host-to-device copy.
    void upload(const T* src, std::size_t n)
    {
        resize(n);
        if (n == 0) return;
        CS_CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }

    /// Resizes dst to size() and copies device-to-host.
    void download(std::vector<T>& dst) const
    {
        dst.resize(size_);
        if (size_ == 0) return;
        CS_CUDA_CHECK(cudaMemcpy(dst.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
    }
    std::vector<T> download() const
    {
        std::vector<T> v;
        download(v);
        return v;
    }
    /// Copies the first n elements to host memory (n <= size()).
    void download(T* dst, std::size_t n) const
    {
        if (n == 0) return;
        CS_CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /// resize(other.size()) followed by a device-to-device copy.
    void copy_from(const DeviceArray<T>& other)
    {
        if (this == &other) return;
        resize(other.size_);
        if (size_ == 0) return;
        CS_CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_, bytes(), cudaMemcpyDeviceToDevice));
    }

    void swap(DeviceArray& o) noexcept
    {
        std::swap(ptr_, o.ptr_);
        std::swap(size_, o.size_);
        std::swap(cap_, o.cap_);
    }

    /// Frees the allocation; size() and capacity() become 0.
    void release()
    {
        if (ptr_) CS_CUDA_CHECK(cudaFree(ptr_));
        ptr_ = nullptr;
        size_ = 0;
        cap_ = 0;
    }

private:
    void reallocate(std::size_t new_cap, bool keep_contents)
    {
        T* fresh = nullptr;
        detail::trace_alloc(new_cap * sizeof(T), sizeof(T), cap_ * sizeof(T));
        CS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fresh), new_cap * sizeof(T)));
        if (keep_contents && size_ > 0) {
            const cudaError_t e = cudaMemcpy(fresh, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
            if (e != cudaSuccess) {
                cudaFree(fresh);
                CS_CUDA_CHECK(e);
            }
        }
        if (ptr_) {
            const cudaError_t e = cudaFree(ptr_);
            if (e != cudaSuccess) {
                cudaFree(fresh);
                CS_CUDA_CHECK(e);
            }
        }
        ptr_ = fresh;
        cap_ = new_cap;
    }

    void release_nothrow() noexcept
    {
        if (ptr_) (void)cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = 0;
        cap_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t cap_ = 0;
};

template <typename T>
void swap(DeviceArray<T>& a, DeviceArray<T>& b) noexcept
{
    a.swap(b);
}

/// One value of T on the device (e.g. a reduction result or a counter).
template <typename T>
class DeviceVar {
    static_assert(std::is_trivially_copyable<T>::value, "DeviceVar<T> requires a trivially copyable T");

public:
    DeviceVar() { CS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T))); }
    explicit DeviceVar(const T& init) : DeviceVar() { set(init); }

    DeviceVar(const DeviceVar&) = delete;
    DeviceVar& operator=(const DeviceVar&) = delete;
    DeviceVar(DeviceVar&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    DeviceVar& operator=(DeviceVar&& o) noexcept
    {
        if (this != &o) {
            if (ptr_) (void)cudaFree(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }
    ~DeviceVar()
    {
        if (ptr_) (void)cudaFree(ptr_);
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    void set(const T& v) { CS_CUDA_CHECK(cudaMemcpy(ptr_, &v, sizeof(T), cudaMemcpyHostToDevice)); }
    T get() const
    {
        T v;
        CS_CUDA_CHECK(cudaMemcpy(&v, ptr_, sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void zero() { CS_CUDA_CHECK(cudaMemset(ptr_, 0, sizeof(T))); }

private:
    T* ptr_ = nullptr;
};

}  // namespace cs

#endif  // CS_DEVICE_BUFFER_CUH
