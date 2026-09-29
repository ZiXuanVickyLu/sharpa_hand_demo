#ifndef BUFFER2D_CUH
#define BUFFER2D_CUH
#include <cuda/api.hpp>
#include <numeric>
#ifdef CCCL_VERSION_GREATER_EQUAL_13_0
#include <cccl/thrust/scan.h>
#include <cccl/thrust/device_ptr.h>
#else
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#endif

namespace cs {
    template<typename T>
    class Buffer2D{
        public:
            Buffer2D(std::vector<std::vector<T>> buffer_2D);
            ~Buffer2D() = default;
            
            // Delete copy constructor and assignment (unique_span is move-only)
            Buffer2D(const Buffer2D&) = delete;
            Buffer2D& operator=(const Buffer2D&) = delete;
            
            // Default move constructor and assignment
            Buffer2D(Buffer2D&&) = default;
            Buffer2D& operator=(Buffer2D&&) = default;
            
            void sync_to_host();
            cuda::unique_span<T> d_buffer;
            std::vector<T> h_buffer;
            cuda::unique_span<T> d_offset;
            cuda::unique_span<T> d_inner_size;
            std::vector<T> h_inner_size;
            int outer_size;
            int total_size;
    };

    template<typename T>
    Buffer2D<T>::Buffer2D(std::vector<std::vector<T>> buffer_2D)
        : outer_size(0), total_size(0) {
        auto device = cuda::device::current::get();
        outer_size = static_cast<int>(buffer_2D.size());
        
        if (outer_size == 0) {
            // Empty buffer - nothing to allocate
            return;
        }
        
        h_inner_size.resize(outer_size);

        d_offset = cuda::memory::make_unique_span<T>(device, outer_size);
        d_inner_size = cuda::memory::make_unique_span<T>(device, outer_size);
        
        // make the flattened buffer for vector of vector
        std::vector<T> flattened_buffer;
        size_t idx = 0;
        for(const auto& row : buffer_2D){
            flattened_buffer.insert(flattened_buffer.end(), row.begin(), row.end());
            h_inner_size[idx] = static_cast<T>(row.size());
            idx++;
        }
        total_size = std::accumulate(h_inner_size.begin(), h_inner_size.end(), 0);
        
        if (total_size > 0) {
            d_buffer = cuda::memory::make_unique_span<T>(device, total_size);
            // copy the flattened buffer to the device
            cuda::memory::copy(d_buffer.data(), flattened_buffer.data(), sizeof(T) * total_size);
        }
        
        // d_inner is each row's size
        cuda::memory::copy(d_inner_size.data(), h_inner_size.data(), sizeof(T) * outer_size);
        
        // d_offset is the prefix sum of the inner size
        // Use thrust::device_ptr to wrap raw pointers for Thrust operations
        thrust::device_ptr<T> d_inner_ptr(d_inner_size.data());
        thrust::device_ptr<T> d_offset_ptr(d_offset.data());
        thrust::exclusive_scan(d_inner_ptr, d_inner_ptr + outer_size, d_offset_ptr);
    }

    template<typename T>
    void Buffer2D<T>::sync_to_host(){
        if (total_size == 0) {
            h_buffer.clear();
            return;
        }
        // copy the buffer to host
        h_buffer.resize(total_size);
        cuda::memory::copy(h_buffer.data(), d_buffer.data(), sizeof(T) * total_size);
    }
}
#endif // BUFFER2D_CUH