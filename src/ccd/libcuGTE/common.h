
#ifndef COMMON_CUH
#define COMMON_CUH
#include <cuda_runtime.h>
namespace gte {
    #ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
    #else
    #define CUDA_CALLABLE inline
    #endif

    #ifdef __CUDACC__
    #define DEVICE_CALLABLE __device__
    #else
    #define DEVICE_CALLABLE inline
    #endif

    #ifdef __CUDACC__
    #define HOST_CALLABLE __host__
    #else
    #define HOST_CALLABLE inline
    #endif
} // namespace gte
#endif // COMMON_CUH