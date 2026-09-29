// cuda_check.h - error checking for CUDA runtime calls and kernel launches.
//
//   CS_CUDA_CHECK(expr)      evaluates a cudaError_t expression and throws std::runtime_error
//                            with the error string, the expression and file:line on failure.
//   CS_CUDA_KERNEL_CHECK()   checks cudaGetLastError() after a launch; with CS_DEBUG_SYNC defined
//                            it also synchronises the device so asynchronous faults surface at
//                            the launch that caused them.
#pragma once
#ifndef CS_CUDA_CHECK_H
#define CS_CUDA_CHECK_H

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cs {

class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t code, const std::string& what) : std::runtime_error(what), code_(code) {}
    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

namespace detail {

[[noreturn]] inline void throw_cuda_error(cudaError_t code, const char* expr, const char* file, int line)
{
    std::string msg = "CUDA error ";
    msg += cudaGetErrorName(code);
    msg += " (";
    msg += cudaGetErrorString(code);
    msg += ") in '";
    msg += expr;
    msg += "' at ";
    msg += file;
    msg += ":";
    msg += std::to_string(line);
    throw cuda_error(code, msg);
}

}  // namespace detail
}  // namespace cs

#define CS_CUDA_CHECK(expr)                                                                   \
    do {                                                                                      \
        const cudaError_t cs_cuda_check_err_ = (expr);                                        \
        if (cs_cuda_check_err_ != cudaSuccess)                                                \
            ::cs::detail::throw_cuda_error(cs_cuda_check_err_, #expr, __FILE__, __LINE__);    \
    } while (0)

// Set (per thread) while a CUDA graph is being captured by stream capture (linsys/pcg.cu): a
// device synchronisation from the capturing thread would invalidate the capture, so the
// debug-sync kernel check only records the launch error while it is non-zero.
namespace cs {
namespace detail {
inline thread_local int g_stream_capture_depth = 0;
// CS_SYNC_KERNELS=1 in the environment makes every kernel check synchronise the device in a
// release build too, so an asynchronous fault is reported at the launch that caused it.
inline bool sync_kernels()
{
    static const bool v = std::getenv("CS_SYNC_KERNELS") != nullptr;
    return v;
}
}  // namespace detail
}  // namespace cs

#ifdef CS_DEBUG_SYNC
#define CS_CUDA_KERNEL_CHECK()                                                                \
    do {                                                                                      \
        CS_CUDA_CHECK(cudaGetLastError());                                                    \
        if (::cs::detail::g_stream_capture_depth == 0) CS_CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#else
#define CS_CUDA_KERNEL_CHECK()                                                                \
    do {                                                                                      \
        CS_CUDA_CHECK(cudaGetLastError());                                                    \
        if (::cs::detail::sync_kernels() && ::cs::detail::g_stream_capture_depth == 0)        \
            CS_CUDA_CHECK(cudaDeviceSynchronize());                                           \
    } while (0)
#endif

// CS_SYNC_POINT("label"): with CS_SYNC_KERNELS=1 set, synchronise here and report a pending
// asynchronous fault with the label (for library calls that launch without a kernel check).
#include <stdexcept>
#include <string>
#define CS_SYNC_POINT(label)                                                                   \
    do {                                                                                       \
        if (::cs::detail::sync_kernels() && ::cs::detail::g_stream_capture_depth == 0) {       \
            const cudaError_t e_ = cudaDeviceSynchronize();                                    \
            if (e_ != cudaSuccess)                                                             \
                throw std::runtime_error(std::string("CS_SYNC_POINT ") + (label) + ": " +      \
                                         cudaGetErrorString(e_));                              \
        }                                                                                      \
    } while (0)

#endif  // CS_CUDA_CHECK_H
