#include "bland/ops/ops_frequency.hpp"

#include "bland/ndarray.hpp"
#include "bland/ndarray_slice.hpp"

#include "internal/shape_helpers.hpp"
#include "device_dispatch.hpp"

#if BLAND_CUDA_CODE
#include "cuda/frequency_cuda.cuh"
#endif // BLAND_CUDA_CODE
#include "cpu/frequency_cpu_impl.hpp"

#include <fmt/format.h>

using namespace bland;


/*
 * Externally exposed function implementations
 */

ndarray bland::fft_shift_mag_square(ndarray x) {
    // For now, this only supports real to complex fft for float32
    auto dtype = x.dtype();
    if (x.dtype() != ndarray::datatype::float32) {
        auto err_msg = fmt::format("ERROR: fft_shift_abs_square got an unsupported datatype {}.{}.{}\n", dtype.code, dtype.bits, dtype.lanes);
        // throw std::runtime_error("bland::fft: Unsupported datatype");
        throw std::runtime_error(err_msg);
    }

    auto out = ndarray(x.shape(), x.dtype(), x.device());

    auto compute_device = x.device();

    if (compute_device.device_type == kDLCPU || compute_device.device_type == kDLCUDAHost) {
        return cpu::fft_shift_mag_square(x, out);
    } else if (compute_device.device_type == kDLCUDA || compute_device.device_type == kDLCUDAManaged) {
    #if BLAND_CUDA_CODE
        return cuda::fft_shift_mag_square(x, out);
    #endif // BLAND_CUDA_CODE
        throw std::runtime_error("BLISS not build with CUDA support but got an array with CUDA");
    } else {
        throw std::runtime_error("Unsupported compute device");
    }
}

ndarray bland::fft(ndarray x) {

    // For now, this only supports real to complex fft for float32
    if (x.dtype() != ndarray::datatype::float32) {
        fmt::print("ERROR: fft got an unsupported datatype\n");
        throw std::runtime_error("bland::fft: Unsupported datatype");
    }

    // 
    auto out = ndarray(x.shape(), ndarray::datatype::cfloat32, x.device());

    auto compute_device = x.device();

    if (compute_device.device_type == kDLCPU || compute_device.device_type == kDLCUDAHost) {
        return cpu::fft(x, out);
    } else if (compute_device.device_type == kDLCUDA || compute_device.device_type == kDLCUDAManaged) {
    #if BLAND_CUDA_CODE
        return cuda::fft(x, out);
    #endif // BLAND_CUDA_CODE
        throw std::runtime_error("BLISS not build with CUDA support but got an array with CUDA");
    } else {
        throw std::runtime_error("Unsupported compute device");
    }
}
