#pragma once

#include "bland/ndarray.hpp"

#include <shape_helpers.hpp>

#include <stdexcept>
#include <type_traits>

using namespace bland;

// TODO: we can likely do better than this, but it's simple and does the trick for now
template <typename FuncCPU, typename FuncCUDA, typename L, typename R>
inline ndarray device_dispatch(FuncCPU cpu_func, FuncCUDA cuda_func, L lhs, R rhs) {
    
    ndarray::dev compute_device = default_device;
    // Check device consistency between arguments and set appropriate compute device
    if constexpr (std::is_same<L, ndarray>::value && std::is_same<R, ndarray>::value) {
        if (lhs.device() == rhs.device()) {
            compute_device = rhs.device();
        } else {
            throw std::runtime_error("ERROR: got two arrays on different devices");
        }
    } else if constexpr (std::is_same<L, ndarray>::value) {
        compute_device = lhs.device();
    } else if constexpr (std::is_same<R, ndarray>::value) {
        compute_device = rhs.device();
    }

    // Call the correct function impl based on compute device
    if (compute_device.device_type == kDLCPU || compute_device.device_type == kDLCUDAHost) {
        return cpu_func(lhs, rhs);
    } else if (compute_device.device_type == kDLCUDA || compute_device.device_type == kDLCUDAManaged) {
    #if BLAND_CUDA_CODE
        if constexpr (!std::is_null_pointer<FuncCUDA>::value) {
            return cuda_func(lhs, rhs);
        } else {
            throw std::runtime_error("No CUDA function provided");
        }
    #endif // BLAND_CUDA_CODE
        throw std::runtime_error("BLISS not build with CUDA support but got an array with CUDA");
    } else {
        throw std::runtime_error("Unsupported compute device");
    }
}
