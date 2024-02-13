#pragma once

#include "bland/ndarray.hpp"

#include <shape_helpers.hpp>

#include <stdexcept>
#include <type_traits>

using namespace bland;

// TODO: we can likely do better than this, but it's simple and does the trick for now
template <typename FuncCPU, typename FuncCUDA, typename T>
ndarray device_dispatch(FuncCPU cpu_func, FuncCUDA cuda_func, ndarray a, T b) {
    auto compute_device = a.device();
    std::vector<int64_t> out_shape;
    if constexpr (std::is_same<T, ndarray>::value) {
        out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    } else {
        out_shape = a.shape();
    }
    auto out = ndarray(out_shape, a.dtype(), compute_device);

    if constexpr (std::is_same<T, ndarray>::value) {
        // If b is an ndarray, check the device
        if (compute_device != b.device()) {
            throw std::runtime_error("Mismatched compute devices between arguments");
        }
    }

    if (compute_device.device_type == kDLCPU) {
        return cpu_func(a, b);
    } else if (compute_device.device_type == kDLCUDA) {
    #if BLAND_CUDA_CODE
        if (cuda_func != nullptr) {
            return cuda_func(a, b);
        } else {
            throw std::runtime_error("No CUDA function provided");
        }
    #endif // BLAND_CUDA_CODE
        throw std::runtime_error("BLISS not build with CUDA support but got an array with CUDA");
    } else {
        throw std::runtime_error("Unsupported compute device");
    }
}
