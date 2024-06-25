#include "bland/ops/ops.hpp"
#include "bland/config.hpp"

#include "bland/ndarray.hpp"
#include "bland/ndarray_slice.hpp"

#include "device_dispatch.hpp"
// #include "internal/dispatcher.hpp"
#include "internal/shape_helpers.hpp"
#include <dlpack/dlpack.h> // consider if bland_tensor_internals.hpp which includes this is more appropriate

#include "cpu/elementwise_unary_op.hpp"
#include "cpu/ops_cpu.hpp"
#if BLAND_CUDA_CODE
#include "cuda/ops_cuda.cuh"
#endif

#include <fmt/core.h>

#include <cstdlib>

#if BLAND_CUDA
#include <cuda_runtime.h> // cudaMalloc
#endif                    // BLAND_CUDA

using namespace bland;

/**
 * Copy (data)
 */
ndarray bland::copy(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return copy(a, out);
}

ndarray bland::copy(ndarray a, ndarray &out) {
    fmt::print("bland::copy\n");
    return device_dispatch(cpu::copy,
                            #if BLAND_CUDA_CODE
                            cuda::copy,
                            #else
                            nullptr,
                            #endif
                            a, out);
}

ndarray bland::to(ndarray src, DLDevice dest_dev) {
    auto source_dev = src.device();
    if (source_dev == dest_dev) {
        // device type and id match, no need to move any data
        return src;
    }
    // create a new tensor on the given device, and copy data over
    auto dst        = bland::ndarray(src.shape(), src.dtype(), dest_dev);
    auto dst_ptr    = dst.data_ptr<void>();
    auto src_ptr    = src.data_ptr<void>();

    // TODO: will probably want to handle slices more efficiently in the future
    auto copy_size = src.numel() * src.dtype().bits / 8;

#ifndef BLAND_CUDA
    // Handle this error once, everything else still needs compile guard but not error handling
    if (source_dev.device_type == ndarray::dev::cuda.device_type ||
        source_dev.device_type == ndarray::dev::cuda_host.device_type ||
        source_dev.device_type == ndarray::dev::cuda_managed.device_type ||
        dest_dev.device_type == ndarray::dev::cuda.device_type ||
        dest_dev.device_type == ndarray::dev::cuda_host.device_type ||
        dest_dev.device_type == ndarray::dev::cuda_managed.device_type) {
        throw std::runtime_error("bland::to got device on cuda but built without cuda support");
    }
#endif

    // TODO: fill in the device transport
    if (source_dev.device_type == ndarray::dev::cpu.device_type) {
        if (dest_dev.device_type == ndarray::dev::cuda.device_type ||
            dest_dev.device_type == ndarray::dev::cuda_managed.device_type) {
            // source cpu, dest cuda device
#if BLAND_CUDA
            if (!g_config.check_is_valid_cuda_device(dest_dev.device_id)) {
                throw std::runtime_error("copying data to an invalid cuda device");
            }
            cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyHostToDevice);
#endif // BLAND_CUDA
        } else if (dest_dev.device_type == ndarray::dev::cuda_host.device_type) {
            // source cpu, dest cuda host
#if BLAND_CUDA
            cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyHostToHost);
#endif // BLAND_CUDA
        } else if (dest_dev.device_type == ndarray::dev::cpu.device_type) {
            if (source_dev.device_id != dest_dev.device_id) {
                // cpu -> cpu, but different device_id, probably a numa thing
                memcpy(dst_ptr, src_ptr, copy_size);
            } else {
                // should not get here (should have returned the src as our result since no copy is required)
                throw std::runtime_error("bland to: cpu->cpu reached unexpected code path");
            }
        } else {
            // unsupported so far
            throw std::runtime_error("bland to: destination device type not supported yet");
        }
    } else if (source_dev.device_type == ndarray::dev::cuda.device_type ||
               source_dev.device_type == ndarray::dev::cuda_managed.device_type) {
#if BLAND_CUDA
        if (dest_dev.device_type == ndarray::dev::cpu.device_type ||
            dest_dev.device_type == ndarray::dev::cuda_host.device_type) {
            // source cuda, dest cpu
            cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToHost);
        } else if (dest_dev.device_type == ndarray::dev::cuda_managed.device_type) {
            if (dest_dev.device_id != source_dev.device_id) {
                if (!g_config.check_is_valid_cuda_device(dest_dev.device_id)) {
                    throw std::runtime_error("copying data to an invalid cuda device");
                }
                // Crossing cuda devices
                cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice);
            } else {
                // should not get here (should have returned the src as our result since no copy is required)
                throw std::runtime_error("bland to: cuda->cuda reached unexpected code path");
            }
        }
#endif // BLAND_CUDA
    } else {
        throw std::runtime_error("bland to: received an array on an unsupported device");
    }

    return dst;
}
ndarray bland::to(ndarray a, std::string_view dest_dev) {
    return to(a, bland::ndarray::dev(dest_dev));
}

// Decide if this should exist or should just be the base case for those recursive calls...
ndarray_slice bland::slice(const ndarray &a, int64_t dim, int64_t start, int64_t end, int64_t stride) {

    ndarray_slice sliced(a);

    if (start < 0) {
        // Negative is an offset from the end
        start = a.size(dim) + start - 1;
    } else if (start > a.size(dim)) {
        throw std::out_of_range("Invalid start for stride (begins beyond the end)");
    }
    if (end < 0) {
        // Negative is an offset from the end
        end = a.size(dim) + end - 1;
    } else if (end > a.size(dim)) {
        end = a.size(dim);
    }

    if (end < start) {
        auto error_message =
                fmt::format("slice: end index ({}) is less than start index ({}). Invalid slice.", end, start);
        throw std::runtime_error(error_message);
    }

    // The offset needs to use the current stride so it's an offset
    // of the current axis rather than the number of items at new stride
    sliced._tensor._offsets[dim] += start * sliced._tensor.strides[dim];

    sliced._tensor.shape[dim] = (end - start) / stride;
    sliced._tensor.strides[dim] *= stride;

    return sliced;
}

// Base case for the recursive call
ndarray process_slice_specs(const ndarray_slice &sliced_result) {
    return sliced_result;
}

template <typename... Args>
ndarray process_slice_specs(const ndarray_slice &sliced_result, slice_spec slice_dim, Args... args) {
    ndarray new_slice = slice(sliced_result, slice_dim.dim, slice_dim.start, slice_dim.end, slice_dim.stride);

    // Recursively process the remaining slice_spec arguments
    return process_slice_specs(new_slice, args...);
}

template <typename... Args>
ndarray_slice bland::slice(const ndarray &a, slice_spec slice_dim, Args... args) {
    ndarray sliced_result = slice(a, slice_dim.dim, slice_dim.start, slice_dim.end, slice_dim.stride);

    // Process the remaining slice_spec arguments
    // TODO: there are bug-covered dragons surrounding the variations of assignment and
    // copy constructors such that assigning to sliced_result from a numpy-borrowed tensor
    // will produce a tensor with null data, but with a new variable it's fine.
    auto sliced_result1 = process_slice_specs(sliced_result, args...);

    return sliced_result1;
}

// Explicit instantiation of the slice function (all this to avoid va_args)
// If this winds up not being enough regularly, we might actually need to just move the definition to header...
template ndarray_slice bland::slice(const ndarray &a, slice_spec slice_dim);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice
bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice
bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);

template <typename T>
ndarray bland::fill(ndarray out, T value) {
    auto compute_device = out.device();
#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::fill(out, value);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::fill(out, value);
    } else {
        throw std::runtime_error("unsupported device for masked_stddev");
    }
}

template ndarray bland::fill<float>(ndarray out, float v);
// template ndarray bland::fill<double>(ndarray out, double v);
template ndarray bland::fill<uint8_t>(ndarray out, uint8_t v);
// template ndarray bland::fill<uint16_t>(ndarray out, uint16_t v);
template ndarray bland::fill<uint32_t>(ndarray out, uint32_t v);
// template ndarray bland::fill<uint64_t>(ndarray out, uint64_t v);
// template ndarray bland::fill<int8_t>(ndarray out, int8_t v);
// template ndarray bland::fill<int16_t>(ndarray out, int16_t v);
template ndarray bland::fill<int32_t>(ndarray out, int32_t v);
// template ndarray bland::fill<int64_t>(ndarray out, int64_t v);


ndarray bland::square(ndarray a, ndarray out) {
    return device_dispatch(cpu::square,
                            #if BLAND_CUDA_CODE
                            cuda::square,
                            #else
                            nullptr,
                            #endif
                            a, out);
}

ndarray bland::square(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return square(a, out);
}


ndarray bland::sqrt(ndarray a, ndarray out) {
    return device_dispatch(cpu::sqrt,
                            #if BLAND_CUDA_CODE
                            cuda::sqrt,
                            #else
                            nullptr,
                            #endif
                            a, out);
}

ndarray bland::sqrt(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return sqrt(a, out);
}

ndarray bland::abs(ndarray a, ndarray out) {
    return device_dispatch(cpu::abs,
                            #if BLAND_CUDA_CODE
                            cuda::abs,
                            #else
                            nullptr,
                            #endif
                            a, out);
}

ndarray bland::abs(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return cpu::abs(a, out);
}
