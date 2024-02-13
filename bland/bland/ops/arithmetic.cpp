
#include "bland/ndarray.hpp"
#include "bland/ops_arithmetic.hpp"
#include "device_dispatch.hpp"

#if BLAND_CUDA_CODE
#include "cuda/arithmetic_cuda.cuh"
#endif // BLAND_CUDA_CODE
#include "cpu/arithmetic_cpu.hpp"

#include <shape_helpers.hpp>

using namespace bland;


/*
 * Externally exposed function implementations
 */

// Adds...
template <typename T>
ndarray bland::add(ndarray a, T b) {
    return device_dispatch(cpu::add_cpu<T>,
                            #if BLAND_CUDA_CODE
                            cuda::add_cuda<T>,
                            #else
                            nullptr,
                            #endif
                            a, b);
}

template ndarray bland::add<ndarray>(ndarray a, ndarray b);
template ndarray bland::add<ndarray_slice>(ndarray a, ndarray_slice b);

template ndarray bland::add<float>(ndarray a, float b);
template ndarray bland::add<double>(ndarray a, double b);
template ndarray bland::add<int8_t>(ndarray a, int8_t b);
template ndarray bland::add<int16_t>(ndarray a, int16_t b);
template ndarray bland::add<int32_t>(ndarray a, int32_t b);
template ndarray bland::add<int64_t>(ndarray a, int64_t b);
template ndarray bland::add<uint8_t>(ndarray a, uint8_t b);
template ndarray bland::add<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::add<uint32_t>(ndarray a, uint32_t b);
template ndarray bland::add<uint64_t>(ndarray a, uint64_t b);


// Subtracts...
template <typename T>
ndarray bland::subtract(ndarray a, T b) {
    return device_dispatch(cpu::subtract_cpu<T>, 
                            #if BLAND_CUDA_CODE
                            cuda::subtract_cuda<T>,
                            #else
                            nullptr,
                            #endif
                            a , b);
}

template ndarray bland::subtract<ndarray>(ndarray a, ndarray b);
template ndarray bland::subtract<ndarray_slice>(ndarray a, ndarray_slice b);

template ndarray bland::subtract<float>(ndarray a, float b);
template ndarray bland::subtract<double>(ndarray a, double b);
template ndarray bland::subtract<int8_t>(ndarray a, int8_t b);
template ndarray bland::subtract<int16_t>(ndarray a, int16_t b);
template ndarray bland::subtract<int32_t>(ndarray a, int32_t b);
template ndarray bland::subtract<int64_t>(ndarray a, int64_t b);
template ndarray bland::subtract<uint8_t>(ndarray a, uint8_t b);
template ndarray bland::subtract<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::subtract<uint32_t>(ndarray a, uint32_t b);
template ndarray bland::subtract<uint64_t>(ndarray a, uint64_t b);


// Multiplies...
template <typename T>
ndarray bland::multiply(ndarray a, T b) {
    return device_dispatch(cpu::multiply_cpu<T>,
                            #if BLAND_CUDA_CODE
                            cuda::multiply_cuda<T>,
                            #else
                            nullptr,
                            #endif
                            a, b);
}

template ndarray bland::multiply<ndarray>(ndarray a, ndarray b);
template ndarray bland::multiply<ndarray_slice>(ndarray a, ndarray_slice b);

template ndarray bland::multiply<float>(ndarray a, float b);
template ndarray bland::multiply<double>(ndarray a, double b);
template ndarray bland::multiply<int8_t>(ndarray a, int8_t b);
template ndarray bland::multiply<int16_t>(ndarray a, int16_t b);
template ndarray bland::multiply<int32_t>(ndarray a, int32_t b);
template ndarray bland::multiply<int64_t>(ndarray a, int64_t b);
template ndarray bland::multiply<uint8_t>(ndarray a, uint8_t b);
template ndarray bland::multiply<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::multiply<uint32_t>(ndarray a, uint32_t b);
template ndarray bland::multiply<uint64_t>(ndarray a, uint64_t b);


// Divides...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::divide(ndarray a, T b) {
    return device_dispatch(cpu::divide_cpu<T>,
                            #if BLAND_CUDA_CODE
                            cuda::divide_cuda<T>,
                            #else
                            nullptr,
                            #endif
                            a, b);
}

template ndarray bland::divide<ndarray>(ndarray a, ndarray b);
template ndarray bland::divide<ndarray_slice>(ndarray a, ndarray_slice b);

template ndarray bland::divide<float>(ndarray a, float b);
template ndarray bland::divide<double>(ndarray a, double b);
template ndarray bland::divide<int8_t>(ndarray a, int8_t b);
template ndarray bland::divide<int16_t>(ndarray a, int16_t b);
template ndarray bland::divide<int32_t>(ndarray a, int32_t b);
template ndarray bland::divide<int64_t>(ndarray a, int64_t b);
template ndarray bland::divide<uint8_t>(ndarray a, uint8_t b);
template ndarray bland::divide<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::divide<uint32_t>(ndarray a, uint32_t b);
template ndarray bland::divide<uint64_t>(ndarray a, uint64_t b);
