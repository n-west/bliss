
#include "bland/ndarray.hpp"
// #include "bland/ops_arithmetic.hpp"

#include "arithmetic_cuda.cuh"

#include "internal/dispatcher.hpp"
#include "elementwise_scalar_op_cuda.cuh"
#include "arithmetic_cuda_impl.cuh"

#include <fmt/format.h>


using namespace bland;
using namespace bland::cuda;

/*
 * Externally exposed function implementations
 */

// Adds...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cuda::add_cuda(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper_cuda, T, elementwise_add_op_ts>(out, a, b);
}
template ndarray bland::cuda::add_cuda<float>(ndarray a, float b);
// template ndarray bland::cuda::add_cuda<double>(ndarray a, double b);
// template ndarray bland::cuda::add_cuda<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cuda::add_cuda<int16_t>(ndarray a, int16_t b);
template ndarray bland::cuda::add_cuda<int32_t>(ndarray a, int32_t b);
template ndarray bland::cuda::add_cuda<int64_t>(ndarray a, int64_t b);
template ndarray bland::cuda::add_cuda<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cuda::add_cuda<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cuda::add_cuda<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cuda::add_cuda<uint64_t>(ndarray a, uint64_t b);


// Subtracts...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cuda::subtract_cuda(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper_cuda, T, elementwise_subtract_op_ts>(out, a, b);
}
template ndarray bland::cuda::subtract_cuda<float>(ndarray a, float b);
// template ndarray bland::cuda::subtract_cuda<double>(ndarray a, double b);
// template ndarray bland::cuda::subtract_cuda<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cuda::subtract_cuda<int16_t>(ndarray a, int16_t b);
template ndarray bland::cuda::subtract_cuda<int32_t>(ndarray a, int32_t b);
template ndarray bland::cuda::subtract_cuda<int64_t>(ndarray a, int64_t b);
template ndarray bland::cuda::subtract_cuda<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cuda::subtract_cuda<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cuda::subtract_cuda<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cuda::subtract_cuda<uint64_t>(ndarray a, uint64_t b);

// Multiplies...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cuda::multiply_cuda(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper_cuda, T, elementwise_multiply_op_ts>(out, a, b);
}

template ndarray bland::cuda::multiply_cuda<float>(ndarray a, float b);
// template ndarray bland::cuda::multiply_cuda<double>(ndarray a, double b);
// template ndarray bland::cuda::multiply_cuda<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cuda::multiply_cuda<int16_t>(ndarray a, int16_t b);
template ndarray bland::cuda::multiply_cuda<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cuda::multiply_cuda<int64_t>(ndarray a, int64_t b);
template ndarray bland::cuda::multiply_cuda<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cuda::multiply_cuda<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cuda::multiply_cuda<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cuda::multiply_cuda<uint64_t>(ndarray a, uint64_t b);

// Divides...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cuda::divide_cuda(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper_cuda, T, elementwise_divide_op_ts>(out, a, b);
}
template ndarray bland::cuda::divide_cuda<float>(ndarray a, float b);
// template ndarray bland::cuda::divide_cuda<double>(ndarray a, double b);
// template ndarray bland::cuda::divide_cuda<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cuda::divide_cuda<int16_t>(ndarray a, int16_t b);
template ndarray bland::cuda::divide_cuda<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cuda::divide_cuda<int64_t>(ndarray a, int64_t b);
template ndarray bland::cuda::divide_cuda<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cuda::divide_cuda<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cuda::divide_cuda<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cuda::divide_cuda<uint64_t>(ndarray a, uint64_t b);
