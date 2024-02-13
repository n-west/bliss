#include "arithmetic_cuda.cuh"

#include "bland/ndarray.hpp"
#include "bland/ops_arithmetic.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op_cuda.cuh"

#include "arithmetic_cuda_impl.cuh"


using namespace bland;
using namespace bland::cuda;


/*
 * Externally exposed function implementations
 */

// Adds...
template <> // Specialize for adding an array
ndarray bland::cuda::add_cuda<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper_cuda<elementwise_add_op_ts>>(out, a, b);
}


template <>
ndarray bland::cuda::add_cuda<ndarray_slice>(ndarray a, ndarray_slice b) {
    return add_cuda(a, ndarray(b));
}

// Subtracts...
template <> // Specialize for adding an array
ndarray bland::cuda::subtract_cuda<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper_cuda<elementwise_subtract_op_ts>>(out, a, b);
}
template <>
ndarray bland::cuda::subtract_cuda<ndarray_slice>(ndarray a, ndarray_slice b) {
    return subtract_cuda(a, ndarray(b));
}

// Multiplies...
template <> // Specialize for adding an array
ndarray bland::cuda::multiply_cuda<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper_cuda<elementwise_multiply_op_ts>>(out, a, b);
}
template <>
ndarray bland::cuda::multiply_cuda<ndarray_slice>(ndarray a, ndarray_slice b) {
    return multiply_cuda(a, ndarray(b));
}

// Divides...
template <> // Specialize for adding an array
ndarray bland::cuda::divide_cuda<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper_cuda<elementwise_divide_op_ts>>(out, a, b);
}
template <>
ndarray bland::cuda::divide_cuda<ndarray_slice>(ndarray a, ndarray_slice b) {
    return divide_cuda(a, ndarray(b));
}
