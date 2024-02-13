#include "arithmetic_cpu.hpp"

#include "bland/ndarray.hpp"
#include "bland/ops_arithmetic.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"

// TODO: will eventually want to move this to this directory
#include "arithmetic_cpu_impl.hpp"


using namespace bland;
using namespace bland::cpu;

/*
 * Externally exposed function implementations
 */

// Adds...
template <> // Specialize for adding an array
ndarray bland::cpu::add_cpu<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper<elementwise_add_op_ts>>(out, a, b);
}

template <>
ndarray bland::cpu::add_cpu<ndarray_slice>(ndarray a, ndarray_slice b) {
    return add_cpu(a, ndarray(b));
}

// Subtracts...
template <> // Specialize for adding an array
ndarray bland::cpu::subtract_cpu<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper<elementwise_subtract_op_ts>>(out, a, b);
}
template <>
ndarray bland::cpu::subtract_cpu<ndarray_slice>(ndarray a, ndarray_slice b) {
    return subtract_cpu(a, ndarray(b));
}

// Multiplies...
template <> // Specialize for adding an array
ndarray bland::cpu::multiply_cpu<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper<elementwise_multiply_op_ts>>(out, a, b);
}
template <>
ndarray bland::cpu::multiply_cpu<ndarray_slice>(ndarray a, ndarray_slice b) {
    return multiply_cpu(a, ndarray(b));
}

// Divides...
template <> // Specialize for adding an array
ndarray bland::cpu::divide_cpu<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper<elementwise_divide_op_ts>>(out, a, b);
}
template <>
ndarray bland::cpu::divide_cpu<ndarray_slice>(ndarray a, ndarray_slice b) {
    return divide_cpu(a, ndarray(b));
}
