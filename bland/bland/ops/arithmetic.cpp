
#include "bland/ndarray.hpp"
#include "bland/ops_arithmetic.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"

#include "arithmetic_impl.hpp"


using namespace bland;


/*
 * Externally exposed function implementations
 */

// Adds...
template <> // Specialize for adding an array
ndarray bland::add<ndarray>(ndarray a, ndarray b) {
    auto out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
    // fmt::print("In this add... a_shape={} + b_shape={} = out_shape {}\n", a.shape(), b.shape(), out_shape);
    auto out = ndarray(out_shape, a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op_ts>(out, a, b);
}

template <>
ndarray bland::add<ndarray_slice>(ndarray a, ndarray_slice b) {
    return add(a, ndarray(b));
}

// Subtracts...
template <> // Specialize for adding an array
ndarray bland::subtract<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_subtract_op_ts>(out, a, b);
}
template <>
ndarray bland::subtract<ndarray_slice>(ndarray a, ndarray_slice b) {
    return subtract(a, ndarray(b));
}

// Multiplies...
template <> // Specialize for adding an array
ndarray bland::multiply<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op_ts>(out, a, b);
}
template <>
ndarray bland::multiply<ndarray_slice>(ndarray a, ndarray_slice b) {
    return multiply(a, ndarray(b));
}

// Divides...
template <> // Specialize for adding an array
ndarray bland::divide<ndarray>(ndarray a, ndarray b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_divide_op_ts>(out, a, b);
}
template <>
ndarray bland::divide<ndarray_slice>(ndarray a, ndarray_slice b) {
    return divide(a, ndarray(b));
}
