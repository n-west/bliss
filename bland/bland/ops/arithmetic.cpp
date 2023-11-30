
#include "bland/ndarray.hpp"
#include "bland/ops.hpp"

#include "assignment_op.hpp"
#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"
#include "elementwise_unary_op.hpp"
#include "shape_helpers.hpp"

#include <cmath> // std::sqrt, std::pow

using namespace bland;

// Elementwise operations
template <typename A, typename B>
struct elementwise_add_op {
    static inline A call(const A &a, const B &b) { return a + b; }
};

template <typename A, typename B>
struct elementwise_subtract_op {
    static inline A call(const A &a, const B &b) { return a - b; }
};

template <typename A, typename B>
struct elementwise_multiply_op {
    static inline A call(const A &a, const B &b) { return a * b; }
};

template <typename A, typename B>
struct elementwise_divide_op {
    static inline A call(const A &a, const B &b) { return a / b; }
};

/*
 * Externally exposed function implementations
 */

// Adds...
template <> // Specialize for adding an array
ndarray bland::add<ndarray>(ndarray a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}
template <> // Specialize for adding an array_slice
ndarray bland::add<ndarray_slice>(ndarray a, ndarray_slice b) {
    // This needing to exist is the glories of the c++ type system
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::add(ndarray a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_add_op>(a, b);
}
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

// TODO: is this ill-formed adding two slices? What is the returned slice a slice of?
template <>
ndarray_slice bland::add<ndarray_slice>(ndarray_slice a, ndarray_slice b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}
template <>
ndarray_slice bland::add<ndarray>(ndarray_slice a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}

template <typename T>
ndarray_slice bland::add(ndarray_slice a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_add_op>(a, b);
}

template ndarray_slice bland::add<float>(ndarray_slice a, float b);
template ndarray_slice bland::add<double>(ndarray_slice a, double b);
template ndarray_slice bland::add<int8_t>(ndarray_slice a, int8_t b);
template ndarray_slice bland::add<int16_t>(ndarray_slice a, int16_t b);
template ndarray_slice bland::add<int32_t>(ndarray_slice a, int32_t b);
template ndarray_slice bland::add<int64_t>(ndarray_slice a, int64_t b);
template ndarray_slice bland::add<uint8_t>(ndarray_slice a, uint8_t b);
template ndarray_slice bland::add<uint16_t>(ndarray_slice a, uint16_t b);
template ndarray_slice bland::add<uint32_t>(ndarray_slice a, uint32_t b);
template ndarray_slice bland::add<uint64_t>(ndarray_slice a, uint64_t b);

// Subtracts...
template <> // Specialize for adding an array
ndarray bland::subtract<ndarray>(ndarray a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_subtract_op>(a, b);
}
template <> // Specialize for adding an array_slice
ndarray bland::subtract<ndarray_slice>(ndarray a, ndarray_slice b) {
    // This needing to exist is the glories of the c++ type system
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}

template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::subtract(ndarray a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_subtract_op>(a, b);
}
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

// TODO: is this ill-formed subtracting two slices? What is the returned slice a slice of?
template <>
ndarray_slice bland::subtract<ndarray_slice>(ndarray_slice a, ndarray_slice b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_subtract_op>(a, b);
}
template <>
ndarray_slice bland::subtract<ndarray>(ndarray_slice a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}

template <typename T>
ndarray_slice bland::subtract(ndarray_slice a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_subtract_op>(a, b);
}

template ndarray_slice bland::subtract<float>(ndarray_slice a, float b);
template ndarray_slice bland::subtract<double>(ndarray_slice a, double b);
template ndarray_slice bland::subtract<int8_t>(ndarray_slice a, int8_t b);
template ndarray_slice bland::subtract<int16_t>(ndarray_slice a, int16_t b);
template ndarray_slice bland::subtract<int32_t>(ndarray_slice a, int32_t b);
template ndarray_slice bland::subtract<int64_t>(ndarray_slice a, int64_t b);
template ndarray_slice bland::subtract<uint8_t>(ndarray_slice a, uint8_t b);
template ndarray_slice bland::subtract<uint16_t>(ndarray_slice a, uint16_t b);
template ndarray_slice bland::subtract<uint32_t>(ndarray_slice a, uint32_t b);
template ndarray_slice bland::subtract<uint64_t>(ndarray_slice a, uint64_t b);

// Multiplies...
template <> // Specialize for adding an array
ndarray bland::multiply<ndarray>(ndarray a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op>(a, b);
}
template <> // Specialize for adding a slice
ndarray bland::multiply<ndarray_slice>(ndarray a, ndarray_slice b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op>(a, b);
}
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::multiply(ndarray a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_multiply_op>(a, b);
}

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

// ndarray_slice bland::multiply(ndarray a, ndarray_slice b) {
//     // This needing to exist is the glories of the c++ type system
//     return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op>(a, b);
// }

// TODO: is this ill-formed multiplying two slices? What is the returned slice a slice of?
template <>
ndarray_slice bland::multiply<ndarray_slice>(ndarray_slice a, ndarray_slice b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op>(a, b);
}
template <>
ndarray_slice bland::multiply<ndarray>(ndarray_slice a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_multiply_op>(a, b);
}

template <typename T>
ndarray_slice bland::multiply(ndarray_slice a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_multiply_op>(a, b);
}

template ndarray_slice bland::multiply<float>(ndarray_slice a, float b);
template ndarray_slice bland::multiply<double>(ndarray_slice a, double b);
template ndarray_slice bland::multiply<int8_t>(ndarray_slice a, int8_t b);
template ndarray_slice bland::multiply<int16_t>(ndarray_slice a, int16_t b);
template ndarray_slice bland::multiply<int32_t>(ndarray_slice a, int32_t b);
template ndarray_slice bland::multiply<int64_t>(ndarray_slice a, int64_t b);
template ndarray_slice bland::multiply<uint8_t>(ndarray_slice a, uint8_t b);
template ndarray_slice bland::multiply<uint16_t>(ndarray_slice a, uint16_t b);
template ndarray_slice bland::multiply<uint32_t>(ndarray_slice a, uint32_t b);
template ndarray_slice bland::multiply<uint64_t>(ndarray_slice a, uint64_t b);

// Divides...
template <> // Specialize for adding an array
ndarray bland::divide<ndarray>(ndarray a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_divide_op>(a, b);
}
template <> // Specialize for adding an array_slice
ndarray bland::divide<ndarray_slice>(ndarray a, ndarray_slice b) {
    // This needing to exist is the glories of the c++ type system
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_add_op>(a, b);
}

template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::divide(ndarray a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_divide_op>(a, b);
}
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

// TODO: is this ill-formed dividing two slices? What is the returned slice a slice of?
template <>
ndarray_slice bland::divide<ndarray_slice>(ndarray_slice a, ndarray_slice b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_divide_op>(a, b);
}
template <>
ndarray_slice bland::divide<ndarray>(ndarray_slice a, ndarray b) {
    return dispatch<elementwise_binary_op_impl_wrapper, elementwise_divide_op>(a, b);
}

template <typename T>
ndarray_slice bland::divide(ndarray_slice a, T b) {
    return dispatch<scalar_op_impl_wrapper, T, elementwise_divide_op>(a, b);
}

template ndarray_slice bland::divide<float>(ndarray_slice a, float b);
template ndarray_slice bland::divide<double>(ndarray_slice a, double b);
template ndarray_slice bland::divide<int8_t>(ndarray_slice a, int8_t b);
template ndarray_slice bland::divide<int16_t>(ndarray_slice a, int16_t b);
template ndarray_slice bland::divide<int32_t>(ndarray_slice a, int32_t b);
template ndarray_slice bland::divide<int64_t>(ndarray_slice a, int64_t b);
template ndarray_slice bland::divide<uint8_t>(ndarray_slice a, uint8_t b);
template ndarray_slice bland::divide<uint16_t>(ndarray_slice a, uint16_t b);
template ndarray_slice bland::divide<uint32_t>(ndarray_slice a, uint32_t b);
template ndarray_slice bland::divide<uint64_t>(ndarray_slice a, uint64_t b);

template <typename A>
struct elementwise_square_op {
    static inline A call(const A &a) { return a * a; }
};

ndarray bland::square(ndarray a) {
    return dispatch<unary_op_impl_wrapper, elementwise_square_op>(a);
}

template <typename A>
struct elementwise_sqrt_op {
    static inline A call(const A &a) { return std::sqrt(a); }
};

ndarray bland::sqrt(ndarray a) {
    return dispatch<unary_op_impl_wrapper, elementwise_sqrt_op>(a);
}
