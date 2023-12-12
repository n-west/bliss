
#include "bland/ndarray.hpp"
#include "bland/ops_arithmetic.hpp"

// #include "assignment_op.hpp"
#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"

#include "arithmetic_impl.hpp"


using namespace bland;


/*
 * Externally exposed function implementations
 */

// Adds...

template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::add(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_new3<scalar_op_impl_wrapper, T, elementwise_add_op_ts>(out, a, b);
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


// Subtracts...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::subtract(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_new3<scalar_op_impl_wrapper, T, elementwise_subtract_op_ts>(out, a, b);
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

// Multiplies...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::multiply(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_new3<scalar_op_impl_wrapper, T, elementwise_multiply_op_ts>(out, a, b);
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

// Divides...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::divide(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_new3<scalar_op_impl_wrapper, T, elementwise_divide_op_ts>(out, a, b);
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
