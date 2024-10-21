
#include "bland/ndarray.hpp"
// #include "bland/ops_arithmetic.hpp"

#include "arithmetic_cpu.hpp"

#include "internal/dispatcher.hpp"
#include "elementwise_scalar_op.hpp"
#include "arithmetic_cpu_impl.hpp"


using namespace bland;
using namespace bland::cpu;

/*
 * Externally exposed function implementations
 */

// Adds...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cpu::add_cpu(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper, T, elementwise_add_op_ts>(out, a, b);
}
template ndarray bland::cpu::add_cpu<float>(ndarray a, float b);
// template ndarray bland::cpu::add_cpu<double>(ndarray a, double b);
// template ndarray bland::cpu::add_cpu<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cpu::add_cpu<int16_t>(ndarray a, int16_t b);
template ndarray bland::cpu::add_cpu<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cpu::add_cpu<int64_t>(ndarray a, int64_t b);
template ndarray bland::cpu::add_cpu<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cpu::add_cpu<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cpu::add_cpu<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cpu::add_cpu<uint64_t>(ndarray a, uint64_t b);


// Subtracts...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cpu::subtract_cpu(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper, T, elementwise_subtract_op_ts>(out, a, b);
}
template ndarray bland::cpu::subtract_cpu<float>(ndarray a, float b);
// template ndarray bland::cpu::subtract_cpu<double>(ndarray a, double b);
// template ndarray bland::cpu::subtract_cpu<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cpu::subtract_cpu<int16_t>(ndarray a, int16_t b);
template ndarray bland::cpu::subtract_cpu<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cpu::subtract_cpu<int64_t>(ndarray a, int64_t b);
template ndarray bland::cpu::subtract_cpu<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cpu::subtract_cpu<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cpu::subtract_cpu<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cpu::subtract_cpu<uint64_t>(ndarray a, uint64_t b);

// Multiplies...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cpu::multiply_cpu(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper, T, elementwise_multiply_op_ts>(out, a, b);
}

template ndarray bland::cpu::multiply_cpu<float>(ndarray a, float b);
// template ndarray bland::cpu::multiply_cpu<double>(ndarray a, double b);
// template ndarray bland::cpu::multiply_cpu<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cpu::multiply_cpu<int16_t>(ndarray a, int16_t b);
template ndarray bland::cpu::multiply_cpu<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cpu::multiply_cpu<int64_t>(ndarray a, int64_t b);
template ndarray bland::cpu::multiply_cpu<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cpu::multiply_cpu<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cpu::multiply_cpu<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cpu::multiply_cpu<uint64_t>(ndarray a, uint64_t b);

// Divides...
template <typename T> // the scalar case (explicitly instantiated below)
ndarray bland::cpu::divide_cpu(ndarray a, T b) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return dispatch_nd_sc<scalar_op_impl_wrapper, T, elementwise_divide_op_ts>(out, a, b);
}
template ndarray bland::cpu::divide_cpu<float>(ndarray a, float b);
// template ndarray bland::cpu::divide_cpu<double>(ndarray a, double b);
// template ndarray bland::cpu::divide_cpu<int8_t>(ndarray a, int8_t b);
// template ndarray bland::cpu::divide_cpu<int16_t>(ndarray a, int16_t b);
template ndarray bland::cpu::divide_cpu<int32_t>(ndarray a, int32_t b);
// template ndarray bland::cpu::divide_cpu<int64_t>(ndarray a, int64_t b);
template ndarray bland::cpu::divide_cpu<uint8_t>(ndarray a, uint8_t b);
// template ndarray bland::cpu::divide_cpu<uint16_t>(ndarray a, uint16_t b);
template ndarray bland::cpu::divide_cpu<uint32_t>(ndarray a, uint32_t b);
// template ndarray bland::cpu::divide_cpu<uint64_t>(ndarray a, uint64_t b);
