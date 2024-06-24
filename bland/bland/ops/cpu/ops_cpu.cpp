
#include "ops_cpu.hpp"

#include "arithmetic_cpu_impl.hpp"
#include "elementwise_unary_op.hpp"
#include "assignment_op.hpp"

#include "internal/dispatcher.hpp"

#include "bland/ndarray.hpp"


using namespace bland;
using namespace bland::cpu;


ndarray bland::cpu::copy(ndarray a, ndarray &out) {
    fmt::print("cpu copy\n");
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_copy_op>(out, a);
}

ndarray bland::cpu::square(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_square_op>(out, a);
}

ndarray bland::cpu::sqrt(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_sqrt_op>(out, a);
}

ndarray bland::cpu::abs(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_abs_op>(out, a);
}

template <typename T>
ndarray bland::cpu::fill(ndarray out, T value) {
    return dispatch_new<cpu::assignment_op_impl_wrapper, T>(out, value);
}

template ndarray bland::cpu::fill<float>(ndarray out, float v);
// template ndarray bland::cpu::fill<double>(ndarray out, double v);
// template ndarray bland::cpu::fill<int8_t>(ndarray out, int8_t v);
// template ndarray bland::cpu::fill<int16_t>(ndarray out, int16_t v);
template ndarray bland::cpu::fill<int32_t>(ndarray out, int32_t v);
// template ndarray bland::cpu::fill<int64_t>(ndarray out, int64_t v);
template ndarray bland::cpu::fill<uint8_t>(ndarray out, uint8_t v);
// template ndarray bland::cpu::fill<uint16_t>(ndarray out, uint16_t v);
template ndarray bland::cpu::fill<uint32_t>(ndarray out, uint32_t v);
// template ndarray bland::cpu::fill<uint64_t>(ndarray out, uint64_t v);
