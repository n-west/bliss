
#include "ops_cpu.hpp"

#include "arithmetic_cpu_impl.hpp"
#include "elementwise_unary_op.hpp"
#include "dispatcher.hpp"

#include "bland/ndarray.hpp"


using namespace bland;
using namespace bland::cpu;


ndarray bland::cpu::copy(ndarray a, ndarray &out) {
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
