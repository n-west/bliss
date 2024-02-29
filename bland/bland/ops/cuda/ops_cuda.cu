
#include "ops_cuda.cuh"

#include "arithmetic_cuda_impl.cuh"
#include "elementwise_unary_op.cuh"
#include "dispatcher.hpp"

#include "bland/ndarray.hpp"

#include <fmt/format.h>


using namespace bland;
using namespace bland::cuda;


ndarray bland::cuda::copy(ndarray a, ndarray &out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_copy_op>(out, a);
}

ndarray bland::cuda::square(ndarray a, ndarray& out) {
    fmt::print("in the cuda square(ndarray)\n");
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_square_op>(out, a);
}

ndarray bland::cuda::sqrt(ndarray a, ndarray& out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_sqrt_op>(out, a);
}

ndarray bland::cuda::abs(ndarray a, ndarray& out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_abs_op>(out, a);
}
