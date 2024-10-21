#include "comparison_cuda.cuh"

#include "bland/ndarray.hpp"
#include "bland/ndarray_slice.hpp"

#include "internal/dispatcher.hpp"
#include "elementwise_binary_op_cuda.cuh"
#include "elementwise_scalar_op_cuda.cuh"

#include "comparison_cuda_impl.cuh"

using namespace bland;
using namespace bland::cuda;

/**
 * greater than
*/
template <typename L, typename R>
ndarray bland::cuda::greater_than(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, L, greater_than_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, R, greater_than_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::None, elementwise_binary_op_impl_wrapper_cuda<greater_than_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::greater_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::greater_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cuda::greater_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::greater_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::greater_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::greater_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::greater_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::greater_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::greater_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::greater_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::greater_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::greater_than<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cuda::greater_than<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::greater_than<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::greater_than<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::greater_than<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than<int64_t, ndarray>(int64_t lhs, ndarray rhs);

template ndarray bland::cuda::greater_than<uint8_t, ndarray_slice>(uint8_t lhs, ndarray_slice rhs);
template ndarray bland::cuda::greater_than<ndarray_slice, uint8_t>(ndarray_slice lhs, uint8_t rhs);


/**
 * Greater than equal to
 */
template <typename L, typename R>
ndarray bland::cuda::greater_than_equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, L, greater_than_equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, R, greater_than_equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::None, elementwise_binary_op_impl_wrapper_cuda<greater_than_equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::greater_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::greater_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cuda::greater_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::greater_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::greater_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::greater_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::greater_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::greater_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::greater_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::greater_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::greater_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::greater_than_equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cuda::greater_than_equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::greater_than_equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than_equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::greater_than_equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than_equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than_equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than_equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::greater_than_equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::greater_than_equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than
 */
template <typename L, typename R>
ndarray bland::cuda::less_than(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, L, less_than_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, R, less_than_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::None, elementwise_binary_op_impl_wrapper_cuda<less_than_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::less_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::less_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cuda::less_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::less_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::less_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::less_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::less_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::less_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::less_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::less_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::less_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::less_than<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cuda::less_than<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::less_than<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::less_than<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::less_than<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than or equal to
 */
template <typename L, typename R>
ndarray bland::cuda::less_than_equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, L, less_than_equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, R, less_than_equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::None, elementwise_binary_op_impl_wrapper_cuda<less_than_equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::less_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::less_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cuda::less_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::less_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::less_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::less_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::less_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::less_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::less_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::less_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::less_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::less_than_equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cuda::less_than_equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::less_than_equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than_equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::less_than_equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than_equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than_equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than_equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::less_than_equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::less_than_equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * logical and (&)
 */
template <typename L, typename R>
ndarray bland::cuda::logical_and(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        //static_assert(!std::is_floating_point_v<L>, "Cannot perform logical_and (&) with floating point array");
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::NoFloat, scalar_op_impl_wrapper_cuda, uint8_t, L, logical_and_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        //static_assert(!std::is_floating_point_v<R>, "Cannot perform logical_and (&) with floating point array");
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::NoFloat, scalar_op_impl_wrapper_cuda, uint8_t, R, logical_and_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::NoFloat, elementwise_binary_op_impl_wrapper_cuda<logical_and_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::logical_and<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::logical_and<ndarray, double>(ndarray lhs, double rhs);
// template ndarray bland::cuda::logical_and<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::logical_and<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::logical_and<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::logical_and<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::logical_and<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::logical_and<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::logical_and<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::logical_and<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::logical_and<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::logical_and<double, ndarray>(double lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::logical_and<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::logical_and<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::logical_and<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::logical_and<int64_t, ndarray>(int64_t lhs, ndarray rhs);

template ndarray bland::cuda::logical_and<uint8_t, ndarray_slice>(uint8_t lhs, ndarray_slice rhs);
template ndarray bland::cuda::logical_and<ndarray_slice, uint8_t>(ndarray_slice lhs, uint8_t rhs);

/**
 * equal_to
 */
template <typename L, typename R>
ndarray bland::cuda::equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, L, equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch_nd_sc<Constraints::None, scalar_op_impl_wrapper_cuda, uint8_t, R, equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return constrained_dispatch<Constraints::None, elementwise_binary_op_impl_wrapper_cuda<equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cuda::equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::cuda::equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cuda::equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cuda::equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::cuda::equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cuda::equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::cuda::equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::cuda::equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::cuda::equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cuda::equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::cuda::equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::cuda::equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cuda::equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cuda::equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::cuda::equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cuda::equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::cuda::equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::cuda::equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::cuda::equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cuda::equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::cuda::equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);

