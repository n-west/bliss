#include "comparison_cpu.hpp"

#include "bland/ndarray.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"

#include "comparison_cpu_impl.hpp"

using namespace bland;
using namespace bland::cpu;

/**
 * greater than
*/
template <typename L, typename R>
ndarray bland::cpu::greater_than(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, greater_than_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, greater_than_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<greater_than_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::greater_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::greater_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::greater_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::greater_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::greater_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::greater_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::greater_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::greater_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::greater_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::greater_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::greater_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::greater_than<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Greater than equal to
 */
template <typename L, typename R>
ndarray bland::cpu::greater_than_equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, greater_than_equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, greater_than_equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<greater_than_equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::greater_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::greater_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::greater_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::greater_than_equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::greater_than_equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than
 */
template <typename L, typename R>
ndarray bland::cpu::less_than(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, less_than_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, less_than_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<less_than_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::less_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::less_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::less_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::less_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::less_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::less_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::less_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::less_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::less_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::less_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::less_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::less_than<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::less_than<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::less_than<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than or equal to
 */
template <typename L, typename R>
ndarray bland::cpu::less_than_equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, less_than_equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, less_than_equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<less_than_equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::less_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::less_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::less_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::less_than_equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::less_than_equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * logical and (&)
 */
template <typename L, typename R>
ndarray bland::cpu::logical_and(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, logical_and_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, logical_and_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<logical_and_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::logical_and<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::logical_and<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::logical_and<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::logical_and<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::logical_and<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::logical_and<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::logical_and<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::logical_and<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::logical_and<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::logical_and<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::logical_and<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::logical_and<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::logical_and<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * equal to
 */
template <typename L, typename R>
ndarray bland::cpu::equal_to(L lhs, R rhs) {
    if constexpr (std::is_arithmetic<L>::value) {
        auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, L, equal_to_impl>(out, rhs, lhs);
    } else if constexpr (std::is_arithmetic<R>::value) {
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch_new3<scalar_op_impl_wrapper, uint8_t, R, equal_to_impl>(out, lhs, rhs);
    } else {
        // TODO: more sanity checking...
        auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
        return dispatch<elementwise_binary_op_impl_wrapper<equal_to_impl>, uint8_t>(out, lhs, rhs);
    }
}

// Explicit instantiations
template ndarray bland::cpu::equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

template ndarray bland::cpu::equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::cpu::equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::cpu::equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::cpu::equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::cpu::equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::cpu::equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::cpu::equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::cpu::equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::cpu::equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::cpu::equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::cpu::equal_to<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<int8_t, ndarray>(int8_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<int32_t, ndarray>(int32_t lhs, ndarray rhs);
template ndarray bland::cpu::equal_to<int64_t, ndarray>(int64_t lhs, ndarray rhs);


struct count_impl {
    template <typename in_datatype>
    static inline int64_t call(const ndarray &a) {

        auto a_data    = a.data_ptr<in_datatype>();
        auto a_shape   = a.shape();
        auto a_strides = a.strides();
        auto a_offset  = a.offsets();

        std::vector<int64_t> input_index(a_shape.size(), 0);
        int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);

        int64_t count = 0;
        auto    numel = a.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            if (a_data[a_linear_index]) {
                ++count;
            }

            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int dim = a_shape.size() - 1; dim >= 0; --dim) {
                // If we're not at the end of this dim, keep going
                ++input_index[dim];
                a_linear_index += a_strides[dim];
                if (input_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    input_index[dim] = 0;
                    a_linear_index -= (a_shape[dim]) * a_strides[dim];
                }
            }
        }

        return count;
    }
};

int64_t bland::cpu::count_true(ndarray x) {
    return dispatch_summary<count_impl>(x);
}
