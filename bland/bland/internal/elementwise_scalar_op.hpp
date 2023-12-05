#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <typeinfo>
// #include <fmt/core.h>
// #include <iostream>

namespace bland {
struct ndarray;

/**
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
template <typename A, typename B, template <typename, typename> class Op>
ndarray elementwise_scalar_op(const ndarray &a, const B &b, ndarray out) {
    auto a_data = a.data_ptr<A>();
    auto a_shape = a.shape();
    auto a_strides = a.strides();
    const auto a_offset = a.offsets();

    auto    out_data = out.data_ptr<A>();
    auto out_shape = out.shape();
    auto out_strides = out.strides();
    const auto out_offset = out.offsets();

    // Current (multi-dimensional) index to read into a, out
    std::vector<int64_t> nd_index(out_shape.size(), 0);

    int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t out_linear_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // Finally... do the actual op
        out_data[out_linear_index] = Op<A, B>::call(a_data[a_linear_index], b);

        // Increment the multi-dimensional index
        for (int i = out_shape.size() - 1; i >= 0; --i) {
            // If we're not at the end of this dim, keep going
            if (++nd_index[i] != out_shape[i]) {
                a_linear_index += a_strides[i];
                out_linear_index += out_strides[i];
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                a_linear_index -= (a_shape[i] - 1) * a_strides[i];
                out_linear_index -= (out_shape[i] - 1) * out_strides[i];
                nd_index[i] = 0;
            }
        }
    }

    return out;
}

/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
struct scalar_op_impl_wrapper {

    template <typename A_type, typename B_type, template <typename, typename> class Op>
    static inline ndarray call(const ndarray &a, const B_type &b, ndarray out) {
        return elementwise_scalar_op<A_type, B_type, Op>(a, b, out);
    }

    template <typename A_type, typename B_type, template <typename, typename> class Op>
    static inline ndarray call(const ndarray &a, const B_type &b) {
        // Create output array with appropriate shape and data type
        // TODO: figure out type promotion for the return type
        ndarray out(a.shape(), a.dtype(), a.device());

        return elementwise_scalar_op<A_type, B_type, Op>(a, b, out);
    }
};


}