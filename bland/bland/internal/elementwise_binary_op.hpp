#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>
#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <numeric> // needed for accumulate
#include <stdexcept>

namespace bland {

/**
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
template <typename Out, typename A, typename B, class Op>
ndarray elementwise_binary_op(ndarray &out, const ndarray &a, const ndarray &b) {
    auto a_data = a.data_ptr<A>();
    auto b_data = b.data_ptr<B>();

    const auto a_offset   = a.offsets();
    const auto b_offset   = b.offsets();
    const auto out_offset = out.offsets();

    auto       out_data  = out.data_ptr<Out>();
    const auto out_shape = out.shape();

    const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
    const auto b_strides   = compute_broadcast_strides(b.shape(), b.strides(), out_shape);
    const auto out_strides = out.strides();
    // fmt::print("Inside the op, shapes: {} = {} + {}\n", out.shape(), a.shape(), b.shape());
    // fmt::print("Inside the op, strides: {} = {} + {}\n", out_strides, a_strides, b_strides);

    const auto a_shape = compute_broadcast_shape(a.shape(), out_shape);
    const auto b_shape = compute_broadcast_shape(b.shape(), out_shape);

    auto numel = out.numel();

    constexpr int64_t    LAST_DIM_UNROLL_FACTOR = 8;
    std::vector<int64_t> nd_index(out_shape.size(), 0);

    int64_t a_index   = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t b_index   = std::accumulate(b_offset.begin(), b_offset.end(), 0);
    int64_t out_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    int64_t       n               = 0;
    const auto    ndim            = out_shape.size();
    const int64_t out_ldim_stride = out_strides[ndim - 1];
    const int64_t a_ldim_stride   = a_strides[ndim - 1];
    const int64_t b_ldim_stride   = b_strides[ndim - 1];

    // Main loop
    int64_t unroll_size = LAST_DIM_UNROLL_FACTOR;
    for (; n < numel; n += unroll_size) {
        auto items_left_in_last_dim = (out_shape[ndim - 1]-1) - nd_index[ndim - 1];
        unroll_size                 = LAST_DIM_UNROLL_FACTOR;
        if (items_left_in_last_dim >= LAST_DIM_UNROLL_FACTOR) {
            out_data[out_index + 0 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 0 * a_ldim_stride], b_data[b_index + 0 * b_ldim_stride]);
            out_data[out_index + 1 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 1 * a_ldim_stride], b_data[b_index + 1 * b_ldim_stride]);
            out_data[out_index + 2 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 2 * a_ldim_stride], b_data[b_index + 2 * b_ldim_stride]);
            out_data[out_index + 3 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 3 * a_ldim_stride], b_data[b_index + 3 * b_ldim_stride]);
            out_data[out_index + 4 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 4 * a_ldim_stride], b_data[b_index + 4 * b_ldim_stride]);
            out_data[out_index + 5 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 5 * a_ldim_stride], b_data[b_index + 5 * b_ldim_stride]);
            out_data[out_index + 6 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 6 * a_ldim_stride], b_data[b_index + 6 * b_ldim_stride]);
            out_data[out_index + 7 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 7 * a_ldim_stride], b_data[b_index + 7 * b_ldim_stride]);
        } else {
            unroll_size = 1+items_left_in_last_dim;
            for (int j = 0; j < unroll_size; ++j) {
                out_data[out_index + j * out_ldim_stride] = Op::template call<Out, A, B>(
                        a_data[a_index + j * a_ldim_stride], b_data[b_index + j * b_ldim_stride]);
            }
        }

        // ***********************
        // ***********************
        // NOTE TO SELF
        // I think this indexing is wrong because it mismatches with the UNROLL_FACTOR loop increment
        // and at best its just confusing
        // ***********************
        // ***********************
        for (int i = ndim - 1; i >= 0; --i) {
            if (i == ndim - 1) {
                // This is unrolled
                nd_index[i] += unroll_size;
                a_index += unroll_size * a_strides[i];
                b_index += unroll_size * b_strides[i];
                out_index += unroll_size * out_strides[i];
            } else {
                // This is the standard not-unrolled case
                nd_index[i] += 1;
                a_index += a_strides[i];
                b_index += b_strides[i];
                out_index += out_strides[i];
            }
            if (nd_index[i] < out_shape[i]) {
                break;
            } else {
                // We've gone beyond the boundary, we'll already have done the work... so just reset
                a_index -= (a_shape[i]) * a_strides[i];
                b_index -= (b_shape[i]) * b_strides[i];
                out_index -= (out_shape[i]) * out_strides[i];
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
struct elementwise_binary_op_impl_wrapper {
    // An output tensor is provided
    template <typename Out, typename A_type, typename B_type, class Op>
    static inline ndarray call(ndarray out, const ndarray &a, const ndarray &b) {
        // Check that this operation is possible
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        // TODO: check/validate output shape!
        if (a.ndim() == b.ndim()) {
            for (int64_t dim = 0; dim < a.ndim(); ++dim) {
                if (a_shape[dim] != b_shape[dim] && a_shape[dim] != 1 && b_shape[dim] != 1) {
                    throw std::runtime_error(
                            "elementwise_binary_op: inputs match ndim but shapes are not compatible or broadcastable");
                }
            }
            // TODO: check if this can be broadcasted....
        }
        return elementwise_binary_op<Out, A_type, B_type, Op>(out, a, b);
    }
};

} // namespace bland
