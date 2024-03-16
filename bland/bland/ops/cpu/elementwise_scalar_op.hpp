#pragma once

#include "bland/ndarray.hpp"
#include "internal/shape_helpers.hpp"

#include <typeinfo>
#include <numeric> // accumulate

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
template <typename Out, typename A, typename B, class Op>
ndarray elementwise_scalar_op(ndarray out, const ndarray &a, const B &b) {
    auto       a_data    = a.data_ptr<A>();
    auto       a_shape   = a.shape();
    auto       a_strides = a.strides();
    const auto a_offset  = a.offsets();

    auto       out_data    = out.data_ptr<Out>();
    auto       out_shape   = out.shape();
    auto       out_strides = out.strides();
    const auto out_offset  = out.offsets();

    // Current (multi-dimensional) index to read into a, out
    std::vector<int64_t> nd_index(out_shape.size(), 0);

    int64_t a_linear_index   = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t out_linear_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // Finally... do the actual op
        out_data[out_linear_index] = Op::template call<Out, A, B>(a_data[a_linear_index], b);

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

template <typename Out, typename A, typename B, class Op>
ndarray elementwise_scalar_op_fast(ndarray out, const ndarray &a, const B &b) {
    auto       a_data    = a.data_ptr<A>();
    auto       a_shape   = a.shape();
    auto       a_strides = a.strides();
    const auto a_offset  = a.offsets();

    // TODO: it's possible we're broadcasting a in to out, so need to compute_broadcasat_shape/strides
    auto       out_data    = out.data_ptr<Out>();
    auto       out_shape   = out.shape();
    auto       out_strides = out.strides();
    const auto out_offset  = out.offsets();

    constexpr int64_t LAST_DIM_UNROLL_FACTOR = 8;
    // Current (multi-dimensional) index to read into a, out
    std::vector<int64_t> nd_index(out_shape.size(), 0);

    int64_t a_index   = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t out_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    const auto    ndim            = out_shape.size();
    int64_t       unroll_dim      = ndim - 1;
    int64_t       unroll_size     = LAST_DIM_UNROLL_FACTOR;
    const int64_t out_ldim_stride = out_strides[unroll_dim];
    const int64_t a_ldim_stride   = a_strides[unroll_dim];

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        auto items_left_in_last_dim = (out_shape[unroll_dim] - 1) - nd_index[unroll_dim];
        unroll_size                 = LAST_DIM_UNROLL_FACTOR;
        if (items_left_in_last_dim >= LAST_DIM_UNROLL_FACTOR) {
            out_data[out_index + 0 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 0 * a_ldim_stride], b);
            out_data[out_index + 1 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 1 * a_ldim_stride], b);
            out_data[out_index + 2 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 2 * a_ldim_stride], b);
            out_data[out_index + 3 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 3 * a_ldim_stride], b);
            out_data[out_index + 4 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 4 * a_ldim_stride], b);
            out_data[out_index + 5 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 5 * a_ldim_stride], b);
            out_data[out_index + 6 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 6 * a_ldim_stride], b);
            out_data[out_index + 7 * out_ldim_stride] = Op::template call<Out, A, B>(
                    a_data[a_index + 7 * a_ldim_stride], b);
        } else {
            unroll_size = 1 + items_left_in_last_dim;
            for (int j = 0; j < unroll_size; ++j) {
                out_data[out_index + j * out_ldim_stride] = Op::template call<Out, A, B>(
                        a_data[a_index + j * a_ldim_stride], b);
            }
        }

        for (int dim = ndim - 1; dim >= 0; --dim) {
            if (dim == unroll_dim) {
                // This is unrolled
                nd_index[dim] += unroll_size;
                a_index += unroll_size * a_strides[dim];
                out_index += unroll_size * out_strides[dim];
            } else {
                // This is the standard not-unrolled case
                nd_index[dim] += 1;
                a_index += a_strides[dim];
                out_index += out_strides[dim];
            }
            if (nd_index[dim] < out_shape[dim]) {
                break;
            } else {
                // We've gone beyond the boundary, we'll already have done the work... so just reset
                a_index -= (a_shape[dim]) * a_strides[dim];
                out_index -= (out_shape[dim]) * out_strides[dim];
                nd_index[dim] = 0;
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

    template <typename Out, typename A_type, typename B_type, class Op>
    static inline ndarray call(ndarray out, const ndarray &a, const B_type &b) {
        return elementwise_scalar_op<Out, A_type, B_type, Op>(out, a, b);
    }

};

} // namespace bland