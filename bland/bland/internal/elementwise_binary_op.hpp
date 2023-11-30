#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <numeric>

namespace bland {
struct ndarray;

namespace optimize_detail {

template <typename A,
          typename B,
          template <typename, typename>
          class Op,
          int64_t LAST_DIM_UNROLL_FACTOR = 32,
          int64_t A_LDIM_STRIDE          = 0,
          int64_t B_LDIM_STRIDE          = 0,
          int64_t OUT_LDIM_STRIDE        = 0>
struct elementwise_binary_op {
    static void apply(A                          *out_data,
                      A                          *a_data,
                      B                          *b_data,
                      const std::vector<int64_t> &out_shape,
                      const std::vector<int64_t> &out_strides,
                      const std::vector<int64_t> &out_offset,
                      const std::vector<int64_t> &a_shape,
                      const std::vector<int64_t> &a_strides,
                      const std::vector<int64_t> &a_offset,
                      const std::vector<int64_t> &b_shape,
                      const std::vector<int64_t> &b_strides,
                      const std::vector<int64_t> &b_offset,
                      const int64_t               numel) {
        // Current (multi-dimensional) index to read into a, b, out
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
        for (; n < numel; n += LAST_DIM_UNROLL_FACTOR) {
            auto    items_left_in_last_dim = out_shape[ndim - 1] - nd_index[ndim - 1];
            int64_t unroll_size            = LAST_DIM_UNROLL_FACTOR;
            if (items_left_in_last_dim >= LAST_DIM_UNROLL_FACTOR) {
                // for (int j = 0; j < LAST_DIM_UNROLL_FACTOR; ++j) {
                    out_data[out_index + 0 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 0 * a_ldim_stride], b_data[b_index + 0 * b_ldim_stride]);
                    out_data[out_index + 1 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 1 * a_ldim_stride], b_data[b_index + 1 * b_ldim_stride]);
                    out_data[out_index + 2 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 2 * a_ldim_stride], b_data[b_index + 2 * b_ldim_stride]);
                    out_data[out_index + 3 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 3 * a_ldim_stride], b_data[b_index + 3 * b_ldim_stride]);
                    out_data[out_index + 4 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 4 * a_ldim_stride], b_data[b_index + 4 * b_ldim_stride]);
                    out_data[out_index + 5 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 5 * a_ldim_stride], b_data[b_index + 5 * b_ldim_stride]);
                    out_data[out_index + 6 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 6 * a_ldim_stride], b_data[b_index + 6 * b_ldim_stride]);
                    out_data[out_index + 7 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 7 * a_ldim_stride], b_data[b_index + 7 * b_ldim_stride]);
                // }
            } else {
                unroll_size = items_left_in_last_dim;
                for (int j = 0; j < items_left_in_last_dim; ++j) {
                    out_data[out_index + j * out_ldim_stride] =
                            Op<A, B>::call(a_data[a_index + j * a_ldim_stride], b_data[b_index + j * b_ldim_stride]);
                }
            }

            // Increment the multi-dimensional index
            for (int i = ndim - 1; i >= 0; --i) {
                if (i == ndim - 1) {
                    nd_index[i] += unroll_size;
                    a_index += unroll_size * a_strides[i];
                    b_index += unroll_size * b_strides[i];
                    out_index += unroll_size * out_strides[i];
                } else {
                    nd_index[i] += 1;
                    a_index += a_strides[i];
                    b_index += b_strides[i];
                    out_index += out_strides[i];
                }

                if (nd_index[i] < out_shape[i]) {
                    break;
                } else if (i != ndim - 1 || n + LAST_DIM_UNROLL_FACTOR >= numel) {
                    // Reset the linear index for the current dimension
                    if (i == ndim - 1) {
                        a_index -= (unroll_size - 1) * a_strides[i];
                        b_index -= (unroll_size - 1) * b_strides[i];
                        out_index -= (unroll_size - 1) * out_strides[i];
                    } else {
                        a_index -= (a_shape[i] - 1) * a_strides[i];
                        b_index -= (b_shape[i] - 1) * b_strides[i];
                        out_index -= (out_shape[i] - 1) * out_strides[i];
                    }
                    nd_index[i] = 0;
                }
            }
        }
    }
};

// The specialization on stride==1 doesn't seem to help as much as I thought, maybe delete in favor of simplicity
// There is also a cpu governor issue which shows up when I am on battery power
template <typename A, typename B, template <typename, typename> class Op, int64_t LAST_DIM_UNROLL_FACTOR>
struct elementwise_binary_op<A, B, Op, LAST_DIM_UNROLL_FACTOR, 1, 1, 1> {
    static void apply(A                          *out_data,
                      A                          *a_data,
                      B                          *b_data,
                      const std::vector<int64_t> &out_shape,
                      const std::vector<int64_t> &out_strides,
                      const std::vector<int64_t> &out_offset,
                      const std::vector<int64_t> &a_shape,
                      const std::vector<int64_t> &a_strides,
                      const std::vector<int64_t> &a_offset,
                      const std::vector<int64_t> &b_shape,
                      const std::vector<int64_t> &b_strides,
                      const std::vector<int64_t> &b_offset,
                      const int64_t               numel) {
        // Current (multi-dimensional) index to read into a, b, out
        std::vector<int64_t> nd_index(out_shape.size(), 0);

        int64_t a_index   = std::accumulate(a_offset.begin(), a_offset.end(), 0);
        int64_t b_index   = std::accumulate(b_offset.begin(), b_offset.end(), 0);
        int64_t out_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

        const auto        ndim            = out_shape.size();
        constexpr int64_t out_ldim_stride = 1;
        constexpr int64_t a_ldim_stride   = 1;
        constexpr int64_t b_ldim_stride   = 1;

        // Main loop
        for (int64_t n = 0; n < numel; n += LAST_DIM_UNROLL_FACTOR) {
            auto    items_left_in_last_dim = out_shape[ndim - 1] - nd_index[ndim - 1];
            int64_t unroll_size            = LAST_DIM_UNROLL_FACTOR;
            if (items_left_in_last_dim >= LAST_DIM_UNROLL_FACTOR) {
                // for (int j = 0; j < LAST_DIM_UNROLL_FACTOR; ++j) {
                    out_data[out_index + 0 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 0 * a_ldim_stride], b_data[b_index + 0 * b_ldim_stride]);
                    out_data[out_index + 1 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 1 * a_ldim_stride], b_data[b_index + 1 * b_ldim_stride]);
                    out_data[out_index + 2 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 2 * a_ldim_stride], b_data[b_index + 2 * b_ldim_stride]);
                    out_data[out_index + 3 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 3 * a_ldim_stride], b_data[b_index + 3 * b_ldim_stride]);
                    out_data[out_index + 4 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 4 * a_ldim_stride], b_data[b_index + 4 * b_ldim_stride]);
                    out_data[out_index + 5 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 5 * a_ldim_stride], b_data[b_index + 5 * b_ldim_stride]);
                    out_data[out_index + 6 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 6 * a_ldim_stride], b_data[b_index + 6 * b_ldim_stride]);
                    out_data[out_index + 7 * out_ldim_stride] = Op<A, B>::call(a_data[a_index + 7 * a_ldim_stride], b_data[b_index + 7 * b_ldim_stride]);
                // }
            } else {
                unroll_size = items_left_in_last_dim;
                for (int j = 0; j < items_left_in_last_dim; ++j) {
                    out_data[out_index + j * out_ldim_stride] =
                            Op<A, B>::call(a_data[a_index + j * a_ldim_stride], b_data[b_index + j * b_ldim_stride]);
                }
            }

            // Increment the multi-dimensional index
            for (int64_t i = ndim - 1; i >= 0; --i) {
                if (i == ndim - 1) {
                    nd_index[i] += unroll_size;
                    a_index += unroll_size * a_strides[i];
                    b_index += unroll_size * b_strides[i];
                    out_index += unroll_size * out_strides[i];
                } else {
                    nd_index[i] += 1;
                    a_index += a_strides[i];
                    b_index += b_strides[i];
                    out_index += out_strides[i];
                }

                if (nd_index[i] < out_shape[i]) {
                    break;
                } else if (i != ndim - 1 || n + LAST_DIM_UNROLL_FACTOR >= numel) {
                    // Reset the linear index for the current dimension
                    if (i == ndim - 1) {
                        a_index -= (unroll_size - 1) * a_strides[i];
                        b_index -= (unroll_size - 1) * b_strides[i];
                        out_index -= (unroll_size - 1) * out_strides[i];
                    } else {
                        a_index -= (a_shape[i] - 1) * a_strides[i];
                        b_index -= (b_shape[i] - 1) * b_strides[i];
                        out_index -= (out_shape[i] - 1) * out_strides[i];
                    }
                    nd_index[i] = 0;
                }
            }
        }
    }
};

} // namespace optimize_detail

/**
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
template <typename A, typename B, template <typename, typename> class Op>
ndarray elementwise_binary_op(const ndarray &a, const ndarray &b, ndarray &out) {
    auto a_data = a.data_ptr<A>();
    auto b_data = b.data_ptr<B>();

    const auto a_offset   = a.offsets();
    const auto b_offset   = b.offsets();
    const auto out_offset = out.offsets();

    // Create output array with appropriate shape and data type
    // TODO: figure out type promotion for the return type
    auto       out_data  = out.data_ptr<A>();
    const auto out_shape = out.shape();

    const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
    const auto b_strides   = compute_broadcast_strides(b.shape(), b.strides(), out_shape);
    const auto out_strides = out.strides();

    const auto a_shape = compute_broadcast_shape(a.shape(), out_shape);
    const auto b_shape = compute_broadcast_shape(b.shape(), out_shape);

    if (out_strides.back() == 1 && a_strides.back() == 1 && b_strides.back() == 1) {
        optimize_detail::elementwise_binary_op<A, B, Op, 8, 1, 1, 1>::apply(out_data,
                                                                             a_data,
                                                                             b_data,
                                                                             out_shape,
                                                                             out_strides,
                                                                             out_offset,
                                                                             a_shape,
                                                                             a_strides,
                                                                             a_offset,
                                                                             b_shape,
                                                                             b_strides,
                                                                             b_offset,
                                                                             out.numel());
    } else {
        optimize_detail::elementwise_binary_op<A, B, Op, 8, 0, 0, 0>::apply(out_data,
                                                                             a_data,
                                                                             b_data,
                                                                             out_shape,
                                                                             out_strides,
                                                                             out_offset,
                                                                             a_shape,
                                                                             a_strides,
                                                                             a_offset,
                                                                             b_shape,
                                                                             b_strides,
                                                                             b_offset,
                                                                             out.numel());
    }

    return out;
}

/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
struct elementwise_binary_op_impl_wrapper {
    // Create new tensor to store output
    template <typename A_type, typename B_type, template <typename, typename> class Op>
    static inline ndarray call(const ndarray &a, const ndarray &b) {
        std::vector<int64_t> out_shape = expand_shapes_to_broadcast(a.shape(), b.shape());
        ndarray              out(out_shape, a.dtype(), a.device());
        return elementwise_binary_op<A_type, B_type, Op>(a, b, out);
    }

    // An output tensor is provided
    template <typename A_type, typename B_type, template <typename, typename> class Op>
    static inline ndarray call(const ndarray &a, const ndarray &b, ndarray &out) {
        return elementwise_binary_op<A_type, B_type, Op>(a, b, out);
    }
};

} // namespace bland
