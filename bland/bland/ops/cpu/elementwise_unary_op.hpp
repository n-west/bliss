#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include "fmt/format.h"
#include "fmt/ranges.h"

#include <numeric> // accumulate

namespace bland {
struct ndarray;

namespace cpu {

/**
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
template <typename Out, typename A, template <typename, typename> class Op>
ndarray elementwise_unary_op(ndarray &out, const ndarray &a) {
    auto a_data    = a.data_ptr<A>();
    auto a_shape   = a.shape();
    auto a_strides = a.strides();
    auto a_offset  = a.offsets();

    // The out array may actually be a slice! So need to respect its strides and offsets
    auto out_data    = out.data_ptr<Out>();
    auto out_shape   = out.shape();
    auto out_strides = out.strides();
    auto out_offset  = out.offsets();

    a_shape = compute_broadcast_shape(a_shape, out_shape);
    a_strides = compute_broadcast_strides(a_shape, a_strides, out_shape);
    if (a.ndim() == out.ndim()) {
        for (int dim = 0; dim < a.ndim(); ++dim) {
            if (a_shape[dim] != out_shape[dim] && a_shape[dim] != 1) {
                throw std::runtime_error("outshape does not equal in shape. Cannot store output in provided tensor");
            }
        }
    } else {
        throw std::runtime_error("out rank does not match in rank. Cannot store output in provided tensor");
    }

    // Current (multi-dimensional) index for a and out
    std::vector<int64_t> nd_index(out_shape.size(), 0);

    int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t out_linear_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // Finally... do the actual op
        out_data[out_linear_index] = Op<Out, A>::call(a_data[a_linear_index]);

        // Increment the multi-dimensional index
        for (int i = nd_index.size() - 1; i >= 0; --i) {
            if (++nd_index[i] != out_shape[i]) {
                a_linear_index += a_strides[i];
                out_linear_index += out_strides[i];
                break;
            } else {
                a_linear_index -= (a_shape[i] -1) * a_strides[i];
                out_linear_index -= (out_shape[i] -1) * out_strides[i];
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
struct unary_op_impl_wrapper {
    template <typename Out, typename A, template <typename, typename> class Op>
    static inline ndarray call(ndarray &out, const ndarray &a) {
        return elementwise_unary_op<Out, A, Op>(out, a);
    }
};

} // namespace cpu
} // namespace bland