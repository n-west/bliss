#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

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
template <typename A, template <typename> class Op>
ndarray elementwise_unary_op(const ndarray &a, ndarray &out) {
    auto a_data    = a.data_ptr<A>();
    auto a_shape   = a.shape();
    auto a_strides = a.strides();
    auto a_offset  = a.offsets();

    // The out array may actually be a slice! So need to respect its strides and offsets
    auto out_data    = out.data_ptr<A>();
    auto out_shape   = out.shape();
    auto out_strides = out.strides();
    auto out_offset  = out.offsets();

    if (a.ndim() == out.ndim()) {
        for (int dim = 0; dim < a.ndim(); ++dim) {
            if (a_shape[dim] != out_shape[dim]) {
                throw std::runtime_error("outshape does not equal in shape. Cannot store output in provided tensor");
            }
        }
    } else {
        throw std::runtime_error("out rank does not match in rank. Cannot store output in provided tensor");
    }

    // Current (multi-dimensional) index for a and out
    std::vector<int64_t> nd_index(a_shape.size(), 0);

    int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);
    int64_t out_linear_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // Finally... do the actual op
        out_data[out_linear_index] = Op<A>::call(a_data[a_linear_index]);

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
    template <typename A_type, template <typename> class Op>
    static inline ndarray call(const ndarray &a) {
        ndarray out(a.shape(), a.dtype(), a.device());
        return elementwise_unary_op<A_type, Op>(a, out);
    }

    template <typename A_type, template <typename> class Op>
    static inline ndarray call(const ndarray &a, ndarray &out) {
        return elementwise_unary_op<A_type, Op>(a, out);
    }
};

} // namespace bland