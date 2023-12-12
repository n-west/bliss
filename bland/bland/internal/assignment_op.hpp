#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

namespace bland {
struct ndarray;

/**
 * 
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
template <typename Out, typename S>
ndarray assignment_op(ndarray &out, const S &a) {
    // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
    auto out_data    = out.data_ptr<Out>();
    auto out_shape   = out.shape();
    auto out_strides = out.strides();
    auto out_offset  = out.offsets();

    // Current (multi-dimensional) index to read into a
    std::vector<int64_t> out_index(out_shape.size(), 0);

    for (int64_t n = 0; n < out.numel(); ++n) {
        // Compute the linear index for a by accumulating stride over the given multi-dim index
        // this is very important for broadcasting and for any
        int64_t out_linear_index = 0;
        for (int i = 0; i < out_shape.size(); ++i) {
            out_linear_index += out_offset[i] + (out_index[i] % out_shape[i]) * out_strides[i];
        }
        // Finally... do the actual op
        out_data[out_linear_index] = static_cast<Out>(a);

        // Increment the multi-dimensional index
        for (int i = out_shape.size() - 1; i >= 0; --i) {
            // If we're not at the end of this dim, keep going
            if (++out_index[i] != out_shape[i]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                out_index[i] = 0;
            }
        }
    }

    return out;
}

/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
struct assignment_op_impl_wrapper {
    template <typename Out, typename S>
    static inline ndarray call(ndarray &out, const S &value) {
        return assignment_op<Out, S>(out, value);
    }
};

} // namespace bland