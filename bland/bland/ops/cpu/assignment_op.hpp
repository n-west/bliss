#pragma once

#include "bland/ndarray.hpp"
#include "internal/shape_helpers.hpp"

namespace bland {
struct ndarray;

namespace cpu {

/**
 * Fill out with the value of a cast to the datatype of out.
 */
template <typename Out, typename S>
ndarray assignment_op(ndarray &out, const S &a) {
    // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
    auto out_data    = out.data_ptr<Out>();
    auto out_shape   = out.shape();
    auto out_strides = out.strides();
    auto out_offset  = out.offsets();

    // Current (multi-dimensional) index to read into out
    auto ndim = out_shape.size();
    std::vector<int64_t> nd_index(ndim, 0);
    int64_t out_linear_index = std::accumulate(out_offset.begin(), out_offset.end(), 0);

    auto numel = out.numel();
    for (int64_t n = 0; n < numel; ++n) {
        // do the actual op
        out_data[out_linear_index] = static_cast<Out>(a);

        // Increment the multi-dimensional index
        for (int i = ndim - 1; i >= 0; --i) {
            // If we're not at the end of this dim, keep going
            if (++nd_index[i] != out_shape[i]) {
                out_linear_index += out_strides[i];
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
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
struct assignment_op_impl_wrapper {
    template <typename Out, typename S>
    static inline ndarray call(ndarray &out, const S &value) {
        // TODO: there's a common case of out being contiguous that can likely be
        // much faster than this general implementation handling striding.
        // implement that.
        return assignment_op<Out, S>(out, value);
    }
};

} // namespace cpu
} // namespace bland