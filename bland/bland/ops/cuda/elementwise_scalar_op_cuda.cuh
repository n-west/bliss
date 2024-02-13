#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <typeinfo>
#include <numeric> // accumulate

namespace bland {
struct ndarray;

/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
//  template <class Op>
 struct scalar_op_impl_wrapper_cuda {

    template <typename Out, typename A_type, typename B_type, class Op>
    static inline ndarray call(ndarray out, const ndarray &a, const B_type &b) {
        // Check that this operation is possible
        auto a_shape = a.shape();
        // TODO: check/validate output shape!
        // if (a.ndim() == b.ndim()) {
        //     for (int64_t dim = 0; dim < a.ndim(); ++dim) {
        //         if (a_shape[dim] != b_shape[dim] && a_shape[dim] != 1 && b_shape[dim] != 1) {
        //             throw std::runtime_error(
        //                     "elementwise_binary_op: inputs match ndim but shapes are not compatible or broadcastable");
        //         }
        //     }
        //     // TODO: check if this can be broadcasted....
        // }

        auto a_data = a.data_ptr<A_type>();
        
        const auto a_offset   = a.offsets();
        const auto out_offset = out.offsets();
        
        auto       out_data  = out.data_ptr<Out>();
        const auto out_shape = out.shape();
        
        const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
        const auto out_strides = out.strides();
        
        a_shape = compute_broadcast_shape(a_shape, out_shape);


        int block_size = 256; // TODO: for some small numels we might still want to reduce this
        auto numel = out.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(16, (numel+block_size-1) / block_size);
        // elementwise_binary_op_cuda_impl<Out, A_type, B_type, Op><<<num_blocks, block_size>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()),
        //                                                         a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()),
        //                                                         b_data, thrust::raw_pointer_cast(dev_b_shape.data()), thrust::raw_pointer_cast(dev_b_strides.data()),
        //                                                         out.ndim(), numel);
        return out;
    }

};

} // namespace bland