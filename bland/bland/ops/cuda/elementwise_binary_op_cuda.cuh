#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>
#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include <numeric> // needed for accumulate
#include <stdexcept>

namespace bland {


template <typename Out, typename A, typename B, class Op>
__global__ void
elementwise_binary_op_cuda_impl(Out* out_data, int64_t* out_shape, int64_t* out_strides,
                                A* a_data, int64_t* a_shape, int64_t* a_strides,
                                B* b_data, int64_t* b_shape, int64_t* b_strides,
                                int64_t ndim, int64_t numel) {
    auto worker_id = gridDim.x * blockIdx.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    // printf("out_shape is %lld and stride is %lld\n", out_shape[0], out_strides[0]);
    // printf("a_shape is %lld and stride is %lld\n", a_shape[0], a_strides[0]);
    // printf("b_shape is %lld and stride is %lld\n", b_shape[0], b_strides[0]);
    // int64_t out_index = 0; // TODO: get the offsets in (can be done on cpu)
    // int64_t a_index   = 0;
    // int64_t b_index   = 0;
    // Initialize nd index based on work ids. The last dim may not have enough work
    // for the entire grid, so have to check for wrap-around to other dimensions
    int64_t* nd_index = (int64_t*) malloc(ndim * sizeof(int64_t));
        auto flattened_work_item = worker_id;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        nd_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }

    for (int64_t n = worker_id; n < numel; n += grid_size) {
        int64_t out_index = 0, a_index = 0, b_index = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            out_index += nd_index[dim] * out_strides[dim];
            a_index += nd_index[dim] * a_strides[dim];
            b_index += nd_index[dim] * b_strides[dim];
        }
        out_data[out_index] = Op::template call<Out, A, B>(a_data[a_index], b_data[b_index]);
        // printf("storing %f [%lld] = f(%f [%lld], %f [%lld]) at ind=%i with tid=%i.%i\n", out_data[out_index], out_index,
        //                                                                         a_data[a_index], a_index,
        //                                                                         b_data[b_index], b_index,
        //                                                                         n, threadIdx.x, blockIdx.x);

        // Update the indices for the next iteration
        auto increment_amount = grid_size;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            nd_index[dim] += increment_amount;
            if (nd_index[dim] < out_shape[dim]) {
                break;
            } else {
                // The remainder might be multiples of dim sizes
                increment_amount = nd_index[dim] / out_shape[dim];
                nd_index[dim] = nd_index[dim] % out_shape[dim];
            }
        }
    }
    free(nd_index);
}

/**
 * Perform an elementwise binary operation such as add, sub, mul, div as indicated
 * in the Op parameter (which will have the underlying datatypes passed through
 * as template parameters to the op) between two tensors with underlying datatypes
 * A and B.
 *
 * Currently the result datatype will be the same as A, but we should fix that!
 */
// template <typename Out, typename A, typename B, class Op>
// ndarray elementwise_binary_op_cuda(ndarray &out, const ndarray &a, const ndarray &b) {

// }

/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
template <class Op>
struct elementwise_binary_op_impl_wrapper_cuda {
    // An output tensor is provided
    template <typename Out, typename A_type, typename B_type>
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
        // int num_blocks = 16;
        // return elementwise_binary_op_cuda<Out, A_type, B_type, Op>(out, a, b);
        auto a_data = a.data_ptr<A_type>();
        auto b_data = b.data_ptr<B_type>();
        
        const auto a_offset   = a.offsets();
        const auto b_offset   = b.offsets();
        const auto out_offset = out.offsets();
        
        auto       out_data  = out.data_ptr<Out>();
        const auto out_shape = out.shape();
        
        const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
        const auto b_strides   = compute_broadcast_strides(b.shape(), b.strides(), out_shape);
        const auto out_strides = out.strides();
        
        a_shape = compute_broadcast_shape(a_shape, out_shape);
        b_shape = compute_broadcast_shape(b_shape, out_shape);
        
        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());
        
        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());
        
        thrust::device_vector<int64_t> dev_b_shape(b_shape.begin(), b_shape.end());
        thrust::device_vector<int64_t> dev_b_strides(b_strides.begin(), b_strides.end());
        
        int block_size = 256; // TODO: for some small numels we might still want to reduce this
        auto numel = out.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(16, (numel+block_size-1) / block_size);
        elementwise_binary_op_cuda_impl<Out, A_type, B_type, Op><<<num_blocks, block_size>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()),
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()),
                                                                b_data, thrust::raw_pointer_cast(dev_b_shape.data()), thrust::raw_pointer_cast(dev_b_strides.data()),
                                                                out.ndim(), numel);
        
        return out;
    }
};

} // namespace bland
