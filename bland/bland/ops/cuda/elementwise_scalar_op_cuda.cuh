#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <typeinfo>
#include <numeric> // accumulate

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include <fmt/format.h>

namespace bland {

template <typename T>
struct is_floating {
    static constexpr bool value = false;
};

template <>
struct is_floating<float> {
    static constexpr bool value = true;
};

template <typename Out, typename A, typename B, class Op>
__global__ void elementwise_scalar_op_cuda_impl(Out* out_data, int64_t* out_shape, int64_t* out_strides,
                                A* a_data, int64_t* a_shape, int64_t* a_strides,
                                B scalar_val, int64_t ndim, int64_t numel) {
    auto worker_id = gridDim.x * blockIdx.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    int64_t* nd_index = (int64_t*) malloc(ndim * sizeof(int64_t));
    auto flattened_work_item = worker_id;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        nd_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }

    for (int64_t n = worker_id; n < numel; n += grid_size) {
        int64_t out_index = 0, a_index = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            out_index += nd_index[dim] * out_strides[dim];
            a_index += nd_index[dim] * a_strides[dim];
        }
        out_data[out_index] = Op::template call<Out, A, B>(a_data[a_index], scalar_val);
        // if constexpr (is_floating<B>::value ) {
        // printf("storing %f [%lld] = f(%f [%lld], %f) at ind=%i with tid=%i.%i\n", out_data[out_index], out_index,
        //                                                                         a_data[a_index], a_index,
        //                                                                         scalar_val,
        //                                                                         n, threadIdx.x, blockIdx.x);
        // }

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
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
struct scalar_op_impl_wrapper_cuda {
    template <typename Out, typename A_type, typename B_type, class Op>
    static inline ndarray call(ndarray out, const ndarray &a, const B_type &b) {
        // Check that this operation is possible
        auto a_shape = a.shape();
        // TODO: check/validate output shape!
        auto a_data = a.data_ptr<A_type>();

        const auto a_offset   = a.offsets();
        const auto out_offset = out.offsets();

        auto       out_data  = out.data_ptr<Out>();
        const auto out_shape = out.shape();
        
        const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
        const auto out_strides = out.strides();
        
        a_shape = compute_broadcast_shape(a_shape, out_shape);

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());
        
        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        int block_size = 256; // TODO: for some small numels we might still want to reduce this
        auto numel = out.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(1, (numel+block_size-1) / block_size);
        elementwise_scalar_op_cuda_impl<Out, A_type, B_type, Op><<<num_blocks, block_size>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()),
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()),
                                                                b,
                                                                out.ndim(), numel);
        // Might want to drop these in a debug mode...
        // Synchronize the device
        // cudaError_t syncErr = cudaDeviceSynchronize();

        // // Check for errors in kernel launch
        // cudaError_t launchErr = cudaGetLastError();
        // if (syncErr != cudaSuccess) 
        // {
        //     printf("Error during kernel execution: %s\n", cudaGetErrorString(syncErr));
        // }

        // if (launchErr != cudaSuccess) 
        // {
        //     printf("Error during kernel launch: %s\n", cudaGetErrorString(launchErr));
        // }
        return out;
    }
};

} // namespace bland