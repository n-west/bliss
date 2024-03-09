#pragma once

#pragma once

#include "bland/ndarray.hpp"
#include "shape_helpers.hpp"

#include <typeinfo>
#include <numeric> // accumulate

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace bland {
namespace cuda {


// TODO: have a safe default if ndim > MAX_NDIM
constexpr int MAX_NDIM=4;

template <typename Out, typename A, template <typename, typename> class Op>
__global__ void elementwise_unary_op_impl(Out* out_data, int64_t* out_shape, int64_t* out_strides,
                                A* a_data, int64_t* a_shape, int64_t* a_strides,
                                int64_t ndim, int64_t numel) {
    auto worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    // int64_t* nd_index = (int64_t*) malloc(ndim * sizeof(int64_t));
    int64_t nd_index[MAX_NDIM] = {0};
    auto flattened_work_item = worker_id;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        nd_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }
    // printf("initial index: [%i]: [%lld, %lld]\n", worker_id, nd_index[0], nd_index[1]);

    for (int64_t n = worker_id; n < numel; n += grid_size) {
        int64_t out_index = 0, a_index = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            out_index += nd_index[dim] * out_strides[dim];
            a_index += nd_index[dim] * a_strides[dim];
        }
        out_data[out_index] = Op<Out, A>::call(a_data[a_index]);
        // __syncthreads();
        // printf("storing %i [%lld] = f(%i [%lld]) at ind=%i  ([%lld, %lld]) with tid=%i.%i\n", out_data[out_index], out_index,
        //                                                                         a_data[a_index], a_index,
        //                                                                         n, nd_index[0], nd_index[1], blockIdx.x, threadIdx.x);

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
    // free(nd_index);
}
/**
 * template wrapper around a template function which calls the function
 * with the given template datatypes
 */
struct unary_op_impl_wrapper {
    template <typename Out, typename A_type, template <typename, typename> class Op>
    static inline ndarray call(ndarray out, const ndarray &a) {
        const auto a_offset   = a.offsets();
        const auto out_offset = out.offsets();
        int64_t a_ptr_offset = std::accumulate(a_offset.begin(), a_offset.end(), 0LL);
        int64_t out_ptr_offset = std::accumulate(out_offset.begin(), out_offset.end(), 0LL);

        auto a_data = a.data_ptr<A_type>() + a_ptr_offset;
        auto a_shape = a.shape();

        auto       out_data  = out.data_ptr<Out>() + out_ptr_offset;
        const auto out_shape = out.shape();

        const auto a_strides   = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
        const auto out_strides = out.strides();

        a_shape = compute_broadcast_shape(a_shape, out_shape);

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());
        
        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        int block_size = 512; // TODO: for some small numels we might still want to reduce this
        auto numel = out.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(32, (numel+block_size-1) / block_size);
        // cudaDeviceSynchronize();

        elementwise_unary_op_impl<Out, A_type, Op><<<num_blocks, block_size>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()),
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()),
                                                                out.ndim(), numel);
        // // Synchronize the device
        // cudaError_t syncErr = cudaDeviceSynchronize();

        // // Check for errors in kernel launch
        // cudaError_t launchErr = cudaGetLastError();
        // if (syncErr != cudaSuccess) {
        //     printf("Error during kernel execution: %s\n", cudaGetErrorString(syncErr));
        // } else {
        //     // printf("No error during kernel exec\n");
        // }

        // if (launchErr != cudaSuccess) {
        //     printf("Error during kernel launch: %s\n", cudaGetErrorString(launchErr));
        // } else {
        //     // printf("no launch error\n");
        // }
        return out;
    }
};

} // namespace cuda
} // namespace bland