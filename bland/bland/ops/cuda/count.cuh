#pragma once

#include "bland/ndarray.hpp"
#include "internal/shape_helpers.hpp"

#include <typeinfo>
#include <numeric> // accumulate

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

namespace bland {

template <typename A>
__global__ void count_true_kernel(uint32_t *count,
                                A* a_data, int64_t* a_shape, int64_t* a_strides,
                                int64_t ndim, int64_t numel) {
    extern __shared__ int64_t shared_mem[];
    int64_t* nd_index = shared_mem + threadIdx.x * ndim;
    uint32_t* block_sum = (uint32_t*)(shared_mem + blockDim.x * ndim);

    auto worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    uint32_t local_count = 0;

    for (int64_t n = worker_id; n < numel; n += grid_size) {
        auto flattened_work_item = n;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            nd_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }

        int64_t a_index = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            a_index += nd_index[dim] * a_strides[dim];
        }

        auto val = a_data[a_index];
        if (val) {
            local_count++;
        }
    }

    block_sum[threadIdx.x] = local_count;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(count, block_sum[0]);
    }
}

struct count_launcher {
    template <typename in_datatype>
    static inline int64_t call(const ndarray &a) {

        auto a_shape = a.shape();
        auto a_strides = a.strides();
        const auto a_offset   = a.offsets();

        auto a_data = a.data_ptr<in_datatype>();

        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        uint32_t* device_sum;  // Host result pointer
        // Allocate pinned host memory
        cudaHostAlloc((void**)&device_sum, sizeof(uint32_t), cudaHostAllocDefault);
        *device_sum = 0;

        int block_size = 256; // TODO: for some small numels we might still want to reduce this
        auto numel = a.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(256, (numel+block_size-1) / block_size);
        size_t shared_memory_size = block_size * (a.ndim() * sizeof(int64_t) + sizeof(uint32_t));
        count_true_kernel<in_datatype><<<num_blocks, block_size, shared_memory_size>>>(device_sum,
                                    a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()),
                                    a.ndim(), numel);
        cudaStreamSynchronize(0);
        int64_t host_sum = static_cast<int64_t>(*device_sum);  // The result can be accessed directly on the host

        // Free pinned host memory
        cudaFreeHost(device_sum);

        return host_sum;
    }
};

} // namespace bland