
#include "ops_cuda.cuh"

#include "arithmetic_cuda_impl.cuh"
#include "elementwise_unary_op.cuh"
#include "internal/dispatcher.hpp"

#include "bland/ndarray.hpp"

#include <fmt/format.h>


using namespace bland;
using namespace bland::cuda;


ndarray bland::cuda::copy(ndarray a, ndarray &out) {
    fmt::print("cuda::copy\n");
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_copy_op>(out, a);
}

ndarray bland::cuda::square(ndarray a, ndarray& out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_square_op>(out, a);
}

ndarray bland::cuda::sqrt(ndarray a, ndarray& out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_sqrt_op>(out, a);
}

ndarray bland::cuda::abs(ndarray a, ndarray& out) {
    return dispatch_new2<cuda::unary_op_impl_wrapper, cuda::elementwise_abs_op>(out, a);
}


/**
 * Fill out with the value of a cast to the datatype of out.
 */
// TODO: have a safe default if ndim > MAX_NDIM
constexpr int ASSIGN_MAX_NDIM=4;

template <typename Out, typename S>
__global__ void assignment_kernel(Out* out_data, int64_t* out_shape, int64_t* out_strides,
                                S value,
                                int64_t ndim, int64_t numel) {
    auto worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_size = gridDim.x * blockDim.x;

    extern __shared__ char smem[];
    int64_t* sh_out_shape = reinterpret_cast<int64_t*>(smem);
    int smem_end = sizeof(int64_t) * ndim;

    int64_t* sh_out_strides = reinterpret_cast<int64_t*>(smem + smem_end);
    smem_end += sizeof(int64_t) * ndim;

    for (int sh_index = threadIdx.x; sh_index < ndim * 2; sh_index += blockDim.x) {
        if (sh_index < ndim) {
            sh_out_shape[sh_index] = out_shape[sh_index];
        } else if (sh_index < ndim*2) {
            sh_out_strides[sh_index - ndim] = out_strides[sh_index - ndim];
        }
    }
    __syncthreads();

    int64_t nd_index[ASSIGN_MAX_NDIM] = {0};
    auto flattened_work_item = worker_id;
    for (int dim = ndim - 1; dim >= 0; --dim) {
        nd_index[dim] = flattened_work_item % sh_out_shape[dim];
        flattened_work_item /= sh_out_shape[dim];
    }
    for (int64_t n = worker_id; n < numel; n += grid_size) {
        int64_t out_index = 0;
        for (int dim = 0; dim < ndim; ++dim) {
            out_index += nd_index[dim] * sh_out_strides[dim];
        }
        out_data[out_index] = value;
        auto increment_amount = grid_size;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            nd_index[dim] += increment_amount;
            if (nd_index[dim] < sh_out_shape[dim]) {
                break;
            } else {
                // The remainder might be multiples of dim sizes
                increment_amount = nd_index[dim] / sh_out_shape[dim];
                nd_index[dim] = nd_index[dim] % sh_out_shape[dim];
            }
        }
    }
}

struct assignment_op_impl_wrapper {
    template <typename Out, typename S>
    static inline ndarray call(ndarray &out, const S &value) {
        const auto out_offset = out.offsets();
        int64_t out_ptr_offset = std::accumulate(out_offset.begin(), out_offset.end(), 0LL);

        auto       out_data  = out.data_ptr<Out>() + out_ptr_offset;
        const auto out_shape = out.shape();
        const auto out_strides = out.strides();

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());

        int block_size = 256; // TODO: for some small numels we might still want to reduce this
        auto numel = out.numel();
        // TODO: do some benchmarking to get a better default max number of blocks
        int num_blocks = std::min<int>(112, (numel+block_size-1) / block_size);
        // cudaDeviceSynchronize();
        auto smem = sizeof(int64_t) * out.ndim()*2;
        assignment_kernel<Out, S><<<num_blocks, block_size, smem>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()),
value,
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

template <typename T>
ndarray bland::cuda::fill(ndarray out, T value) {
    return dispatch_new<assignment_op_impl_wrapper, T>(out, value);
}

template ndarray bland::cuda::fill<float>(ndarray out, float v);
// template ndarray bland::cuda::fill<double>(ndarray out, double v);
// template ndarray bland::cuda::fill<int8_t>(ndarray out, int8_t v);
// template ndarray bland::cuda::fill<int16_t>(ndarray out, int16_t v);
template ndarray bland::cuda::fill<int32_t>(ndarray out, int32_t v);
// template ndarray bland::cuda::fill<int64_t>(ndarray out, int64_t v);
template ndarray bland::cuda::fill<uint8_t>(ndarray out, uint8_t v);
// template ndarray bland::cuda::fill<uint16_t>(ndarray out, uint16_t v);
template ndarray bland::cuda::fill<uint32_t>(ndarray out, uint32_t v);
// template ndarray bland::cuda::fill<uint64_t>(ndarray out, uint64_t v);
