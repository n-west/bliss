#include "statistical.cuh"

#include "bland/ndarray.hpp"

#include "dispatcher.hpp"
#include "shape_helpers.hpp"

#include "count.cuh" // kernel definition for count

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include <algorithm> // std::find
#include <numeric>   // std::accumulate

using namespace bland;
using namespace bland::cuda;


template <typename T>
struct Accumulator {
    using type = T;
};

template <>
struct Accumulator<float> {
    using type = double;
};

template <>
struct Accumulator<int8_t> {
    using type = int16_t;
};

template <>
struct Accumulator<int16_t> {
    using type = int32_t;
};

template <>
struct Accumulator<int32_t> {
    using type = int64_t;
};

// TODO: have a safe default if ndim > MAX_NDIM
constexpr int MAX_NDIM=4;

enum class reductiontype {
    sum,
    mean,
    stddev,
    var
};


template<typename T>
__device__ void fast2sum(T& a, T& b) {
    auto s = a + b;
    auto z = s - a;
    auto t = b - z;
    a = s;
    b = t;
}
/**
 * one step of kahan sum for corrected accumulation. Inputs modified in-place
 * where `a` should be the accumulated value and `b` is a new input (replaced with
 * the correction that should be applied to next input)
*/
template <typename T>
__device__ void kahan(T& a, T& b) {
    auto t = a + b;
    b = (t - a) - b;
    a = t;
}


template <typename T>
struct is_floating {
    static constexpr bool value = false;
};

template <>
struct is_floating<float> {
    static constexpr bool value = true;
};
template <>
struct is_floating<double> {
    static constexpr bool value = true;
};

// Non-masked reductions
template <typename out_datatype, typename in_datatype, reductiontype Op>
__global__ void reduction_impl(out_datatype* out_data, int64_t* out_shape, int64_t* out_strides, int64_t out_ndim, int64_t out_numel,
                        in_datatype* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    using accumulator_type = in_datatype;
    auto tid = threadIdx.x;

    // Assume we've been given shared memory with enough space for each thread
    // to hold an accumulator of the in_datatype size (and one for x^2 if stddev or var)
    // Break that shared memory up in to appropriate chunks easily indexed by smem
    extern __shared__ char shared_memory[];
    accumulator_type* sdata = reinterpret_cast<accumulator_type*>(shared_memory);
    accumulator_type sdata_c = 0;
    accumulator_type* s2data = sdata + blockDim.x;
    accumulator_type s2data_c = 0;

    int64_t out_index[MAX_NDIM] = {0};
    int64_t in_index[MAX_NDIM] = {0};
    // int64_t mask_index[MAX_NDIM] = {0};
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    auto numel = out_numel;
    for (int64_t i = 0; i < numel; ++i) {

        // Reset accumulators
        sdata[tid] = 0;
        if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
            s2data[tid] = 0;
        }
        // Deep copy the input index which is advancing over non-reduced inputs
        // to reduced_nd_index which will be used in inner loop to advance through
        // reduced inputs 
        for (int dim=0; dim < a_ndim; ++dim) {
            reduced_nd_index[dim] = in_index[dim];
        }
        auto increment_size = threadIdx.x;
        for (int dim = a_ndim - 1; dim >= 0; --dim) {
            // increment the index if this dim is being reduced
            bool reduce_this_dim = false;
            for (int ra=0; ra < nreduced_dim; ++ra) {
                if (dim == reduced_axes[ra]) {
                    reduce_this_dim = true;
                    break;
                }
            }
            if (reduce_this_dim) {
                reduced_nd_index[dim] += increment_size;
                if (reduced_nd_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    // The remainder might be multiples of dim sizes
                    increment_size = reduced_nd_index[dim] / a_shape[dim];
                    reduced_nd_index[dim] = reduced_nd_index[dim] % a_shape[dim];
                }
            }
        }

        // Iterate through the inputs to reduce to this output....
        for (int64_t jj=tid; jj < reduction_factor; jj += blockDim.x) {
            int64_t in_linear_index = 0;
            for (int dim = 0; dim < a_ndim; ++dim) {
                in_linear_index += reduced_nd_index[dim] * a_strides[dim];
            }

            // Read the input data and accumulate it to this threads accumulator
            auto in_val = a_data[in_linear_index];
            
            // use a corrected sum (kahan sum) to preserve precision
            if constexpr (is_floating<in_datatype>::value) {
                auto y = in_val - sdata_c;
                kahan(sdata[tid], y);
                sdata_c = y;
            } else {
                // naive sum
                sdata[tid] += in_val;
            }
            
            // stddev and variance will also keep track of x^2 accumulation
            if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
                auto x_square = in_val * in_val;
                if constexpr (is_floating<in_datatype>::value) {
                    auto y2 = x_square - s2data_c;
                    kahan(s2data[tid], y2);
                    s2data_c = y2;
                } else {
                    s2data[tid] += x_square;
                }
            }
            // Increment nd_indix for reduced dimensions
            auto increment_size = blockDim.x;
            for (int dim = a_ndim - 1; dim >= 0; --dim) {
                // increment the index if this dim is being reduced
                bool reduce_this_dim = false;
                for (int ra=0; ra < nreduced_dim; ++ ra) {
                    if (dim == reduced_axes[ra]) {
                        reduce_this_dim = true;
                        break;
                    }
                }
                if (reduce_this_dim) {
                    reduced_nd_index[dim] += increment_size;
                    if (reduced_nd_index[dim] < a_shape[dim]) {
                        break;
                    } else {
                        // The remainder might be multiples of dim sizes
                        increment_size = reduced_nd_index[dim] / a_shape[dim];
                        reduced_nd_index[dim] = reduced_nd_index[dim] % a_shape[dim];
                    }
                }
            }
        }

        int64_t out_linear_index = 0;
        for (int dim = 0; dim < out_ndim; ++dim) {
            out_linear_index += out_index[dim] * out_strides[dim];
        }

        // Normalize accumulated inputs as needed.
        if constexpr (Op == reductiontype::mean) {
            // printf("(%i) The partial sum is %f\n", tid, sdata[tid]);
            sdata[tid] /= reduction_factor;
        } else if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
            sdata[tid] /= reduction_factor;
            s2data[tid] /= reduction_factor;
        }
        __syncthreads();
        for (int s=blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) {
                // __syncthreads();
                // printf("(%i) reduced sum is sdata[%i] + sdata[%i]. (%f + %f)\n", tid, tid, tid+s, sdata[tid], sdata[tid+s]);
                sdata[tid] += sdata[tid+s];
                if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
                    s2data[tid] += s2data[tid+s];
                }
            }
            __syncthreads();
        }
        // Write the final accumulated value to global output
        if (tid == 0) {
            if constexpr (Op == reductiontype::stddev) {
                // out_data[out_linear_index] = static_cast<out_datatype>((s2data[0] - sdata[0]*sdata[0]));
                // printf("E[X^2] = %f - E[X]^2 = %f\n", s2data[0], sdata[0]*sdata[0]);
                auto diff = (s2data[0] - sdata[0]*sdata[0]);
                diff = max((in_datatype)0, diff);
                if constexpr (is_floating<in_datatype>::value) {
                    out_data[out_linear_index] = static_cast<out_datatype>(sqrt(diff));
                } else {
                    out_data[out_linear_index] = static_cast<out_datatype>(sqrt((float)diff));
                }
            } else if constexpr (Op == reductiontype::var) {
                // Var is E[x^2] - E[x]^2, both were calculated at once to avoid redundant passes through data
                auto diff = (s2data[0] - sdata[0]*sdata[0]);
                diff = max((in_datatype)0, diff);
                out_data[out_linear_index] = static_cast<out_datatype>(diff);
            } else {
                // sum, mean
                out_data[out_linear_index] = static_cast<out_datatype>(sdata[0]);
            }
        }

        // Increment out and in pointers to next non-reduced dims
        for (int dim = out_ndim - 1; dim >= 0; --dim) {
            if (++out_index[dim] != out_shape[dim]) {
                break;
            } else {
                out_index[dim] = 0;
            }
        }
        for (int dim = a_ndim - 1; dim >= 0; --dim) {
            // increment the index if this dim is being reduced
            bool reduce_this_dim = false;
            for (int ra=0; ra < nreduced_dim; ++ra) {
                if (dim == reduced_axes[ra]) {
                    reduce_this_dim = true;
                    break;
                }
            }
            if (!reduce_this_dim) {
                if (++in_index[dim] != a_shape[dim]) {
                    break;
                } else {
                    in_index[dim] = 0;
                }
            }
        }
    }
}

template <typename out_datatype, typename in_datatype, reductiontype Op>
__global__ void reduction_impl(out_datatype* out_data, int64_t* out_shape, int64_t* out_strides, int64_t out_ndim, int64_t out_numel,
                        in_datatype* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        uint8_t* mask_data, int64_t* mask_shape, int64_t* mask_strides, int64_t mask_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    using accumulator_type = in_datatype;
    auto tid = threadIdx.x;

    // Assume we've been given shared memory with enough space for each thread
    // to hold an accumulator of the in_datatype size (and one for x^2 if stddev or var)
    // Break that shared memory up in to appropriate chunks easily indexed by smem
    extern __shared__ char shared_memory[];
    accumulator_type* sdata = reinterpret_cast<accumulator_type*>(shared_memory);
    accumulator_type sdata_c = 0;
    accumulator_type* s2data = sdata + blockDim.x;
    accumulator_type s2data_c = 0;
    int count_offset = blockDim.x;
    if constexpr(Op == reductiontype::stddev || Op == reductiontype::var) {
        count_offset = 2*blockDim.x;
    }
    uint32_t* scount = reinterpret_cast<uint32_t*>(sdata + count_offset);

    int64_t out_index[MAX_NDIM] = {0};
    int64_t in_index[MAX_NDIM] = {0};
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    auto numel = out_numel;
    for (int64_t i = 0; i < numel; ++i) {
        // Reset accumulators
        sdata[tid] = 0;
        scount[tid] = 0;
        if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
            s2data[tid] = 0;
        }
        // Deep copy the input index which is advancing over non-reduced inputs
        // to reduced_nd_index which will be used in inner loop to advance through
        // reduced inputs 
        for (int dim=0; dim < a_ndim; ++dim) {
            reduced_nd_index[dim] = in_index[dim];
        }
        auto increment_size = threadIdx.x;
        for (int dim = a_ndim - 1; dim >= 0; --dim) {
            // increment the index if this dim is being reduced
            bool reduce_this_dim = false;
            for (int ra=0; ra < nreduced_dim; ++ra) {
                if (dim == reduced_axes[ra]) {
                    reduce_this_dim = true;
                    break;
                }
            }
            if (reduce_this_dim) {
                reduced_nd_index[dim] += increment_size;
                if (reduced_nd_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    // The remainder might be multiples of dim sizes
                    increment_size = reduced_nd_index[dim] / a_shape[dim];
                    reduced_nd_index[dim] = reduced_nd_index[dim] % a_shape[dim];
                }
            }
        }

        // Iterate through the inputs to reduce to this output....
        for (int64_t jj=tid; jj < reduction_factor; jj += blockDim.x) {
            int64_t in_linear_index = 0;
            int64_t mask_linear_index = 0;
            for (int dim = 0; dim < a_ndim; ++dim) {
                in_linear_index += reduced_nd_index[dim] * a_strides[dim];
                mask_linear_index += reduced_nd_index[dim] * mask_strides[dim];
            }

            if (mask_data[mask_linear_index] == 0) {
                scount[tid] += 1;

                // Read the input data and accumulate it to this threads accumulator
                auto in_val = a_data[in_linear_index];
                
                // use a corrected sum (kahan sum) to preserve precision
                if constexpr (is_floating<in_datatype>::value) {
                    auto y = in_val - sdata_c;
                    kahan(sdata[tid], y);
                    sdata_c = y;
                } else {
                    // naive sum
                    sdata[tid] += in_val;
                }
                
                // stddev and variance will also keep track of x^2 accumulation
                if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
                    auto x_square = in_val * in_val;
                    if constexpr (is_floating<in_datatype>::value) {
                        auto y2 = x_square - s2data_c;
                        kahan(s2data[tid], y2);
                        s2data_c = y2;
                    } else {
                        s2data[tid] += x_square;
                    }
                }
            }

            // Increment nd_index for reduced dimensions
            auto increment_size = blockDim.x;
            for (int dim = a_ndim - 1; dim >= 0; --dim) {
                // increment the index if this dim is being reduced
                bool reduce_this_dim = false;
                for (int ra=0; ra < nreduced_dim; ++ ra) {
                    if (dim == reduced_axes[ra]) {
                        reduce_this_dim = true;
                        break;
                    }
                }
                if (reduce_this_dim) {
                    reduced_nd_index[dim] += increment_size;
                    if (reduced_nd_index[dim] < a_shape[dim]) {
                        break;
                    } else {
                        // The remainder might be multiples of dim sizes
                        increment_size = reduced_nd_index[dim] / a_shape[dim];
                        reduced_nd_index[dim] = reduced_nd_index[dim] % a_shape[dim];
                    }
                }
            }
        }

        int64_t out_linear_index = 0;
        for (int dim = 0; dim < out_ndim; ++dim) {
            out_linear_index += out_index[dim] * out_strides[dim];
        }

        __syncthreads();
        for (int s=blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) {
                // __syncthreads();
                // printf("(%i) reduced sum is sdata[%i] + sdata[%i]. (%f + %f)\n", tid, tid, tid+s, sdata[tid], sdata[tid+s]);
                sdata[tid] += sdata[tid+s];
                scount[tid] += scount[tid+s];
                if constexpr (Op == reductiontype::stddev || Op == reductiontype::var) {
                    s2data[tid] += s2data[tid+s];
                }
            }
            __syncthreads();
        }
        // Write the final accumulated value to global output
        if (tid == 0) {
            // TODO: guard against scount == 0
            if constexpr (Op == reductiontype::stddev) {
                sdata[0] /= scount[0];
                s2data[0] /= scount[0];
                // out_data[out_linear_index] = static_cast<out_datatype>((s2data[0] - sdata[0]*sdata[0]));
                // printf("E[X^2] = %f - E[X]^2 = %f\n", s2data[0], sdata[0]*sdata[0]);
                auto diff = (s2data[0] - sdata[0]*sdata[0]);
                diff = max((in_datatype)0, diff);
                if constexpr (is_floating<in_datatype>::value) {
                    out_data[out_linear_index] = static_cast<out_datatype>(sqrt(diff));
                } else {
                    out_data[out_linear_index] = static_cast<out_datatype>(sqrt((float)diff));
                }
            } else if constexpr (Op == reductiontype::var) {
                // Var is E[x^2] - E[x]^2, both were calculated at once to avoid redundant passes through data
                sdata[0] /= scount[0];
                s2data[0] /= scount[0];
                auto diff = (s2data[0] - sdata[0]*sdata[0]);
                diff = max((in_datatype)0, diff);
                out_data[out_linear_index] = static_cast<out_datatype>(diff);
            } else if constexpr (Op == reductiontype::mean) {
                out_data[out_linear_index] = static_cast<out_datatype>(sdata[0] / scount[0]);
            } else {
                // sum
                out_data[out_linear_index] = static_cast<out_datatype>(sdata[0]);
            }
        }

        // Increment out and in pointers to next non-reduced dims
        for (int dim = out_ndim - 1; dim >= 0; --dim) {
            if (++out_index[dim] != out_shape[dim]) {
                break;
            } else {
                out_index[dim] = 0;
            }
        }
        for (int dim = a_ndim - 1; dim >= 0; --dim) {
            // increment the index if this dim is being reduced
            bool reduce_this_dim = false;
            for (int ra=0; ra < nreduced_dim; ++ra) {
                if (dim == reduced_axes[ra]) {
                    reduce_this_dim = true;
                    break;
                }
            }
            if (!reduce_this_dim) {
                if (++in_index[dim] != a_shape[dim]) {
                    break;
                } else {
                    in_index[dim] = 0;
                }
            }
        }
    }
}


template <reductiontype Reduction>
struct statistical_launch_wrapper {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray out, const ndarray &a, std::vector<int64_t> reduced_axes) {
        auto       out_data  = out.data_ptr<out_datatype>();
        const auto out_offset = out.offsets();
        const auto out_shape = out.shape();
        const auto out_strides = out.strides();        

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());

        auto a_shape = a.shape();
        auto a_data = a.data_ptr<in_datatype>();
        const auto a_offset  = a.offsets();
        const auto a_strides = a.strides();

        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }
        thrust::device_vector<int64_t> dev_reduced_axes(reduced_axes.begin(), reduced_axes.end());

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, a.device().device_id);

        // TODO: could use more elegance here
        int block_size = 512;
        if (reduced_elements < 512) {
            block_size = reduced_elements | (reduced_elements >> 1);
            block_size |= (block_size >> 2);
            block_size |= (block_size >> 4);
            block_size |= (block_size >> 8);
            block_size |= (block_size >> 16);
            block_size = block_size - (block_size >> 1);
        }
        auto out_numel = out.numel();
        // TODO: the current way reductions are written limits us to 1 block per reduction. That's actually
        // fine, but an improvement would be to do multiple reductions at once (if needed), so increase block
        // size to get parallelized reductions.
        int num_blocks = 1;
        int smem_per_block = sizeof(in_datatype) * block_size; // this should be sizeof the accumulator type
        if (Reduction == reductiontype::stddev || Reduction == reductiontype::var) {
            // TODO: Check we have enough smem
            smem_per_block *= 2; // need to keep an E[X^2] which will take double the smem
        }
        reduction_impl<out_datatype, in_datatype, Reduction><<<num_blocks, block_size, smem_per_block>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()), dev_a_shape.size(),
                                                                thrust::raw_pointer_cast(dev_reduced_axes.data()), dev_reduced_axes.size(), reduced_elements
                                                                );
        // auto launch_ret = cudaDeviceSynchronize();
        // auto kernel_ret = cudaGetLastError();
        // if (launch_ret != cudaSuccess) {
        //     fmt::print("cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
        // }
        // if (kernel_ret != cudaSuccess) {
        //     fmt::print("cuda launch got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
        // }

        return out;
    }

    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray out, const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
        auto       out_data  = out.data_ptr<out_datatype>();
        const auto out_offset = out.offsets();
        const auto out_shape = out.shape();
        const auto out_strides = out.strides();        

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());

        auto a_shape = a.shape();
        auto a_data = a.data_ptr<in_datatype>();
        const auto a_offset   = a.offsets();
        const auto a_strides   = a.strides();

        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        auto mask_shape = mask.shape();
        auto mask_data = mask.data_ptr<uint8_t>();
        const auto mask_offset   = mask.offsets();
        const auto mask_strides  = mask.strides();

        thrust::device_vector<int64_t> dev_mask_shape(mask_shape.begin(), mask_shape.end());
        thrust::device_vector<int64_t> dev_mask_strides(mask_strides.begin(), mask_strides.end());

        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }
        thrust::device_vector<int64_t> dev_reduced_axes(reduced_axes.begin(), reduced_axes.end());

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, a.device().device_id);

        // TODO: could use more elegance here
        int block_size = 512;
        int required_smem_per_thread = sizeof(in_datatype) + 4; // 4B for the mask count
        if (Reduction == reductiontype::stddev || Reduction == reductiontype::var) {
            required_smem_per_thread += sizeof(in_datatype);
            if constexpr (sizeof(in_datatype) == 8) {
                // Need 8B for X accum, 8B for X^2 accum, 4B for count = 20B/thread -> ~200
                block_size = 192;
            } else if (sizeof(in_datatype) == 4) {
                // Need 4B for X accum, 4B for X^2 accum, 4B for count = 12B/thread -> ~341
                block_size = 320;
            }
        } else if constexpr (sizeof(in_datatype) == 8) {
            // Need 8B for X accum, 4B for count = 12B/thread -> ~341
            block_size = 320;
        }

        if (reduced_elements < block_size) {
            // Reduce to a block size that is a power of 2 but less than number of items that need reducing
            block_size = reduced_elements | (reduced_elements >> 1);
            block_size |= (block_size >> 2);
            block_size |= (block_size >> 4);
            block_size |= (block_size >> 8);
            block_size |= (block_size >> 16);
            block_size = block_size - (block_size >> 1);
        }
        auto out_numel = out.numel();
        // TODO: the current way reductions are written limits us to 1 block per reduction. That's actually
        // fine, but an improvement would be to do multiple reductions at once (if needed), so increase block
        // size to get parallelized reductions.
        int num_blocks = 1;
        int smem_per_block = required_smem_per_thread * block_size; // this should be sizeof the accumulator type
        reduction_impl<out_datatype, in_datatype, Reduction><<<num_blocks, block_size, smem_per_block>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()), dev_a_shape.size(),
                                                                mask_data, thrust::raw_pointer_cast(dev_mask_shape.data()), thrust::raw_pointer_cast(dev_mask_strides.data()), dev_mask_shape.size(),
                                                                // mask_data, nullptr, nullptr, mask_shape.size(),
                                                                thrust::raw_pointer_cast(dev_reduced_axes.data()), dev_reduced_axes.size(), reduced_elements
                                                                );
        // auto launch_ret = cudaDeviceSynchronize();
        // auto kernel_ret = cudaGetLastError();
        // if (launch_ret != cudaSuccess) {
        //     fmt::print("cuda launch got error {} ({})\n", launch_ret, cudaGetErrorString(launch_ret));
        // }
        // if (kernel_ret != cudaSuccess) {
        //     fmt::print("cuda launch got error {} ({})\n", kernel_ret, cudaGetErrorString(kernel_ret));
        // }

        return out;
    }
};


ndarray bland::cuda::sum(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::sum>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_sum(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::sum>>(out, a, mask, reduced_axes);
}

ndarray bland::cuda::mean(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::mean>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_mean(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::mean>>(out, a, mask, reduced_axes);
}

ndarray bland::cuda::stddev(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::stddev>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_stddev(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::stddev>>(out, a, mask, reduced_axes);
}

ndarray bland::cuda::var(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::var>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_var(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<statistical_launch_wrapper<reductiontype::var>>(out, a, mask, reduced_axes);
}

int64_t bland::cuda::count_true(ndarray x) {
    return dispatch_summary<count_launcher>(x);
}

