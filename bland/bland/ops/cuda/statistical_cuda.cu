#include "statistical.cuh"

#include "bland/ndarray.hpp"

#include "internal/dispatcher.hpp"
#include "internal/shape_helpers.hpp"

#include "count.cuh" // kernel definition for count

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <thrust/device_vector.h>
#include <cuda/std/limits>
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

enum class comparison_reductiontype {
    max
};

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
    auto outer_workid = blockIdx.x;
    auto outer_gridsize = gridDim.x;

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
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    // Initialize ndindex per threadblock
    auto flattened_work_item = outer_workid;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        out_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = a_ndim - 1; dim >= 0; --dim) {
        bool reduce_this_dim = false;
        for (int ra=0; ra < nreduced_dim; ++ra) {
            if (dim == reduced_axes[ra]) {
                reduce_this_dim = true;
                break;
            }
        }
        if (!reduce_this_dim) {
            in_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }
    }

    auto numel = out_numel;
    for (int64_t i = outer_workid; i < numel; i+=outer_gridsize) {
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

        increment_size = gridDim.x;
        // Increment out and in pointers to next non-reduced dims
        for (int dim = out_ndim - 1; dim >= 0; --dim) {
            out_index[dim] += increment_size;
            if (out_index[dim] < out_shape[dim]) {
                break;
            } else {
                increment_size = out_index[dim] / out_shape[dim];
                out_index[dim] = out_index[dim] % out_shape[dim];
            }
        }

        increment_size = gridDim.x;
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
                in_index[dim] += increment_size;
                if (in_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    increment_size = in_index[dim] / a_shape[dim];
                    in_index[dim] = in_index[dim] % a_shape[dim];
                }
            }
        }
    }
}


// Non-masked reductions, hard coded to float for now
// template <typename out_datatype, typename in_datatype, reductiontype Op>
__global__ void mean_stddev_impl(float* mean_out_data, int64_t* mean_out_shape, int64_t* mean_out_strides, int64_t mean_out_ndim, int64_t mean_out_numel,
                        float* stddev_out_data, int64_t* stddev_out_shape, int64_t* stddev_out_strides, int64_t stddev_out_ndim, int64_t stddev_out_numel,
                        float* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    using out_datatype = float;
    using in_datatype = float;
    using accumulator_type = float;
    auto tid = threadIdx.x;
    auto outer_workid = blockIdx.x;
    auto outer_gridsize = gridDim.x;

    // Assume we've been given shared memory with enough space for each thread
    // to hold an accumulator of the in_datatype size (and one for x^2 if stddev or var)
    // Break that shared memory up in to appropriate chunks easily indexed by smem
    extern __shared__ char shared_memory[];
    accumulator_type* sdata = reinterpret_cast<accumulator_type*>(shared_memory);
    accumulator_type sdata_c = 0;
    accumulator_type* s2data = sdata + blockDim.x;
    accumulator_type s2data_c = 0;

    int64_t mean_out_index[MAX_NDIM] = {0};
    int64_t stddev_out_index[MAX_NDIM] = {0};
    int64_t in_index[MAX_NDIM] = {0};
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    // Initialize ndindex per threadblock
    auto flattened_work_item = outer_workid;
    for (int dim = mean_out_ndim - 1; dim >= 0; --dim) {
        mean_out_index[dim] = flattened_work_item % mean_out_shape[dim];
        flattened_work_item /= mean_out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = stddev_out_ndim - 1; dim >= 0; --dim) {
        stddev_out_index[dim] = flattened_work_item % mean_out_shape[dim];
        flattened_work_item /= stddev_out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = a_ndim - 1; dim >= 0; --dim) {
        bool reduce_this_dim = false;
        for (int ra=0; ra < nreduced_dim; ++ra) {
            if (dim == reduced_axes[ra]) {
                reduce_this_dim = true;
                break;
            }
        }
        if (!reduce_this_dim) {
            in_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }
    }

    auto numel = mean_out_numel;
    for (int64_t i = outer_workid; i < numel; i+=outer_gridsize) {
        // Reset accumulators
        sdata[tid] = 0;
        s2data[tid] = 0;
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
            
            // keep track of x^2 accumulation for stddev
            auto x_square = in_val * in_val;
            if constexpr (is_floating<in_datatype>::value) {
                auto y2 = x_square - s2data_c;
                kahan(s2data[tid], y2);
                s2data_c = y2;
            } else {
                s2data[tid] += x_square;
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
        int64_t stddev_out_linear_index = 0;
        for (int dim = 0; dim < mean_out_ndim; ++dim) {
            out_linear_index += mean_out_index[dim] * mean_out_strides[dim];
            stddev_out_linear_index += stddev_out_index[dim] * stddev_out_strides[dim];
        }

        // Normalize accumulated inputs as needed.
        sdata[tid] /= reduction_factor;
        s2data[tid] /= reduction_factor;
        __syncthreads();
        for (int s=blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) {
                // __syncthreads();
                // printf("(%i) reduced sum is sdata[%i] + sdata[%i]. (%f + %f)\n", tid, tid, tid+s, sdata[tid], sdata[tid+s]);
                sdata[tid] += sdata[tid+s];
                s2data[tid] += s2data[tid+s];
            }
            __syncthreads();
        }
        // Write the final accumulated value to global output
        if (tid == 0) {
            // out_data[out_linear_index] = static_cast<out_datatype>((s2data[0] - sdata[0]*sdata[0]));
            // printf("E[X^2] = %f - E[X]^2 = %f\n", s2data[0], sdata[0]*sdata[0]);
            auto diff = (s2data[0] - sdata[0]*sdata[0]);
            diff = max((in_datatype)0, diff);
            mean_out_data[out_linear_index] = static_cast<out_datatype>(sdata[0]);
            stddev_out_data[stddev_out_linear_index] = static_cast<out_datatype>(sqrt(diff));
        }

        increment_size = gridDim.x;
        // Increment out and in pointers to next non-reduced dims
        for (int dim = mean_out_ndim - 1; dim >= 0; --dim) {
            mean_out_index[dim] += increment_size;
            if (mean_out_index[dim] < mean_out_shape[dim]) {
                break;
            } else {
                increment_size = mean_out_index[dim] / mean_out_shape[dim];
                mean_out_index[dim] = mean_out_index[dim] % mean_out_shape[dim];
            }
        }
        for (int dim = stddev_out_ndim - 1; dim >= 0; --dim) {
            stddev_out_index[dim] += increment_size;
            if (stddev_out_index[dim] < stddev_out_shape[dim]) {
                break;
            } else {
                increment_size = stddev_out_index[dim] / stddev_out_shape[dim];
                stddev_out_index[dim] = stddev_out_index[dim] % stddev_out_shape[dim];
            }
        }

        increment_size = gridDim.x;
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
                in_index[dim] += increment_size;
                if (in_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    increment_size = in_index[dim] / a_shape[dim];
                    in_index[dim] = in_index[dim] % a_shape[dim];
                }
            }
        }
    }
}

/**
 * masked version
 */
template <typename out_datatype, typename in_datatype, reductiontype Op>
__global__ void reduction_impl(out_datatype* out_data, int64_t* out_shape, int64_t* out_strides, int64_t out_ndim, int64_t out_numel,
                        in_datatype* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        uint8_t* mask_data, int64_t* mask_shape, int64_t* mask_strides, int64_t mask_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    using accumulator_type = in_datatype;
    auto tid = threadIdx.x;
    auto outer_workid = blockIdx.x;
    auto outer_gridsize = gridDim.x;

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

    // Initialize ndindex per threadblock
    auto flattened_work_item = outer_workid;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        out_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = a_ndim - 1; dim >= 0; --dim) {
        bool reduce_this_dim = false;
        for (int ra=0; ra < nreduced_dim; ++ra) {
            if (dim == reduced_axes[ra]) {
                reduce_this_dim = true;
                break;
            }
        }
        if (!reduce_this_dim) {
            in_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }
    }

    auto numel = out_numel;
    for (int64_t i = outer_workid; i < numel; i+=outer_gridsize) {
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

        increment_size = gridDim.x;
        // Increment out and in pointers to next non-reduced dims
        for (int dim = out_ndim - 1; dim >= 0; --dim) {
            out_index[dim] += increment_size;
            if (out_index[dim] < out_shape[dim]) {
                break;
            } else {
                increment_size = out_index[dim] / out_shape[dim];
                out_index[dim] = out_index[dim] % out_shape[dim];
            }
        }

        increment_size = gridDim.x;
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
                in_index[dim] += increment_size;
                if (in_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    increment_size = in_index[dim] / a_shape[dim];
                    in_index[dim] = in_index[dim] % a_shape[dim];
                }
            }
        }
    }
}


__global__ void mean_stddev_impl(float* mean_out_data, int64_t* mean_out_shape, int64_t* mean_out_strides, int64_t mean_out_ndim, int64_t mean_out_numel,
                        float* stddev_out_data, int64_t* stddev_out_shape, int64_t* stddev_out_strides, int64_t stddev_out_ndim, int64_t stddev_out_numel,
                        float* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        uint8_t* mask_data, int64_t* mask_shape, int64_t* mask_strides, int64_t mask_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    using out_datatype = float;
    using in_datatype = float;
    using accumulator_type = float;
    auto tid = threadIdx.x;
    auto outer_workid = blockIdx.x;
    auto outer_gridsize = gridDim.x;

    // Assume we've been given shared memory with enough space for each thread
    // to hold an accumulator of the in_datatype size (and one for x^2 if stddev or var)
    // Break that shared memory up in to appropriate chunks easily indexed by smem
    extern __shared__ char shared_memory[];
    accumulator_type* sdata = reinterpret_cast<accumulator_type*>(shared_memory);
    accumulator_type sdata_c = 0;
    accumulator_type* s2data = sdata + blockDim.x;
    accumulator_type s2data_c = 0;

    int count_offset = blockDim.x;
    count_offset = 2*blockDim.x;
    uint32_t* scount = reinterpret_cast<uint32_t*>(sdata + count_offset);

    int64_t out_index[MAX_NDIM] = {0};
    int64_t in_index[MAX_NDIM] = {0};
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    // Initialize ndindex per threadblock
    auto flattened_work_item = outer_workid;
    for (int dim = mean_out_ndim - 1; dim >= 0; --dim) {
        out_index[dim] = flattened_work_item % mean_out_shape[dim];
        flattened_work_item /= mean_out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = a_ndim - 1; dim >= 0; --dim) {
        bool reduce_this_dim = false;
        for (int ra=0; ra < nreduced_dim; ++ra) {
            if (dim == reduced_axes[ra]) {
                reduce_this_dim = true;
                break;
            }
        }
        if (!reduce_this_dim) {
            in_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }
    }

    auto numel = mean_out_numel;
    for (int64_t i = outer_workid; i < numel; i+=outer_gridsize) {
        // Reset accumulators
        sdata[tid] = 0;
        scount[tid] = 0;
        s2data[tid] = 0;
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
                auto x_square = in_val * in_val;
                auto y2 = x_square - s2data_c;
                kahan(s2data[tid], y2);
                s2data_c = y2;

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
        for (int dim = 0; dim < mean_out_ndim; ++dim) {
            out_linear_index += out_index[dim] * mean_out_strides[dim];
        }

        __syncthreads();
        for (int s=blockDim.x/2; s > 0; s>>=1) {
            if (tid < s) {
                // __syncthreads();
                // printf("(%i) reduced sum is sdata[%i] + sdata[%i]. (%f + %f)\n", tid, tid, tid+s, sdata[tid], sdata[tid+s]);
                sdata[tid] += sdata[tid+s];
                scount[tid] += scount[tid+s];
                s2data[tid] += s2data[tid+s];
            }
            __syncthreads();
        }
        // Write the final accumulated value to global output
        if (tid == 0) {
            // TODO: guard against scount == 0

                sdata[0] /= scount[0];
                s2data[0] /= scount[0];
                // out_data[out_linear_index] = static_cast<out_datatype>((s2data[0] - sdata[0]*sdata[0]));
                // printf("E[X^2] = %f - E[X]^2 = %f\n", s2data[0], sdata[0]*sdata[0]);
                auto diff = (s2data[0] - sdata[0]*sdata[0]);
                diff = max((in_datatype)0, diff);
                mean_out_data[out_linear_index] = static_cast<out_datatype>(sdata[0]);
                stddev_out_data[out_linear_index] = static_cast<out_datatype>(sqrt(diff));
        }

        increment_size = gridDim.x;
        // Increment out and in pointers to next non-reduced dims
        for (int dim = mean_out_ndim - 1; dim >= 0; --dim) {
            out_index[dim] += increment_size;
            if (out_index[dim] < mean_out_shape[dim]) {
                break;
            } else {
                increment_size = out_index[dim] / mean_out_shape[dim];
                out_index[dim] = out_index[dim] % mean_out_shape[dim];
            }
        }

        increment_size = gridDim.x;
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
                in_index[dim] += increment_size;
                if (in_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    increment_size = in_index[dim] / a_shape[dim];
                    in_index[dim] = in_index[dim] % a_shape[dim];
                }
            }
        }
    }
}


/**
 * max,...
 */
template <typename out_datatype, comparison_reductiontype Op>
__global__ void comparison_reduction_impl(out_datatype* out_data, int64_t* out_shape, int64_t* out_strides, int64_t out_ndim, int64_t out_numel,
                        out_datatype* a_data, int64_t* a_shape, int64_t* a_strides, int64_t a_ndim,
                        // uint8_t* mask_data, int64_t* mask_shape, int64_t* mask_strides, int64_t mask_ndim,
                        int64_t* reduced_axes, int64_t nreduced_dim, int64_t reduction_factor) {
    // A good primer on reduction kernels: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    auto tid = threadIdx.x;
    auto outer_workid = blockIdx.x;
    auto outer_gridsize = gridDim.x;

    // Assume we've been given shared memory with enough space for each thread
    // to hold its stat
    // Break that shared memory up in to appropriate chunks easily indexed by smem
    extern __shared__ char shared_memory[];
    out_datatype* sdata = reinterpret_cast<out_datatype*>(shared_memory);
    // where do we initialize_sdata?

    // datatype sdata_c = 0; // need this to be min
    // int count_offset = blockDim.x;
    // if constexpr(Op == reductiontype::stddev || Op == reductiontype::var) {
    //     count_offset = 2*blockDim.x;
    // }
    // uint32_t* scount = reinterpret_cast<uint32_t*>(sdata + count_offset);

    int64_t out_index[MAX_NDIM] = {0};
    int64_t in_index[MAX_NDIM] = {0};
    int64_t reduced_nd_index[MAX_NDIM] = {0};

    // Initialize ndindex per threadblock
    auto flattened_work_item = outer_workid;
    for (int dim = out_ndim - 1; dim >= 0; --dim) {
        out_index[dim] = flattened_work_item % out_shape[dim];
        flattened_work_item /= out_shape[dim];
    }
    flattened_work_item = outer_workid;
    for (int dim = a_ndim - 1; dim >= 0; --dim) {
        bool reduce_this_dim = false;
        for (int ra=0; ra < nreduced_dim; ++ra) {
            if (dim == reduced_axes[ra]) {
                reduce_this_dim = true;
                break;
            }
        }
        if (!reduce_this_dim) {
            in_index[dim] = flattened_work_item % a_shape[dim];
            flattened_work_item /= a_shape[dim];
        }
    }

    auto numel = out_numel;
    for (int64_t i = outer_workid; i < numel; i+=outer_gridsize) {
        // Reset accumulators
        // sdata[tid] = 0; // Initialize to datatype min value
        sdata[tid] = ::cuda::std::numeric_limits<out_datatype>::lowest();
        // scount[tid] = 0;
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
            // int64_t mask_linear_index = 0;
            for (int dim = 0; dim < a_ndim; ++dim) {
                in_linear_index += reduced_nd_index[dim] * a_strides[dim];
                // mask_linear_index += reduced_nd_index[dim] * mask_strides[dim];
            }

            // Read the input data
            auto in_val = a_data[in_linear_index];
            if (in_val > sdata[tid]) {
                sdata[tid] = in_val;
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
                if (sdata[tid+s] > sdata[tid] ) {
                    sdata[tid] = sdata[tid+s];
                }

            }
            __syncthreads();
        }
        // Write the final accumulated value to global output
        if (tid == 0) {
            out_data[out_linear_index] = static_cast<out_datatype>(sdata[0]);
        }

        increment_size = gridDim.x;
        // Increment out and in pointers to next non-reduced dims
        for (int dim = out_ndim - 1; dim >= 0; --dim) {
            out_index[dim] += increment_size;
            if (out_index[dim] < out_shape[dim]) {
                break;
            } else {
                increment_size = out_index[dim] / out_shape[dim];
                out_index[dim] = out_index[dim] % out_shape[dim];
            }
        }

        increment_size = gridDim.x;
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
                in_index[dim] += increment_size;
                if (in_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    increment_size = in_index[dim] / a_shape[dim];
                    in_index[dim] = in_index[dim] % a_shape[dim];
                }
            }
        }
    }
}


template <reductiontype Reduction>
struct statistical_launch_wrapper {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray out, const ndarray &a, std::vector<int64_t> reduced_axes) {
        const auto out_offset     = out.offsets();
        const auto out_shape      = out.shape();
        const auto out_strides    = out.strides();
        int64_t    out_ptr_offset = std::accumulate(out_offset.begin(), out_offset.end(), 0LL);
        auto       out_data       = out.data_ptr<out_datatype>() + out_ptr_offset;

        thrust::device_vector<int64_t> dev_out_shape(out_shape.begin(), out_shape.end());
        thrust::device_vector<int64_t> dev_out_strides(out_strides.begin(), out_strides.end());

        const auto a_offset     = a.offsets();
        const auto a_shape      = a.shape();
        const auto a_strides    = a.strides();
        int64_t    a_ptr_offset = std::accumulate(a_offset.begin(), a_offset.end(), 0LL);
        auto       a_data       = a.data_ptr<in_datatype>() + a_ptr_offset;

        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }
        thrust::device_vector<int64_t> dev_reduced_axes(reduced_axes.begin(), reduced_axes.end());

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, a.device().device_id);

        int block_size = 512;
        int required_smem_per_thread = sizeof(in_datatype);
        if (Reduction == reductiontype::stddev || Reduction == reductiontype::var) {
            // need to keep an E[X^2] which will take double the smem
            required_smem_per_thread += sizeof(in_datatype);
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
        // TODO (low prio): find a good maximum # blocks
        int num_blocks = std::min<int64_t>(1024, out_numel);
        int smem_per_block = required_smem_per_thread * block_size;
        if (Reduction == reductiontype::stddev || Reduction == reductiontype::var) {
            // TODO: Check we have enough smem
            smem_per_block *= 2; 
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
        const auto a_offset  = a.offsets();
        const auto a_strides = a.strides();

        thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
        thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

        auto mask_shape = mask.shape();
        auto mask_data = mask.data_ptr<uint8_t>();
        const auto mask_offset   = mask.offsets();
        const auto mask_strides  = mask.strides();

        thrust::device_vector<int64_t> dev_mask_shape(mask_shape.begin(), mask_shape.end());
        thrust::device_vector<int64_t> dev_mask_strides(mask_strides.begin(), mask_strides.end());

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
            // need to keep an E[X^2] which will take double the smem
            required_smem_per_thread += sizeof(in_datatype);
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
        // TODO (low prio): find a good maximum # blocks
        int num_blocks = std::min<int64_t>(1024, out_numel);
        int smem_per_block = required_smem_per_thread * block_size;
        reduction_impl<out_datatype, in_datatype, Reduction><<<num_blocks, block_size, smem_per_block>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                                a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()), dev_a_shape.size(),
                                                                mask_data, thrust::raw_pointer_cast(dev_mask_shape.data()), thrust::raw_pointer_cast(dev_mask_strides.data()), dev_mask_shape.size(),
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

template <comparison_reductiontype Reduction>
struct comparison_reduction_launch_wrapper {
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

        int block_size = 512;
        int required_smem_per_thread = sizeof(out_datatype);
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
        int num_blocks = std::min<int64_t>(1024, out_numel);
        int smem_per_block = required_smem_per_thread * block_size;
        comparison_reduction_impl<out_datatype, Reduction><<<num_blocks, block_size, smem_per_block>>>(out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
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
};


std::pair<ndarray, ndarray> mean_stddev_wrapper(ndarray mean_out, ndarray stddev_out, const ndarray &a, std::vector<int64_t> reduced_axes) {
    // This is mostly a copy of the template stuct of statistical_reduction_wrapper but uses a fixed float dtype and two output arrays
    using out_datatype = float;
    using in_datatype = float;
    auto       mean_out_data  = mean_out.data_ptr<out_datatype>();
    const auto mean_out_offset = mean_out.offsets();
    const auto mean_out_shape = mean_out.shape();
    const auto mean_out_strides = mean_out.strides();        

    auto       stddev_out_data  = stddev_out.data_ptr<out_datatype>();
    const auto stddev_out_offset = stddev_out.offsets();
    const auto stddev_out_shape = stddev_out.shape();
    const auto stddev_out_strides = stddev_out.strides();        

    thrust::device_vector<int64_t> dev_out_shape(mean_out_shape.begin(), mean_out_shape.end());
    thrust::device_vector<int64_t> dev_out_strides(mean_out_strides.begin(), mean_out_strides.end());

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

    int block_size = 512;
    int required_smem_per_thread = sizeof(in_datatype);
    // need to keep an E[X^2] which will take double the smem
    required_smem_per_thread += sizeof(in_datatype);
    if (reduced_elements < block_size) {
        // Reduce to a block size that is a power of 2 but less than number of items that need reducing
        block_size = reduced_elements | (reduced_elements >> 1);
        block_size |= (block_size >> 2);
        block_size |= (block_size >> 4);
        block_size |= (block_size >> 8);
        block_size |= (block_size >> 16);
        block_size = block_size - (block_size >> 1);
    }
    auto out_numel = mean_out.numel();
    // TODO (low prio): find a good maximum # blocks
    int num_blocks = std::min<int64_t>(1024, out_numel);
    int smem_per_block = required_smem_per_thread * block_size;
    // TODO: Check we have enough smem
    smem_per_block *= 2; 
    mean_stddev_impl<<<num_blocks, block_size, smem_per_block>>>(mean_out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                            stddev_out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
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

    return std::make_pair(mean_out, stddev_out);
}


std::pair<ndarray, ndarray> mean_stddev_wrapper(ndarray mean_out, ndarray stddev_out, const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
    // This is mostly a copy of the template stuct of statistical_reduction_wrapper but uses a fixed float dtype and two output arrays
    using out_datatype = float;
    using in_datatype = float;
    auto       mean_out_data  = mean_out.data_ptr<out_datatype>();
    const auto mean_out_offset = mean_out.offsets();
    const auto mean_out_shape = mean_out.shape();
    const auto mean_out_strides = mean_out.strides();

    auto       stddev_out_data  = stddev_out.data_ptr<out_datatype>();
    const auto stddev_out_offset = stddev_out.offsets();
    const auto stddev_out_shape = stddev_out.shape();
    const auto stddev_out_strides = stddev_out.strides();

    thrust::device_vector<int64_t> dev_out_shape(mean_out_shape.begin(), mean_out_shape.end());
    thrust::device_vector<int64_t> dev_out_strides(mean_out_strides.begin(), mean_out_strides.end());

    auto a_shape = a.shape();
    auto a_data = a.data_ptr<in_datatype>();
    const auto a_offset  = a.offsets();
    const auto a_strides = a.strides();

    thrust::device_vector<int64_t> dev_a_shape(a_shape.begin(), a_shape.end());
    thrust::device_vector<int64_t> dev_a_strides(a_strides.begin(), a_strides.end());

    auto mask_shape = mask.shape();
    auto mask_data = mask.data_ptr<uint8_t>();
    const auto mask_offset   = mask.offsets();
    const auto mask_strides  = mask.strides();

    thrust::device_vector<int64_t> dev_mask_shape(mask_shape.begin(), mask_shape.end());
    thrust::device_vector<int64_t> dev_mask_strides(mask_strides.begin(), mask_strides.end());

    int64_t reduced_elements = 1;
    for (auto &d : reduced_axes) {
        reduced_elements *= a_shape[d];
    }
    thrust::device_vector<int64_t> dev_reduced_axes(reduced_axes.begin(), reduced_axes.end());

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, a.device().device_id);

    int block_size = 512;
    int required_smem_per_thread = sizeof(in_datatype) +4; // 4B for the mask count
    // need to keep an E[X^2] which will take double the smem
    required_smem_per_thread += sizeof(in_datatype);
    if (reduced_elements < block_size) {
        // Reduce to a block size that is a power of 2 but less than number of items that need reducing
        block_size = reduced_elements | (reduced_elements >> 1);
        block_size |= (block_size >> 2);
        block_size |= (block_size >> 4);
        block_size |= (block_size >> 8);
        block_size |= (block_size >> 16);
        block_size = block_size - (block_size >> 1);
    }
    auto out_numel = mean_out.numel();
    // TODO (low prio): find a good maximum # blocks
    int num_blocks = std::min<int64_t>(1024, out_numel);
    int smem_per_block = required_smem_per_thread * block_size;
    // TODO: Check we have enough smem
    smem_per_block *= 2; 
    mean_stddev_impl<<<num_blocks, block_size, smem_per_block>>>(mean_out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                            stddev_out_data, thrust::raw_pointer_cast(dev_out_shape.data()), thrust::raw_pointer_cast(dev_out_strides.data()), dev_out_shape.size(), out_numel,
                                                            a_data, thrust::raw_pointer_cast(dev_a_shape.data()), thrust::raw_pointer_cast(dev_a_strides.data()), dev_a_shape.size(),
                                                            mask_data, thrust::raw_pointer_cast(dev_mask_shape.data()), thrust::raw_pointer_cast(dev_mask_strides.data()), dev_mask_shape.size(),
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

    return std::make_pair(mean_out, stddev_out);
}


ndarray bland::cuda::max(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    // return dispatch_new4<comparison_reduction_launch_wrapper<comparison_reductiontype::max>>(out, a, reduced_axes);
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, comparison_reduction_launch_wrapper<comparison_reductiontype::max>>(out, a, reduced_axes);
}


ndarray bland::cuda::sum(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::sum>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_sum(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::sum>>(out, a, mask, reduced_axes);
}

ndarray bland::cuda::mean(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::mean>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_mean(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::mean>>(out, a, mask, reduced_axes);
}

ndarray bland::cuda::stddev(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::stddev>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_stddev(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::stddev>>(out, a, mask, reduced_axes);
}

std::pair<ndarray, ndarray> bland::cuda::mean_stddev(const ndarray &a, ndarray &out_mean, ndarray &out_stddev, std::vector<int64_t> reduced_axes) {
    return mean_stddev_wrapper(out_mean, out_stddev, a, reduced_axes);
}

std::pair<ndarray, ndarray> bland::cuda::masked_mean_stddev(const ndarray &a, const ndarray &mask, ndarray &out_mean, ndarray &out_stddev, std::vector<int64_t> reduced_axes) {
    return mean_stddev_wrapper(out_mean, out_stddev, a, mask, reduced_axes);
}

ndarray bland::cuda::var(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::var>>(out, a, reduced_axes);
}

ndarray bland::cuda::masked_var(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return constrained_dispatch_nd<Constraints::NoInt | Constraints::NoUInt, statistical_launch_wrapper<reductiontype::var>>(out, a, mask, reduced_axes);
}

int64_t bland::cuda::count_true(ndarray x) {
    return dispatch_summary<count_launcher>(x);
}

