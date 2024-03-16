#include "statistical_cpu.hpp"
#include "ops_cpu.hpp" // square (for var), copy (standardized_moment)

#include "bland/ndarray.hpp"

#include "internal/dispatcher.hpp"

#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"

#include "internal/shape_helpers.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm> // std::find
#include <numeric>   // std::accumulate

using namespace bland;
using namespace bland::cpu;

constexpr float eps = 1e-8f;

struct mean_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, std::vector<int64_t> reduced_axes) {
        auto a_data = a.data_ptr<in_datatype>();

        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a.shape()[d];
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto out_data = out.data_ptr<out_datatype>();

        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        constexpr bool is_floating_point = std::is_floating_point<out_datatype>::value;
        using accumulator_type = std::conditional_t<is_floating_point, double, out_datatype>;

        out_datatype scale = 1.0 / static_cast<out_datatype>(reduced_elements - 1);
        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto             reduce_nd_index = input_index;
            accumulator_type mean            = 0;
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                mean += a_data[input_linear_index];
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis]) * out_strides[axis];
            }

            out_data[out_linear_index] = static_cast<out_datatype>(mean / reduced_elements);
            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

struct masked_mean_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto                 out_data    = out.data_ptr<out_datatype>();
        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 a_data    = a.data_ptr<in_datatype>();
        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }

        if (mask.dtype().code != ndarray::datatype::uint8.code && mask.dtype().bits != 8) {
            throw std::runtime_error("masked_mean: mask dtype is not uint8_t");
        }
        auto mask_data    = mask.data_ptr<uint8_t>();
        auto mask_shape   = mask.shape();
        auto mask_strides = mask.strides();
        auto mask_offset  = mask.offsets();

        if (a.ndim() == mask.ndim()) {
            for (int dim = 0; dim < a.ndim(); ++dim) {
                if (a_shape[dim] != mask_shape[dim]) {
                    throw std::runtime_error("mask_mean: a shape does not match mask shape");
                }
            }
        } else {
            throw std::runtime_error("mask_mean: dims of a shape does not match dims of mask shape");
        }

        constexpr bool is_floating_point = std::is_floating_point<out_datatype>::value;
        using accumulator_type = std::conditional_t<is_floating_point, double, out_datatype>;

        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto             reduce_nd_index  = input_index;
            accumulator_type mean             = 0;
            int64_t          elements_in_mean = 0;
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                if (mask_data[input_linear_index] == 0) {
                    mean += a_data[input_linear_index];
                    elements_in_mean += 1;
                }
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis]) * out_strides[axis];
            }

            if (elements_in_mean == 0) {
                throw std::runtime_error("masked_mean: there are no non-masked elements to take the mean of");
            }
            out_data[out_linear_index] = static_cast<out_datatype>(mean / (elements_in_mean - 1));
            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

ndarray bland::cpu::mean(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<mean_impl>(out, a, reduced_axes);
}


ndarray bland::cpu::masked_mean(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<masked_mean_impl>(out, a, mask, reduced_axes);
}

struct stddev_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, std::vector<int64_t> reduced_axes) {

        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto                 out_data    = out.data_ptr<out_datatype>();
        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 a_data    = a.data_ptr<in_datatype>();
        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        // TODO, do E[x^2] - E[x]^2 to reduce a redundant pass through data
        auto means = ndarray(out.shape(), out.dtype(), out.device());
        mean_impl::call<out_datatype, in_datatype>(means, a, reduced_axes);

        auto                 mean_data    = means.data_ptr<out_datatype>();
        auto                 mean_shape   = means.shape();
        auto                 mean_strides = means.strides();
        auto                 mean_offsets = means.offsets();
        std::vector<int64_t> mean_index(mean_shape.size(), 0);

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }

        constexpr bool is_floating_point = std::is_floating_point<out_datatype>::value;
        using accumulator_type = std::conditional_t<is_floating_point, double, out_datatype>;

        // TODO (flexibility): add correction option (noff in torch)
        out_datatype scale = 1.0 / static_cast<out_datatype>(reduced_elements - 1);

        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto             reduce_nd_index = input_index;
            accumulator_type dev             = 0;

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t mean_linear_index = 0;
            for (int axis = 0; axis < mean_shape.size(); ++axis) {
                mean_linear_index += mean_offsets[axis] + (mean_index[axis] % mean_shape[axis]) * mean_strides[axis];
            }

            auto this_reduction_mean = mean_data[mean_linear_index];
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                auto deviation = (a_data[input_linear_index] - this_reduction_mean);
                dev += (deviation * deviation * scale);
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis] % out_shape[axis]) * out_strides[axis];
            }

            out_data[out_linear_index] = static_cast<out_datatype>(std::sqrt(dev));

            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            for (int axis = mean_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++mean_index[axis] != mean_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    mean_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

ndarray bland::cpu::stddev(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<stddev_impl>(out, a, reduced_axes);
}

struct masked_stddev_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto                 out_data    = out.data_ptr<out_datatype>();
        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 a_data    = a.data_ptr<in_datatype>();
        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        if (mask.dtype().code != ndarray::datatype::uint8.code && mask.dtype().bits != 8) {
            throw std::runtime_error("masked_mean: mask dtype is not uint8_t");
        }
        auto mask_data    = mask.data_ptr<uint8_t>();
        auto mask_shape   = mask.shape();
        auto mask_strides = mask.strides();
        auto mask_offset  = mask.offsets();

        if (a.ndim() == mask.ndim()) {
            for (int dim = 0; dim < a.ndim(); ++dim) {
                if (a_shape[dim] != mask_shape[dim]) {
                    throw std::runtime_error("mask_mean: a shape does not match mask shape");
                }
            }
        } else {
            throw std::runtime_error("mask_mean: dims of a shape does not match dims of mask shape");
        }

        // TODO, do E[x^2] - E[x]^2 to reduce a redundant pass through data
        auto means = ndarray(out.shape(), out.dtype(), out.device());
        masked_mean_impl::call<out_datatype, in_datatype>(means, a, mask, reduced_axes);

        auto                 mean_data    = means.data_ptr<out_datatype>();
        auto                 mean_shape   = means.shape();
        auto                 mean_strides = means.strides();
        auto                 mean_offsets = means.offsets();
        std::vector<int64_t> mean_index(mean_shape.size(), 0);

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a_shape[d];
        }

        constexpr bool is_floating_point = std::is_floating_point<out_datatype>::value;
        using accumulator_type = std::conditional_t<is_floating_point, double, out_datatype>;

        // TODO (flexibility): add correction option (noff in torch)
        out_datatype scale = 1.0 / static_cast<out_datatype>(reduced_elements - 1);

        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto             reduce_nd_index = input_index;
            accumulator_type dev             = 0;
            int64_t          elements_in_dev = 0;

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t mean_linear_index = 0;
            for (int axis = 0; axis < mean_shape.size(); ++axis) {
                mean_linear_index += mean_offsets[axis] + (mean_index[axis] % mean_shape[axis]) * mean_strides[axis];
            }

            auto this_reduction_mean = mean_data[mean_linear_index];
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                if (mask_data[input_linear_index] == 0) {
                    auto deviation = (a_data[input_linear_index] - this_reduction_mean);
                    dev += (deviation * deviation);
                    elements_in_dev += 1;
                }
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis] % out_shape[axis]) * out_strides[axis];
            }

            if (elements_in_dev == 0) {
                throw std::runtime_error("masked_stddev: there are no non-masked elements to take the mean of");
            }
            out_data[out_linear_index] = static_cast<out_datatype>(std::sqrt(dev / elements_in_dev));

            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            for (int axis = mean_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++mean_index[axis] != mean_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    mean_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

ndarray bland::cpu::masked_stddev(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<masked_stddev_impl>(out, a, mask, reduced_axes);
}

ndarray bland::cpu::var(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    cpu::stddev(a, out, reduced_axes);
    return cpu::square(out, out);
}

ndarray bland::cpu::masked_var(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    cpu::masked_stddev(a, mask, out, reduced_axes);
    return cpu::square(out, out);
}

struct standardized_moment_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, std::vector<int64_t> reduced_axes, int degree) {

        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // TODO, allow passing in means as an arg
        auto means = ndarray(out.shape(), out.dtype(), out.device());
        mean_impl::call<out_datatype, in_datatype>(means, a, reduced_axes);

        auto a_data = a.data_ptr<in_datatype>();

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a.shape()[d];
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto out_data = out.data_ptr<out_datatype>();

        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 mean_data    = means.data_ptr<out_datatype>();
        auto                 mean_shape   = means.shape();
        auto                 mean_strides = means.strides();
        auto                 mean_offsets = means.offsets();
        std::vector<int64_t> mean_index(mean_shape.size(), 0);

        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        // TODO (flexibility): add correction option (noff in torch)
        out_datatype scale = 1.0 / static_cast<out_datatype>(reduced_elements - 1);

        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto   reduce_nd_index = input_index;
            double moment          = 0; // This is the moment
            double squared_dev     = 0; // This will basically be variance

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t mean_linear_index = 0;
            for (int axis = 0; axis < mean_shape.size(); ++axis) {
                mean_linear_index += mean_offsets[axis] + (mean_index[axis] % mean_shape[axis]) * mean_strides[axis];
            }

            auto this_reduction_mean = mean_data[mean_linear_index];
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                auto deviation = (a_data[input_linear_index] - this_reduction_mean);
                moment += std::pow(deviation, degree) * scale;
                squared_dev += deviation * deviation * scale;
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis] % out_shape[axis]) * out_strides[axis];
            }

            out_data[out_linear_index] = moment / (eps + std::pow(squared_dev, static_cast<float>(degree) / 2.0f));
            // out_data[out_linear_index] = static_cast<out_datatype>(std::sqrt(squared_dev));

            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            for (int axis = mean_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++mean_index[axis] != mean_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    mean_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

ndarray bland::cpu::standardized_moment(const ndarray &a, int degree, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<standardized_moment_impl>(out, a, reduced_axes, degree);
}

// median
// TODO: generalize this to an ndarray op
float bland::cpu::median(const ndarray &x, std::vector<int64_t> axes) {
    const size_t size = x.numel();

    auto x_copy = bland::ndarray(x.shape(), x.dtype(), x.device());
    x_copy = cpu::copy(x, x_copy);
    auto x_data = x_copy.data_ptr<float>();
    // Compute the position of the median
    auto mid = x_data + size / 2;

    // Use nth_element to rearrange elements such that the element at mid is the element that would be in that
    // position if the whole array was sorted
    std::nth_element(x_data, mid, x_data + size);

    auto median = *mid;

    // If the size is even, we need the mid-point of the other middle point
    if (size % 2 == 0) {
        std::nth_element(x_data, mid - 1, x_data + size);
        median = (median + *(mid - 1)) / 2;
    }

    return median;
}

/*************
 **** Sum ****
 ************/

struct sum_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, std::vector<int64_t> reduced_axes) {
        auto a_data = a.data_ptr<in_datatype>();

        if (reduced_axes.empty()) {
            for (int axis = 0; axis < a.ndim(); ++axis) {
                reduced_axes.push_back(axis);
            }
        }

        // Number of elements that get reduced to a single output element
        int64_t reduced_elements = 1;
        for (auto &d : reduced_axes) {
            reduced_elements *= a.shape()[d];
        }

        // The out array may actually be a slice! So need to respect its strides, shapes, and offsets
        auto out_data = out.data_ptr<out_datatype>();

        auto                 out_shape   = out.shape();
        auto                 out_strides = out.strides();
        auto                 out_offset  = out.offsets();
        std::vector<int64_t> out_index(out_shape.size(), 0);

        auto                 a_shape   = a.shape();
        auto                 a_strides = a.strides();
        auto                 a_offset  = a.offsets();
        std::vector<int64_t> input_index(a_shape.size(), 0);

        // Loop over the dimensions of the array and perform the reduction operation
        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto        reduce_nd_index = input_index;
            in_datatype total           = 0;
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                total += a_data[input_linear_index];
                // Increment the multi-dimensional index
                for (int i = reduced_axes.size() - 1; i >= 0; --i) {
                    auto d = reduced_axes[i];
                    // If we're not at the end of this dim, keep going
                    if (++reduce_nd_index[d] != a_shape[d]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        reduce_nd_index[d] = 0;
                    }
                }
            }

            int64_t out_linear_index = 0;
            for (int axis = 0; axis < out_shape.size(); ++axis) {
                out_linear_index += out_offset[axis] + (out_index[axis]) * out_strides[axis];
            }

            out_data[out_linear_index] = total;

            // Increment the multi-dimensional output index
            for (int axis = out_shape.size() - 1; axis >= 0; --axis) {
                // If we're not at the end of this dim, keep going
                if (++out_index[axis] != out_shape[axis]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    out_index[axis] = 0;
                }
            }
            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int axis = a_shape.size() - 1; axis >= 0; --axis) {
                if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                    // If we're not at the end of this dim, keep going
                    if (++input_index[axis] != a_shape[axis]) {
                        break;
                    } else {
                        // Otherwise, set it to 0 and move down to the next dim
                        input_index[axis] = 0;
                    }
                }
            }
        }

        return out;
    }
};

ndarray bland::cpu::sum(const ndarray &a, ndarray &out, std::vector<int64_t> axes) {
    return dispatch_new<sum_impl>(out, a, axes);
}


struct count_impl {
    template <typename in_datatype>
    static inline int64_t call(const ndarray &a) {

        auto a_data    = a.data_ptr<in_datatype>();
        auto a_shape   = a.shape();
        auto a_strides = a.strides();
        auto a_offset  = a.offsets();

        std::vector<int64_t> input_index(a_shape.size(), 0);
        int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);

        int64_t count = 0;
        auto    numel = a.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            if (a_data[a_linear_index]) {
                ++count;
            }

            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int dim = a_shape.size() - 1; dim >= 0; --dim) {
                // If we're not at the end of this dim, keep going
                ++input_index[dim];
                a_linear_index += a_strides[dim];
                if (input_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    input_index[dim] = 0;
                    a_linear_index -= (a_shape[dim]) * a_strides[dim];
                }
            }
        }

        return count;
    }
};

int64_t bland::cpu::count_true(ndarray x) {
    return dispatch_summary<count_impl>(x);
}
