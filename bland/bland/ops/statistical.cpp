#include "bland/ndarray.hpp"
#include "bland/ops_statistical.hpp"
#include "bland/ops.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"
#include "shape_helpers.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm> // std::find
#include <numeric> // std::accumulate

using namespace bland;

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

        out_datatype scale = 1.0 / static_cast<out_datatype>(reduced_elements);
        // Loop over the dimensions of the array and perform the reduction operation
        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto         reduce_nd_index = input_index;
            out_datatype mean            = 0;
            for (int jj = 0; jj < reduced_elements; ++jj) {
                int64_t input_linear_index = 0;
                // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
                for (int axis = 0; axis < a_shape.size(); ++axis) {
                    input_linear_index += a_offset[axis] + (reduce_nd_index[axis] % a_shape[axis]) * a_strides[axis];
                }
                if (std::is_same<out_datatype, float>() || std::is_same<out_datatype, double>()) {
                    mean += (a_data[input_linear_index] * scale);
                } else {
                    mean += a_data[input_linear_index];
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

            if (std::is_same<out_datatype, float>() || std::is_same<out_datatype, double>()) {
                out_data[out_linear_index] = static_cast<out_datatype>(mean);
            } else {
                out_data[out_linear_index] = static_cast<out_datatype>(mean) / reduced_elements;
            }
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

ndarray bland::mean(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<mean_impl>(out, a, reduced_axes);
}

ndarray bland::mean(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return mean(a, out, reduced_axes);
}

struct stddev_impl {
    template <typename out_datatype, typename in_datatype>
    static inline ndarray call(ndarray &out, const ndarray &a, std::vector<int64_t> reduced_axes) {

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
        // Loop over the dimensions of the array and perform the reduction operation
        auto numel = out.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            auto   reduce_nd_index = input_index;
            double dev             = 0;

            // TODO (perf): move this inside the nd_index increment similar to the elementwise binary ops
            int64_t mean_linear_index = 0;
            for (int axis = 0; axis < mean_shape.size(); ++axis) {
                mean_linear_index += mean_offsets[axis] + (mean_index[axis] % mean_shape[axis]) * mean_strides[axis];
            }

            auto this_reduction_mean = mean_data[mean_linear_index];
            // fmt::print("We think this reduction mean is \n", this_reduction_mean);
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

ndarray bland::stddev(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<stddev_impl>(out, a, reduced_axes);
}
ndarray bland::stddev(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return stddev(a, out, reduced_axes);
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
        // Loop over the dimensions of the array and perform the reduction operation
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

            out_data[out_linear_index] = moment / std::pow(squared_dev, static_cast<float>(degree)/2.0f);
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

ndarray bland::standardized_moment(const ndarray &a, int degree, ndarray &out, std::vector<int64_t> reduced_axes) {
    return dispatch_new<standardized_moment_impl>(out, a, reduced_axes, degree);
}
ndarray bland::standardized_moment(const ndarray &a, int degree, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return standardized_moment(a, degree, out, reduced_axes);
}

// median
// TODO: generalize this to an ndarray op
float bland::median(const ndarray &x, std::vector<int64_t> axes) {
    const size_t size = x.numel();

    auto x_copy = copy(x);
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

ndarray bland::sum(const ndarray &a, ndarray &out, std::vector<int64_t> axes) {
    return dispatch_new<sum_impl>(out, a, axes);
}

ndarray bland::sum(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return sum(a, out, reduced_axes);
}