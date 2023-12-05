

#include "bland/ops.hpp"
#include "bland/ndarray.hpp"

#include "assignment_op.hpp"
#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"
#include "elementwise_unary_op.hpp"
#include "shape_helpers.hpp"

#include <fmt/core.h>

#include <algorithm> // std::find
#include <numeric>   // std::accumulate

using namespace bland;

/**
 * Copy (data)
 */
template <typename A>
struct elementwise_copy_op {
    static inline A call(const A a) { return a; }
};

ndarray bland::copy(ndarray a) {
    return dispatch<unary_op_impl_wrapper, elementwise_copy_op>(a);
}

ndarray bland::copy(ndarray a, ndarray &out) {
    return dispatch<unary_op_impl_wrapper, elementwise_copy_op>(a, out);
}

// Decide if this should exist or should just be the base case for those recursive calls...
ndarray_slice bland::slice(const ndarray &a, int64_t dim, int64_t start, int64_t end, int64_t stride) {
    // fmt::print("slice impl input data is at {}\n", a.data_ptr<void>());

    ndarray_slice sliced(a);

    if (start < 0) {
        // Negative is an offset from the end
        start = a.size(dim) + start - 1;
    } else if (start > a.size(dim)) {
        throw std::out_of_range("Invalid start for stride (begins beyond the end)");
    }
    if (end < 0) {
        // Negative is an offset from the end
        end = a.size(dim) + end - 1;
    } else if (end > a.size(dim)) {
        end = a.size(dim);
    }

    // The offset needs to use the current stride so it's an offset
    // of the current axis rather than the number of items at new stride
    sliced._tensor._offsets[dim] += start * sliced._tensor.strides[dim];

    sliced._tensor.shape[dim] = (end - start) / stride;
    sliced._tensor.strides[dim] *= stride;
    // fmt::print("slice impl output data is at {}\n", sliced.data_ptr<void>());

    return sliced;
}

// Base case for the recursive call
ndarray process_slice_specs(const ndarray_slice &sliced_result) {
    // fmt::print("process_slice_specs w/o args input data is at {}\n", sliced_result.data_ptr<void>());
    return sliced_result;
}

template <typename... Args>
ndarray process_slice_specs(const ndarray_slice &sliced_result, slice_spec slice_dim, Args... args) {
    // fmt::print("process_slice_specs w/ args input data is at {}\n", sliced_result.data_ptr<void>());
    ndarray new_slice = slice(sliced_result, slice_dim.dim, slice_dim.start, slice_dim.end, slice_dim.stride);
    // fmt::print("process_slice_specs new slice data is at {}\n", new_slice.data_ptr<void>());

    // Recursively process the remaining slice_spec arguments
    return process_slice_specs(new_slice, args...);
}

template <typename... Args>
ndarray_slice bland::slice(const ndarray &a, slice_spec slice_dim, Args... args) {
    // fmt::print("exposed slice: input to slice data is at {}\n", a.data_ptr<void>());
    ndarray sliced_result = slice(a, slice_dim.dim, slice_dim.start, slice_dim.end, slice_dim.stride);
    // fmt::print("exposed slice: sliced data is at {}\n", sliced_result.data_ptr<void>());

    // Process the remaining slice_spec arguments
    // TODO: there are bug-covered dragons surrounding the variations of assignment and
    // copy constructors such that assigning to sliced_result from a numpy-borrowed tensor
    // will produce a tensor with null data, but with a new variable it's fine.
    auto sliced_result1 = process_slice_specs(sliced_result, args...);

    // fmt::print("exposed slice: slice return will be at {}\n", sliced_result1.data_ptr<void>());

    return sliced_result1;
}

// Explicit instantiation of the slice function (all this to avoid va_args)
// If this winds up not being enough regularly, we might actually need to just move the definition to header...
template ndarray_slice bland::slice(const ndarray &a, slice_spec slice_dim);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice
bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice
bland::slice(const ndarray &a, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);
template ndarray_slice bland::slice(const ndarray &a,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec,
                                    slice_spec);

template <typename T>
ndarray bland::fill(ndarray a, T v) {
    return dispatch<assignment_op_impl_wrapper, T>(v, a);
}

template ndarray bland::fill<float>(ndarray a, float v);
template ndarray bland::fill<double>(ndarray a, double v);
template ndarray bland::fill<int8_t>(ndarray a, int8_t v);
template ndarray bland::fill<int16_t>(ndarray a, int16_t v);
template ndarray bland::fill<int32_t>(ndarray a, int32_t v);
template ndarray bland::fill<int64_t>(ndarray a, int64_t v);

struct sum_impl {
    template <typename in_datatype, typename out_datatype>
    static inline ndarray call(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
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
    return dispatch<sum_impl>(a, out, axes);
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




template <typename A, typename B>
struct greater_than_equals_to_op {
    static bool call(const A &a, const B &b) {
        return a >= b;
    }
};

template <typename A, typename B>
struct greater_than_op {
    static bool call(const A &a, const B &b) {
        return a > b;
    }
};

template <typename A, typename B>
struct less_than_op {
    static bool call(const A &a, const B &b) {
        return a < b;
    }
};

template <typename A, typename B>
struct less_than_equals_to_op {
    static bool call(const A &a, const B &b) {
        return a > b;
    }
};

template <typename S>
ndarray bland::operator >=(S lhs, ndarray rhs) {
    // lhs >= rhs translates to rhs < lhs
    ndarray out(rhs.shape());
    return dispatch<scalar_op_impl_wrapper, S, less_than_op>(rhs, lhs);
}

template ndarray bland::operator>=(float lhs, ndarray rhs);
template ndarray bland::operator>=(double lhs, ndarray rhs);
// template ndarray bland::operator>=(int8_t lhs, ndarray rhs);
// template ndarray bland::operator>=(int16_t lhs, ndarray rhs);
// template ndarray bland::operator>=(int32_t lhs, ndarray rhs);
// template ndarray bland::operator>=(int64_t lhs, ndarray rhs);
// template ndarray bland::operator>=(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator>=(uint16_t lhs, ndarray rhs);
// template ndarray bland::operator>=(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator>=(uint64_t lhs, ndarray rhs);

template <typename S>
ndarray bland::operator >(S lhs, ndarray rhs) {

}

template <typename S>
ndarray bland::operator <=(S lhs, ndarray rhs) {

}

template <typename S>
ndarray bland::operator <(S lhs, ndarray rhs) {

}

template <typename S>
ndarray bland::operator ==(S lhs, ndarray rhs) {

}
