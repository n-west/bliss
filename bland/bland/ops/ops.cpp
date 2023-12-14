

#include "bland/ops.hpp"
#include "bland/ndarray.hpp"

#include "assignment_op.hpp"
#include "dispatcher.hpp"
// #include "elementwise_binary_op.hpp"
#include "elementwise_scalar_op.hpp"
#include "elementwise_unary_op.hpp"
#include "shape_helpers.hpp"

#include <fmt/core.h>

#include <cstdlib>

using namespace bland;

/**
 * Copy (data)
 */
template <typename Out, typename A>
struct elementwise_copy_op {
    static inline Out call(const A &a) { return static_cast<Out>(a); }
};

ndarray bland::copy(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return copy(a, out);
}

ndarray bland::copy(ndarray a, ndarray &out) {
    return dispatch_new2<unary_op_impl_wrapper, elementwise_copy_op>(out, a);
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
ndarray bland::fill(ndarray out, T value) {
    return dispatch_new<assignment_op_impl_wrapper, T>(out, value);
}


template ndarray bland::fill<float>(ndarray out, float v);
template ndarray bland::fill<double>(ndarray out, double v);
template ndarray bland::fill<int8_t>(ndarray out, int8_t v);
template ndarray bland::fill<int16_t>(ndarray out, int16_t v);
template ndarray bland::fill<int32_t>(ndarray out, int32_t v);
template ndarray bland::fill<int64_t>(ndarray out, int64_t v);



template <typename Out, typename A>
struct elementwise_square_op {
    static inline Out call(const A &a) { return static_cast<Out>(a * a); }
};

ndarray bland::square(ndarray a, ndarray out) {
    return dispatch_new2<unary_op_impl_wrapper, elementwise_square_op>(out, a);
}

ndarray bland::square(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return square(a, out);
}

template <typename Out, typename A>
struct elementwise_sqrt_op {
    static inline Out call(const A &a) { return static_cast<Out>(std::sqrt(a)); }
};

ndarray bland::sqrt(ndarray a, ndarray out) {
    return dispatch_new2<unary_op_impl_wrapper, elementwise_sqrt_op>(out, a);
}

ndarray bland::sqrt(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return sqrt(a, out);
}

template <typename Out, typename A>
struct elementwise_abs_op {
    static inline Out call(const A &a) { return static_cast<Out>(std::abs(a)); }
};

template <typename Out>
struct elementwise_abs_op<Out, uint8_t> {
    static inline Out call(const uint8_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint16_t> {
    static inline Out call(const uint16_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint32_t> {
    static inline Out call(const uint32_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint64_t> {
    static inline Out call(const uint64_t &a) { return static_cast<Out>(a); }
};

ndarray bland::abs(ndarray a, ndarray out) {
    return dispatch_new2<unary_op_impl_wrapper, elementwise_abs_op>(out, a);
}

ndarray bland::abs(ndarray a) {
    auto out = ndarray(a.shape(), a.dtype(), a.device());
    return abs(a, out);
}


