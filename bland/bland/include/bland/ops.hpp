#pragma once

#include "ops_arithmetic.hpp"
#include "ops_shape.hpp"
#include "ops_statistical.hpp"

#include <cstdint>
#include <limits>

namespace bland {

struct ndarray;
struct ndarray_slice;

ndarray copy(ndarray a);
ndarray copy(ndarray a, ndarray &out);

ndarray square(ndarray a);
ndarray square(ndarray a, ndarray out);
ndarray sqrt(ndarray a);
ndarray sqrt(ndarray a, ndarray out);

ndarray_slice slice(const ndarray &a, int64_t dim, int64_t start, int64_t end, int64_t stride = 1);
struct slice_spec {
    /**
     * dimension to slice
     */
    int64_t dim;
    /**
     * Start (in number of items) index of the slice
     */
    int64_t start = 0;
    /**
     * End (in number of items) index of the slice
     */
    int64_t end = std::numeric_limits<int64_t>::max();
    /**
     * Stride (in number of items along this dimension) to stride in the slice
     *
     * \internal the new stride is the product of this stride and the existing stride. So a stride of 1 leaves the same
     * spacing between elements and a stride of 2 would double it.
     */
    int64_t stride = 1;

  public:
    slice_spec(int64_t dim) : dim(dim) {}
    slice_spec(int64_t dim, int64_t start = 0, int64_t end = std::numeric_limits<int64_t>::max(), int64_t stride = 1) :
            dim(dim), start(start), end(end), stride(stride) {}
};

template <typename... Args>
ndarray_slice slice(const ndarray &a, slice_spec slice_dim, Args... args);

template <typename T>
ndarray fill(ndarray out, T value);


} // namespace bland
