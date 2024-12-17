#pragma once

#include "ndarray.hpp"

namespace bland {


/**
 * An ndarray_slice is type-system syntactic sugar that allows storing results in to
 * and existing array. This is especially convenient when we've sliced an array and
 * want to store results in the slice of that array.
 */
class ndarray_slice : public ndarray {
  public:
    ndarray_slice(const ndarray &other);

    /**
     * Copy the data from the other slice into this slice
     * 
     * If the rhs slice is equal to this slice, just the shape, strides, and offset are updated
     * with no copying.
     */
    ndarray_slice &operator=(const ndarray_slice &rhs);

    /**
     * Copy the data from the other ndarray into this slice
     */
    ndarray_slice &operator=(const ndarray &rhs);

  protected:
    friend ndarray_slice slice(const ndarray &, int64_t, int64_t, int64_t, int64_t);
};

} // namespace bland
