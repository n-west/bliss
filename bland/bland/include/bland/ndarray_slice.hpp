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

    ndarray_slice &operator=(const ndarray_slice &rhs);

    ndarray_slice &operator=(const ndarray &rhs);

  protected:
    friend ndarray_slice slice(const ndarray &, int64_t, int64_t, int64_t, int64_t);
};

} // namespace bland
