#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace bland {

struct ndarray;
struct ndarray_slice;


template <typename T>
ndarray add(ndarray a, T b);

// ndarray_slice add(ndarray a, ndarray_slice b);

// template <typename T>
// ndarray_slice add(ndarray_slice a, T b);

template <typename T>
ndarray subtract(ndarray a, T b);

// ndarray_slice subtract(ndarray a, ndarray_slice b);

// template <typename T>
// ndarray_slice subtract(ndarray_slice a, T b);

template <typename T>
ndarray multiply(ndarray a, T b);

// ndarray_slice multiply(ndarray a, ndarray_slice b);

// template <typename T>
// ndarray_slice multiply(ndarray_slice a, T b);

template <typename T>
ndarray divide(ndarray a, T b);

// ndarray_slice divide(ndarray a, ndarray_slice b);

// template <typename T>
// ndarray_slice divide(ndarray_slice a, T b);


} // namespace bland