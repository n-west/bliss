#pragma once

#include <cstdint>
#include <type_traits>

namespace bland {
struct ndarray;
struct ndarray_slice;

namespace cuda {

template <typename L, typename R>
ndarray greater_than(L lhs, R rhs);

template <typename L, typename R>
ndarray greater_than_equal_to(L lhs, R rhs);

template <typename L, typename R>
ndarray less_than(L lhs, R rhs);

template <typename L, typename R>
ndarray less_than_equal_to(L lhs, R rhs);

template <typename L, typename R>
ndarray logical_and(L lhs, R rhs);

template <typename L, typename R>
ndarray equal_to(L lhs, R rhs);

/**
 * return the number of elements that will evaluate to true
 */
int64_t count_true(ndarray x);

} // namespace cuda
} // namespace bland