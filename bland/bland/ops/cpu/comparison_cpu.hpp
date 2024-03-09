#pragma once

#include <type_traits>
#include <cstdint>

namespace bland {
struct ndarray;
struct ndarray_slice;

namespace cpu {

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

} // namespace cpu
} // namespace bland
