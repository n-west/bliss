#pragma once

#include <type_traits>

namespace bland {

struct ndarray;
struct ndarray_slice;

template <typename L, typename R>
ndarray greater_than(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator>(L lhs, R rhs);


template <typename L, typename R>
ndarray greater_than_equal_to(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator>=(L lhs, R rhs);


template <typename L, typename R>
ndarray less_than(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator<(L lhs, R rhs);


template <typename L, typename R>
ndarray less_than_equal_to(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator<=(L lhs, R rhs);


template <typename L, typename R>
ndarray logical_and(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator&(L lhs, R rhs);


template <typename L, typename R>
ndarray equal_to(L lhs, R rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
operator==(L lhs, R rhs);

/**
 * absolute(lhs - rhs) <= (atol + rtol * absolute(rhs))
 *
 * pairwise for all elements in lhs and rhs
 */
// bool isclose(ndarray lhs, ndarray rhs, float relative_tolerance, float absolute_tolerance);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> approx_equal(ndarray lhs, T rhs, float relative_tolerance,
// float absolute_tolerance); template <typename T> std::enable_if_t<std::is_arithmetic<T>::value, ndarray>
// approx_equal(T lhs, ndarray rhs, float relative_tolerance, float absolute_tolerance);


} // namespace bland