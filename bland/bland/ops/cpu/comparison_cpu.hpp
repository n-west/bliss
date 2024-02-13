#pragma once

#include <type_traits>
#include <cstdint>

namespace bland {
struct ndarray;
struct ndarray_slice;

namespace cpu {



ndarray greater_than(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> greater_than(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> greater_than(T lhs, ndarray rhs);
ndarray                                                 operator>(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator>(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator>(T lhs, ndarray rhs);

ndarray greater_than_equal_to(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> greater_than_equal_to(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> greater_than_equal_to(T lhs, ndarray rhs);
ndarray                                                 operator>=(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator>=(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator>=(T lhs, ndarray rhs);

ndarray less_than(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> less_than(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> less_than(T lhs, ndarray rhs);
ndarray                                                 operator<(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator<(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator<(T lhs, ndarray rhs);

ndarray less_than_equal_to(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> less_than_equal_to(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> less_than_equal_to(T lhs, ndarray rhs);
ndarray                                                 operator<=(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator<=(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator<=(T lhs, ndarray rhs);

ndarray logical_and(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> logical_and(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> logical_and(T lhs, ndarray rhs);
ndarray                                                 operator&(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator&(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator&(T lhs, ndarray rhs);

ndarray equal_to(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> equal_to(ndarray lhs, T rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> equal_to(T lhs, ndarray rhs);
ndarray                                                 operator==(ndarray lhs, ndarray rhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator==(ndarray rhs, T lhs);
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator==(T lhs, ndarray rhs);

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

/**
 * return the number of elements that will evaluate to true
 *
 * TODO: optionally add an dim parameter
 */
int64_t count_true(ndarray x);

} // namespace cpu
} // namespace bland
