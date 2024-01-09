#pragma once

namespace bland {

struct ndarray;
struct ndarray_slice;

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

// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> logical_or(ndarray rhs, T lhs);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator |(ndarray rhs, T lhs);

// template <typename T>
// ndarray equal_to(ndarray lhs, T rhs);
// template <typename T>
// ndarray operator ==(T lhs, ndarray rhs);

// bool approx_equal(const ndarray &a, const ndarray &b);

} // namespace bland