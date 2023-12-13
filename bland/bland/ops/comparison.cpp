
#include "bland/ndarray.hpp"
#include "bland/ops_comparison.hpp"

#include "dispatcher.hpp"
#include "elementwise_binary_op.hpp"

#include "arithmetic_impl.hpp"
#include <type_traits>

using namespace bland;

struct greater_than_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a > b);
    }
};

struct greater_than_equal_to_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a >= b);
    }
};

struct less_than_equal_to_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a <= b);
    }
};

struct less_than_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a < b);
    }
};

struct equal_to_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a == b);
    }
};

struct approx_equal_to_impl {
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b, float tol) {
        return static_cast<Out>(std::abs(a - b) < tol);
    }
};

ndarray bland::greater_than(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, greater_than_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::greater_than(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_equal_to_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::greater_than(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, greater_than_impl>(out, a, b);
}

template ndarray bland::greater_than<double>(ndarray lhs, double rhs);
template ndarray bland::greater_than<float>(ndarray lhs, float rhs);
template ndarray bland::greater_than<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::greater_than<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::greater_than<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::greater_than<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::greater_than<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::greater_than<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::greater_than<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::greater_than<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::greater_than<double>(double lhs, ndarray rhs);
template ndarray bland::greater_than<float>(float lhs, ndarray rhs);
template ndarray bland::greater_than<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::greater_than<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::greater_than<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::greater_than<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::greater_than<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::greater_than<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::greater_than<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::greater_than<int64_t>(int64_t lhs, ndarray rhs);

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>(ndarray lhs, T rhs) {
    return greater_than(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>(T lhs, ndarray rhs) {
    return greater_than(lhs, rhs);
}

template ndarray bland::operator> <double>(ndarray lhs, double rhs);
template ndarray bland::operator> <float>(ndarray lhs, float rhs);
template ndarray bland::operator> <uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator> <uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator> <uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator> <uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator> <int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator> <int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator> <int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator> <int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator> <double>(double lhs, ndarray rhs);
template ndarray bland::operator> <float>(float lhs, ndarray rhs);
template ndarray bland::operator> <uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator> <uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator> <uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator> <uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator> <int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator> <int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator> <int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator> <int64_t>(int64_t lhs, ndarray rhs);

/**
 * Greater than equal to
*/
ndarray bland::greater_than_equal_to(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, lhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, greater_than_equal_to_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::greater_than_equal_to(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::greater_than_equal_to(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, greater_than_equal_to_impl>(out, a, b);
}

template ndarray bland::greater_than_equal_to<double>(ndarray lhs, double rhs);
template ndarray bland::greater_than_equal_to<float>(ndarray lhs, float rhs);
template ndarray bland::greater_than_equal_to<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::greater_than_equal_to<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::greater_than_equal_to<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::greater_than_equal_to<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::greater_than_equal_to<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::greater_than_equal_to<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::greater_than_equal_to<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::greater_than_equal_to<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::greater_than_equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<int64_t>(int64_t lhs, ndarray rhs);

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>=(ndarray lhs, T rhs) {
    return greater_than_equal_to(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>=(T lhs, ndarray rhs) {
    return greater_than_equal_to(lhs, rhs);
}

template ndarray bland::operator>= <double>(ndarray lhs, double rhs);
template ndarray bland::operator>= <float>(ndarray lhs, float rhs);
template ndarray bland::operator>= <uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator>= <uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator>= <uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator>= <uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator>= <int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator>= <int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator>= <int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator>= <int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator>= <double>(double lhs, ndarray rhs);
template ndarray bland::operator>= <float>(float lhs, ndarray rhs);
template ndarray bland::operator>= <uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator>= <uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator>= <uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator>= <uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator>= <int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator>= <int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator>= <int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator>= <int64_t>(int64_t lhs, ndarray rhs);

/**
 * Less than
*/
ndarray bland::less_than(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, less_than_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::less_than(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, greater_than_equal_to_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::less_than(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_impl>(out, a, b);
}

template ndarray bland::less_than<double>(ndarray lhs, double rhs);
template ndarray bland::less_than<float>(ndarray lhs, float rhs);
template ndarray bland::less_than<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::less_than<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::less_than<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::less_than<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::less_than<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::less_than<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::less_than<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::less_than<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::less_than<double>(double lhs, ndarray rhs);
template ndarray bland::less_than<float>(float lhs, ndarray rhs);
template ndarray bland::less_than<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::less_than<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::less_than<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::less_than<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::less_than<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::less_than<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::less_than<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::less_than<int64_t>(int64_t lhs, ndarray rhs);

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator<(ndarray lhs, T rhs) {
    return less_than(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator<(T lhs, ndarray rhs) {
    return less_than(lhs, rhs);
}

template ndarray bland::operator< <double>(ndarray lhs, double rhs);
template ndarray bland::operator< <float>(ndarray lhs, float rhs);
template ndarray bland::operator< <uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator< <uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator< <uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator< <uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator< <int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator< <int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator< <int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator< <int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator< <double>(double lhs, ndarray rhs);
template ndarray bland::operator< <float>(float lhs, ndarray rhs);
template ndarray bland::operator< <uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator< <uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator< <uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator< <uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator< <int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator< <int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator< <int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator< <int64_t>(int64_t lhs, ndarray rhs);


// template <typename T>
// ndarray less_than_equal_to(ndarray lhs, T rhs) {
//     auto out = ndarray(lhs.shape(), datatype::uint8);
//     return dispatch<elementwise_binary_op_impl_wrapper, less_than_equal_to_impl>(out, lhs, rhs);
// }
// template <typename T>
// ndarray less_than_equal_to(T lhs, ndarray rhs) {
//     auto out = ndarray(lhs.shape(), datatype::uint8);
//     return dispatch<elementwise_binary_op_impl_wrapper, less_than_equal_to_impl>(out, lhs, rhs);
// }
// template <typename T>
// ndarray operator <=(ndarray lhs, T rhs) {
//     return less_than_equal_to(lhs, rhs);
// }
// template <typename T>
// ndarray operator <=(T lhs, ndarray rhs) {
//     return less_than_equal_to(lhs, rhs);
// }

// template <typename T>
// ndarray equal_to(ndarray lhs, T rhs) {
//     auto out = ndarray(lhs.shape(), datatype::uint8);
//     return dispatch<elementwise_binary_op_impl_wrapper, equal_to_impl>(out, lhs, rhs);
// }
// template <typename T>
// ndarray operator ==(T lhs, ndarray rhs) {
//     return equal_to(lhs, rhs);
// }

// bool approx_equal(const ndarray &a, const ndarray &b);
