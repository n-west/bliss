
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

struct logical_and_impl {
    // TODO: think about the Out type...
    template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) {
        if constexpr (std::is_floating_point_v<A> || std::is_floating_point_v<B> || std::is_floating_point_v<Out>) {
            throw std::runtime_error("logical_and_impl: cannot perform logical_and on floating point types");
        } else {
            return static_cast<Out>(a & b);
        }
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
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, greater_than_impl>(out, rhs, lhs);
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

ndarray bland::operator>(ndarray lhs, ndarray rhs) {
    return greater_than(lhs, rhs);
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>(ndarray lhs, T rhs) {
    return greater_than(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator>(T lhs, ndarray rhs) {
    return greater_than(lhs, rhs);
}

template ndarray bland::operator><double>(ndarray lhs, double rhs);
template ndarray bland::operator><float>(ndarray lhs, float rhs);
template ndarray bland::operator><uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator><uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator><uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator><uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator><int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator><int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator><int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator><int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator><double>(double lhs, ndarray rhs);
template ndarray bland::operator><float>(float lhs, ndarray rhs);
template ndarray bland::operator><uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator><uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator><uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator><uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator><int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator><int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator><int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator><int64_t>(int64_t lhs, ndarray rhs);

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
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, greater_than_equal_to_impl>(out, rhs, lhs);
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

template ndarray bland::operator>=<double>(ndarray lhs, double rhs);
template ndarray bland::operator>=<float>(ndarray lhs, float rhs);
template ndarray bland::operator>=<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator>=<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator>=<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator>=<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator>=<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator>=<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator>=<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator>=<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator>=<double>(double lhs, ndarray rhs);
template ndarray bland::operator>=<float>(float lhs, ndarray rhs);
template ndarray bland::operator>=<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator>=<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator>=<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator>=<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator>=<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator>=<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator>=<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator>=<int64_t>(int64_t lhs, ndarray rhs);

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
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_impl>(out, rhs, lhs);
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

ndarray bland::operator<(ndarray lhs, ndarray rhs) {
    return less_than(lhs, rhs);
}

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

/**
 * Less than or equal to
 */
ndarray bland::less_than_equal_to(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, less_than_equal_to_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::less_than_equal_to(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_equal_to_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::less_than_equal_to(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, less_than_equal_to_impl>(out, a, b);
}

template ndarray bland::less_than_equal_to<double>(ndarray lhs, double rhs);
template ndarray bland::less_than_equal_to<float>(ndarray lhs, float rhs);
template ndarray bland::less_than_equal_to<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::less_than_equal_to<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::less_than_equal_to<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::less_than_equal_to<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::less_than_equal_to<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::less_than_equal_to<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::less_than_equal_to<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::less_than_equal_to<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::less_than_equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<int64_t>(int64_t lhs, ndarray rhs);

ndarray bland::operator<=(ndarray lhs, ndarray rhs) {
    return less_than(lhs, rhs);
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator<=(ndarray lhs, T rhs) {
    return less_than(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator<=(T lhs, ndarray rhs) {
    return less_than(lhs, rhs);
}

template ndarray bland::operator<=<double>(ndarray lhs, double rhs);
template ndarray bland::operator<=<float>(ndarray lhs, float rhs);
template ndarray bland::operator<=<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator<=<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator<=<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator<=<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator<=<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator<=<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator<=<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator<=<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator<=<double>(double lhs, ndarray rhs);
template ndarray bland::operator<=<float>(float lhs, ndarray rhs);
template ndarray bland::operator<=<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator<=<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator<=<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator<=<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator<=<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator<=<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator<=<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator<=<int64_t>(int64_t lhs, ndarray rhs);

/**
 * logical and (&)
 */
ndarray bland::logical_and(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, logical_and_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::logical_and(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, logical_and_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::logical_and(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, logical_and_impl>(out, a, b);
}

// logical_and doesn't make sense on float types, but left commented out as a reminder this isn't a mistake of omission
// template ndarray bland::logical_and<double>(ndarray lhs, double rhs);
// template ndarray bland::logical_and<float>(ndarray lhs, float rhs);
template ndarray bland::logical_and<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::logical_and<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::logical_and<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::logical_and<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::logical_and<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::logical_and<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::logical_and<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::logical_and<int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::logical_and<double>(double lhs, ndarray rhs);
// template ndarray bland::logical_and<float>(float lhs, ndarray rhs);
template ndarray bland::logical_and<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::logical_and<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::logical_and<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::logical_and<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::logical_and<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::logical_and<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::logical_and<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::logical_and<int64_t>(int64_t lhs, ndarray rhs);

ndarray bland::operator&(ndarray lhs, ndarray rhs) {
    return logical_and(lhs, rhs);
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator&(ndarray lhs, T rhs) {
    return logical_and(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator&(T lhs, ndarray rhs) {
    return logical_and(lhs, rhs);
}

template ndarray bland::operator&<double>(ndarray lhs, double rhs);
template ndarray bland::operator&<float>(ndarray lhs, float rhs);
template ndarray bland::operator&<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator&<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator&<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator&<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator&<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator&<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator&<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator&<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator&<double>(double lhs, ndarray rhs);
template ndarray bland::operator&<float>(float lhs, ndarray rhs);
template ndarray bland::operator&<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator&<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator&<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator&<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator&<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator&<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator&<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator&<int64_t>(int64_t lhs, ndarray rhs);

// ndarray equal_to(ndarray lhs, ndarray rhs);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> equal_to(ndarray lhs, T rhs);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> equal_to(T lhs, ndarray rhs);
// ndarray                                                 operator==(ndarray lhs, ndarray rhs);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator==(ndarray rhs, T lhs);
// template <typename T>
// std::enable_if_t<std::is_arithmetic<T>::value, ndarray> operator==(T lhs, ndarray rhs);

/**
 * logical and (&)
 */
ndarray bland::equal_to(ndarray lhs, ndarray rhs) {
    auto out = ndarray(lhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch<elementwise_binary_op_impl_wrapper, uint8_t, equal_to_impl>(out, lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::equal_to(T lhs, ndarray rhs) {
    auto out = ndarray(rhs.shape(), ndarray::datatype::uint8, rhs.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, equal_to_impl>(out, rhs, lhs);
}
template <typename T> // the scalar case (explicitly instantiated below)
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::equal_to(ndarray a, T b) {
    auto out = ndarray(a.shape(), ndarray::datatype::uint8, a.device());
    return dispatch_new3<scalar_op_impl_wrapper, uint8_t, T, equal_to_impl>(out, a, b);
}

template ndarray bland::equal_to<double>(ndarray lhs, double rhs);
template ndarray bland::equal_to<float>(ndarray lhs, float rhs);
template ndarray bland::equal_to<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::equal_to<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::equal_to<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::equal_to<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::equal_to<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::equal_to<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::equal_to<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::equal_to<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::equal_to<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::equal_to<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::equal_to<int64_t>(int64_t lhs, ndarray rhs);

ndarray bland::operator==(ndarray lhs, ndarray rhs) {
    return equal_to(lhs, rhs);
}

template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator==(ndarray lhs, T rhs) {
    return equal_to(lhs, rhs);
}
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, ndarray> bland::operator==(T lhs, ndarray rhs) {
    return equal_to(lhs, rhs);
}

template ndarray bland::operator==<double>(ndarray lhs, double rhs);
template ndarray bland::operator==<float>(ndarray lhs, float rhs);
template ndarray bland::operator==<uint8_t>(ndarray lhs, uint8_t rhs);
template ndarray bland::operator==<uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator==<uint32_t>(ndarray lhs, uint32_t rhs);
template ndarray bland::operator==<uint64_t>(ndarray lhs, uint64_t rhs);
template ndarray bland::operator==<int8_t>(ndarray lhs, int8_t rhs);
template ndarray bland::operator==<int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator==<int32_t>(ndarray lhs, int32_t rhs);
template ndarray bland::operator==<int64_t>(ndarray lhs, int64_t rhs);

template ndarray bland::operator==<double>(double lhs, ndarray rhs);
template ndarray bland::operator==<float>(float lhs, ndarray rhs);
template ndarray bland::operator==<uint8_t>(uint8_t lhs, ndarray rhs);
template ndarray bland::operator==<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator==<uint32_t>(uint32_t lhs, ndarray rhs);
template ndarray bland::operator==<uint64_t>(uint64_t lhs, ndarray rhs);
template ndarray bland::operator==<int8_t>(int8_t lhs, ndarray rhs);
template ndarray bland::operator==<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::operator==<int32_t>(int32_t lhs, ndarray rhs);
template ndarray bland::operator==<int64_t>(int64_t lhs, ndarray rhs);

struct count_impl {
    template <typename in_datatype>
    static inline int64_t call(const ndarray &a) {

        auto a_data    = a.data_ptr<in_datatype>();
        auto a_shape   = a.shape();
        auto a_strides = a.strides();
        auto a_offset  = a.offsets();

        std::vector<int64_t> input_index(a_shape.size(), 0);
        int64_t a_linear_index = std::accumulate(a_offset.begin(), a_offset.end(), 0);

        int64_t count = 0;
        auto    numel = a.numel();
        for (int i = 0; i < numel; ++i) {
            // Make a copy of the current input index, we'll fix the non-summed dims
            // and iterate over the reduced dims accumulating the total
            if (a_data[a_linear_index]) {
                ++count;
            }

            // Increment the multi-dimensional input index
            // TODO: I think I can dedupe this with above by checking if axis is in reduce axis but that may actually be
            // less efficient
            for (int dim = a_shape.size() - 1; dim >= 0; --dim) {
                // If we're not at the end of this dim, keep going
                ++input_index[dim];
                a_linear_index += a_strides[dim];
                if (input_index[dim] < a_shape[dim]) {
                    break;
                } else {
                    // Otherwise, set it to 0 and move down to the next dim
                    input_index[dim] = 0;
                    a_linear_index -= (a_shape[dim]) * a_strides[dim];
                }
            }
        }

        return count;
    }
};

int64_t bland::count_true(ndarray x) {
    return dispatch_summary<count_impl>(x);
}
