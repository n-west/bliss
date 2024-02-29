#pragma once

#include <cmath> // sqrt, abs

namespace bland
{

/**
 * Template struct to allow passing an elementwise addition as a template argument
*/
struct elementwise_add_op_ts {
    /**
     * elementwise addition between two scalars in an ndarray with type `A` and `B`
    */
template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a + b); }
};

/**
 * Template struct to allow passing an elementwise subtraction as a template argument
*/
struct elementwise_subtract_op_ts {
    /**
     * elementwise subtraction between two scalars in an ndarray with type `A` and `B`
    */
template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a - b); }
};

/**
 * Template struct to allow passing an elementwise multiplication as a template argument
*/
struct elementwise_multiply_op_ts {
    /**
     * elementwise multiplication between two scalars in an ndarray with type `A` and `B`
    */
template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a * b); }
};

/**
 * Template struct to allow passing an elementwise division as a template argument
*/
struct elementwise_divide_op_ts {
    /**
     * elementwise division between two scalars in an ndarray with type `A` and `B`
    */
template <typename Out, typename A, typename B>
    static inline Out call(const A &a, const B &b) { return static_cast<Out>(a / b); }
};


template <typename Out, typename A>
struct elementwise_copy_op {
    static inline Out call(const A &a) { return static_cast<Out>(a); }
};

template <typename Out, typename A>
struct elementwise_square_op {
    static inline Out call(const A &a) { return static_cast<Out>(a * a); }
};

template <typename Out, typename A>
struct elementwise_sqrt_op {
    static inline Out call(const A &a) { return static_cast<Out>(std::sqrt(a)); }
};

template <typename Out, typename A>
struct elementwise_abs_op {
    static inline Out call(const A &a) { return static_cast<Out>(std::abs(a)); }
};

template <typename Out>
struct elementwise_abs_op<Out, uint8_t> {
    static inline Out call(const uint8_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint16_t> {
    static inline Out call(const uint16_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint32_t> {
    static inline Out call(const uint32_t &a) { return static_cast<Out>(a); }
};
template <typename Out>
struct elementwise_abs_op<Out, uint64_t> {
    static inline Out call(const uint64_t &a) { return static_cast<Out>(a); }
};

} // namespace bland
