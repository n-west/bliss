#pragma once

#include <stdexcept>
#include <type_traits>


namespace bland {

struct greater_than_impl {
    template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a > b);
    }
};

struct greater_than_equal_to_impl {
    template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a >= b);
    }
};

struct less_than_equal_to_impl {
    template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a <= b);
    }
};

struct less_than_impl {
    template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a < b);
    }
};

struct equal_to_impl {
    template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        return static_cast<Out>(a == b);
    }
};

struct logical_and_impl {
    // TODO: think about the Out type...
    template <typename Out, typename A, typename B, 
          typename = std::enable_if_t<!std::is_floating_point_v<A> && 
                                      !std::is_floating_point_v<B> && 
                                      !std::is_floating_point_v<Out>>>
    //template <typename Out, typename A, typename B>
    static __device__ inline Out call(const A &a, const B &b) {
        // if constexpr (std::is_floating_point_v<A> || std::is_floating_point_v<B> || std::is_floating_point_v<Out>) {
        //     // TODO: how to signal an error from device
        // } else {
        //     return static_cast<Out>(a & b);
        // }
        return static_cast<Out>(a & b);
    }
};
} // namespace
