#include "bland/ops/ops_comparison.hpp"

#include "bland/ndarray.hpp"
#include "bland/ndarray_slice.hpp"

#include "device_dispatch.hpp"

#if BLAND_CUDA_CODE
#include "cuda/comparison_cuda.cuh"
#endif // BLAND_CUDA_CODE
#include "cpu/comparison_cpu.hpp"

#include <type_traits>

using namespace bland;


template <typename L, typename R>
ndarray bland::greater_than(L lhs, R rhs) {
    return device_dispatch(cpu::greater_than<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::greater_than<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::greater_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::greater_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::greater_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::greater_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::greater_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::greater_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::greater_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::greater_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::greater_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::greater_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::greater_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::greater_than<double>(double lhs, ndarray rhs);
template ndarray bland::greater_than<float>(float lhs, ndarray rhs);
template ndarray bland::greater_than<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::greater_than<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::greater_than<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::greater_than<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::greater_than<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::greater_than<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::greater_than<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::greater_than<int64_t>(int64_t lhs, ndarray rhs);

template ndarray bland::greater_than<uint8_t, ndarray_slice>(uint8_t lhs, ndarray_slice rhs);
template ndarray bland::greater_than<ndarray_slice, uint8_t>(ndarray_slice lhs, uint8_t rhs);


template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator>(L lhs, R rhs) {
    return greater_than(lhs, rhs);
}

template ndarray bland::operator><ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator><ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::operator><ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator><ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator><ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator><ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator><ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator><ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator><ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator><ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator><ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator><double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::operator><float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator><uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator><uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator><uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator><uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator><int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator><int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator><int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator><int64_t, ndarray>(int64_t lhs, ndarray rhs);

/**
 * Greater than equal to
 */
template <typename L, typename R>
ndarray bland::greater_than_equal_to(L lhs, R rhs) {
    return device_dispatch(cpu::greater_than_equal_to<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::greater_than_equal_to<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::greater_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::greater_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::greater_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::greater_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::greater_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::greater_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::greater_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::greater_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::greater_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::greater_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::greater_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::greater_than_equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::greater_than_equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::greater_than_equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::greater_than_equal_to<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::greater_than_equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::greater_than_equal_to<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::greater_than_equal_to<int64_t>(int64_t lhs, ndarray rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator>=(L lhs, R rhs) {
    return greater_than_equal_to(lhs, rhs);
}

template ndarray bland::operator>=<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator>=<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::operator>=<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator>=<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator>=<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator>=<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator>=<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator>=<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator>=<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator>=<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator>=<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator>=<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::operator>=<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator>=<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator>=<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator>=<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator>=<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator>=<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator>=<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator>=<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator>=<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than
 */
template <typename L, typename R>
ndarray bland::less_than(L lhs, R rhs) {
    return device_dispatch(cpu::less_than<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::less_than<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::less_than<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::less_than<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::less_than<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::less_than<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::less_than<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::less_than<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::less_than<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::less_than<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::less_than<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::less_than<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::less_than<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::less_than<double>(double lhs, ndarray rhs);
template ndarray bland::less_than<float>(float lhs, ndarray rhs);
template ndarray bland::less_than<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::less_than<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::less_than<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::less_than<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::less_than<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::less_than<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::less_than<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::less_than<int64_t>(int64_t lhs, ndarray rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator<(L lhs, R rhs) {
    return less_than(lhs, rhs);
}

template ndarray bland::operator< <ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator< <ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::operator< <ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator< <ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator< <ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator< <ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator< <ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator< <ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator< <ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator< <ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator< <ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator< <double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::operator< <float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator< <uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator< <uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator< <uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator< <uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator< <int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator< <int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator< <int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator< <int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * Less than or equal to
 */
template <typename L, typename R>
ndarray bland::less_than_equal_to(L lhs, R rhs) {
    return device_dispatch(cpu::less_than_equal_to<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::less_than_equal_to<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::less_than_equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::less_than_equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::less_than_equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::less_than_equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::less_than_equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::less_than_equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::less_than_equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::less_than_equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::less_than_equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::less_than_equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::less_than_equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::less_than_equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::less_than_equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::less_than_equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::less_than_equal_to<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::less_than_equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::less_than_equal_to<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::less_than_equal_to<int64_t>(int64_t lhs, ndarray rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator<=(L lhs, R rhs) {
    return less_than_equal_to(lhs, rhs);
}

template ndarray bland::operator<=<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator<=<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::operator<=<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator<=<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator<=<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator<=<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator<=<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator<=<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator<=<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator<=<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator<=<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator<=<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::operator<=<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator<=<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator<=<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator<=<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator<=<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator<=<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator<=<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator<=<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator<=<int64_t, ndarray>(int64_t lhs, ndarray rhs);


/**
 * logical and (&)
 */
template <typename L, typename R>
ndarray bland::logical_and(L lhs, R rhs) {
    return device_dispatch(cpu::logical_and<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::logical_and<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::logical_and<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::logical_and<ndarray, double>(ndarray lhs, double rhs);
// template ndarray bland::logical_and<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::logical_and<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::logical_and<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::logical_and<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::logical_and<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::logical_and<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::logical_and<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::logical_and<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::logical_and<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::logical_and<double>(double lhs, ndarray rhs);
// template ndarray bland::logical_and<float>(float lhs, ndarray rhs);
template ndarray bland::logical_and<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::logical_and<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::logical_and<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::logical_and<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::logical_and<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::logical_and<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::logical_and<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::logical_and<int64_t>(int64_t lhs, ndarray rhs);

template ndarray bland::logical_and<uint8_t, ndarray_slice>(uint8_t lhs, ndarray_slice rhs);
template ndarray bland::logical_and<ndarray_slice, uint8_t>(ndarray_slice lhs, uint8_t rhs);


template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator&(L lhs, R rhs) {
    return logical_and(lhs, rhs);
}

template ndarray bland::operator&<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator&<ndarray, double>(ndarray lhs, double rhs);
// template ndarray bland::operator&<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator&<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator&<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator&<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator&<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator&<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator&<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator&<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator&<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator&<double, ndarray>(double lhs, ndarray rhs);
// template ndarray bland::operator&<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator&<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator&<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator&<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator&<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator&<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator&<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator&<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator&<int64_t, ndarray>(int64_t lhs, ndarray rhs);

template ndarray bland::operator&<uint8_t, ndarray_slice>(uint8_t lhs, ndarray_slice rhs);
template ndarray bland::operator&<ndarray_slice, uint8_t>(ndarray_slice lhs, uint8_t rhs);


/**
 * equal_to (==)
 */
template <typename L, typename R>
ndarray bland::equal_to(L lhs, R rhs) {
    return device_dispatch(cpu::equal_to<L, R>,
                            #if BLAND_CUDA_CODE
                            cuda::equal_to<L, R>,
                            #else
                            nullptr,
                            #endif
                            lhs, rhs);
}

template ndarray bland::equal_to<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::equal_to<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::equal_to<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::equal_to<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::equal_to<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::equal_to<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::equal_to<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::equal_to<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::equal_to<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::equal_to<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::equal_to<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::equal_to<double>(double lhs, ndarray rhs);
template ndarray bland::equal_to<float>(float lhs, ndarray rhs);
template ndarray bland::equal_to<uint8_t>(uint8_t lhs, ndarray rhs);
// template ndarray bland::equal_to<uint16_t>(uint16_t lhs, ndarray rhs);
template ndarray bland::equal_to<uint32_t>(uint32_t lhs, ndarray rhs);
// template ndarray bland::equal_to<uint64_t>(uint64_t lhs, ndarray rhs);
// template ndarray bland::equal_to<int8_t>(int8_t lhs, ndarray rhs);
// template ndarray bland::equal_to<int16_t>(int16_t lhs, ndarray rhs);
template ndarray bland::equal_to<int32_t>(int32_t lhs, ndarray rhs);
// template ndarray bland::equal_to<int64_t>(int64_t lhs, ndarray rhs);

template <typename L, typename R>
std::enable_if_t<std::is_base_of<ndarray, std::decay_t<L>>::value || std::is_base_of<ndarray, std::decay_t<R>>::value, ndarray>
bland::operator==(L lhs, R rhs) {
    return equal_to(lhs, rhs);
}

template ndarray bland::operator==<ndarray, ndarray>(ndarray lhs, ndarray rhs);

// template ndarray bland::operator==<ndarray, double>(ndarray lhs, double rhs);
template ndarray bland::operator==<ndarray, float>(ndarray lhs, float rhs);
template ndarray bland::operator==<ndarray, uint8_t>(ndarray lhs, uint8_t rhs);
// template ndarray bland::operator==<ndarray, uint16_t>(ndarray lhs, uint16_t rhs);
template ndarray bland::operator==<ndarray, uint32_t>(ndarray lhs, uint32_t rhs);
// template ndarray bland::operator==<ndarray, uint64_t>(ndarray lhs, uint64_t rhs);
// template ndarray bland::operator==<ndarray, int8_t>(ndarray lhs, int8_t rhs);
// template ndarray bland::operator==<ndarray, int16_t>(ndarray lhs, int16_t rhs);
template ndarray bland::operator==<ndarray, int32_t>(ndarray lhs, int32_t rhs);
// template ndarray bland::operator==<ndarray, int64_t>(ndarray lhs, int64_t rhs);

// template ndarray bland::operator==<double, ndarray>(double lhs, ndarray rhs);
template ndarray bland::operator==<float, ndarray>(float lhs, ndarray rhs);
template ndarray bland::operator==<uint8_t, ndarray>(uint8_t lhs, ndarray rhs);
// template ndarray bland::operator==<uint16_t, ndarray>(uint16_t lhs, ndarray rhs);
template ndarray bland::operator==<uint32_t, ndarray>(uint32_t lhs, ndarray rhs);
// template ndarray bland::operator==<uint64_t, ndarray>(uint64_t lhs, ndarray rhs);
// template ndarray bland::operator==<int8_t, ndarray>(int8_t lhs, ndarray rhs);
// template ndarray bland::operator==<int16_t, ndarray>(int16_t lhs, ndarray rhs);
template ndarray bland::operator==<int32_t, ndarray>(int32_t lhs, ndarray rhs);
// template ndarray bland::operator==<int64_t, ndarray>(int64_t lhs, ndarray rhs);

