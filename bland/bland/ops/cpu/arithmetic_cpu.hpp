#pragma once


namespace bland {

struct ndarray;
struct ndarray_slice;

namespace cpu {


template <typename T>
ndarray add_cpu(ndarray a, T b);

template <typename T>
ndarray subtract_cpu(ndarray a, T b);

template <typename T>
ndarray multiply_cpu(ndarray a, T b);

template <typename T>
ndarray divide_cpu(ndarray a, T b);

} // namespace cpu

} // namespace bland