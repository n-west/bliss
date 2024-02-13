#pragma once


namespace bland {

struct ndarray;
struct ndarray_slice;

namespace cuda {


template <typename T>
ndarray add_cuda(ndarray a, T b);

template <typename T>
ndarray subtract_cuda(ndarray a, T b);

template <typename T>
ndarray multiply_cuda(ndarray a, T b);

template <typename T>
ndarray divide_cuda(ndarray a, T b);

} // namespace cuda

} // namespace bland