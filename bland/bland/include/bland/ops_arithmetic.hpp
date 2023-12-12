#pragma once


namespace bland {

struct ndarray;
struct ndarray_slice;


template <typename T>
ndarray add(ndarray a, T b);

template <typename T>
ndarray subtract(ndarray a, T b);

template <typename T>
ndarray multiply(ndarray a, T b);

template <typename T>
ndarray divide(ndarray a, T b);


} // namespace bland