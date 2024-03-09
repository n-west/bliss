#pragma once

#include <bland/ndarray.hpp>


namespace bland {
namespace cpu {

ndarray copy(ndarray a, ndarray &out);
ndarray square(ndarray a, ndarray& out);
ndarray sqrt(ndarray a, ndarray& out);
ndarray abs(ndarray a, ndarray& out);

template <typename T>
ndarray fill(ndarray out, T value);

} // namespace cpu
} // namespace bland
