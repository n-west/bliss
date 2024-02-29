#pragma once

#include <bland/ndarray.hpp>


namespace bland {
namespace cpu {

ndarray copy(ndarray a, ndarray &out);
ndarray square(ndarray a, ndarray& out);
ndarray sqrt(ndarray a, ndarray& out);
ndarray abs(ndarray a, ndarray& out);

} // namespace cpu
} // namespace bland
