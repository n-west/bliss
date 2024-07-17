#pragma once

#include <vector>
#include <cstdint>

namespace bland {

struct ndarray;
struct ndarray_slice;

ndarray reshape(ndarray x, std::vector<int64_t> new_shape);

} //namespace bland