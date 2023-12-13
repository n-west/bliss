#pragma once

#include <cstdint>
#include <vector>

namespace bland {

struct ndarray;
struct ndarray_slice;

/**
 * Reductions
*/
ndarray sum(const ndarray &a, std::vector<int64_t> axes={});
ndarray sum(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

ndarray mean(const ndarray &a, std::vector<int64_t> axes={});
ndarray mean(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

ndarray stddev(const ndarray &a, std::vector<int64_t> axes={});
ndarray stddev(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

ndarray standardized_moment(const ndarray &a, int degree, std::vector<int64_t> axes={});
ndarray standardized_moment(const ndarray &a, int degree, ndarray &out, std::vector<int64_t> axes={});


// TODO
float median(const ndarray &a, std::vector<int64_t> axes={});
// ndarray median(const ndarray &a, std::vector<int64_t> axes={});
// ndarray median(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

} // namespace bland