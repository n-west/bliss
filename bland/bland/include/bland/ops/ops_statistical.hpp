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

ndarray masked_mean(const ndarray &a, const ndarray &mask, std::vector<int64_t> axes={});
ndarray masked_mean(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> axes={});

ndarray stddev(const ndarray &a, std::vector<int64_t> axes={});
ndarray stddev(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

ndarray masked_stddev(const ndarray &a, const ndarray &mask, std::vector<int64_t> axes={});
ndarray masked_stddev(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> axes={});

ndarray var(const ndarray &a, std::vector<int64_t> axes={});
ndarray var(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});

ndarray masked_var(const ndarray &a, const ndarray &mask, std::vector<int64_t> axes={});
ndarray masked_var(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> axes={});

ndarray standardized_moment(const ndarray &a, int degree, std::vector<int64_t> axes={});
ndarray standardized_moment(const ndarray &a, int degree, ndarray &out, std::vector<int64_t> axes={});


// TODO
float median(const ndarray &a, std::vector<int64_t> axes={});
// ndarray median(const ndarray &a, std::vector<int64_t> axes={});
// ndarray median(const ndarray &a, ndarray &out, std::vector<int64_t> axes={});


/**
 * return the number of elements that will evaluate to true
 *
 * TODO: optionally add an dim parameter
 */
int64_t count_true(ndarray x);


} // namespace bland