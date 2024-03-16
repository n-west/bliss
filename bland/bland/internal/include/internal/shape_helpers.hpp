#pragma once

#include <stdexcept>
#include <vector>

namespace bland {
std::vector<int64_t> expand_shapes_to_broadcast(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape);

std::vector<int64_t> compute_broadcast_offset(const std::vector<int64_t> &shape,
                                              const std::vector<int64_t> &offsets,
                                              const std::vector<int64_t> &broadcast_shape);

std::vector<int64_t> compute_broadcast_strides(const std::vector<int64_t> &shape,
                                               const std::vector<int64_t> &strides,
                                               const std::vector<int64_t> &out_shape);

std::vector<int64_t> compute_broadcast_shape(std::vector<int64_t> shape, std::vector<int64_t> out_shape);

} // namespace bland
