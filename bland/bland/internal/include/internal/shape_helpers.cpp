
#include "shape_helpers.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <stdexcept>

using namespace bland;

std::vector<int64_t> bland::expand_shapes_to_broadcast(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape) {
    size_t               ndim = std::max(a_shape.size(), b_shape.size());
    std::vector<int64_t> out_shape(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        int64_t a_dim = i < a_shape.size() ? a_shape[a_shape.size() - i - 1] : 1;
        int64_t b_dim = i < b_shape.size() ? b_shape[b_shape.size() - i - 1] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            auto error_message = fmt::format(
                    "Shapes of input arrays ({} and {}) are not compatible for broadcasting.", a_shape, b_shape);
            throw std::invalid_argument(error_message);
        }

        out_shape[ndim - i - 1] = std::max(a_dim, b_dim);
    }

    return out_shape;
}

std::vector<int64_t> bland::compute_broadcast_offset(const std::vector<int64_t> &shape,
                                                     const std::vector<int64_t> &offsets,
                                                     const std::vector<int64_t> &broadcast_shape) {
    std::vector<int64_t> broadcast_offsets(broadcast_shape.size(), 0);
    int                  offset = broadcast_shape.size() - shape.size();
    for (int i = 0; i < shape.size(); ++i) {
        // TODO, I'm not sure this will be right in all cases
        if (shape[i] != 1) {
            broadcast_offsets[i + offset] = offsets[i];
        }
    }
    return broadcast_offsets;
}

std::vector<int64_t> bland::compute_broadcast_strides(const std::vector<int64_t> &shape,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<int64_t> &out_shape) {
    std::vector<int64_t> broadcast_strides(out_shape.size(), 0);

    int offset = out_shape.size() - shape.size();
    for (int i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1) {
            broadcast_strides[i + offset] = strides[i];
        }
    }
    return broadcast_strides;
}

std::vector<int64_t> bland::compute_broadcast_shape(std::vector<int64_t> shape, std::vector<int64_t> out_shape) {
    std::vector<int64_t> broadcast_shape(out_shape.size());

    size_t input_shape_index = 0;
    for (size_t i = 0; i < out_shape.size(); ++i) {
        if (shape[input_shape_index] == out_shape[i] || shape[input_shape_index] == 1) {
            broadcast_shape[i] = out_shape[i];
            input_shape_index += 1;
        } else {
            broadcast_shape[i] = 1;
        }
    }

    return broadcast_shape;
}
