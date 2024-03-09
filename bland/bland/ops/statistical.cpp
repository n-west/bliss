#include "bland/ndarray.hpp"
#include "bland/ops_statistical.hpp"

#include "device_dispatch.hpp"

#include "shape_helpers.hpp"

#include "cpu/statistical_cpu.hpp"
#if BLAND_CUDA_CODE
#include "cuda/statistical.cuh"
#endif // BLAND_CUDA_CODE

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm> // std::find
#include <numeric>   // std::accumulate

using namespace bland;


ndarray bland::mean(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    auto compute_device = a.device();
    if (out.device() != compute_device) {
        throw std::runtime_error("out array is not on same device as input");
    }
    if (reduced_axes.empty()) {
        for (int axis = 0; axis < a.ndim(); ++axis) {
            reduced_axes.push_back(axis);
        }
    }

#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::mean(a, out, reduced_axes);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::mean(a, out, reduced_axes);
    } else {
        throw std::runtime_error("unsupported device for mean");
    }
}

ndarray bland::mean(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return mean(a, out, reduced_axes);
}

ndarray bland::masked_mean(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    auto compute_device = a.device();

    if (out.device() != compute_device) {
        throw std::runtime_error("out array is not on same device as input");
    }
    if (mask.device() != compute_device) {
        throw std::runtime_error("mask is not on same device as input");
    }
    if (reduced_axes.empty()) {
        for (int axis = 0; axis < a.ndim(); ++axis) {
            reduced_axes.push_back(axis);
        }
    }

#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::masked_mean(a, mask, out, reduced_axes);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::masked_mean(a, mask, out, reduced_axes);
    } else {
        throw std::runtime_error("unsupported device for masked_mean");
    }
}

ndarray bland::masked_mean(const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return masked_mean(a, mask, out, reduced_axes);
}

ndarray bland::stddev(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    auto compute_device = a.device();

    if (out.device() != compute_device) {
        throw std::runtime_error("out array is not on same device as input");
    }
    if (reduced_axes.empty()) {
        for (int axis = 0; axis < a.ndim(); ++axis) {
            reduced_axes.push_back(axis);
        }
    }

#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::stddev(a, out, reduced_axes);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::stddev(a, out, reduced_axes);
    } else {
        throw std::runtime_error("unsupported device for stddev");
    }
}
ndarray bland::stddev(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return stddev(a, out, reduced_axes);
}

ndarray bland::masked_stddev(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    auto compute_device = a.device();

    if (out.device() != compute_device) {
        throw std::runtime_error("out array is not on same device as input");
    }
    if (mask.device() != compute_device) {
        throw std::runtime_error("mask is not on same device as input");
    }
    if (reduced_axes.empty()) {
        for (int axis = 0; axis < a.ndim(); ++axis) {
            reduced_axes.push_back(axis);
        }
    }

#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::masked_stddev(a, mask, out, reduced_axes);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::masked_stddev(a, mask, out, reduced_axes);
    } else {
        throw std::runtime_error("unsupported device for masked_stddev");
    }
}

ndarray bland::masked_stddev(const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return masked_stddev(a, mask, out, reduced_axes);
}


// TODO: we have a cuda var, implement a cpu one and dispatch appropriately...
ndarray bland::var(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    stddev(a, out, reduced_axes);
    return square(out);
}
ndarray bland::var(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out = stddev(a, reduced_axes);
    return square(out);
}

ndarray bland::masked_var(const ndarray &a, const ndarray &mask, ndarray &out, std::vector<int64_t> reduced_axes) {
    masked_stddev(a, mask, out, reduced_axes);
    return square(out);
}
ndarray bland::masked_var(const ndarray &a, const ndarray &mask, std::vector<int64_t> reduced_axes) {
    auto out = masked_stddev(a, mask, reduced_axes);
    return square(out);
}

ndarray bland::standardized_moment(const ndarray &a, int degree, ndarray &out, std::vector<int64_t> reduced_axes) {
    return cpu::standardized_moment(a, degree, out, reduced_axes);
}
ndarray bland::standardized_moment(const ndarray &a, int degree, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return standardized_moment(a, degree, out, reduced_axes);
}

float bland::median(const ndarray &x, std::vector<int64_t> reduced_axes) {
    return cpu::median(x, reduced_axes);
}

ndarray bland::sum(const ndarray &a, ndarray &out, std::vector<int64_t> reduced_axes) {
    auto compute_device = a.device();

    if (out.device() != compute_device) {
        throw std::runtime_error("out array is not on same device as input");
    }
    if (reduced_axes.empty()) {
        for (int axis = 0; axis < a.ndim(); ++axis) {
            reduced_axes.push_back(axis);
        }
    }
#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::sum(a, out, reduced_axes);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::sum(a, out, reduced_axes);
    } else {
        throw std::runtime_error("unsupported device for sum");
    }
}

ndarray bland::sum(const ndarray &a, std::vector<int64_t> reduced_axes) {
    auto out_shape = std::vector<int64_t>();
    auto a_shape   = a.shape();
    if (!reduced_axes.empty()) {
        for (int64_t axis = 0; axis < a_shape.size(); ++axis) {
            if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
                out_shape.push_back(a_shape[axis]);
            }
        }
    }
    // output shape will be empty either because axes is empty OR is all dims
    if (out_shape.empty()) {
        out_shape = {1};
    }
    ndarray out(out_shape, a.dtype(), a.device());
    return sum(a, out, reduced_axes);
}

int64_t bland::count_true(ndarray x) {
    auto compute_device = x.device();

#if BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cuda.device_type || compute_device.device_type == ndarray::dev::cuda_managed.device_type) {
        return cuda::count_true(x);
    } else
#endif // BLAND_CUDA_CODE
    if (compute_device.device_type == ndarray::dev::cpu.device_type) {
        return cpu::count_true(x);
    } else {
        throw std::runtime_error("unsupported device for count_true");
    }

}
