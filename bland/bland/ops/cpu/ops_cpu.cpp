
#include "ops_cpu.hpp"

#include "arithmetic_cpu_impl.hpp"
#include "elementwise_unary_op.hpp"
#include "assignment_op.hpp"

#include "internal/dispatcher.hpp"
#include "bland_helpers/file_helpers.hpp"

#include "bland/ndarray.hpp"

#include <fstream>


using namespace bland;
using namespace bland::cpu;

void bland::cpu::write_to_file(ndarray x, std::string_view file_path) {
    using file_dtype = float;
    
    std::ofstream out_file(file_path.data(), std::ios::binary);
    if (!out_file) {
        throw std::runtime_error("Unable to open file for writing.");
    }

    // do a typed dispatch :shrug:
    auto x_data = x.data_ptr<file_dtype>(); // TODO: templatize and dispatch
    auto x_shape = x.shape();
    auto x_strides = x.strides();
    auto x_offset = x.offsets();

    // Current (multi-dimensional) index for a and out
    std::vector<int64_t> nd_index(x.ndim(), 0);

    int64_t x_linear_index = std::accumulate(x_offset.begin(), x_offset.end(), 0);

    auto numel = x.numel();
    for (int64_t n = 0; n < numel; ++n) {
        auto val = x_data[x_linear_index];
        out_file.write(reinterpret_cast<const char*>(&val), sizeof(file_dtype));

        // Increment the multi-dimensional index
        for (int i = nd_index.size() - 1; i >= 0; --i) {
            if (++nd_index[i] != x_shape[i]) {
                x_linear_index += x_strides[i];
                break;
            } else {
                x_linear_index -= (x_shape[i] -1) * x_strides[i];
                nd_index[i] = 0;
            }
        }
    }
    out_file.close();
}

ndarray bland::cpu::read_from_file(std::string_view file_path, ndarray::datatype dtype) {
    std::ifstream in_file(file_path.data(), std::ios::binary);
    if (!in_file) {
        throw std::runtime_error(fmt::format("Could not open file ({}) for reading.", file_path));
    }

    // Get the number of bytes, create a buffer big enough, and read it all in
    auto file_length_bytes = get_ifstream_file_length(in_file);

    int64_t numel = file_length_bytes / (dtype.bits/8);
    auto x = ndarray({numel}, dtype);
    auto x_ptr = x.data_ptr<char>();
    in_file.read(x_ptr, file_length_bytes);

    return x;
}

ndarray bland::cpu::copy(ndarray a, ndarray &out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_copy_op>(out, a);
}

ndarray bland::cpu::square(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_square_op>(out, a);
}

ndarray bland::cpu::sqrt(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_sqrt_op>(out, a);
}

ndarray bland::cpu::abs(ndarray a, ndarray& out) {
    return dispatch_new2<cpu::unary_op_impl_wrapper, elementwise_abs_op>(out, a);
}

template <typename T>
ndarray bland::cpu::fill(ndarray out, T value) {
    return dispatch_scalar<cpu::assignment_op_impl_wrapper, T>(out, value);
}

template ndarray bland::cpu::fill<float>(ndarray out, float v);
// template ndarray bland::cpu::fill<double>(ndarray out, double v);
// template ndarray bland::cpu::fill<int8_t>(ndarray out, int8_t v);
// template ndarray bland::cpu::fill<int16_t>(ndarray out, int16_t v);
template ndarray bland::cpu::fill<int32_t>(ndarray out, int32_t v);
// template ndarray bland::cpu::fill<int64_t>(ndarray out, int64_t v);
template ndarray bland::cpu::fill<uint8_t>(ndarray out, uint8_t v);
// template ndarray bland::cpu::fill<uint16_t>(ndarray out, uint16_t v);
template ndarray bland::cpu::fill<uint32_t>(ndarray out, uint32_t v);
// template ndarray bland::cpu::fill<uint64_t>(ndarray out, uint64_t v);
