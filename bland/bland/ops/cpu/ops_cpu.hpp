#pragma once

#include <bland/ndarray.hpp>


namespace bland {
namespace cpu {

void write_to_file(ndarray x, std::string_view file_path);
ndarray read_from_file(std::string_view file_path, ndarray::datatype dtype);

ndarray copy(ndarray a, ndarray &out);
ndarray square(ndarray a, ndarray& out);
ndarray sqrt(ndarray a, ndarray& out);
ndarray abs(ndarray a, ndarray& out);

template <typename T>
ndarray fill(ndarray out, T value);

} // namespace cpu
} // namespace bland
