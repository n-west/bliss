
#include "bland/ndarray.hpp"
#include "bland/ndarray_slice.hpp"
#include "bland/ops/ops_arithmetic.hpp"
#include "bland/ops/ops.hpp" // to, fill
#include <bland/config.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <complex> // std::complex
#include <numeric> // accumulate
#include <regex> // regex (parsing device id)

using namespace bland;

/**
 * ndarray impl
 */

bland::ndarray::datatype::datatype(std::string_view dtype_str) {
    if (dtype_str == "float" || dtype_str == "float32") {
        code  = kDLFloat;
        bits  = 32;
        lanes = 1;
    } else if (dtype_str == "double" || dtype_str == "float64") {
        code  = kDLFloat;
        bits  = 64;
        lanes = 1;
    } else if (dtype_str == "int" || dtype_str == "int64") {
        code  = kDLInt;
        bits  = 64;
        lanes = 1;
    } else if (dtype_str == "int32") {
        code  = kDLInt;
        bits  = 32;
        lanes = 1;
    } else if (dtype_str == "int16") {
        code  = kDLInt;
        bits  = 16;
        lanes = 1;
    } else if (dtype_str == "int8") {
        code  = kDLInt;
        bits  = 8;
        lanes = 1;
    } else if (dtype_str == "uint" || dtype_str == "uint64") {
        code  = kDLUInt;
        bits  = 64;
        lanes = 1;
    } else if (dtype_str == "uint32") {
        code  = kDLUInt;
        bits  = 32;
        lanes = 1;
    } else if (dtype_str == "uint16") {
        code  = kDLUInt;
        bits  = 16;
        lanes = 1;
    } else if (dtype_str == "uint8") {
        code  = kDLUInt;
        bits  = 8;
        lanes = 1;
    }
}
bland::ndarray::datatype::datatype(DLDataType dtype) : DLDataType(dtype) {}

bool bland::ndarray::datatype::operator==(const datatype&other) {
    return this->code == other.code && this->bits == other.bits && this->lanes == other.lanes;
}

bool bland::ndarray::datatype::operator!=(const datatype&other) {
    return !(*this == other);
}

std::string bland::ndarray::datatype::repr() {
    auto r = fmt::format("ndarray::datatype (.code={}, .bits={}, .lanes={})\n", static_cast<int8_t>(code), bits, lanes);
    return r;
}

bland::ndarray::dev::dev(DLDevice d) : DLDevice(d) {}

bool bland::ndarray::dev::operator==(const dev &other) {
    return this->device_type == other.device_type && this->device_id == other.device_id;
}

bool bland::ndarray::dev::operator==(const DLDevice &other) {
    return this->device_type == other.device_type && this->device_id == other.device_id;
}

bool bland::ndarray::dev::operator!=(const dev &other) {
    return !(*this == other);
}

std::string bland::ndarray::dev::repr() {
    auto r = fmt::format("ndarray::dev (.device_type={}, .device_id={})\n", static_cast<int8_t>(device_type), device_id);
    return r;
}

bland::ndarray::dev::dev(std::string_view dev_str) {
    if (dev_str == "cpu") {
		device_type = DLDeviceType::kDLCPU;
		device_id   = 0;
    } else if (dev_str.substr(0, 4) == "cuda") {
        device_type = DLDeviceType::kDLCUDA;
        device_id   = 0;
        if (dev_str.size() > 4 && dev_str[4] == ':') {
            device_id = std::stoi(std::string(dev_str.substr(5)));
        }
    } else if (dev_str.substr(0, 11) == "cudamanaged") {
        device_type = DLDeviceType::kDLCUDAManaged;
        device_id   = 0;
        if (dev_str.size() > 11 && dev_str[11] == ':') {
            device_id = std::stoi(std::string(dev_str.substr(12)));
        }
    } else if (dev_str.substr(0, 8) == "cudahost") {
        device_type = DLDeviceType::kDLCUDAHost;
        device_id   = 0;
        if (dev_str.size() > 8 && dev_str[8] == ':') {
            device_id = std::stoi(std::string(dev_str.substr(9)));
        }
    } else {
        throw std::runtime_error("Device type not supported in bland yet");
	}
}

bland::ndarray::ndarray(DLManagedTensor tensor) : _tensor(tensor) {
    // Create a new NDArray that shares the memory with the DLManagedTensor
    // Call the deleter function when the NDArray is destroyed
}

bland::ndarray::ndarray(std::vector<int64_t> dims, datatype dtype, DLDevice device) :
        _tensor(detail::blandDLTensor(dims, dtype, device, {})) {
    int64_t stride = 1; // Stride for the last dimension
    for (size_t i = 0; i < dims.size(); ++i) {
        size_t j           = dims.size() - i - 1; // Reverse index for row-major
        _tensor.shape[j]   = dims[j];             // Copy dimension to shape
        _tensor.strides[j] = stride;              // Set stride for this dimension
        stride *= _tensor.shape[j];               // Update stride for the next dimension
    }
}

template <typename T>
void initialize_memory(T *data, int64_t numel, T value) {
    // We know this is a dense array
    for (int n = 0; n < numel; ++n) {
        data[n] = value;
    }
}

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type *>
bland::ndarray::ndarray(std::vector<int64_t> dims, T initial_value, datatype dtype, DLDevice device) :
        _tensor(detail::blandDLTensor(dims, dtype, device, {})) {
    int64_t stride = 1; // Stride for the last dimension
    for (size_t i = 0; i < dims.size(); ++i) {
        size_t j           = dims.size() - i - 1; // Reverse index for row-major
        _tensor.shape[j]   = dims[j];             // Copy dimension to shape
        _tensor.strides[j] = stride;              // Set stride for this dimension
        stride *= _tensor.shape[j];               // Update stride for the next dimension
    }

    fill(*this, initial_value);
}

template bland::ndarray::ndarray(std::vector<int64_t> dims, float initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, double initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, int8_t initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, int16_t initial_value, datatype dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, int32_t initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, int64_t initial_value, datatype dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint8_t initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, uint16_t initial_value, datatype dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint32_t initial_value, datatype dtype, DLDevice device);
// template bland::ndarray::ndarray(std::vector<int64_t> dims, uint64_t initial_value, datatype dtype, DLDevice device);

DLManagedTensor *bland::ndarray::get_managed_tensor() {
    return _tensor.to_dlpack();
}

std::vector<int64_t> bland::ndarray::shape() const {
    return std::vector<int64_t>(_tensor.shape, _tensor.shape + _tensor.ndim);
}

int64_t bland::ndarray::size(int64_t axis) const {
    if (axis < _tensor.ndim) {
        return _tensor.shape[axis];
    } else {
        throw std::invalid_argument("bad axis for ndim");
    }
}

std::vector<int64_t> bland::ndarray::strides() const {
    return std::vector<int64_t>(_tensor.strides, _tensor.strides + _tensor.ndim);
}

std::vector<int64_t> bland::ndarray::offsets() const {
    return _tensor._offsets;
}

bland::ndarray::datatype bland::ndarray::dtype() const {
    return _tensor.dtype;
}

bland::ndarray::dev bland::ndarray::device() const {
    return _tensor.device;
}

int64_t bland::ndarray::numel() const {
    int64_t num_elements = std::accumulate(_tensor.shape, _tensor.shape + _tensor.ndim, 1, std::multiplies<int64_t>());
    return num_elements;
}

int64_t bland::ndarray::ndim() const {
    return _tensor.ndim;
}

template <typename T>
T bland::ndarray::scalarize() const {
    if (numel() > 1) {
        throw std::runtime_error("scalarize: There is more than one element, so not a scalar. Cannot scalarize");
    }
    // You can only scalarize a cpu value...
    auto cpu_copy = bland::to(*this, "cpu");
    // TODO also check datatype makes sense to scalarize
    return cpu_copy.data_ptr<T>()[0];
}

template float    bland::ndarray::scalarize<float>() const;
template double   bland::ndarray::scalarize<double>() const;
template int8_t   bland::ndarray::scalarize<int8_t>() const;
template int16_t  bland::ndarray::scalarize<int16_t>() const;
template int32_t  bland::ndarray::scalarize<int32_t>() const;
template int64_t  bland::ndarray::scalarize<int64_t>() const;
template uint8_t  bland::ndarray::scalarize<uint8_t>() const;
template uint16_t bland::ndarray::scalarize<uint16_t>() const;
template uint32_t bland::ndarray::scalarize<uint32_t>() const;
template uint64_t bland::ndarray::scalarize<uint64_t>() const;

template <typename T>
T bland::ndarray::scalarize(const std::vector<int64_t> &nd_index) const {

    if (nd_index.size() != ndim()) {
        throw std::runtime_error("scalarize: wrong number of dimensions indexing to array");
    }

    auto linear_offset = 0;
    for (int dim = 0; dim < nd_index.size(); ++dim) {
        linear_offset +=  _tensor._offsets[dim] +  nd_index[dim] * _tensor.strides[dim]; // TODO: also add offset
    }
    // TODO also check datatype makes sense to scalarize
    return data_ptr<T>()[linear_offset];
}

template float    bland::ndarray::scalarize<float>(const std::vector<int64_t> &) const;
template double   bland::ndarray::scalarize<double>(const std::vector<int64_t> &) const;
template int8_t   bland::ndarray::scalarize<int8_t>(const std::vector<int64_t> &) const;
template int16_t  bland::ndarray::scalarize<int16_t>(const std::vector<int64_t> &) const;
template int32_t  bland::ndarray::scalarize<int32_t>(const std::vector<int64_t> &) const;
template int64_t  bland::ndarray::scalarize<int64_t>(const std::vector<int64_t> &) const;
template uint8_t  bland::ndarray::scalarize<uint8_t>(const std::vector<int64_t> &) const;
template uint16_t bland::ndarray::scalarize<uint16_t>(const std::vector<int64_t> &) const;
template uint32_t bland::ndarray::scalarize<uint32_t>(const std::vector<int64_t> &) const;
template uint64_t bland::ndarray::scalarize<uint64_t>(const std::vector<int64_t> &) const;

ndarray bland::ndarray::to(const ndarray::dev &dest) {
    return bland::to(*this, dest);
}

ndarray bland::ndarray::to(std::string_view dest) {
    return bland::to(*this, dest);
}

template <typename datatype>
std::string pretty_print(const ndarray &a) {
    // (multi-dimensional) index to read into a
    std::vector<int64_t> index(a.shape().size(), 0);

    std::string type_specifier{};
    std::string dtype_pp{};
    if (std::is_same<datatype, float>()) {
        dtype_pp = "float32";
        type_specifier = "{:f}";
    } else if (std::is_same<datatype, double>()) {
        dtype_pp = "float64";
    } else if (std::is_same<datatype, std::complex<float>>()) {
        dtype_pp = "complex<float32>";
    } else if (std::is_same<datatype, int8_t>()) {
        dtype_pp = "int8_t";
    } else if (std::is_same<datatype, int16_t>()) {
        dtype_pp = "int16_t";
    } else if (std::is_same<datatype, int32_t>()) {
        dtype_pp = "int32_t";
    } else if (std::is_same<datatype, int64_t>()) {
        dtype_pp = "int64_t";
    } else if (std::is_same<datatype, uint8_t>()) {
        dtype_pp = "uint8_t";
    } else if (std::is_same<datatype, uint16_t>()) {
        dtype_pp = "uint16_t";
    } else if (std::is_same<datatype, uint32_t>()) {
        dtype_pp = "uint32_t";
    } else if (std::is_same<datatype, uint64_t>()) {
        dtype_pp = "uint64_t";
    }
    std::string dev_pp{};
    auto        dev = a.device();
    if (dev.device_type == bland::ndarray::dev::cpu.device_type) {
        dev_pp = fmt::format("cpu:{}", dev.device_id);
    } else if (dev.device_type == bland::ndarray::dev::cuda.device_type) {
        dev_pp = fmt::format("cuda:{}", dev.device_id);
    } else if (dev.device_type == bland::ndarray::dev::cuda_managed.device_type) {
        dev_pp = fmt::format("cuda_managed:{}", dev.device_id);
    } else if (dev.device_type == bland::ndarray::dev::cuda_host.device_type) {
        dev_pp = fmt::format("cuda_host:{}", dev.device_id);
    } else {
        dev_pp = "not implemented device";
    }
    std::string output_repr;
    fmt::format_to(std::back_inserter(output_repr), "bland shape {} with dtype {} on device {}:\n", a.shape(), dtype_pp, dev_pp);
    int count = 0;
    auto a_cpu = bland::to(a, "cpu");
    auto a_data = a_cpu.data_ptr<datatype>();

    // TODO: condense printing large arrays, no one needs or wants to see thousands of items
    auto numel = std::min<int64_t>(a_cpu.numel(), 25);
    for (int64_t n = 0; n < numel; ++n) {
        // Finally... do the actual op
        int64_t linear_index = 0;
        for (int i = 0; i < a_cpu.shape().size(); ++i) {
            linear_index += a_cpu.offsets()[i] + (index[i] % a_cpu.shape()[i]) * a_cpu.strides()[i];
        }
        fmt::format_to(std::back_inserter(output_repr), "{} ", a_data[linear_index]);

        // Increment the multi-dimensional index
        for (int i = a_cpu.shape().size() - 1; i >= 0; --i) {
            // If we're not at the end of this dim, keep going
            if (++index[i] != a_cpu.shape()[i]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                index[i] = 0;
                fmt::format_to(std::back_inserter(output_repr), "\n");
            }
        }
    }

    return output_repr;
}

std::string bland::ndarray::repr() const {
    auto dtype = _tensor.dtype;
    switch (dtype.code) {
    case kDLComplex: {
        switch (dtype.bits) {
        case 64: {
            return pretty_print<std::complex<float>>(*this);
        } break;
        case 128: {
            return pretty_print<std::complex<double>>(*this);
        } break;
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    } break;
    case kDLFloat: {
        switch (dtype.bits) {
        case 32: {
            return pretty_print<float>(*this);
        } break;
        case 64: {
            return pretty_print<double>(*this);
        } break;
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
    } break;
    case kDLInt: {
        switch (dtype.bits) {
        case 8: {
            return pretty_print<int8_t>(*this);
        } break;
        case 16: {
            return pretty_print<int16_t>(*this);
        } break;
        case 32: {
            return pretty_print<int32_t>(*this);
        } break;
        case 64: {
            return pretty_print<int64_t>(*this);
        } break;
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
    } break;
    case kDLUInt: {
        switch (dtype.bits) {
        case 8: {
            return pretty_print<uint8_t>(*this);
        } break;
        case 16: {
            return pretty_print<uint16_t>(*this);
        } break;
        case 32: {
            return pretty_print<uint32_t>(*this);
        } break;
        case 64: {
            return pretty_print<uint64_t>(*this);
        } break;
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
    } break;
    default:
        auto err = fmt::format("Unsupported datatype code {}", static_cast<int>(dtype.code));
        throw std::runtime_error(err);
    }
}

template <typename T>
ndarray bland::ndarray::add(const T &b) const {
    return bland::add(*this, b);
}

template <typename T>
ndarray bland::ndarray::operator+(const T &b) const {
    return this->add(b);
}

template ndarray bland::ndarray::operator+<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::operator+<ndarray_slice>(const ndarray_slice &b) const;
template ndarray bland::ndarray::operator+<float>(const float &b) const;
// template ndarray bland::ndarray::operator+<double>(const double &b) const;
// template ndarray bland::ndarray::operator+<int8_t>(const int8_t &b) const;
// template ndarray bland::ndarray::operator+<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator+<int32_t>(const int32_t &b) const;
// template ndarray bland::ndarray::operator+<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator+<uint8_t>(const uint8_t &b) const;
// template ndarray bland::ndarray::operator+<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator+<uint32_t>(const uint32_t &b) const;
// template ndarray bland::ndarray::operator+<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator+(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs + lhs;
}

template ndarray bland::operator+<float>(const float &lhs, const ndarray &rhs);
// template ndarray bland::operator+<double>(const double &lhs, const ndarray &rhs);
// template ndarray bland::operator+<int8_t>(const int8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator+<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<int32_t>(const int32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator+<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator+<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator+<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

template <typename T>
ndarray bland::ndarray::subtract(const T &b) const {
    return bland::subtract(*this, b);
}

template <typename T>
ndarray bland::ndarray::operator-(const T &b) const {
    return this->subtract(b);
}

template ndarray bland::ndarray::operator-<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::operator-<float>(const float &b) const;
// template ndarray bland::ndarray::operator-<double>(const double &b) const;
// template ndarray bland::ndarray::operator-<int8_t>(const int8_t &b) const;
// template ndarray bland::ndarray::operator-<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator-<int32_t>(const int32_t &b) const;
// template ndarray bland::ndarray::operator-<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator-<uint8_t>(const uint8_t &b) const;
// template ndarray bland::ndarray::operator-<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator-<uint32_t>(const uint32_t &b) const;
// template ndarray bland::ndarray::operator-<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator-(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs - lhs;
}

template ndarray bland::operator-<float>(const float &lhs, const ndarray &rhs);
// template ndarray bland::operator-<double>(const double &lhs, const ndarray &rhs);
// template ndarray bland::operator-<int8_t>(const int8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator-<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<int32_t>(const int32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator-<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator-<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator-<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

template <typename T>
ndarray bland::ndarray::multiply(const T &b) const {
    return bland::multiply(*this, b);
}

template ndarray bland::ndarray::multiply<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::multiply<float>(const float &b) const;
// template ndarray bland::ndarray::multiply<double>(const double &b) const;
// template ndarray bland::ndarray::multiply<int8_t>(const int8_t &b) const;
// template ndarray bland::ndarray::multiply<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::multiply<int32_t>(const int32_t &b) const;
// template ndarray bland::ndarray::multiply<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::multiply<uint8_t>(const uint8_t &b) const;
// template ndarray bland::ndarray::multiply<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::multiply<uint32_t>(const uint32_t &b) const;
// template ndarray bland::ndarray::multiply<uint64_t>(const uint64_t &b) const;

template <typename T>
ndarray bland::ndarray::operator*(const T &b) const {
    return this->multiply(b);
}

template ndarray bland::ndarray::operator*<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::operator*<float>(const float &b) const;
// template ndarray bland::ndarray::operator*<double>(const double &b) const;
// template ndarray bland::ndarray::operator*<int8_t>(const int8_t &b) const;
// template ndarray bland::ndarray::operator*<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator*<int32_t>(const int32_t &b) const;
// template ndarray bland::ndarray::operator*<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator*<uint8_t>(const uint8_t &b) const;
// template ndarray bland::ndarray::operator*<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator*<uint32_t>(const uint32_t &b) const;
// template ndarray bland::ndarray::operator*<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator*(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs * lhs;
}

template ndarray bland::operator*<float>(const float &lhs, const ndarray &rhs);
// template ndarray bland::operator*<double>(const double &lhs, const ndarray &rhs);
// template ndarray bland::operator*<int8_t>(const int8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator*<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<int32_t>(const int32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator*<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator*<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator*<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

template <typename T>
ndarray bland::ndarray::divide(const T &b) const {
    return bland::divide(*this, b);
}

template <typename T>
ndarray bland::ndarray::operator/(const T &b) const {
    return this->divide(b);
}

template ndarray bland::ndarray::operator/<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::operator/<float>(const float &b) const;
// template ndarray bland::ndarray::operator/<double>(const double &b) const;
// template ndarray bland::ndarray::operator/<int8_t>(const int8_t &b) const;
// template ndarray bland::ndarray::operator/<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator/<int32_t>(const int32_t &b) const;
// template ndarray bland::ndarray::operator/<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator/<uint8_t>(const uint8_t &b) const;
// template ndarray bland::ndarray::operator/<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator/<uint32_t>(const uint32_t &b) const;
// template ndarray bland::ndarray::operator/<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator/(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs * lhs;
}

template ndarray bland::operator/<float>(const float &lhs, const ndarray &rhs);
// template ndarray bland::operator/<double>(const double &lhs, const ndarray &rhs);
// template ndarray bland::operator/<int8_t>(const int8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator/<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<int32_t>(const int32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator/<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
// template ndarray bland::operator/<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
// template ndarray bland::operator/<uint64_t>(const uint64_t &lhs, const ndarray &rhs);


ndarray bland::ndarray::reshape(const std::vector<int64_t> &new_shape) {
    int64_t num_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int64_t>());

    if (num_elements != numel()) {
        auto error_message = fmt::format("reshape {} to {} does not match number of elements ({} to {})",
                                         shape(),
                                         new_shape,
                                         numel(),
                                         num_elements);
        throw std::runtime_error(error_message);
    }

    if (new_shape.size() != _tensor.ndim) {
        // This won't always be right and I think sometimes it might be required to make
        // a copy. That's what pytorch does if the new shape isn't compatible due to strides
        int64_t new_ndim = new_shape.size();
        // We can collapse all offsets to the last dimension
        auto new_offsets   = std::vector<int64_t>(new_ndim, 0);
        new_offsets.back() = std::accumulate(_tensor._offsets.begin(), _tensor._offsets.end(), 0);
        _tensor._offsets   = new_offsets;

        // Number of dims changing means we need to update strides
        auto    new_strides        = std::vector<int64_t>(new_ndim, 0);
        int     old_stride_axis    = _tensor.ndim - 1;
        int64_t accumulated_stride = 1;
        for (int axis = new_ndim - 1; axis >= 0; --axis) {
            if (old_stride_axis > 0) {
                accumulated_stride *= _tensor.strides[old_stride_axis];
                new_strides[axis] = _tensor.strides[old_stride_axis];
                old_stride_axis -= 1;
            } else {
                new_strides[axis] = accumulated_stride;
                accumulated_stride *= new_shape[axis];
            }
        }
        _tensor._strides_ownership = new_strides;
        _tensor.strides            = _tensor._strides_ownership.data();
        _tensor.ndim               = new_ndim;
    } else {
        throw std::runtime_error("This reshape would require a copy which isn't yet implemented");
    }

    _tensor._shape_ownership = new_shape;
    _tensor.shape            = _tensor._shape_ownership.data();

    return *this;
}

ndarray bland::ndarray::squeeze(int64_t squeeze_axis) {
    auto original_squeeze = squeeze_axis;
    if (squeeze_axis < 0) {
        squeeze_axis = ndim() + squeeze_axis;
    }

    if (squeeze_axis < 0) {
        auto error_message =
                fmt::format("ndarray::squeeze({}) cannot squeeze negative axis", original_squeeze, squeeze_axis);
        throw std::runtime_error(error_message);
    }

    if (squeeze_axis > ndim() - 1) {
        auto error_message = fmt::format("ndarray::squeeze({}) cannot squeeze axis {} of tensor with {} dims\n",
                                         original_squeeze,
                                         squeeze_axis,
                                         ndim());
        throw std::runtime_error(error_message);
    }

    if (_tensor.shape[squeeze_axis] != 1) {
        auto error_message = fmt::format("ndarray::squeeze({}) cannot squeeze axis {} with size {}",
                                         original_squeeze,
                                         squeeze_axis,
                                         shape()[squeeze_axis]);
        throw std::runtime_error(error_message);
    }

    auto new_shape   = std::vector<int64_t>();
    auto new_strides = std::vector<int64_t>();
    auto new_offsets = std::vector<int64_t>();
    for (int axis = 0; axis < _tensor.ndim; ++axis) {
        if (axis != squeeze_axis) {
            new_shape.push_back(_tensor.shape[axis]);
            new_strides.push_back(_tensor.strides[axis]);
            new_offsets.push_back(_tensor._offsets[axis]);
        }
    }
    _tensor._shape_ownership   = new_shape;
    _tensor.shape              = _tensor._shape_ownership.data();
    _tensor._strides_ownership = new_strides;
    _tensor.strides            = _tensor._strides_ownership.data();
    _tensor._offsets           = new_offsets;
    _tensor.ndim               = _tensor._shape_ownership.size();
    return *this;
}

ndarray bland::ndarray::unsqueeze(const int64_t unsqueeze_axis) {
    auto    new_shape      = std::vector<int64_t>();
    auto    new_strides    = std::vector<int64_t>();
    auto    new_offsets    = std::vector<int64_t>();
    int offset = 0;
    for (int axis = 0; axis < _tensor.ndim + 1; ++axis) {
        if (axis == unsqueeze_axis) {
            new_shape.push_back(1);
            new_strides.push_back(0);
            new_offsets.push_back(0);
            offset += 1;
        } else {
            new_shape.push_back(_tensor.shape[axis-offset]);
            new_strides.push_back(_tensor.strides[axis-offset]);
            new_offsets.push_back(_tensor._offsets[axis-offset]);
        }
    }
    _tensor._shape_ownership   = new_shape;
    _tensor.shape              = _tensor._shape_ownership.data();
    _tensor._strides_ownership = new_strides;
    _tensor.strides            = _tensor._strides_ownership.data();
    _tensor._offsets           = new_offsets;
    _tensor.ndim               = _tensor._shape_ownership.size();
    return *this;
}

