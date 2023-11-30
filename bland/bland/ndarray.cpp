
#include "bland/ndarray.hpp"

#include "bland/ops.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <numeric> // accumulate

using namespace bland;

namespace bland::detail {

// Effectively a from_dlpack method...
blandDLTensor::blandDLTensor(DLManagedTensor other_tensor) {
    // TODO: check this is used...
    // fmt::print("Construct blandDLTensor from DLManagedTensor\n");
    DLTensor::ndim = other_tensor.dl_tensor.ndim;
    // Deep copy the shape and stides to RAII container
    _shape_ownership   = std::vector<int64_t>(other_tensor.dl_tensor.shape,
                                            other_tensor.dl_tensor.shape + other_tensor.dl_tensor.ndim);
    DLTensor::shape    = _shape_ownership.data();
    _strides_ownership = std::vector<int64_t>(other_tensor.dl_tensor.strides,
                                              other_tensor.dl_tensor.strides + other_tensor.dl_tensor.ndim);
    DLTensor::strides  = _strides_ownership.data();

    // dlpack can't support offsets, so they must be 0
    blandDLTensor::_offsets = std::vector<int64_t>(DLTensor::ndim, 0);

    DLTensor::device  = other_tensor.dl_tensor.device;
    DLTensor::dtype   = other_tensor.dl_tensor.dtype;
    DLTensor::data    = other_tensor.dl_tensor.data;
    DLTensor::shape   = other_tensor.dl_tensor.shape;
    DLTensor::strides = other_tensor.dl_tensor.strides;
    // TODO, need to keep a reference to this and call its deleter eventually
}

blandDLTensor::blandDLTensor(const std::vector<int64_t> &shape,
                             DLDataType                  dtype,
                             DLDevice                    device,
                             std::vector<int64_t>        strides) {
    // fmt::print("Construct blandDLTensor from individual specs\n");

    DLTensor::ndim                  = shape.size();
    blandDLTensor::_shape_ownership = std::vector<int64_t>(shape);
    blandDLTensor::shape            = _shape_ownership.data();

    blandDLTensor::_offsets = std::vector<int64_t>(DLTensor::ndim, 0);

    if (strides.empty()) {
        blandDLTensor::_strides_ownership = std::vector<int64_t>(DLTensor::ndim, 0);
        blandDLTensor::strides            = _strides_ownership.data();

        int64_t stride = 1; // Stride for the last dimension
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t j             = shape.size() - i - 1; // Reverse index for row-major
            DLTensor::strides[j] = stride;               // Set stride for this dimension
            stride *= shape[j];                          // Update stride for the next dimension
        }
    } else {
        blandDLTensor::_strides_ownership = std::vector<int64_t>(strides);
        blandDLTensor::strides            = _strides_ownership.data();
    }

    DLTensor::dtype       = dtype;
    DLTensor::device      = device;
    DLTensor::byte_offset = 0;
    if (DLTensor::device.device_type == DLDeviceType::kDLCPU) {
        // DLPack API "requires" 256-byte alignment. TODO: understand lanes > 1 better...
        int64_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        // _data_ownership = std::shared_ptr<void>(malloc(num_elements * (DLTensor::dtype.bits / 8)), free);
        void *ptr = nullptr;
        auto  res = posix_memalign(&ptr, 256, num_elements * (DLTensor::dtype.bits / 8));
        if (res == ENOMEM) {
            throw std::runtime_error("Not enough memory to allocate for this tensor");
        } else if (res == EINVAL) {
            throw std::runtime_error("Alignment of 256B not suitable for this platform");
        }
        _data_ownership = std::shared_ptr<void>(ptr, free);
    } else if (DLTensor::device.device_type == DLDeviceType::kDLCUDA) { /*todo*/
    } else {
        throw std::runtime_error("Unsupported device");
    }
    DLTensor::data = _data_ownership.get();
}

// Copy constructor
blandDLTensor::blandDLTensor(const blandDLTensor &other) {
    // fmt::print("Construct blandDLTensor from copy of another blandDLTensor\n");

    this->byte_offset     = other.byte_offset;
    this->_data_ownership = other._data_ownership; // shared_ptr increases refcount
    // This might have been called with a DLTensor from a DLManagedTensor that is just a view from
    // another framework. The safest thing to do is always copy the data ptr rather than `get`ing
    // the data ptr from our data_ownership (which will be null if this is a view from another framework)
    // Consider making a copy constructor that takes a DLTensor to avoid confusion (but increase repetition)
    this->data = other.data;

    this->device = other.device;
    this->dtype  = other.dtype;
    this->ndim   = other.ndim;

    this->_offsets = other._offsets;

    // Deep copy shape and strides, setting the shape, strides ptrs
    // to our owned copy.
    this->_shape_ownership = other._shape_ownership;
    this->shape            = this->_shape_ownership.data();

    this->_strides_ownership = other._strides_ownership;
    this->strides            = this->_strides_ownership.data();
}

// Copy assignment
blandDLTensor &blandDLTensor::operator=(const blandDLTensor &other) {
    if (this != &other) {
        this->byte_offset     = other.byte_offset;
        this->_data_ownership = other._data_ownership; // shared_ptr increases refcount
        this->data            = static_cast<void *>(this->_data_ownership.get());

        this->device = other.device;
        this->dtype  = other.dtype;
        this->ndim   = other.ndim;

        this->_offsets = other._offsets;

        // Deep copy shape and strides, setting the shape, strides ptrs
        // to our owned copy.
        this->_shape_ownership = other._shape_ownership;
        this->shape            = this->_shape_ownership.data();

        this->_strides_ownership = other._strides_ownership;
        this->strides            = this->_strides_ownership.data();
    }
    return *this;
}

// Method to export to other frameworks
DLManagedTensor *blandDLTensor::to_dlpack() {
    // TODO: if we have a non-zero offset we need to make a copy
    // under some conditions we should be able to use the byte_offset instead of copying
    // but need to think through corner cases
    bool copy_required = false;
    for (const auto &offset : _offsets) {
        if (offset > 0) {
            copy_required = true;
            break;
        }
    }
    if (copy_required) {
        // auto dense_copy = copy(*this);
        fmt::print("deep copy of data buf required");
        // TODO actually do the copy...
    }

    // Create a DLManagedTensor that shares the memory with the NDArray
    // Set the deleter function to be called when the DLManagedTensor is destroyed
    DLManagedTensor *tensor = new DLManagedTensor;
    // tensor->dl_tensor       = *this; // Copy constructor of blandDLTensor (increases data refcount)
    // this->_data_ownership.

    // Dynamically allocate a new blandDLTensor to co-own underlying data
    auto context_owner  = new blandDLTensor(*this);
    tensor->dl_tensor   = *context_owner;
    tensor->manager_ctx = context_owner;

    tensor->deleter = [](DLManagedTensor *self) {
        // We're basically letting numpy actually delete data. We *want* numpy to just allow us to delete...
        // we need to check the refcounts before just deleting I think
        auto context = static_cast<blandDLTensor *>(self->manager_ctx);
        delete context;
    };
    return tensor;
}

} // namespace bland::detail

/**
 * ndarray impl
 */

bland::ndarray::ndarray(DLManagedTensor tensor) : _tensor(tensor) {
    // Create a new NDArray that shares the memory with the DLManagedTensor
    // Call the deleter function when the NDArray is destroyed
}

bland::ndarray::ndarray(std::vector<int64_t> dims, DLDataType dtype, DLDevice device) :
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

template <typename T>
bland::ndarray::ndarray(std::vector<int64_t> dims, T initial_value, DLDataType dtype, DLDevice device) :
        _tensor(detail::blandDLTensor(dims, dtype, device, {})) {
    int64_t stride = 1; // Stride for the last dimension
    for (size_t i = 0; i < dims.size(); ++i) {
        size_t j           = dims.size() - i - 1; // Reverse index for row-major
        _tensor.shape[j]   = dims[j];             // Copy dimension to shape
        _tensor.strides[j] = stride;              // Set stride for this dimension
        stride *= _tensor.shape[j];               // Update stride for the next dimension
    }

    switch (dtype.code) {
    case kDLFloat: {
        switch (dtype.bits) {
        case 32:
            initialize_memory(data_ptr<float>(), numel(), static_cast<float>(initial_value));
            break;
        case 64:
            initialize_memory(data_ptr<double>(), numel(), static_cast<double>(initial_value));
            break;
        default:
            throw std::runtime_error("Unsupported float bitwidth");
        }
        break;
    }
    case kDLInt: {
        switch (dtype.bits) {
        case 8:
            initialize_memory(data_ptr<int8_t>(), numel(), static_cast<int8_t>(initial_value));
            break;
        case 16:
            initialize_memory(data_ptr<int16_t>(), numel(), static_cast<int16_t>(initial_value));
            break;
        case 32:
            initialize_memory(data_ptr<int32_t>(), numel(), static_cast<int32_t>(initial_value));
            break;
        case 64:
            initialize_memory(data_ptr<int64_t>(), numel(), static_cast<int64_t>(initial_value));
            break;
        default:
            throw std::runtime_error("Unsupported int bitwidth");
        }
        break;
    }
    case kDLUInt: {
        switch (dtype.bits) {
        case 8:
            initialize_memory(data_ptr<uint8_t>(), numel(), static_cast<uint8_t>(initial_value));
            break;
        case 16:
            initialize_memory(data_ptr<uint16_t>(), numel(), static_cast<uint16_t>(initial_value));
            break;
        case 32:
            initialize_memory(data_ptr<uint32_t>(), numel(), static_cast<uint32_t>(initial_value));
            break;
        case 64:
            initialize_memory(data_ptr<uint64_t>(), numel(), static_cast<uint64_t>(initial_value));
            break;
        default:
            throw std::runtime_error("Unsupported uint bitwidth");
        }
        break;
    }
    default:
        auto error_message =
                fmt::format("ndarray (ctor): unsupported datatype code {} with {} bits", dtype.code, dtype.bits);
        throw std::runtime_error(error_message);
    }
}

template bland::ndarray::ndarray(std::vector<int64_t> dims, float initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, double initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, int8_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, int16_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, int32_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, int64_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint8_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint16_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint32_t initial_value, DLDataType dtype, DLDevice device);
template bland::ndarray::ndarray(std::vector<int64_t> dims, uint64_t initial_value, DLDataType dtype, DLDevice device);

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

DLDataType bland::ndarray::dtype() const {
    return _tensor.dtype;
}

DLDevice bland::ndarray::device() const {
    return _tensor.device;
}

int64_t bland::ndarray::numel() const {
    int64_t num_elements = std::accumulate(_tensor.shape, _tensor.shape + _tensor.ndim, 1, std::multiplies<int64_t>());
    return num_elements;
}

int64_t bland::ndarray::ndim() const {
    return _tensor.ndim;
}

template <typename datatype>
std::string pretty_print(const ndarray &a) {
    auto a_data = a.data_ptr<datatype>();

    // (multi-dimensional) index to read into a
    std::vector<int64_t> index(a.shape().size(), 0);

    std::string dtype_pp{};
    if (std::is_same<datatype, float>()) {
        dtype_pp = "float32";
    } else if (std::is_same<datatype, double>()) {
        dtype_pp = "float64";
    } else if (std::is_same<datatype, int8_t>()) {
        dtype_pp = "int8_t";
    } else if (std::is_same<datatype, int16_t>()) {
        dtype_pp = "int16_t";
    } else if (std::is_same<datatype, int32_t>()) {
        dtype_pp = "int32_t";
    } else if (std::is_same<datatype, int64_t>()) {
        dtype_pp = "int64_t";
    }
    fmt::memory_buffer output_repr;
    fmt::format_to(output_repr, "bland {}:\n", dtype_pp);
    for (int64_t n = 0; n < a.numel(); ++n) {
        // Finally... do the actual op
        int64_t linear_index = 0;
        for (int i = 0; i < a.shape().size(); ++i) {
            linear_index += a.offsets()[i] + (index[i] % a.shape()[i]) * a.strides()[i];
        }
        fmt::format_to(output_repr, "{} ", a_data[linear_index]);

        // Increment the multi-dimensional index
        for (int i = a.shape().size() - 1; i >= 0; --i) {
            // If we're not at the end of this dim, keep going
            if (++index[i] != a.shape()[i]) {
                break;
            } else {
                // Otherwise, set it to 0 and move down to the next dim
                index[i] = 0;
                fmt::format_to(output_repr, "\n");
            }
        }
    }

    return fmt::to_string(output_repr);
}

std::string bland::ndarray::repr() const {
    auto dtype = _tensor.dtype;
    switch (dtype.code) {
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
template ndarray bland::ndarray::operator+<double>(const double &b) const;
template ndarray bland::ndarray::operator+<int8_t>(const int8_t &b) const;
template ndarray bland::ndarray::operator+<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator+<int32_t>(const int32_t &b) const;
template ndarray bland::ndarray::operator+<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator+<uint8_t>(const uint8_t &b) const;
template ndarray bland::ndarray::operator+<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator+<uint32_t>(const uint32_t &b) const;
template ndarray bland::ndarray::operator+<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator+(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs + lhs;
}

template ndarray bland::operator+<float>(const float &lhs, const ndarray &rhs);
template ndarray bland::operator+<double>(const double &lhs, const ndarray &rhs);
template ndarray bland::operator+<int8_t>(const int8_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<int32_t>(const int32_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
template ndarray bland::operator+<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

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
template ndarray bland::ndarray::operator-<double>(const double &b) const;
template ndarray bland::ndarray::operator-<int8_t>(const int8_t &b) const;
template ndarray bland::ndarray::operator-<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator-<int32_t>(const int32_t &b) const;
template ndarray bland::ndarray::operator-<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator-<uint8_t>(const uint8_t &b) const;
template ndarray bland::ndarray::operator-<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator-<uint32_t>(const uint32_t &b) const;
template ndarray bland::ndarray::operator-<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator-(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs - lhs;
}

template ndarray bland::operator-<float>(const float &lhs, const ndarray &rhs);
template ndarray bland::operator-<double>(const double &lhs, const ndarray &rhs);
template ndarray bland::operator-<int8_t>(const int8_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<int32_t>(const int32_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
template ndarray bland::operator-<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

template <typename T>
ndarray bland::ndarray::multiply(const T &b) const {
    return bland::multiply(*this, b);
}

template ndarray bland::ndarray::multiply<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::multiply<float>(const float &b) const;
template ndarray bland::ndarray::multiply<double>(const double &b) const;
template ndarray bland::ndarray::multiply<int8_t>(const int8_t &b) const;
template ndarray bland::ndarray::multiply<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::multiply<int32_t>(const int32_t &b) const;
template ndarray bland::ndarray::multiply<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::multiply<uint8_t>(const uint8_t &b) const;
template ndarray bland::ndarray::multiply<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::multiply<uint32_t>(const uint32_t &b) const;
template ndarray bland::ndarray::multiply<uint64_t>(const uint64_t &b) const;

template <typename T>
ndarray bland::ndarray::operator*(const T &b) const {
    return this->multiply(b);
}

template ndarray bland::ndarray::operator*<ndarray>(const ndarray &b) const;
template ndarray bland::ndarray::operator*<float>(const float &b) const;
template ndarray bland::ndarray::operator*<double>(const double &b) const;
template ndarray bland::ndarray::operator*<int8_t>(const int8_t &b) const;
template ndarray bland::ndarray::operator*<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator*<int32_t>(const int32_t &b) const;
template ndarray bland::ndarray::operator*<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator*<uint8_t>(const uint8_t &b) const;
template ndarray bland::ndarray::operator*<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator*<uint32_t>(const uint32_t &b) const;
template ndarray bland::ndarray::operator*<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator*(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs * lhs;
}

template ndarray bland::operator*<float>(const float &lhs, const ndarray &rhs);
template ndarray bland::operator*<double>(const double &lhs, const ndarray &rhs);
template ndarray bland::operator*<int8_t>(const int8_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<int32_t>(const int32_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
template ndarray bland::operator*<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

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
template ndarray bland::ndarray::operator/<double>(const double &b) const;
template ndarray bland::ndarray::operator/<int8_t>(const int8_t &b) const;
template ndarray bland::ndarray::operator/<int16_t>(const int16_t &b) const;
template ndarray bland::ndarray::operator/<int32_t>(const int32_t &b) const;
template ndarray bland::ndarray::operator/<int64_t>(const int64_t &b) const;
template ndarray bland::ndarray::operator/<uint8_t>(const uint8_t &b) const;
template ndarray bland::ndarray::operator/<uint16_t>(const uint16_t &b) const;
template ndarray bland::ndarray::operator/<uint32_t>(const uint32_t &b) const;
template ndarray bland::ndarray::operator/<uint64_t>(const uint64_t &b) const;

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type bland::operator/(const T       &lhs,
                                                                                      const ndarray &rhs) {
    return rhs * lhs;
}

template ndarray bland::operator/<float>(const float &lhs, const ndarray &rhs);
template ndarray bland::operator/<double>(const double &lhs, const ndarray &rhs);
template ndarray bland::operator/<int8_t>(const int8_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<int16_t>(const int16_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<int32_t>(const int32_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<int64_t>(const int64_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint8_t>(const uint8_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint16_t>(const uint16_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint32_t>(const uint32_t &lhs, const ndarray &rhs);
template ndarray bland::operator/<uint64_t>(const uint64_t &lhs, const ndarray &rhs);

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
    int64_t current_stride = 1;
    for (int axis = 0; axis < _tensor.ndim + 1; ++axis) {
        if (axis == unsqueeze_axis) {
            new_shape.push_back(1);
            new_strides.push_back(current_stride);
            new_offsets.push_back(0);
            axis -= 1;
        } else {
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

template <typename... Args>
ndarray_slice bland::ndarray::slice(Args... args) {
    return bland::slice(*this, args...);
}

template ndarray_slice bland::ndarray::slice(slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec, slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice
        bland::ndarray::slice(slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec, slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec);
template ndarray_slice bland::ndarray::slice(slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec,
                                             slice_spec);

/*******************************
 ******* ndarray_slice *********
 ********************************/

ndarray_slice::ndarray_slice(const ndarray &other) : ndarray(other) {}

ndarray_slice &ndarray_slice::operator=(const ndarray_slice &rhs) {

    copy(rhs, *this);
    return *this;
};

ndarray_slice &ndarray_slice::operator=(const ndarray &rhs) {
    copy(rhs, *this);
    return *this;
};
