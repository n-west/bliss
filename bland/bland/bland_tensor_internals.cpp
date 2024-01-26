
#include "bland/detail/bland_tensor_internals.hpp"

using namespace bland;

namespace bland::detail {

blandDLTensor::blandDLTensor() {
    DLTensor::data = nullptr;
    DLTensor::shape = nullptr; // Should we actually make this valid but 0?
    DLTensor::ndim = 0;
    DLTensor::strides = nullptr;
}

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
