#pragma once

#include "ops.hpp"

#include <dlpack/dlpack.h>

#include <memory>
#include <string_view>
#include <vector>

namespace bland {

#define default_device                                                                                                 \
    DLDevice { .device_type = kDLCPU, .device_id = 0 }

#define default_dtype                                                                                                  \
    DLDataType { .code = kDLFloat, .bits = 32 }

namespace detail {

struct blandDLTensor : public DLTensor {
  public:
    /**
     * Create a new `blandDLTensor` from a DLManagedTensor (borrow the tensor from another library or framework)
     */
    blandDLTensor(DLManagedTensor other_tensor);
    /**
     * Create a new `blandDLTensor` with unitialized memory with the given shape, dtype, device, strides.
     */
    blandDLTensor(const std::vector<int64_t> &shape,
                  DLDataType                  dtype,
                  DLDevice                    device,
                  std::vector<int64_t>        strides = {});
    /**
     * Copy constructor to copy metadata while ensuring zero-copy of underlying data buffer
     */
    blandDLTensor(const blandDLTensor &other);

    /**
     * Assignment operator which ensures the data buffer is not copied but all required dynamic memory is valid
     */
    blandDLTensor &operator=(const blandDLTensor &other);

    /**
     * Create a DLManagedTensor to allow other frameworks to borrow/view this tensor with zero-copy of underlying data
     */
    DLManagedTensor *to_dlpack();

    // Internally bland can create sophisticated views with an offset per dim
    std::vector<int64_t> _offsets;

    // Memory management of dynamic buffers
    /**
     * shared_ptr to underlying data buffer ensures the data is valid as long as any tensor (borrowed or not) is using
     * it
     */
    std::shared_ptr<void> _data_ownership;
    /**
     * RAII guarantee that the shape of this tensor is always valid
     */
    std::vector<int64_t> _shape_ownership;
    /**
     * RAII guarantee that the shape of this tensor is always valid
     */
    std::vector<int64_t> _strides_ownership;
};

} // namespace detail

/**
 * `ndarray` is an array with n dimensions backed by an opaque data buffer existing on `device` interpreted as a runtime
 * `dtype`. The underlying elements are interpreted to have a given shape, strides, offset with a rich library of
 * operations to perform DSP and scientific computing. `ndarray` is backed by a `dltensor` which supports the `dlpack`
 * protocol for zero-copy interaction between C++ and python libraries supporting the dlpack protocol (numpy, cupy,
 * matplotlib, h5py, pytorch, tensorflow)
 */
class ndarray {
  public:
    /**
     * Syntactic sugar around dtypes
     */
    struct datatype : DLDataType {
        datatype(std::string_view dtype);
        datatype(DLDataType dtype);

        bool operator==(const datatype &other);
        bool operator!=(const datatype &other);

        static constexpr DLDataType float32 = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1};
        static constexpr DLDataType float64 = DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 64, .lanes = 1};
        static constexpr DLDataType int8    = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 8, .lanes = 1};
        static constexpr DLDataType int16   = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 16, .lanes = 1};
        static constexpr DLDataType int32   = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
        static constexpr DLDataType int64   = DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 64, .lanes = 1};
        static constexpr DLDataType uint8   = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1};
        static constexpr DLDataType uint16  = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 16, .lanes = 1};
        static constexpr DLDataType uint32  = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1};
        static constexpr DLDataType uint64  = DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 64, .lanes = 1};
    };

    struct dev : DLDevice {
        dev(std::string_view dev);
        dev(DLDevice);
        bool operator==(const dev &other);

        static constexpr DLDevice cpu          = DLDevice{.device_type = kDLCPU, .device_id = 0};
        static constexpr DLDevice cuda         = DLDevice{.device_type = kDLCUDA, .device_id = 0};
        static constexpr DLDevice cuda_host    = DLDevice{.device_type = kDLCUDAHost, .device_id = 0};
        static constexpr DLDevice opencl       = DLDevice{.device_type = kDLOpenCL, .device_id = 0};
        static constexpr DLDevice vulkan       = DLDevice{.device_type = kDLVulkan, .device_id = 0};
        static constexpr DLDevice metal        = DLDevice{.device_type = kDLMetal, .device_id = 0};
        static constexpr DLDevice vpi          = DLDevice{.device_type = kDLVPI, .device_id = 0};
        static constexpr DLDevice rocm         = DLDevice{.device_type = kDLROCM, .device_id = 0};
        static constexpr DLDevice rocm_host    = DLDevice{.device_type = kDLROCMHost, .device_id = 0};
        static constexpr DLDevice extdev       = DLDevice{.device_type = kDLExtDev, .device_id = 0};
        static constexpr DLDevice cuda_managed = DLDevice{.device_type = kDLCUDAManaged, .device_id = 0};
        static constexpr DLDevice oneapi       = DLDevice{.device_type = kDLOneAPI, .device_id = 0};
        static constexpr DLDevice webgpu       = DLDevice{.device_type = kDLWebGPU, .device_id = 0};
        static constexpr DLDevice hexagon      = DLDevice{.device_type = kDLHexagon, .device_id = 0};
    };

    // ndarray() = default; // warning: explicitly defaulted default constructor is implicitly deleted
    // note: default constructor of 'ndarray' is implicitly deleted because field '_tensor' has no default constructor
    ndarray(DLManagedTensor);

    /**
     * Construct an ndarray with the given dims and uninitialized memory
     */
    ndarray(std::vector<int64_t> dims, datatype dtype = datatype::float32, DLDevice device = default_device);

    /**
     * Construct an ndarray with the given shape and fill every element with the given `initial_val`
     */
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    ndarray(std::vector<int64_t> dims,
            T                    initial_val,
            datatype             dtype  = datatype::float32,
            DLDevice             device = default_device);

    template <typename T>
    T *data_ptr() const {
        return static_cast<T *>(_tensor.data);
    }

    /**
     * Return a DLManagedTensor meant to zero-copy share data to other frameworks. This is effectively
     * a `to_dlpack` method in the language of dlpack.
     *
     * \internal The DLManagedTensor holds an opaquely packaged shared_ptr as the context. That shared pointer makes
     * sure all dynamic memory is alive as long as this DLManagedTensor is valid. When the deleter method is called, the
     * shared_ptr is casted back, deleted (decreasing ref count) so that memory might be freed when no one needs it.
     */
    DLManagedTensor *get_managed_tensor();

    /**
     * Return the shape (size of each dimension) of the array
     */
    std::vector<int64_t> shape() const;

    /**
     * Return the size of the given dimension
     */
    int64_t size(int64_t axis) const;

    /**
     * Return the strides of the array in number of items
     *
     * The stride is the distance (in items) between two adjacent elements along a dimension. For example
     * a compact 1d array will have strides {1} because items are adjacent in memory. A compact 2d array
     * with size (10, 10) will have strides {10, 1} because we default to "C" (rather than Fortan) convention
     * of the last dimension being the most compact (row major). The stride of 10 means to index array[1,0]
     * the data pointer needs to be incremented 10 items (item size is determined by dtype).
     *
     * A stride operation is typically the only reason this will change from row-major C-style conventions
     */
    std::vector<int64_t> strides() const;

    std::vector<int64_t> offsets() const;

    datatype dtype() const;

    dev device() const;

    int64_t numel() const;

    int64_t ndim() const;

    template <typename T>
    T scalarize() const;

    /**
     * Return a string representation of this tensor. Format is
     * a pretty print including the datatype, shape, device and
     * formatted data
     */
    std::string repr() const;

    template <typename T>
    ndarray add(const T &b) const;
    template <typename T>
    ndarray operator+(const T &b) const;

    template <typename T>
    ndarray subtract(const T &b) const;
    template <typename T>
    ndarray operator-(const T &b) const;

    template <typename T>
    ndarray multiply(const T &b) const;
    template <typename T>
    ndarray operator*(const T &b) const;

    template <typename T>
    ndarray divide(const T &b) const;
    template <typename T>
    ndarray operator/(const T &b) const;

    ndarray reshape(const std::vector<int64_t> &new_shape);
    ndarray squeeze(int64_t squeeze_axis);
    ndarray unsqueeze(int64_t unsqueeze_axis);
    // ndarray permute(const std::vector<int64_t> &axis_order);

    template <typename... Args>
    ndarray_slice slice(Args... args);

  protected:
    // TODO: make this a ptr type so we can hide definition of blandDLTensor (could make some types of assignment and
    // sharing easier too)
    detail::blandDLTensor _tensor;
};

/**
 * An ndarray_slice is type-system syntactic sugar that allows storing results in to
 * and existing array. This is especially convenient when we've sliced an array and
 * want to store results in the slice of that array.
 */
class ndarray_slice : public ndarray {
  public:
    ndarray_slice(const ndarray &other);

    ndarray_slice &operator=(const ndarray_slice &rhs);

    ndarray_slice &operator=(const ndarray &rhs);

  protected:
    friend ndarray_slice slice(const ndarray &, int64_t, int64_t, int64_t, int64_t);
};

/**
 * Global operators to allow switching operator order. These are only enabled for arithmetic
 * types to avoid ambiguous selection on f(ndarray, ndarray)
 */

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type operator+(const T &lhs, const ndarray &rhs);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type operator-(const T &lhs, const ndarray &rhs);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type operator*(const T &lhs, const ndarray &rhs);
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, ndarray>::type operator/(const T &lhs, const ndarray &rhs);

} // namespace bland
