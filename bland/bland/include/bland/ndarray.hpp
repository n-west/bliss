#pragma once

#include "ops.hpp"

#include <dlpack/dlpack.h>

#include <memory>
#include <vector>

namespace bland {

#define default_device                                                                                                 \
    DLDevice { .device_type = kDLCPU, .device_id = 0 }

#define default_dtype                                                                                                  \
    DLDataType { .code = kDLFloat, .bits = 32 }

namespace detail {

struct blandDLTensor : public DLTensor {
  public:
    // Constructors
    blandDLTensor(DLManagedTensor other_tensor);
    blandDLTensor(const std::vector<int64_t> &shape,
                  DLDataType                  dtype,
                  DLDevice                    device,
                  std::vector<int64_t>        strides = {});
    // Copy constructor
    blandDLTensor(const blandDLTensor &other);

    blandDLTensor &operator=(const blandDLTensor &other);

    DLManagedTensor *to_dlpack();

    // Internally bland can create sophisticated views with an offset per dim
    std::vector<int64_t> _offsets;

    // Memory management of dynamic buffers
    std::shared_ptr<void> _data_ownership;
    std::vector<int64_t>  _shape_ownership;
    std::vector<int64_t>  _strides_ownership;
};

} // namespace detail

class ndarray {
  public:
    // ndarray() = default; // warning: explicitly defaulted default constructor is implicitly deleted
    // note: default constructor of 'ndarray' is implicitly deleted because field '_tensor' has no default constructor
    ndarray(DLManagedTensor);
    ndarray(std::vector<int64_t> dims, DLDataType dtype = default_dtype, DLDevice device = default_device);
    template <typename T>
    ndarray(std::vector<int64_t> dims, T initial_val, DLDataType dtype, DLDevice device = default_device);

    template <typename T>
    T *data_ptr() const {
        return static_cast<T *>(_tensor.data);
    }

    DLManagedTensor *get_managed_tensor();

    std::vector<int64_t> shape() const;

    int64_t size(int64_t axis) const;

    std::vector<int64_t> strides() const;

    std::vector<int64_t> offsets() const;

    DLDataType dtype() const;

    DLDevice device() const;

    int64_t numel() const;

    int64_t ndim() const;

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
