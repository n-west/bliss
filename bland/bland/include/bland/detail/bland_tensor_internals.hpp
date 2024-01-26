#pragma once

#include <dlpack/dlpack.h>

#include <memory> // shared_ptr
#include <vector>
#include <cstdint>

namespace bland {

namespace detail {

struct blandDLTensor : public DLTensor {
  public:

    blandDLTensor();

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

} // namespace bland