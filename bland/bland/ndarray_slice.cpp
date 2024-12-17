#include "bland/ndarray_slice.hpp"

#include "bland/ops/ops.hpp" // bland::slice

using namespace bland;

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
    if (rhs._tensor.data == _tensor.data) {
        // assigning a slice of a slice to itself has been a source of several bugs
        _tensor._shape_ownership = rhs._tensor._shape_ownership;
        _tensor.shape = _tensor._shape_ownership.data();
        _tensor._strides_ownership = rhs._tensor._strides_ownership;
        _tensor.strides = _tensor._strides_ownership.data();
        _tensor._offsets = rhs._tensor._offsets;
    } else {
        copy(rhs, *this);
    }
    return *this;
};

ndarray_slice &ndarray_slice::operator=(const ndarray &rhs) {
    copy(rhs, *this);
    return *this;
};

