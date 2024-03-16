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

    copy(rhs, *this);
    return *this;
};

ndarray_slice &ndarray_slice::operator=(const ndarray &rhs) {
    copy(rhs, *this);
    return *this;
};

