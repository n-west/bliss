

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


#include <bland/ndarray.hpp>
#include "spectrumsum/spectrumsum.hpp"

#include <iostream>

namespace nb = nanobind;


int add(int a, int b) {
    return a + b;
}

int sub(int a, int b) {
    std::cout << "Subtract" << std::endl;
    return a - b;
}

bland::ndarray nb_to_bland(nb::ndarray<> t) {
    DLManagedTensor dl_x{
        .dl_tensor = {
            .data   = t.data(),
            .device = DLDevice{.device_type = DLDeviceType(t.device_type()), .device_id = t.device_id()},
            .ndim   = static_cast<int32_t>(t.ndim()),
            .dtype  = DLDataType{.code = t.dtype().code, .bits = t.dtype().bits, .lanes = t.dtype().lanes},
        }
    };
    dl_x.dl_tensor.shape = reinterpret_cast<int64_t *>(malloc(sizeof(int64_t) * t.ndim()));
    for (int dim = 0; dim < t.ndim(); ++dim) {
        dl_x.dl_tensor.shape[dim] = t.shape(dim);
    }
    dl_x.dl_tensor.strides = reinterpret_cast<int64_t *>(malloc(sizeof(int64_t) * t.ndim()));
    for (int dim = 0; dim < t.ndim(); ++dim) {
        dl_x.dl_tensor.strides[dim] = t.stride(dim);
    }
    dl_x.manager_ctx = t.handle();
    auto our_array = bland::ndarray(dl_x);

    std::cout << "Our array has data at " << our_array.data_ptr<void>() << std::endl;
    return our_array;

    // bland::detail::blandDLTensor ()
}


NB_MODULE(pyspectrumsum, m) {

    m.def("add", &add);
    m.def("sub", &sub);

    nb::enum_<bliss::spectrum_sum_method>(m, "spectrum_sum_method")
        .value("LINEAR_ROUND", bliss::spectrum_sum_method::LINEAR_ROUND)
        .value("TAYLOR_TREE", bliss::spectrum_sum_method::TAYLOR_TREE)
        .value("HOUSTON", bliss::spectrum_sum_method::HOUSTON)
        ;

    nb::class_<bliss::spectrum_sum_options>(m, "spectrum_sum_options")
    .def(nb::init<>())
    .def_rw("method", &bliss::spectrum_sum_options::method)
    .def_rw("desmear", &bliss::spectrum_sum_options::desmear)
    .def_rw("drift_range", &bliss::spectrum_sum_options::drift_range)
    ;

    m.def("spectrum_sum", [](nb::ndarray<> spectrum, bliss::spectrum_sum_options options) {
        std::cout << "*** BLISS ***: In binded spectrum sum" << std::endl;
        auto bland_spectrum = nb_to_bland(spectrum);
        std::cout << "Survived conversion" << std::endl;
        auto detection_plane = bliss::spectrum_sum(bland_spectrum, options);
        return detection_plane;
    });

    // m.def("spectrum_sum", &bliss::spectrum_sum);

}