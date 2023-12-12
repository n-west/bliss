#pragma once

#include "bland/bland.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

bland::ndarray nb_to_bland(nb::ndarray<> t);


void bind_pybland(nb::module_ m) {

    // Define ndarray type
    nb::class_<bland::ndarray>(m, "ndarray")
    .def(nb::init<std::vector<int64_t>>())
    .def("__repr__", &bland::ndarray::repr)
    .def("__dlpack__", [](bland::ndarray &self) {
        // Get the DLManagedTensor from arr
        DLManagedTensor *tensor = self.get_managed_tensor();
        // Create a new Python capsule containing the DLManagedTensor
        PyObject *capsule = PyCapsule_New(tensor, "dltensor", nullptr);
        return nb::capsule(capsule, nanobind::detail::steal_t{});
    })
    .def("shape", &bland::ndarray::shape)
    .def("strides", &bland::ndarray::strides)
    .def("offsets", &bland::ndarray::offsets)
    .def("numel", &bland::ndarray::numel)
    .def("ndim", &bland::ndarray::ndim)
    .def("dtype", &bland::ndarray::dtype)
    .def("device", &bland::ndarray::device)
    .def("size", &bland::ndarray::size)
    ;

    m.def("arange", [](float start, float end, float step) {
        return bland::arange(start, end, step);
    });
    m.def("linspace", [](float start, float end, size_t number_steps) {
        return bland::linspace(start, end, number_steps);
    });

    /**************
     * Arithmetic
     **************/
    m.def("add", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::add(nb_to_bland(a), nb_to_bland(b));
    });
    m.def("add", [](nb::ndarray<> a, float b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("add", [](nb::ndarray<> a, double b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("add", [](nb::ndarray<> a, int8_t b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("add", [](nb::ndarray<> a, int16_t b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("add", [](nb::ndarray<> a, int32_t b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("add", [](nb::ndarray<> a, int64_t b) {
        return bland::add(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::subtract(nb_to_bland(a), nb_to_bland(b));
    });
    m.def("subtract", [](nb::ndarray<> a, float b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, double b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, int8_t b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, int16_t b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, int32_t b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, int64_t b) {
        return bland::subtract(nb_to_bland(a), b);
    });
    m.def("subtract", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::subtract(nb_to_bland(a), nb_to_bland(b));
    });


    m.def("multiply", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::multiply(nb_to_bland(a), nb_to_bland(b));
    });
    m.def("divide", [](nb::ndarray<> a, nb::ndarray<> b) {
        return bland::divide(nb_to_bland(a), nb_to_bland(b));
    });


    /**************
     * Statistical & Reduction ops
    **************/
    m.def("sum", [](nb::ndarray<> a, std::vector<int64_t>axes={}) {
        return bland::sum(nb_to_bland(a), axes);
    }, "a"_a, "axes"_a=std::vector<int64_t>{});

    m.def("mean", [](nb::ndarray<> a, std::vector<int64_t>axes={}) {
        return bland::mean(nb_to_bland(a), axes);
    }, "a"_a, "axes"_a=std::vector<int64_t>{});
    m.def("median", [](nb::ndarray<> a, std::vector<int64_t>axes={}) {
        return bland::median(nb_to_bland(a), axes);
    }, "a"_a, "axes"_a=std::vector<int64_t>{});
    m.def("stddev", [](nb::ndarray<> a, std::vector<int64_t>axes={}) {
        return bland::stddev(nb_to_bland(a), axes);
    }, "a"_a, "axes"_a=std::vector<int64_t>{});
    m.def("standardized_moment", [](nb::ndarray<> a, int degree, std::vector<int64_t>axes={}) {
        return bland::standardized_moment(nb_to_bland(a), degree, axes);
    }, "a"_a, "degree"_a, "axes"_a=std::vector<int64_t>{});


    /**************
     * Shaping ops
    **************/
    m.def("slice", [](nb::ndarray<> a, int64_t dim, int64_t start, int64_t end, int64_t stride=1) {
        auto our_tensor_type = nb_to_bland(a);
        auto sliced = bland::slice(our_tensor_type, dim, start, end, stride);
        auto r = copy(sliced);
        return r;
    });

}


/**
 * The nanobind nb::ndarray<> type does a fantastic job accepting all kinds of ndarrays
 * including torch & numpy. I don't really know how it works, but they make it very hard
 * (impossible?) to get out a DLManagedTensor even though that must be the mechanism they get data
 * from other frameworks. Anyway, build up a DLManagedTensor to construct bland arrays from
 * those other frameworks
 * 
 * This is really a hackjob and certainly leaks memory but not an issue yet, once we start
 * with a lot of conversions and real usage we *do* need to clean this up
 * 
 * I do wonder if our calls can just accept a pycapsule type or something
*/
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
    return our_array;
}
