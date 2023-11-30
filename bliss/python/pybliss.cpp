#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pybliss, m) {

    auto bland_module = m.def_submodule("bland", "Breakthrough Listen Arrays with N Dimensions. A multi-device ndarray library based on dlpack.");
    

}
