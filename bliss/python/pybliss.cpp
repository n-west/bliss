
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <pybland.hpp>

#include "core/pyblisscore.hpp"
#include "file_types/pyfile_types.hpp"
#include "flaggers/pyflaggers.hpp"
#include "estimators/pyestimators.hpp"
#include "drift_search/pydrift_search.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pybliss, m) {

    bind_pycore(m);

    auto bland_module = m.def_submodule("bland", "Breakthrough Listen Arrays with N Dimensions. A multi-device ndarray library based on dlpack.");
    bind_pybland(bland_module);

    auto drift_search_module = m.def_submodule("drift_search", "integrate & search for doppler drifting signals");
    bind_pydrift_search(drift_search_module);

    auto fileio_module = m.def_submodule("io", "File I/O types");
    bind_pyfile_types(fileio_module);

    auto estimators_module = m.def_submodule("estimators", "Estimators");
    bind_pyestimators(estimators_module);

    auto flaggers_module = m.def_submodule("flaggers", "Flaggers");
    bind_pyflaggers(flaggers_module);

}
